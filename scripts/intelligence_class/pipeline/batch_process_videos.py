# batch_process_videos.py
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import random
import cv2
from ultralytics import YOLO


# =========================
# 常量：行为检测 8 类
# =========================
BEHAVIOR_NAMES = ["dx", "dk", "tt", "zt", "js", "zl", "xt", "jz"]


# =========================
# 工具函数
# =========================

def project_root_from_this_file() -> Path:
    """
    约定脚本位置：YOLOv11/scripts/intelligence_class/batch_process_videos.py
    project_root = YOLOv11
    """
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_write_line(fp: Path, line: str) -> None:
    ensure_dir(fp.parent)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def list_view_folders(dataset_root: Path) -> List[Path]:
    """
    视角文件夹：正方视角、斜上方视角1/2、后方视角、教师视角……
    同级还有 案例/readme.txt 等；这里只选“文件夹且非案例”。
    """
    views = []
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name != "案例":
            views.append(child)
    return sorted(views, key=lambda x: x.name)


def iter_mp4_files(folder: Path) -> List[Path]:
    """
    遍历 folder 下所有 mp4（不递归；如果你数据有子目录可改成 rglob）
    """
    return sorted(folder.glob("*.mp4"))


def pick_pose_or_detect_model(project_root: Path, user_model: str) -> Path:
    """
    给原始 pose/detect 分支用（你原来的逻辑）
    4060笔记本友好：优先 n 其次 s。
    """
    if user_model:
        return (project_root / user_model).resolve()

    candidates = [
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
        "yolo11n.pt",
        "yolo11s.pt",
    ]
    for name in candidates:
        p = (project_root / name).resolve()
        if p.exists():
            print(f"[AUTO] pick model = {name}")
            return p

    raise FileNotFoundError(
        "No model found in project root. Put one of: "
        + ", ".join(candidates)
        + " or pass --model <path>"
    )


def resolve_model_path(project_root: Path, model_rel: str) -> Path:
    p = (project_root / model_rel).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return p


def open_video(video_path: Path) -> Tuple[cv2.VideoCapture, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV cannot open video: {video_path}\n"
            "可能原因：编码(HEVC/H.265) 或 OpenCV 缺少解码器。"
        )
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


def open_writer(out_path: Path, fps: float, w: int, h: int, codec: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if writer.isOpened():
        return writer

    # fallback：有些机器 mp4v 不行，试试 avc1
    if codec.lower() != "avc1":
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            return writer

    raise RuntimeError(
        f"VideoWriter open failed: {out_path}\n"
        "可以尝试：--codec avc1 或者改输出为 .avi + XVID。"
    )


def results_to_jsonable_general(r0) -> Dict[str, Any]:
    """
    原始分支（pose/detect）：
    把 ultralytics Results 转成 json 可写结构。
    """
    out: Dict[str, Any] = {}

    boxes = []
    if getattr(r0, "boxes", None) is not None and r0.boxes is not None:
        b = r0.boxes
        xyxy = b.xyxy.cpu().numpy() if b.xyxy is not None else None
        conf = b.conf.cpu().numpy() if b.conf is not None else None
        cls = b.cls.cpu().numpy() if b.cls is not None else None

        n = 0
        if xyxy is not None:
            n = xyxy.shape[0]

        for i in range(n):
            item = {
                "xyxy": [float(x) for x in xyxy[i].tolist()],
                "conf": float(conf[i]) if conf is not None else None,
                "cls": int(cls[i]) if cls is not None else None,
            }
            boxes.append(item)

    out["boxes"] = boxes

    # keypoints (pose)
    if getattr(r0, "keypoints", None) is not None and r0.keypoints is not None:
        kp = r0.keypoints
        if getattr(kp, "xy", None) is not None and kp.xy is not None:
            xy = kp.xy.cpu().numpy()
            out["keypoints_xy"] = [[[float(v) for v in pt] for pt in person] for person in xy.tolist()]
        if getattr(kp, "conf", None) is not None and kp.conf is not None:
            kc = kp.conf.cpu().numpy()
            out["keypoints_conf"] = [[float(v) for v in person] for person in kc.tolist()]

    return out


def results_to_jsonable_behavior(r0) -> Dict[str, Any]:
    """
    behavior_det 分支专用：
    输出 boxes + (cls->name) 显式写入，方便后续做 timeline/统计。
    """
    dets = []
    if getattr(r0, "boxes", None) is not None and r0.boxes is not None and len(r0.boxes) > 0:
        b = r0.boxes
        xyxy = b.xyxy.cpu().numpy() if b.xyxy is not None else None
        conf = b.conf.cpu().numpy() if b.conf is not None else None
        cls = b.cls.cpu().numpy() if b.cls is not None else None

        if xyxy is not None:
            n = xyxy.shape[0]
            for i in range(n):
                ci = int(cls[i]) if cls is not None else -1
                name = BEHAVIOR_NAMES[ci] if 0 <= ci < len(BEHAVIOR_NAMES) else str(ci)
                dets.append({
                    "cls": ci,
                    "name": name,
                    "conf": float(conf[i]) if conf is not None else None,
                    "xyxy": [float(x) for x in xyxy[i].tolist()],
                })

    return {"detections": dets}


def done_outputs_exist(out_overlay: Path, out_jsonl: Path) -> bool:
    """
    更稳的“完成判断”：两个输出都存在且非空才算完成。
    """
    if not out_jsonl.exists() or out_jsonl.stat().st_size <= 1024:
        return False
    if not out_overlay.exists() or out_overlay.stat().st_size <= 1024:
        return False
    return True


# =========================
# 核心处理
# =========================

def process_one_video(
    model: YOLO,
    in_video: Path,
    out_overlay: Path,
    out_jsonl: Path,
    args,
    view_name: str,
    log_path: Path,
    task_name: str,
) -> Tuple[bool, str]:
    """
    返回 (success, status): status in {"ok","skipped"}
    """

    # 跳过策略：只有当 overlay+jsonl 都存在且非空才跳过（更靠谱）
    if not args.force:
        if args.viz:
            if done_outputs_exist(out_overlay, out_jsonl):
                msg = f"[SKIP] {task_name} | {view_name} | {in_video.name}"
                safe_write_line(log_path, msg)
                return True, "skipped"
        else:
            # 不输出 overlay 的情况下，只要 jsonl 存在且非空就跳过
            if out_jsonl.exists() and out_jsonl.stat().st_size > 1024:
                msg = f"[SKIP] {task_name} | {view_name} | {in_video.name} (jsonl-only)"
                safe_write_line(log_path, msg)
                return True, "skipped"

    cap, fps, w, h = open_video(in_video)

    ensure_dir(out_overlay.parent)
    ensure_dir(out_jsonl.parent)

    writer = None
    if args.viz:
        # stride 会导致输出 fps 变化（避免视频看起来快进）
        out_fps = fps / max(int(args.stride), 1)
        writer = open_writer(out_overlay, out_fps, w, h, args.codec)

    # 写 meta（每个视频一个）
    meta = {
        "task": task_name,
        "view": view_name,
        "input_video": str(in_video),
        "output_overlay": str(out_overlay) if args.viz else "",
        "output_jsonl": str(out_jsonl),
        "fps": fps,
        "width": w,
        "height": h,
        "stride": int(args.stride),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "device": args.device,
        "half": bool(args.half) if args.device != -1 else False,
        "model": str(args.model_path_used),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 重跑时清理 jsonl，避免脏拼接
    if out_jsonl.exists():
        out_jsonl.unlink()

    with open(out_jsonl, "w", encoding="utf-8") as jf:
        frame_idx = 0
        processed = 0
        t0 = time.time()

        stride = max(int(args.stride), 1)
        max_frames = int(args.max_frames)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx % stride != 0:
                continue

            results = model.predict(
                frame,
                verbose=False,
                device=args.device,
                half=bool(args.half) if args.device != -1 else False,
                imgsz=int(args.imgsz),
                conf=float(args.conf),
            )

            r0 = results[0]

            if task_name == "behavior_det":
                pred_obj = results_to_jsonable_behavior(r0)
            else:
                pred_obj = results_to_jsonable_general(r0)

            record = {
                "task": task_name,
                "view": view_name,
                "video": in_video.name,
                "frame_idx": frame_idx,
                "time_sec": float(frame_idx / fps) if fps > 1e-6 else None,
                "w": int(w),
                "h": int(h),
                "pred": pred_obj,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.viz and writer is not None:
                annotated = r0.plot()
                writer.write(annotated)

            processed += 1

            if processed % 60 == 0:
                dt = time.time() - t0
                avg_fps = processed / dt if dt > 1e-6 else 0.0
                print(f"[INFO] {task_name} | {view_name} | {in_video.name} | processed={processed} | avg_fps={avg_fps:.2f}")

            if max_frames > 0 and processed >= max_frames:
                print(f"[INFO] {task_name} | {view_name} | {in_video.name} | reach max_frames={max_frames}, stop.")
                break

    cap.release()
    if writer is not None:
        writer.release()

    meta_path = out_jsonl.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    safe_write_line(log_path, f"[OK] {task_name} | {view_name} | {in_video.name}")
    return True, "ok"


def make_output_paths(out_view_dir: Path, stem: str, task: str) -> Tuple[Path, Path]:
    """
    输出命名规则：
      - 原始(pose/detect)： stem.jsonl + stem_overlay.mp4
      - behavior_det：      stem_behavior.jsonl + stem_behavior_overlay.mp4
    """
    if task == "behavior_det":
        out_jsonl = out_view_dir / f"{stem}_behavior.jsonl"
        out_overlay = out_view_dir / f"{stem}_behavior_overlay.mp4"
    else:
        out_jsonl = out_view_dir / f"{stem}.jsonl"
        out_overlay = out_view_dir / f"{stem}_overlay.mp4"
    return out_overlay, out_jsonl


def main():
    project_root = project_root_from_this_file()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/智慧课堂学生行为数据集", help="relative to project root")
    parser.add_argument("--out_root", type=str, default="output/智慧课堂学生行为数据集", help="relative to project root")

    # 只跑某个视角子文件夹名（精确匹配）
    parser.add_argument("--subset", type=str, default="", help="only process this view folder name")
    # 只跑一个视频（相对路径）
    parser.add_argument("--video", type=str, default="", help="only process one video (relative to project root)")

    # 任务分支
    parser.add_argument("--model_task", type=str, default="pose", choices=["pose", "behavior_det"],
                        help="pose: original pipeline (pose/detect). behavior_det: 8-class behavior detector best.pt")

    # 原始分支模型（可选）
    parser.add_argument("--model", type=str, default="", help="model path relative to project root (optional auto-pick)")

    # behavior_det 分支模型
    parser.add_argument("--behavior_model", type=str, default="runs/detect/train/weights/best.pt",
                        help="behavior detector model path relative to project root")

    # 推理参数
    parser.add_argument("--device", type=int, default=0, help="GPU id, use -1 for CPU")
    parser.add_argument("--half", type=int, default=1, help="use FP16 on CUDA (1/0)")
    parser.add_argument("--imgsz", type=int, default=832, help="inference image size (640/832/960 recommended)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")

    # 输出与性能
    parser.add_argument("--viz", type=int, default=1, help="write overlay video (1/0)")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame (>=1)")
    parser.add_argument("--max_frames", type=int, default=0, help="stop after N processed frames (0 = no limit)")
    parser.add_argument("--codec", type=str, default="mp4v", help="mp4v / avc1 ...")

    # 断点/重跑
    parser.add_argument("--force", type=int, default=0, help="force re-run even if outputs exist (1/0)")

    args = parser.parse_args()
    args.viz = int(args.viz)
    args.force = int(args.force)
    args.half = int(args.half)

    data_root = (project_root / args.data_root).resolve()
    out_root = (project_root / args.out_root).resolve()
    ensure_dir(out_root)

    log_path = out_root / "process_log.txt"
    failed_path = out_root / "failed_videos.txt"

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # ========== 选择/加载模型（只加载一次） ==========
    if args.model_task == "behavior_det":
        model_path = resolve_model_path(project_root, args.behavior_model)
        task_name = "behavior_det"
    else:
        model_path = pick_pose_or_detect_model(project_root, args.model)
        task_name = "pose"

    args.model_path_used = str(model_path)

    print(f"[INFO] Project Root: {project_root}")
    print(f"[INFO] Data Root   : {data_root}")
    print(f"[INFO] Out Root    : {out_root}")
    print(f"[INFO] Task        : {task_name}")
    print(f"[INFO] Model       : {model_path}")
    print(f"[INFO] Params      : device={args.device}, half={args.half}, imgsz={args.imgsz}, conf={args.conf}, stride={args.stride}, viz={args.viz}")

    model = YOLO(str(model_path))

    total = 0
    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0
    failed_list: List[str] = []

    # ========== 单视频模式 ==========
    if args.video:
        one = (project_root / args.video).resolve()
        if not one.exists():
            raise FileNotFoundError(f"--video not found: {one}")

        view_name = one.parent.name
        out_view_dir = out_root / view_name
        ensure_dir(out_view_dir)

        stem = one.stem
        out_overlay, out_jsonl = make_output_paths(out_view_dir, stem, task_name)

        total = 1
        try:
            success, status = process_one_video(
                model=model,
                in_video=one,
                out_overlay=out_overlay,
                out_jsonl=out_jsonl,
                args=args,
                view_name=view_name,
                log_path=log_path,
                task_name=task_name,
            )
            if success and status == "ok":
                ok_cnt += 1
            elif success and status == "skipped":
                skip_cnt += 1
        except Exception as e:
            fail_cnt += 1
            failed_list.append(str(one))
            safe_write_line(failed_path, str(one))
            safe_write_line(log_path, f"[FAIL] {task_name} | {view_name} | {one.name} | {repr(e)}")
            print(f"[ERROR] {one} -> {e}")

    # ========== 批处理模式 ==========
    else:
        views = list_view_folders(data_root)
        if args.subset:
            views = [v for v in views if v.name == args.subset]
            if not views:
                raise ValueError(f"--subset={args.subset} not found under {data_root}")

        for view_dir in views:
            view_name = view_dir.name
            out_view_dir = out_root / view_name
            ensure_dir(out_view_dir)

            mp4s = iter_mp4_files(view_dir)
            if not mp4s:
                print(f"[WARN] no mp4 found in {view_dir}")
                continue

            print(f"\n[VIEW] {view_name} | videos={len(mp4s)} | task={task_name}")
            for vp in mp4s:
                total += 1
                stem = vp.stem
                out_overlay, out_jsonl = make_output_paths(out_view_dir, stem, task_name)

                try:
                    success, status = process_one_video(
                        model=model,
                        in_video=vp,
                        out_overlay=out_overlay,
                        out_jsonl=out_jsonl,
                        args=args,
                        view_name=view_name,
                        log_path=log_path,
                        task_name=task_name,
                    )
                    if success and status == "ok":
                        ok_cnt += 1
                    elif success and status == "skipped":
                        skip_cnt += 1
                except Exception as e:
                    fail_cnt += 1
                    failed_list.append(str(vp))
                    safe_write_line(failed_path, str(vp))
                    safe_write_line(log_path, f"[FAIL] {task_name} | {view_name} | {vp.name} | {repr(e)}")
                    print(f"[ERROR] {task_name} | {view_name} | {vp.name} -> {e}")

    # ========== 汇总 ==========
    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"  task    : {task_name}")
    print(f"  total   : {total}")
    print(f"  success : {ok_cnt}")
    print(f"  skipped : {skip_cnt}")
    print(f"  failed  : {fail_cnt}")
    if failed_list:
        print("  failed list:")
        for x in failed_list[:20]:
            print("   -", x)
        if len(failed_list) > 20:
            print(f"   ... and {len(failed_list) - 20} more")
        print(f"[INFO] full failed list saved to: {failed_path}")
    print(f"[INFO] process log saved to     : {log_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
