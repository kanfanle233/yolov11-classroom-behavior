# scripts/intelligence_class/pipeline/01_run_single_video.py
# 目标：
# 1) 兼容你“新结构”的 pipeline 脚本位置
# 2) 在 out_dir 里同时输出两类“对前端友好”的文件 (Legacy + Behavior)
# 3) 【双重验证核心】：集成 ASR (06) 和 对齐模块 (xx_align)
# 4) 【产品化交付】：集成 Summary 和 Aggregate
# 5) 【GitHub Pages支持】：集成 Static Projection 生成 (Step 8)
# 6) 【Web兼容性修复】：集成 FFmpeg H.264 转码，确保生成视频可播放
# 7) 【Timeline增强】：集成 Timeline Viz 生成 (Step 9)

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


# =====================================================================================
# 0) 动态加载 pathing.py
# =====================================================================================

def _guess_project_root_from_this_file() -> Path:
    """pipeline -> intelligence_class -> scripts -> YOLOv11(root)"""
    this_file = Path(__file__).resolve()
    candidate = this_file.parent
    for _ in range(8):
        if (candidate / "data").exists() or (candidate / "scripts").exists():
            return candidate
        candidate = candidate.parent
    return this_file.parents[3]  # fallback


def _load_pathing(project_root: Path):
    cand = [
        project_root / "scripts" / "intelligence_class" / "_utils" / "pathing.py",
        project_root / "scripts" / "intelligence class" / "_utils" / "pathing.py",
    ]
    for p in cand:
        if p.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("ic_pathing", str(p))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            return mod

    class MockPathing:
        @staticmethod
        def find_project_root(_p):  # noqa
            return project_root

        @staticmethod
        def resolve_under_project(root: Path, path: str):
            pp = Path(path)
            return pp if pp.is_absolute() else root / pp

    print("[WARN] pathing.py not found, using fallback.")
    return MockPathing


PROJECT_ROOT_FALLBACK = _guess_project_root_from_this_file()
PATHING = _load_pathing(PROJECT_ROOT_FALLBACK)
find_project_root = PATHING.find_project_root
resolve_under_project = PATHING.resolve_under_project


# =====================================================================================
# 1) 结构与路径
# =====================================================================================

def resolve_paths() -> Tuple[Path, Path, Path, Path]:
    """解析 project_root + scripts_dir + pipeline_dir + tools_dir"""
    current_file = Path(__file__).resolve()
    project_root = find_project_root(current_file) or PROJECT_ROOT_FALLBACK

    scripts_dir = project_root / "scripts"

    pipeline_dir = scripts_dir / "intelligence_class" / "pipeline"
    if not pipeline_dir.exists():
        pipeline_dir = scripts_dir / "intelligence class" / "pipeline"

    tools_dir = scripts_dir / "intelligence_class" / "tools"
    if not tools_dir.exists():
        tools_dir = scripts_dir / "intelligence class" / "tools"

    return project_root, scripts_dir, pipeline_dir, tools_dir


def check_script_exists(script_path: Path) -> bool:
    if not script_path.exists():
        print(f"\n[警告] 找不到脚本文件: {script_path}")
        return False
    return True


def _ensure_dir(p: Path, dry_run: bool) -> None:
    if dry_run:
        return
    p.mkdir(parents=True, exist_ok=True)


def _pick_best_case_det_model(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "models" / "best.pt",
        project_root / "runs" / "detect" / "case_yolo_train" / "weights" / "best.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    best_by_mtime: Optional[Path] = None
    best_mtime = -1.0
    detect_root = project_root / "runs" / "detect"
    if detect_root.exists():
        for p in detect_root.rglob("best.pt"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime > best_mtime:
                best_mtime = mtime
                best_by_mtime = p
    return best_by_mtime


def _pick_best_pose_model(project_root: Path) -> Optional[Path]:
    candidates = [
        project_root / "yolo11l-pose.pt",
        project_root / "yolo11m-pose.pt",
        project_root / "yolo11s-pose.pt",
        project_root / "yolo11n-pose.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    best_by_mtime: Optional[Path] = None
    best_mtime = -1.0
    for p in project_root.glob("*-pose.pt"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best_by_mtime = p
    return best_by_mtime


# =====================================================================================
# 1.5) 视频转码工具
# =====================================================================================

def _convert_to_h264(src: Path, dst: Path):
    """
    用 ffmpeg 转码到 H.264（yuv420p），更适配浏览器播放。
    失败则保留 mp4v 文件，并尝试拷贝一份到目标路径。
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "veryfast",
        str(dst),
    ]

    print(f"[转码] 正在转换为 H.264: {dst.name} ...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[转码] 成功: {dst.name}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        is_missing = isinstance(e, FileNotFoundError)
        reason = "FFmpeg 未安装" if is_missing else f"报错 (Exit: {getattr(e, 'returncode', '?')})"
        print(f"[警告] H.264 转码失败 ({reason})。保留 mp4v 文件。")
        if dst.exists():
            dst.unlink()
        if src.exists():
            shutil.copy2(src, dst)


# =====================================================================================
# 2) Step 0: Case Behavior Detector (best.pt)
# =====================================================================================

CASE_NAMES = ["dx", "dk", "tt", "zt", "js", "zl", "xt", "jz"]


def get_video_info(video_path: Path, fps_fallback: float = 25.0) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_use = float(fps) if fps and fps > 0 else float(fps_fallback)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps_use, n, w, h


def run_case_det(
    video_path: Path,
    out_jsonl: Path,
    model_path: Path,
    conf: float,
    fps_fallback: float,
    dry_run: bool,
    skip_existing: bool,
) -> Tuple[bool, str]:
    print("\n" + "=" * 60)
    print("[执行] 步骤 0: 行为检测 (Case Behavior Detector)")
    print(f"      视频路径: {video_path}")
    print(f"      模型路径: {model_path}")
    print(f"      置信度  : {conf}")
    print("=" * 60)

    if skip_existing and out_jsonl.exists() and out_jsonl.stat().st_size > 128:
        print(f"[跳过] 行为检测结果已存在: {out_jsonl}")
        return True, "Skip Existing"

    if dry_run:
        return True, "Dry Run"

    if not model_path.exists():
        return False, f"找不到行为检测模型: {model_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"无法打开视频: {video_path}"

    fps_use, *_ = get_video_info(video_path, fps_fallback)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    frame_idx = 0
    try:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                res = model(frame, conf=float(conf), verbose=False)[0]

                dets = []
                boxes = getattr(res, "boxes", None)
                if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy().tolist()
                    cls_ids = boxes.cls.cpu().numpy().tolist()
                    confs = boxes.conf.cpu().numpy().tolist()

                    for bb, cid, c in zip(xyxy, cls_ids, confs):
                        cid_int = int(cid)
                        label = CASE_NAMES[cid_int] if 0 <= cid_int < len(CASE_NAMES) else "unknown"
                        dets.append(
                            {
                                "xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                                "cls_id": cid_int,
                                "label": label,
                                "conf": float(c),
                            }
                        )

                rec = {
                    "frame_idx": frame_idx,
                    "time_sec": float(frame_idx / fps_use) if fps_use > 0 else None,
                    "fps": fps_use,
                    "dets": dets,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                frame_idx += 1

    finally:
        cap.release()

    return True, "Success"


# =====================================================================================
# 3) 子进程 runner
# =====================================================================================

def run_step(
    cmd: List[str],
    step_name: str,
    dry_run: bool = False,
    log_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    print("\n" + "=" * 60)
    print(f"[执行] {step_name}")
    print(f"      CMD: {' '.join(cmd)}")
    print("=" * 60)

    if dry_run:
        return True, "Dry Run"

    try:
        if log_dir is None:
            subprocess.run(cmd, check=True)
            return True, "Success"

        log_dir.mkdir(parents=True, exist_ok=True)
        safe_name = step_name.replace(" ", "_").replace(":", "_")
        log_path = log_dir / f"{safe_name}.log"
        with log_path.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        msg = f"步骤执行失败，退出码 {e.returncode}."
        print(f"\n❌ [错误] {step_name} 失败！(Exit: {e.returncode})")
        return False, msg
    except Exception as e:
        print(f"\n❌ [异常] {str(e)}")
        return False, str(e)


# =====================================================================================
# 4) 输出打包 (Export)
# =====================================================================================

def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_json(path: Path, obj: Any, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _copy_jsonl(src: Path, dst: Path, dry_run: bool, skip_existing: bool) -> None:
    if skip_existing and dst.exists() and dst.stat().st_size > 128:
        return
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _overlay_from_case_det(
    video_path: Path,
    case_det_jsonl: Path,
    out_mp4: Path,
    fps_fallback: float,
    dry_run: bool,
    skip_existing: bool,
) -> None:
    if skip_existing and out_mp4.exists() and out_mp4.stat().st_size > 1024:
        return
    if dry_run:
        return

    mp4v_path = out_mp4.with_name(f"{out_mp4.stem}_mp4v.mp4")
    print(f"[绘制] 正在生成行为叠加视频 (mp4v): {mp4v_path.name}")

    fps_use, _, w, h = get_video_info(video_path, fps_fallback)

    det_map: Dict[int, List[Dict[str, Any]]] = {}
    for rec in _read_jsonl(case_det_jsonl):
        fi = int(rec.get("frame_idx", 0))
        det_map[fi] = rec.get("dets", []) or []

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(mp4v_path), fourcc, fps_use, (w, h))

    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = det_map.get(fi, [])
            for d in dets:
                x1, y1, x2, y2 = d.get("xyxy", [0, 0, 0, 0])
                label = str(d.get("label", ""))
                conf = float(d.get("conf", 0.0))
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, (255, 200, 0), 2)
                txt = f"{label} {conf:.2f}"
                cv2.putText(
                    frame, txt,
                    (p1[0], max(18, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2
                )
            vw.write(frame)
            fi += 1
    finally:
        cap.release()
        vw.release()

    _convert_to_h264(mp4v_path, out_mp4)


def export_bundles(
    *,
    video_path: Path,
    out_dir: Path,
    case_id: str,
    video_id: str,
    fps_fallback: float,
    tracks_jsonl: Path,
    case_det_jsonl: Optional[Path],
    dry_run: bool,
    skip_existing: bool,
    export_legacy: bool,
    export_behavior: bool,
    make_overlays: bool,
    view_name: Optional[str] = None,
    pose_model: Optional[str] = None,
    case_det_model: Optional[str] = None,
) -> None:
    fps_use, n_frames, w, h = get_video_info(video_path, fps_fallback)

    legacy_jsonl = out_dir / f"{case_id}.jsonl"
    legacy_meta = out_dir / f"{case_id}.meta.json"
    legacy_overlay = out_dir / f"{case_id}_overlay.mp4"
    legacy_summary = out_dir / f"{case_id}_summary.json"

    beh_jsonl = out_dir / f"{case_id}_behavior.jsonl"
    beh_meta = out_dir / f"{case_id}_behavior.meta.json"
    beh_overlay = out_dir / f"{case_id}_behavior_overlay.mp4"

    if export_behavior and case_det_jsonl and case_det_jsonl.exists():
        _copy_jsonl(case_det_jsonl, beh_jsonl, dry_run=dry_run, skip_existing=skip_existing)
        _write_json(
            beh_meta,
            {
                "video_id": video_id,
                "case_id": case_id,
                "view": view_name,
                "video_path": str(video_path),
                "fps": fps_use,
                "frames": n_frames,
                "labels": CASE_NAMES,
                "source_case_det": str(case_det_jsonl),
                "pose_model": pose_model,
                "case_det_model": case_det_model,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            dry_run=dry_run,
        )
        if make_overlays:
            _overlay_from_case_det(video_path, case_det_jsonl, beh_overlay, fps_fallback, dry_run, skip_existing)

    if export_legacy and tracks_jsonl.exists():
        _copy_jsonl(tracks_jsonl, legacy_jsonl, dry_run=dry_run, skip_existing=skip_existing)
        _write_json(
            legacy_meta,
            {
                "video_id": video_id,
                "case_id": case_id,
                "view": view_name,
                "video_path": str(video_path),
                "fps": fps_use,
                "frames": n_frames,
                "width": w,
                "height": h,
                "source_tracks": str(tracks_jsonl),
                "pose_model": pose_model,
                "case_det_model": case_det_model,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            dry_run=dry_run,
        )
        # 你原来有轨迹叠加（含骨架绘制）那一套；为了“最小可落地”，这里先不强行画骨架
        # 如需恢复原轨迹overlay，可把你原来的 _overlay_from_tracks 整段粘回即可。

    _write_json(legacy_summary, {"info": "See xx_summarize_case.py output for details"}, dry_run=dry_run)


# =====================================================================================
# 5) 主流程
# =====================================================================================

def _normalize_case_id_from_video(video_p: Path) -> str:
    stem = video_p.stem
    # 001 / 0001 / 01 -> 尽量统一成 001 / 006 这种三位
    if stem.isdigit():
        return f"{int(stem):03d}"
    return stem


def run_single_video(
    *,
    video_path: str,
    video_id: str,
    out_dir: str,
    case_id: Optional[str],
    view_name: Optional[str],
    fps: float = 25.0,
    dry_run: bool = False,
    skip_existing: bool = True,
    # case_det
    case_det_model: str = "runs/detect/case_yolo_train/weights/best.pt",
    case_conf: float = 0.25,
    case_det: int = 1,
    # toggles
    run_pose: int = 1,
    run_track: int = 1,
    run_actions: int = 1,
    run_asr: int = 1,
    run_align: int = 1,
    export_legacy: int = 1,
    export_behavior: int = 1,
    make_overlays: int = 1,
    # reporting
    run_summarize: int = 1,
    run_aggregate: int = 1,
    run_projection: int = 1,
    # timeline
    run_timeline: int = 1,
    timeline_source: str = "behavior",                 # auto|actions|behavior
    timeline_tracks: str = "pose_tracks_smooth.jsonl", # 相对 case_dir
    # short video + logging
    short_video: int = 0,
    log_dir: Optional[str] = None,
    # models
    pose_model: str = "yolo11s-pose.pt",
    asr_model: str = "base",
) -> Dict[str, Any]:
    project_root, scripts_dir, pipeline_dir, tools_dir = resolve_paths()

    video_p = Path(video_path).resolve()
    out_p = Path(out_dir).resolve()
    _ensure_dir(out_p, dry_run)

    if not case_id:
        case_id = _normalize_case_id_from_video(video_p)

    # Outputs
    path_pose_jsonl = out_p / "pose_keypoints_v2.jsonl"
    path_track_jsonl = out_p / "pose_tracks_smooth.jsonl"
    path_track_kpts_jsonl = out_p / "pose_tracks_smooth_kpts.jsonl"
    path_actions_jsonl = out_p / "actions.jsonl"
    path_transcript_jsonl = out_p / "transcript.jsonl"
    path_case_det_jsonl = out_p / "case_det.jsonl"

    # Scripts
    script_pose = pipeline_dir / "02_export_keypoints_jsonl.py"
    script_track = pipeline_dir / "03_track_and_smooth_v2.py"
    script_attach = pipeline_dir / "03b_attach_keypoints.py"
    script_actions = pipeline_dir / "04_action_rules.py"
    script_asr = pipeline_dir / "06_run_whisper_asr.py"

    script_align = tools_dir / "xx_align_multimodal.py"
    script_sum = tools_dir / "xx_summarize_case.py"
    script_agg = tools_dir / "xx_aggregate_dataset_report.py"
    script_proj = tools_dir / "xx_generate_static_projections.py"
    script_timeline = tools_dir / "xx_generate_timeline_viz.py"

    python_exe = sys.executable
    log_dir_path = Path(log_dir).resolve() if log_dir else (out_p / "logs")
    if dry_run:
        log_dir_path = None

    if int(short_video) == 1:
        track_min_frames = 10
        track_max_miss = 6
        action_min_track_frames = 10
        action_raise_hand_sec = 0.3
        action_head_down_sec = 0.4
        action_stand_sec = 0.4
    else:
        track_min_frames = 30
        track_max_miss = 10
        action_min_track_frames = 30
        action_raise_hand_sec = 0.5
        action_head_down_sec = 0.7
        action_stand_sec = 0.7

    used_case_det_model: Optional[str] = None
    used_pose_model: Optional[str] = None

    # ========= Step 0: Case Det (best.pt) =========
    if int(case_det) == 1:
        model_case = None
        if str(case_det_model).strip():
            model_case = resolve_under_project(project_root, str(case_det_model))
            if not model_case.exists():
                print(f"[WARN] case_det_model not found: {model_case}")
                model_case = None
        if model_case is None:
            model_case = _pick_best_case_det_model(project_root)
        if model_case is None:
            return {"status": "failed", "error": "case_det_model not found", "video_id": video_id}
        used_case_det_model = str(model_case)
        ok, msg = run_case_det(video_p, path_case_det_jsonl, model_case, float(case_conf), fps, dry_run, skip_existing)
        if not ok:
            return {"status": "failed", "error": msg, "video_id": video_id}

    # ========= Step 1: Pose =========
    if int(run_pose) == 1 and check_script_exists(script_pose):
        pose_model_path = resolve_under_project(project_root, pose_model) if pose_model else None
        if pose_model_path and not pose_model_path.exists():
            print(f"[WARN] pose_model not found: {pose_model_path}")
            pose_model_path = None
        if pose_model_path is None:
            pose_model_path = _pick_best_pose_model(project_root)
        if pose_model_path is None:
            return {"status": "failed", "error": "pose_model not found", "video_id": video_id}
        used_pose_model = str(pose_model_path)
        cmd = [
            python_exe, str(script_pose),
            "--video", str(video_p),
            "--out", str(path_pose_jsonl),
            "--model", str(pose_model_path),
        ]
        ok, msg = run_step(cmd, "步骤 1: 姿态估计", dry_run, log_dir_path)
        if not ok:
            return {"status": "failed", "error": msg, "video_id": video_id}

    # ========= Step 2: Track =========
    if int(run_track) == 1 and check_script_exists(script_track) and check_script_exists(script_attach):
        ok, msg = run_step(
            [
                python_exe,
                str(script_track),
                "--in",
                str(path_pose_jsonl),
                "--out",
                str(path_track_jsonl),
                "--min_frames",
                str(track_min_frames),
                "--max_miss",
                str(track_max_miss),
            ],
            "步骤 2: 轨迹跟踪",
            dry_run,
            log_dir_path,
        )
        if not ok:
            return {"status": "failed", "error": msg, "video_id": video_id}

        ok, msg = run_step(
            [
                python_exe, str(script_attach),
                "--pose", str(path_pose_jsonl),
                "--tracks", str(path_track_jsonl),
                "--out", str(path_track_kpts_jsonl),
                "--iou_thr", "0.5",
            ],
            "步骤 2.5: 关联关键点",
            dry_run,
            log_dir_path,
        )
        if not ok:
            return {"status": "failed", "error": msg, "video_id": video_id}

    # ========= Step 3: Actions =========
    if int(run_actions) == 1 and check_script_exists(script_actions):
        ok, msg = run_step(
            [
                python_exe,
                str(script_actions),
                "--in",
                str(path_track_kpts_jsonl),
                "--out",
                str(path_actions_jsonl),
                "--fps",
                str(float(fps)),
                "--raise_hand_sec",
                str(action_raise_hand_sec),
                "--head_down_sec",
                str(action_head_down_sec),
                "--stand_sec",
                str(action_stand_sec),
                "--min_track_frames",
                str(action_min_track_frames),
            ],
            "步骤 3: 动作规则",
            dry_run,
            log_dir_path,
        )
        if not ok:
            return {"status": "failed", "error": msg, "video_id": video_id}

    # ========= Step 4: ASR =========
    if int(run_asr) == 1 and check_script_exists(script_asr):
        ok, msg = run_step(
            [
                python_exe,
                str(script_asr),
                "--video",
                str(video_p),
                "--out_dir",
                str(out_p),
                "--model",
                str(asr_model),
                "--skip_on_error",
            ],
            "步骤 4: 语音转写",
            dry_run,
            log_dir_path,
        )
        if not ok:
            return {"status": "failed", "error": msg, "video_id": video_id}

    # ========= Export =========
    print("\n" + "=" * 60 + "\n[导出] 正在生成结果包...\n" + "=" * 60)
    export_bundles(
        video_path=video_p,
        out_dir=out_p,
        case_id=case_id,
        video_id=video_id,
        fps_fallback=float(fps),
        tracks_jsonl=path_track_kpts_jsonl if path_track_kpts_jsonl.exists() else path_track_jsonl,
        case_det_jsonl=path_case_det_jsonl if path_case_det_jsonl.exists() else None,
        dry_run=dry_run,
        skip_existing=skip_existing,
        export_legacy=bool(int(export_legacy)),
        export_behavior=bool(int(export_behavior)),
        make_overlays=bool(int(make_overlays)),
        view_name=view_name,
        pose_model=used_pose_model or (str(pose_model) if pose_model else None),
        case_det_model=used_case_det_model or (str(case_det_model) if case_det_model else None),
    )

    # ========= Step 5: Align =========
    if int(run_align) == 1 and check_script_exists(script_align):
        if path_transcript_jsonl.exists() or path_actions_jsonl.exists() or path_case_det_jsonl.exists():
            ok, msg = run_step(
                [python_exe, str(script_align), "--case_dir", str(out_p), "--fps", str(float(fps))],
                "步骤 5: 多模态对齐",
                dry_run,
                log_dir_path,
            )
            if not ok:
                return {"status": "failed", "error": msg, "video_id": video_id}

    # ========= Step 6-8: Reports =========
    if int(run_summarize) == 1 and check_script_exists(script_sum):
        cmd = [
            python_exe,
            str(script_sum),
            "--case_dir",
            str(out_p),
            "--case_id",
            str(case_id),
            "--overwrite",
            "1",
        ]
        if int(short_video) == 1:
            cmd += ["--short_video", "1"]
        run_step(
            cmd,
            "步骤 6: 案例报告",
            dry_run,
            log_dir_path,
        )

    if int(run_aggregate) == 1 and check_script_exists(script_agg):
        run_step(
            [python_exe, str(script_agg), "--ds_root", str(out_p.parent.parent), "--views", str(out_p.parent.name)],
            "步骤 7: 聚合报表",
            dry_run,
            log_dir_path,
        )

    if int(run_projection) == 1 and check_script_exists(script_proj):
        run_step(
            [python_exe, "-W", "ignore", str(script_proj), "--root", str(out_p.parent.parent), "--target_case", str(out_p)],
            "步骤 8: 静态投影",
            dry_run,
            log_dir_path,
        )

    # ========= Step 9: Timeline Viz =========
    if int(run_timeline) == 1 and check_script_exists(script_timeline):
        cmd = [
            python_exe, str(script_timeline),
            "--case_dir", str(out_p),
            "--fps", str(float(fps)),
            "--source", str(timeline_source),
        ]

        tt = str(timeline_tracks).strip()
        if tt:
            tracks_path = Path(tt)
            # ✅ 最关键修复：相对路径按 case_dir（out_p）来拼
            if not tracks_path.is_absolute():
                tracks_path = out_p / tracks_path
            cmd += ["--tracks", str(tracks_path)]

        run_step(cmd, "步骤 9: 生成时间轴可视化", dry_run, log_dir_path)

    return {"status": "success", "video_id": video_id, "case_id": case_id, "out_dir": str(out_p)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--case_id", default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", type=int, default=1)

    # models
    parser.add_argument("--case_det_model", default="runs/detect/case_yolo_train/weights/best.pt")
    parser.add_argument("--case_conf", type=float, default=0.25)
    parser.add_argument("--pose_model", default="yolo11s-pose.pt")
    parser.add_argument("--asr_model", default="base")

    # toggles
    parser.add_argument("--case_det", type=int, default=1)
    parser.add_argument("--run_pose", type=int, default=1)
    parser.add_argument("--run_track", type=int, default=1)
    parser.add_argument("--run_actions", type=int, default=1)
    parser.add_argument("--run_asr", type=int, default=1)
    parser.add_argument("--run_align", type=int, default=1)
    parser.add_argument("--export_legacy", type=int, default=1)
    parser.add_argument("--export_behavior", type=int, default=1)
    parser.add_argument("--make_overlays", type=int, default=1)
    parser.add_argument("--run_summarize", type=int, default=1)
    parser.add_argument("--run_aggregate", type=int, default=1)
    parser.add_argument("--run_projection", type=int, default=1)

    # timeline
    parser.add_argument("--run_timeline", type=int, default=1)
    parser.add_argument("--timeline_source", default="behavior", choices=["auto", "actions", "behavior"])
    parser.add_argument("--timeline_tracks", default="pose_tracks_smooth.jsonl")
    parser.add_argument("--short_video", type=int, default=0, help="optimize thresholds for short videos (6-20s)")
    parser.add_argument("--log_dir", type=str, default=None, help="store subprocess logs (stdout+stderr)")

    args = parser.parse_args()

    res = run_single_video(
        video_path=args.video,
        video_id=args.video_id,
        out_dir=args.out_dir,
        case_id=args.case_id,
        view_name=args.view,
        fps=float(args.fps),
        dry_run=bool(args.dry_run),
        skip_existing=bool(int(args.skip_existing)),
        case_det_model=str(args.case_det_model),
        case_conf=float(args.case_conf),
        case_det=int(args.case_det),
        run_pose=int(args.run_pose),
        run_track=int(args.run_track),
        run_actions=int(args.run_actions),
        run_asr=int(args.run_asr),
        run_align=int(args.run_align),
        export_legacy=int(args.export_legacy),
        export_behavior=int(args.export_behavior),
        make_overlays=int(args.make_overlays),
        run_summarize=int(args.run_summarize),
        run_aggregate=int(args.run_aggregate),
        run_projection=int(args.run_projection),
        run_timeline=int(args.run_timeline),
        timeline_source=str(args.timeline_source),
        timeline_tracks=str(args.timeline_tracks),
        short_video=int(args.short_video),
        log_dir=str(args.log_dir) if args.log_dir else None,
        pose_model=str(args.pose_model),
        asr_model=str(args.asr_model),
    )

    print("\n" + "=" * 50 + "\n PIPELINE FINISHED \n" + "=" * 50)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if res.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
