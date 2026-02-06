import os
import json
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


def safe_mkdir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def parse_classes(s: str):
    """
    "67,73,63" -> [67,73,63]
    "" -> None
    """
    s = (s or "").strip()
    if not s:
        return None
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo11s.pt")

    # ✅ 关键：让你能复现/调参
    parser.add_argument("--conf", type=float, default=0.10, help="confidence threshold (small objects建议 0.08~0.15)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="inference image size (small objects建议 960/1280)")
    parser.add_argument("--device", type=str, default="", help="e.g. 0 or 'cpu'. empty=auto")

    # ✅ 关键：只检测指定类（强烈建议默认不含 person）
    # COCO 常见：67 cell phone, 73 book, 63 laptop, 62 tv, 64 mouse, etc（以你模型 names 为准）
    parser.add_argument("--classes", type=str, default="67,73,63",
                        help="comma-separated class ids to keep, e.g. '67,73,63'. default excludes person.")

    # ✅ 输出一个 demo 视频（像01一样），方便肉眼验证
    parser.add_argument("--out_video", type=str, default="", help="optional: output demo video with boxes")

    # ✅ 可选：每隔多少帧推理一次（加速用，默认每帧）
    parser.add_argument("--stride", type=int, default=1, help="infer every N frames (1=every frame)")

    args = parser.parse_args()

    video_path = (base_dir / args.video).resolve()
    jsonl_path = (base_dir / args.out).resolve()
    safe_mkdir(jsonl_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ✅ 模型路径：允许你传相对路径（项目根目录下）
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (base_dir / model_path).resolve()
    if not model_path.exists():
        # ultralytics 也可能会自动下载，但你现在做科研工程，建议显式失败，别悄悄换权重
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    # ✅ names 映射，后面用来保证 name 一定正确
    names = model.model.names if hasattr(model, "model") else model.names
    if not isinstance(names, dict):
        # 有的版本是 list
        names = {i: n for i, n in enumerate(list(names))}

    keep_classes = parse_classes(args.classes)  # None or [..]
    if keep_classes is not None:
        # 防御：过滤掉越界 id
        keep_classes = [cid for cid in keep_classes if cid in names]

    print(f"[INFO] video   = {video_path}")
    print(f"[INFO] out     = {jsonl_path}")
    print(f"[INFO] model   = {model_path}")
    print(f"[INFO] conf/iou = {args.conf}/{args.iou}, imgsz={args.imgsz}, stride={args.stride}")
    print(f"[INFO] classes = {keep_classes} -> {[names[c] for c in keep_classes] if keep_classes else 'ALL'}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1e-6:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频 writer（可选）
    writer = None
    out_video_path = None
    if args.out_video.strip():
        out_video_path = Path(args.out_video)
        if not out_video_path.is_absolute():
            out_video_path = (base_dir / out_video_path).resolve()
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"[INFO] demo video -> {out_video_path}")

    frame_idx = 0
    wrote_lines = 0

    with open(str(jsonl_path), "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            do_infer = (frame_idx % max(1, args.stride) == 0)

            detections = []
            if do_infer:
                # ✅ ultralytics 原生 classes 过滤：最关键的“防 person 污染”
                results = model.predict(
                    frame,
                    verbose=False,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device if args.device else None,
                    classes=keep_classes  # ✅ 关键：只要指定类
                )
                r = results[0]

                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        cls_id = int(b.cls[0])
                        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                        conf = float(b.conf[0])
                        name = names.get(cls_id, "unknown")

                        # ✅ schema 稳定：每条必有 name/conf/bbox
                        detections.append({
                            "cls_id": cls_id,
                            "name": name,
                            "conf": conf,
                            "bbox": [x1, y1, x2, y2]
                        })

                        # 可视化
                        if writer is not None:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                            cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), max(0, int(y1) - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ✅ 建议：即使本帧没检测到，也可以不写（省空间）
            # 但为了后续对齐更稳：你可以改成“每帧都写”，这里按你之前逻辑：有 det 才写
            if len(detections) > 0:
                rec = {
                    "frame": frame_idx,
                    "t": round(frame_idx / fps, 3),
                    "objects": detections
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote_lines += 1

            if writer is not None:
                cv2.putText(frame, f"t={frame_idx/fps:.2f}s frame={frame_idx}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                writer.write(frame)

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"[INFO] processed {frame_idx} frames, wrote_lines={wrote_lines}, last dets={len(detections)}")

    cap.release()
    if writer is not None:
        writer.release()

    print(f"[DONE] objects jsonl saved: {jsonl_path} (lines={wrote_lines})")
    if out_video_path is not None:
        print(f"[DONE] objects demo video: {out_video_path}")


if __name__ == "__main__":
    main()
