# scripts/02c_export_objects_jsonl_custom.py
import os
import json
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="video path (relative to project root or absolute)")
    parser.add_argument("--out", type=str, required=True, help="output jsonl path")
    parser.add_argument("--model", type=str, required=True, help="yolo det model path (e.g., yolo11s.pt or your custom)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IOU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="inference image size")
    parser.add_argument("--device", type=str, default="", help="cuda:0 / cpu / ''(auto)")
    parser.add_argument("--batch", type=int, default=8, help="batch size for inference")
    parser.add_argument("--stride", type=int, default=1, help="process every N frames (1=all frames)")
    parser.add_argument(
        "--keep",
        type=str,
        default="",
        help="comma-separated class ids to keep (e.g., '67,73,41,63'). empty means keep all classes"
    )
    args = parser.parse_args()

    # Resolve paths
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (base_dir / model_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    keep_ids = None
    if args.keep.strip():
        keep_ids = set(int(x) for x in args.keep.split(",") if x.strip().isdigit())

    print(f"[INFO] video : {video_path}")
    print(f"[INFO] out   : {out_path}")
    print(f"[INFO] model : {model_path}")
    print(f"[INFO] conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, batch={args.batch}, stride={args.stride}")
    if keep_ids is not None:
        print(f"[INFO] keep class ids: {sorted(list(keep_ids))}")

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames = []
    frame_indices = []
    frame_idx = 0
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % args.stride == 0:
                frames.append(frame)
                frame_indices.append(frame_idx)

                # Run when batch full
                if len(frames) >= args.batch:
                    results = model.predict(
                        frames,
                        verbose=False,
                        conf=float(args.conf),
                        iou=float(args.iou),
                        imgsz=int(args.imgsz),
                        device=args.device if args.device else None
                    )

                    for r, fi in zip(results, frame_indices):
                        objs = []
                        if r.boxes is not None and len(r.boxes) > 0:
                            names = r.names
                            xyxy = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            clss = r.boxes.cls.cpu().numpy().astype(int)

                            for box, cf, cid in zip(xyxy, confs, clss):
                                if keep_ids is not None and cid not in keep_ids:
                                    continue
                                x1, y1, x2, y2 = box.tolist()
                                objs.append({
                                    "cls_id": int(cid),
                                    "name": names.get(int(cid), str(int(cid))),
                                    "conf": float(cf),
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                                })

                        if objs:
                            rec = {"frame": int(fi), "t": round(fi / fps, 3), "objects": objs}
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            written += 1

                    frames.clear()
                    frame_indices.clear()

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"[INFO] read {frame_idx} frames... written_lines={written}")

        # Flush remaining
        if frames:
            results = model.predict(
                frames,
                verbose=False,
                conf=float(args.conf),
                iou=float(args.iou),
                imgsz=int(args.imgsz),
                device=args.device if args.device else None
            )
            for r, fi in zip(results, frame_indices):
                objs = []
                if r.boxes is not None and len(r.boxes) > 0:
                    names = r.names
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    for box, cf, cid in zip(xyxy, confs, clss):
                        if keep_ids is not None and cid not in keep_ids:
                            continue
                        x1, y1, x2, y2 = box.tolist()
                        objs.append({
                            "cls_id": int(cid),
                            "name": names.get(int(cid), str(int(cid))),
                            "conf": float(cf),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        })
                if objs:
                    rec = {"frame": int(fi), "t": round(fi / fps, 3), "objects": objs}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1

    cap.release()
    print(f"[DONE] objects.jsonl saved: {out_path}")
    print(f"[DONE] total frames read: {frame_idx}, lines written: {written}")


if __name__ == "__main__":
    main()
