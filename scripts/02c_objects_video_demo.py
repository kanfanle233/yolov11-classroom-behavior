# scripts/02c_objects_video_demo.py
import json
import argparse
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("需要安装 ultralytics: pip install ultralytics") from e


# COCO 类名（ultralytics 内置的模型通常是 COCO）
COCO_NAMES = None

# 你关心的小物体（可按需扩充）
WATCH_NAMES = {
    "cell phone", "book", "laptop", "tablet", "backpack", "handbag", "keyboard", "mouse", "remote"
}

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    base_dir = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--out_jsonl", required=True, type=str)
    ap.add_argument("--out_video", required=True, type=str)
    ap.add_argument("--model", type=str, default=str(base_dir / "yolo11s.pt"))
    ap.add_argument("--conf", type=float, default=0.15, help="小目标建议 0.1~0.2")
    ap.add_argument("--imgsz", type=int, default=960, help="小目标建议 960/1280")
    ap.add_argument("--device", type=str, default="", help="''=auto, '0'=GPU0, 'cpu'")
    ap.add_argument("--only_watch", action="store_true", help="只保留关注物体类（不输出 person）")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_jsonl = Path(args.out_jsonl)
    out_video = Path(args.out_video)
    ensure_parent(out_jsonl)
    ensure_parent(out_video)

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not Path(args.model).exists():
        raise FileNotFoundError(f"model not found: {args.model}")

    model = YOLO(args.model)
    global COCO_NAMES
    # ultralytics 的 names 通常在 model.model.names
    try:
        COCO_NAMES = model.model.names
    except Exception:
        COCO_NAMES = None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    f_out = out_jsonl.open("w", encoding="utf-8")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 推理（stream=False 单帧推理；为了简单稳）
        r = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            verbose=False
        )[0]

        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                conf = float(b.conf[0].item()) if b.conf is not None else None
                xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                name = COCO_NAMES.get(cls_id, str(cls_id)) if isinstance(COCO_NAMES, dict) else str(cls_id)

                if args.only_watch:
                    # 只保留关注类，并显式排除 person
                    if name == "person":
                        continue
                    if name not in WATCH_NAMES:
                        continue

                dets.append({
                    "name": name,
                    "cls": cls_id,
                    "conf": round(conf, 4) if conf is not None else None,
                    "bbox": [round(xyxy[0], 2), round(xyxy[1], 2), round(xyxy[2], 2), round(xyxy[3], 2)]
                })

                # 画框
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        rec = {
            "frame": frame_idx,
            "t": round(frame_idx / fps, 3),
            "fps": fps,
            "detections": dets
        }
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"[INFO] processed {frame_idx} frames, last dets={len(dets)}")

    cap.release()
    writer.release()
    f_out.close()
    print(f"[DONE] jsonl: {out_jsonl}")
    print(f"[DONE] video: {out_video}")

if __name__ == "__main__":
    main()
