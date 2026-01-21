import os
import json
import argparse
import subprocess
from pathlib import Path

import cv2


def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    # 保险：按 frame 排序，方便指针推进
    items.sort(key=lambda x: x.get("frame", 0))
    return items


def mux_audio(noaudio_path: Path, src_video: Path, out_path: Path):
    # 用原视频音轨封装到新视频
    cmd = [
        "ffmpeg", "-y",
        "-i", str(noaudio_path),
        "-i", str(src_video),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Overlay object detection boxes onto video (demo output).")
    parser.add_argument("--video", type=str, required=True, help="video path")
    parser.add_argument("--objects", type=str, required=True, help="objects.jsonl path")
    parser.add_argument("--out", type=str, required=True, help="output mp4 path (final, with audio if mux_audio=1)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--mux_audio", type=int, default=1, help="1=mux original audio via ffmpeg, 0=no")
    parser.add_argument("--show_empty", type=int, default=1, help="1=draw small HUD even if no objects in frame")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    objects_path = Path(args.objects)
    if not objects_path.is_absolute():
        objects_path = (base_dir / objects_path).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_noaudio = out_path.with_name(out_path.stem + "_noaudio.mp4")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not objects_path.exists():
        raise FileNotFoundError(f"Objects jsonl not found: {objects_path}")

    objects = load_jsonl(objects_path)
    print(f"[INFO] video   : {video_path}")
    print(f"[INFO] objects : {objects_path} (lines={len(objects)})")
    print(f"[INFO] out     : {out_path}")
    print(f"[INFO] conf    : {args.conf}, mux_audio={args.mux_audio}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_noaudio), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 用指针推进 objects（避免建超大 dict）
    ptr = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 收集当前帧的 objects（可能一帧多条/也可能跳帧）
        cur_objs = []
        while ptr < len(objects) and objects[ptr].get("frame", -1) < frame_idx:
            ptr += 1
        if ptr < len(objects) and objects[ptr].get("frame", -1) == frame_idx:
            # 如果你的 jsonl 一帧只写一行，这里就是那一行
            rec = objects[ptr]
            for obj in rec.get("objects", []):
                if float(obj.get("conf", 0.0)) >= args.conf:
                    cur_objs.append(obj)

        # HUD：让你知道脚本确实在跑（避免“像原视频”的错觉）
        t = frame_idx / fps
        if args.show_empty:
            cv2.putText(frame, f"t={t:.2f}s  frame={frame_idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 画框
        for obj in cur_objs:
            bbox = obj.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            name = str(obj.get("name", obj.get("cls_id", "obj")))
            conf = float(obj.get("conf", 0.0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"[INFO] processing frame {frame_idx} (t={t:.1f}s)")

    cap.release()
    writer.release()

    if args.mux_audio:
        try:
            mux_audio(out_noaudio, video_path, out_path)
            print(f"[DONE] saved with audio: {out_path}")
        except FileNotFoundError:
            print("[WARN] ffmpeg not found; keeping no-audio output only.")
            out_noaudio.replace(out_path)
    else:
        out_noaudio.replace(out_path)
        print(f"[DONE] saved (no audio): {out_path}")


if __name__ == "__main__":
    main()
