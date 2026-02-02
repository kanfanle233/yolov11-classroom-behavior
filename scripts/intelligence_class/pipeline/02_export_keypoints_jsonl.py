# 02_export_keypoints_jsonl.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO pose keypoints to JSONL (pipeline version)")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    video_p = Path(args.video).resolve()
    out_p = Path(args.out).resolve()
    model_p = Path(args.model).resolve()

    if not video_p.exists():
        raise FileNotFoundError(f"video not found: {video_p}")
    if not model_p.exists():
        raise FileNotFoundError(f"model not found: {model_p}")

    out_p.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_p}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 25.0

    model = YOLO(str(model_p))

    frame_idx = 0
    with open(out_p, "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = model(frame, conf=float(args.conf), verbose=False)[0]
            persons = []

            # ultralytics pose: res.keypoints (n, k, 2/3)
            kps = getattr(res, "keypoints", None)
            boxes = getattr(res, "boxes", None)

            if kps is not None and boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy().tolist()
                confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [None] * len(xyxy)

                # kps.xy: (n, k, 2)
                kxy = kps.xy.cpu().numpy().tolist() if getattr(kps, "xy", None) is not None else None
                kcf = kps.conf.cpu().numpy().tolist() if getattr(kps, "conf", None) is not None else None

                if kxy is not None:
                    for i in range(len(xyxy)):
                        one_kpts = []
                        for j in range(len(kxy[i])):
                            x, y = kxy[i][j]
                            one_kpts.append({
                                "x": float(x) if x is not None else None,
                                "y": float(y) if y is not None else None,
                                "conf": float(kcf[i][j]) if (kcf is not None and kcf[i][j] is not None) else None
                            })

                        persons.append({
                            "track_id": -1,  # tracking 之后再写
                            "bbox": [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])],
                            "conf": float(confs[i]) if confs[i] is not None else None,
                            "keypoints": one_kpts
                        })

            rec = {
                "frame": frame_idx,
                "frame_idx": frame_idx,
                "time_sec": float(frame_idx / fps),
                "fps": fps,
                "persons": persons
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            frame_idx += 1

    cap.release()
    print(f"[DONE] JSONL: {out_p}")


if __name__ == "__main__":
    main()
