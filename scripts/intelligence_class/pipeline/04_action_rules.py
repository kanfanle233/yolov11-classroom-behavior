import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # pipeline -> intelligence_class -> scripts -> YOLOv11

# ====== COCO-Pose 关键点索引 ======
NOSE = 0
LS, RS = 5, 6
LW, RW = 9, 10
LH, RH = 11, 12

REQUIRED_IDXS = [NOSE, LS, RS, LW, RW, LH, RH]

def mid_point(a, b):
    return ((a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0)

def dist_y(a, b):
    return a["y"] - b["y"]

def _kpt_ok(kpts, idx):
    """kpts[idx] 必须是 dict 且含 x,y"""
    if kpts is None:
        return False
    if not isinstance(kpts, (list, tuple)):
        return False
    if idx >= len(kpts):
        return False
    p = kpts[idx]
    if p is None or not isinstance(p, dict):
        return False
    return ("x" in p) and ("y" in p)

def _all_required_ok(kpts):
    return all(_kpt_ok(kpts, i) for i in REQUIRED_IDXS)

def torso_length(kpts):
    """肩中点到髋中点的距离（用于尺度归一化）"""
    # 保险：任何缺失直接返回 None
    if not _all_required_ok(kpts):
        return None
    sh = mid_point(kpts[LS], kpts[RS])
    hp = mid_point(kpts[LH], kpts[RH])
    return abs(hp[1] - sh[1]) + 1e-6

def load_tracks(path, fps=25.0):
    tracks = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            # 兼容字段：frame/frame_idx
            frame = rec.get("frame", rec.get("frame_idx", 0))

            # 兼容字段：t/time_sec
            t = rec.get("t", rec.get("time_sec", frame / float(fps)))

            persons = rec.get("persons", [])
            if not persons:
                continue

            for p in persons:
                tid = p.get("track_id", None)
                if tid is None:
                    continue

                kpts = p.get("keypoints", None)
                bbox = p.get("bbox", None)

                # bbox 或 keypoints 为空就跳过，避免后面炸
                if bbox is None or kpts is None:
                    continue

                tracks[tid].append({
                    "frame": frame,
                    "t": t,
                    "bbox": bbox,
                    "keypoints": kpts
                })
    return tracks

def detect_actions(tracks, fps=25.0):
    events = []

    raise_hand_frames = int(0.5 * fps)   # ≥0.5s
    head_down_frames  = int(0.7 * fps)   # ≥0.7s
    stand_frames      = int(0.7 * fps)

    for tid, seq in tracks.items():
        if len(seq) < 30:
            continue

        heights = [(s["bbox"][3] - s["bbox"][1]) for s in seq if s.get("bbox") is not None]
        if not heights:
            continue
        median_h = sorted(heights)[len(heights)//2]

        rh_cnt = lh_cnt = hd_cnt = st_cnt = 0
        rh_start = lh_start = hd_start = st_start = None

        for s in seq:
            k = s.get("keypoints", None)
            if k is None:
                continue

            torso = torso_length(k)
            if torso is None:
                # 关键点缺失：直接跳过这一帧
                continue

            frame = s["frame"]
            t = s["t"]

            # ====== 举手（左右分别） ======
            if dist_y(k[LW], k[LS]) < -0.15 * torso:
                if lh_cnt == 0:
                    lh_start = (frame, t)
                lh_cnt += 1
            else:
                if lh_cnt >= raise_hand_frames and lh_start is not None:
                    events.append({
                        "track_id": tid,
                        "action": "raise_hand",
                        "side": "left",
                        "start_frame": lh_start[0],
                        "end_frame": frame,
                        "start_time": lh_start[1],
                        "end_time": t,
                        "duration": t - lh_start[1],
                        "confidence": min(1.0, lh_cnt / (2 * raise_hand_frames))
                    })
                lh_cnt = 0
                lh_start = None

            if dist_y(k[RW], k[RS]) < -0.15 * torso:
                if rh_cnt == 0:
                    rh_start = (frame, t)
                rh_cnt += 1
            else:
                if rh_cnt >= raise_hand_frames and rh_start is not None:
                    events.append({
                        "track_id": tid,
                        "action": "raise_hand",
                        "side": "right",
                        "start_frame": rh_start[0],
                        "end_frame": frame,
                        "start_time": rh_start[1],
                        "end_time": t,
                        "duration": t - rh_start[1],
                        "confidence": min(1.0, rh_cnt / (2 * raise_hand_frames))
                    })
                rh_cnt = 0
                rh_start = None

            # ====== 低头 ======
            shoulder_mid_y = (k[LS]["y"] + k[RS]["y"]) / 2.0
            if k[NOSE]["y"] > shoulder_mid_y + 0.12 * torso:
                if hd_cnt == 0:
                    hd_start = (frame, t)
                hd_cnt += 1
            else:
                if hd_cnt >= head_down_frames and hd_start is not None:
                    events.append({
                        "track_id": tid,
                        "action": "head_down",
                        "start_frame": hd_start[0],
                        "end_frame": frame,
                        "start_time": hd_start[1],
                        "end_time": t,
                        "duration": t - hd_start[1],
                        "confidence": min(1.0, hd_cnt / (2 * head_down_frames))
                    })
                hd_cnt = 0
                hd_start = None

            # ====== 站立 ======
            h = s["bbox"][3] - s["bbox"][1]
            if h > 1.18 * median_h:
                if st_cnt == 0:
                    st_start = (frame, t)
                st_cnt += 1
            else:
                if st_cnt >= stand_frames and st_start is not None:
                    events.append({
                        "track_id": tid,
                        "action": "stand",
                        "start_frame": st_start[0],
                        "end_frame": frame,
                        "start_time": st_start[1],
                        "end_time": t,
                        "duration": t - st_start[1],
                        "confidence": min(1.0, st_cnt / (2 * stand_frames))
                    })
                st_cnt = 0
                st_start = None

    return events

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, required=True)
    parser.add_argument("--fps", dest="fps", type=float, default=25.0)
    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tracks = load_tracks(in_path, fps=float(args.fps))
    events = detect_actions(tracks, fps=float(args.fps))

    with open(out_path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("[DONE] actions.jsonl:", out_path)
    print("[INFO] total events:", len(events))

if __name__ == "__main__":
    main()
