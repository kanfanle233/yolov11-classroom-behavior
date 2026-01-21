import os
import json
import cv2
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ====== 可调显示参数 ======
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS = 2

COLOR_MAP = {
    "raise_hand": (0, 0, 255),   # 红
    "head_down": (255, 0, 0),    # 蓝
    "stand": (0, 255, 255)       # 黄
}

MAX_LABELS_PER_FRAME = 4  # 防止刷屏


def load_actions(path):
    """把 actions.jsonl 按 frame 范围索引"""
    actions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except json.JSONDecodeError:
                continue
            actions.append(a)
    return actions


def build_frame_action_index(actions, fps=None, max_frame=None):
    """frame -> list of actions active at this frame
    兼容三种 actions 记录：
      1) start_frame/end_frame
      2) frame（单帧）
      3) start_time/end_time（需要 fps 才能换算为帧）
    """
    frame_map = defaultdict(list)

    for a in actions:
        # 1) 优先走 start_frame/end_frame（最标准）
        if "start_frame" in a and "end_frame" in a:
            sf = int(a["start_frame"])
            ef = int(a["end_frame"])

        # 2) 其次：只有 frame（逐帧微片段）
        elif "frame" in a:
            sf = ef = int(a["frame"])

        # 3) 再其次：只有时间戳（需要 fps）
        elif fps and ("start_time" in a or "end_time" in a):
            st = a.get("start_time")
            et = a.get("end_time", st)
            if st is None:
                continue
            sf = int(round(float(st) * fps))
            ef = int(round(float(et) * fps))
            # ---- 边界修正：防止 ef < sf、负帧 ----
            sf = int(sf)
            ef = int(ef)

            if ef < sf:
                sf, ef = ef, sf

            # 两个都在 0 之前，直接跳过
            if ef < 0:
                continue

            # 起始帧裁剪到 0
            sf = max(sf, 0)


        else:
            # 完全无法索引到帧
            continue

        for fr in range(sf, ef + 1):
            frame_map[fr].append(a)

    return frame_map



def load_tracks(path):
    """frame -> list of persons (with bbox, track_id)"""
    frame_map = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            fr = rec.get("frame")
            persons = rec.get("persons", [])
            if fr is None or not isinstance(persons, list):
                continue

            for p in persons:
                if "bbox" in p and "track_id" in p:
                    frame_map[int(fr)].append(p)
    return frame_map



def draw_bbox_and_id(img, bbox, tid):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"ID {tid}",
        (x1, max(20, y1 - 6)),
        FONT,
        FONT_SCALE,
        (0, 255, 0),
        THICKNESS,
        cv2.LINE_AA
    )


def draw_action_label(img, x, y, text, color):
    cv2.putText(
        img,
        text,
        (x, y),
        FONT,
        FONT_SCALE,
        color,
        THICKNESS,
        cv2.LINE_AA
    )


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.path.join("data", "videos", "demo1.mp4"))
    parser.add_argument("--tracks", type=str, default=os.path.join("output", "pose_tracks_smooth.jsonl"))
    parser.add_argument("--actions", type=str, default=os.path.join("output", "actions.jsonl"))
    parser.add_argument("--out", type=str, default=os.path.join("output", "action_overlay.mp4"))
    args = parser.parse_args()

    video_path = os.path.join(base_dir, args.video)
    tracks_path = os.path.join(base_dir, args.tracks)
    actions_path = os.path.join(base_dir, args.actions)
    out_path = os.path.join(base_dir, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    print("[INFO] loading tracks...")
    frame_tracks = load_tracks(tracks_path)

    print("[INFO] loading actions...")
    actions = load_actions(actions_path)
    frame_actions = build_frame_action_index(actions, fps=fps)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ====== 画 bbox + ID ======
        persons = frame_tracks.get(frame_idx, [])
        for p in persons:
            draw_bbox_and_id(frame, p["bbox"], p["track_id"])

        # ====== 画行为标签（限制数量，防刷屏） ======
        acts = frame_actions.get(frame_idx, [])
        shown = 0
        for a in acts:
            if shown >= MAX_LABELS_PER_FRAME:
                break
            color = COLOR_MAP.get(a["action"], (255, 255, 255))
            label = a["action"]
            if "side" in a:
                label += f" ({a['side']})"
            label += f"  ID:{a['track_id']}"
            draw_action_label(frame, 10, 30 + 22 * shown, label, color)
            shown += 1

        # ====== 画时间 ======
        t = frame_idx / fps
        cv2.putText(
            frame,
            f"t = {t:.2f}s",
            (w - 160, 30),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            THICKNESS,
            cv2.LINE_AA
        )

        writer.write(frame)

        if frame_idx % 200 == 0:
            print(f"[INFO] processed frame {frame_idx}")

        frame_idx += 1

    cap.release()
    writer.release()

    print("[DONE] action overlay video:", out_path)


if __name__ == "__main__":
    main()
