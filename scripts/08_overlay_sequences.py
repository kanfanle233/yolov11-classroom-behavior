# scripts/08_overlay_sequences.py
import json
import os
import argparse
import subprocess
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont  # pip install pillow


def load_jsonl(path: Path):
    data = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    data.append(json.loads(s))
                except Exception:
                    continue
    return data


def draw_text_chinese(img, text, position, font_size, color):
    """使用 Pillow 在 OpenCV 图像上绘制中文"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试加载系统字体，如果没有则用默认（默认不支持中文）
    # 建议把 simhei.ttf 放到项目根目录，或者指定绝对路径
    font_path = "simhei.ttf"
    # Windows 常见路径
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/simhei.ttf"

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def safe_get(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def clamp_text(s: str, max_len: int = 40) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len - 1] + "…"


def normalize_transcript(transcript):
    """
    统一 transcript 为 list[{"start":float,"end":float,"text":str}] 并按 start 排序
    """
    norm = []
    for r in transcript:
        st = safe_get(r, ["start", "begin", "start_time", "begin_time"], None)
        ed = safe_get(r, ["end", "finish", "end_time", "finish_time"], None)
        tx = safe_get(r, ["text", "sentence", "content", "transcript"], "")
        try:
            st = float(st)
            ed = float(ed)
        except Exception:
            continue
        if ed < st:
            st, ed = ed, st
        tx = str(tx).strip()
        if not tx:
            continue
        norm.append({"start": st, "end": ed, "text": tx})
    norm.sort(key=lambda x: (x["start"], x["end"]))
    return norm


def normalize_actions(actions):
    """
    兼容多种 actions.jsonl schema
    [关键修复] 增加了对 start_time / end_time 的支持
    """
    norm = []

    for r in actions:
        # case: nested people list (兼容旧格式，目前主要走 flat record)
        people = safe_get(r, ["people", "persons", "tracks"], None)
        t = safe_get(r, ["t", "time"], None)
        frame = safe_get(r, ["frame"], None)

        if people and isinstance(people, list):
            try:
                t_val = float(t) if t is not None else None
            except Exception:
                t_val = None

            for p in people:
                label = safe_get(p, ["label", "action", "name", "cls", "behavior"], None)
                conf = safe_get(p, ["conf", "score", "prob"], None)
                pid = safe_get(p, ["id", "track_id", "person_id"], None)
                if label is None:
                    continue
                try:
                    conf = float(conf) if conf is not None else None
                except Exception:
                    conf = None
                norm.append({
                    "t": t_val,
                    "start": None,
                    "end": None,
                    "id": pid,
                    "label": str(label),
                    "conf": conf,
                    "_frame": frame
                })
            continue

        # case: flat record (04_complex_logic.py 输出的是这种)
        label = safe_get(r, ["label", "action", "name", "cls", "behavior"], None)
        if label is None:
            continue

        pid = safe_get(r, ["id", "track_id", "person_id"], None)
        conf = safe_get(r, ["conf", "score", "prob"], None)

        # [关键修复] 添加 start_time / end_time 到查找列表
        start = safe_get(r, ["start", "begin", "start_time"], None)
        end = safe_get(r, ["end", "finish", "end_time"], None)

        try:
            t_val = float(t) if t is not None else None
        except Exception:
            t_val = None
        try:
            start_val = float(start) if start is not None else None
        except Exception:
            start_val = None
        try:
            end_val = float(end) if end is not None else None
        except Exception:
            end_val = None
        try:
            conf_val = float(conf) if conf is not None else None
        except Exception:
            conf_val = None

        if end_val is not None and start_val is not None and end_val < start_val:
            start_val, end_val = end_val, start_val

        norm.append({
            "t": t_val,
            "start": start_val,
            "end": end_val,
            "id": pid,
            "label": str(label),
            "conf": conf_val,
            "_frame": frame
        })

    # 排序策略：优先按 t，其次按 start
    def _key(x):
        t = x["t"]
        st = x["start"]
        return (
            0 if t is not None else 1,
            t if t is not None else 1e18,
            st if st is not None else 1e18
        )

    norm.sort(key=_key)
    return norm


def find_active_transcript(transcript, ptr, t):
    while ptr < len(transcript) and transcript[ptr]["end"] < t:
        ptr += 1
    if ptr < len(transcript):
        r = transcript[ptr]
        if r["start"] <= t <= r["end"]:
            return ptr, r
    return ptr, None


def build_actions_index(actions_norm):
    with_t = [a for a in actions_norm if a["t"] is not None]
    with_seg = [a for a in actions_norm if a["t"] is None and a["start"] is not None and a["end"] is not None]
    return with_t, with_seg


def collect_active_actions(with_t, with_seg, ptr_t, ptr_seg, t, window=0.20, max_lines=6):
    act = []

    # 1) 帧级：推进到 t-window 之前
    while ptr_t < len(with_t) and (with_t[ptr_t]["t"] is not None) and with_t[ptr_t]["t"] < (t - window):
        ptr_t += 1
    i = ptr_t
    while i < len(with_t) and with_t[i]["t"] <= (t + window):
        act.append(with_t[i])
        i += 1

    # 2) 区间级：推进到 end < t 的位置
    while ptr_seg < len(with_seg) and with_seg[ptr_seg]["end"] < t:
        ptr_seg += 1
    j = ptr_seg
    while j < len(with_seg) and with_seg[j]["start"] <= t:
        # start <= t，还需要检查 t <= end
        if t <= with_seg[j]["end"]:
            act.append(with_seg[j])
        j += 1

    # 去重：同一个人同一个 label 只保留置信度最高的
    best = {}
    for a in act:
        k = (a.get("id", None), a.get("label", ""))
        prev = best.get(k)
        if prev is None:
            best[k] = a
        else:
            c1 = prev.get("conf", -1.0)
            c2 = a.get("conf", -1.0)
            if c2 is not None and c2 > (c1 if c1 is not None else -1.0):
                best[k] = a

    act = list(best.values())

    # 排序：置信度高的在前
    def _score(a):
        c = a.get("conf", None)
        return -(c if c is not None else 0.0)

    act.sort(key=_score)
    act = act[:max_lines]
    return act, ptr_t, ptr_seg


def main():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=str(base_dir / "data/videos/demo3.mp4"))
    parser.add_argument("--actions", type=str, default=str(base_dir / "output/demo3/actions.jsonl"))
    parser.add_argument("--transcript", type=str, default=str(base_dir / "output/demo3/transcript.jsonl"))
    parser.add_argument("--out_dir", type=str, default=str(base_dir / "output/demo3"))
    parser.add_argument("--name", type=str, default="demo3")
    parser.add_argument("--mux_audio", type=int, default=1)
    parser.add_argument("--out_video", type=str, default="",
                        help="final overlay mp4 path (with audio). if set, override default out_path")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    actions_path = Path(args.actions)
    if not actions_path.is_absolute():
        actions_path = (base_dir / actions_path).resolve()
    transcript_path = Path(args.transcript)
    if not transcript_path.is_absolute():
        transcript_path = (base_dir / transcript_path).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_video:
        out_path = Path(args.out_video)
        if not out_path.is_absolute():
            out_path = (base_dir / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = out_dir / f"{args.name}_overlay.mp4"

    out_path_noaudio = out_dir / f"{args.name}_overlay_noaudio.mp4"

    transcript_raw = load_jsonl(transcript_path)
    actions_raw = load_jsonl(actions_path)

    transcript = normalize_transcript(transcript_raw)
    actions_norm = normalize_actions(actions_raw)
    with_t, with_seg = build_actions_index(actions_norm)

    print(f"[INFO] transcript lines: {len(transcript_raw)} -> normalized: {len(transcript)}")
    print(f"[INFO] actions lines    : {len(actions_raw)} -> normalized: {len(actions_norm)}")
    print(f"[INFO] actions with_t   : {len(with_t)}")
    print(f"[INFO] actions with_seg : {len(with_seg)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1e-6:
        fps = 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 注意：OpenCV 写入的这个中间文件依然是 mp4v 格式，体积很大
    # 但它只是临时的，后续会被 ffmpeg 转码压缩
    writer = cv2.VideoWriter(str(out_path_noaudio), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tr_ptr = 0
    act_ptr_t = 0
    act_ptr_seg = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps

        # HUD
        cv2.putText(frame, f"t={t:.2f}s  frame={frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 动作列表
        active_actions, act_ptr_t, act_ptr_seg = collect_active_actions(
            with_t, with_seg, act_ptr_t, act_ptr_seg, t, window=0.25, max_lines=6
        )

        y0 = 80
        if active_actions:
            # 半透明背景
            cv2.rectangle(frame, (10, 55), (min(620, w - 10), 55 + 30 * (len(active_actions) + 1)), (0, 0, 0), -1)
            frame = draw_text_chinese(frame, f"Actions (near t={t:.2f}s):", (20, 60), 26, (255, 255, 255))
            y = y0
            for a in active_actions:
                pid = a.get("id", None)
                label = a.get("label", "")
                # conf = a.get("conf", None)
                line = f"ID:{pid}  {label}"
                frame = draw_text_chinese(frame, clamp_text(line, 45), (20, y), 26, (255, 255, 0))
                y += 30
        else:
            frame = draw_text_chinese(frame, "Actions: (none)", (20, 70), 24, (200, 200, 200))

        # 字幕
        tr_ptr, seg = find_active_transcript(transcript, tr_ptr, t)
        if seg is not None:
            text = clamp_text(seg["text"], 50)
            cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)
            frame = draw_text_chinese(frame, f"Speech: {text}", (20, h - 70), 30, (255, 255, 255))

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"[INFO] Processing frame {frame_idx} (t={t:.1f}s)")

    cap.release()
    writer.release()

    # 音频封装与转码
    if args.mux_audio:
        try:
            print("[INFO] Starting FFmpeg transcoding (H.264)...")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(out_path_noaudio),  # 视频流 (OpenCV生成)
                    "-i", str(video_path),  # 音频流 (原视频)
                    # === 关键修改 ===
                    "-c:v", "libx264",  # 强制使用 H.264 编码
                    "-crf", "23",  # 压缩质量 (越小越清晰, 23是平衡点)
                    "-preset", "fast",  # 编码速度
                    "-pix_fmt", "yuv420p",  # 确保浏览器兼容性
                    # ===============
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest", str(out_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[DONE] Final video saved to: {out_path}")

            # 清理巨大的临时文件
            if os.path.exists(out_path_noaudio):
                os.remove(out_path_noaudio)
                print(f"[INFO] Cleaned up temporary file: {out_path_noaudio}")

        except Exception as e:
            print(f"[WARN] ffmpeg failed ({e}). Keeping no-audio file.")
            # 如果转码失败，至少保留原文件（虽然很大）
            if out_path.exists(): os.remove(out_path)
            os.rename(out_path_noaudio, out_path)
    else:
        # 如果不合并音频，直接改名（注意：这里生成的文件依然是 mp4v 可能会很大且浏览器不兼容）
        if out_path.exists(): os.remove(out_path)
        os.rename(out_path_noaudio, out_path)
        print(f"[DONE] overlay saved to: {out_path} (Warning: No audio & Codec mp4v)")


if __name__ == "__main__":
    main()
