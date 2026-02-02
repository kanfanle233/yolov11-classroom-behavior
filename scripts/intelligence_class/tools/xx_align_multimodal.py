import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


# ===========================
# 1. 辅助函数：读取数据
# ===========================

def load_jsonl(path: Path) -> List[Dict]:
    data = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    pass
    return data


# ===========================
# 2. 核心逻辑：时序对齐
# ===========================

def compress_frame_dets(
        det_rows: List[Dict],
        fps: float,
        min_duration: float = 1.0,
        gap_tolerance: float = 0.5
) -> List[Dict]:
    """
    将逐帧的 YOLO 检测结果压缩成“事件段”。
    例如：第 10-50 帧连续检测到 sleeping，合并为一个 {start: 0.33, end: 1.66, label: sleeping} 事件。
    """
    if not det_rows:
        return []

    # 1. 按标签分组
    events_by_label = {}  # { "sleeping": [frame_idx, ...], "hand_raise": [...] }

    for row in det_rows:
        frame_idx = row.get("frame_idx")
        dets = row.get("dets", [])
        for d in dets:
            lbl = d.get("label")
            if lbl:
                if lbl not in events_by_label:
                    events_by_label[lbl] = []
                events_by_label[lbl].append(frame_idx)

    # 2. 简单的连通域合并
    compressed_events = []

    for label, frames in events_by_label.items():
        frames.sort()
        if not frames:
            continue

        start_f = frames[0]
        curr_f = frames[0]

        # 允许中间断几帧 (gap_tolerance * fps)
        max_gap = int(gap_tolerance * fps)

        for next_f in frames[1:]:
            if next_f - curr_f <= max_gap:
                # 视为连续
                curr_f = next_f
            else:
                # 断开了，结算上一段
                duration_sec = (curr_f - start_f + 1) / fps
                if duration_sec >= min_duration:
                    compressed_events.append({
                        "type": "behavior_yolo",
                        "label": label,
                        "start": start_f / fps,
                        "end": curr_f / fps,
                        "duration": duration_sec
                    })
                # 开启新一段
                start_f = next_f
                curr_f = next_f

        # 结算最后一段
        duration_sec = (curr_f - start_f + 1) / fps
        if duration_sec >= min_duration:
            compressed_events.append({
                "type": "behavior_yolo",
                "label": label,
                "start": start_f / fps,
                "end": curr_f / fps,
                "duration": duration_sec
            })

    return compressed_events


def align_events(
        visual_events: List[Dict],
        transcript_segs: List[Dict],
        window_sec: float = 2.0
) -> List[Dict]:
    """
    以视觉事件为核心，寻找时间上重叠（overlap）或临近的文本。
    window_sec: 向前后扩展的时间窗 (±2s)
    """
    aligned_results = []

    for ve in visual_events:
        v_start = ve["start"]
        v_end = ve["end"]

        # 定义上下文窗口
        ctx_start = max(0, v_start - window_sec)
        ctx_end = v_end + window_sec

        matched_texts = []

        for seg in transcript_segs:
            t_start = seg["start"]
            t_end = seg["end"]

            # 判断是否有交集 (Intersection)
            # max(start1, start2) < min(end1, end2)
            if max(ctx_start, t_start) < min(ctx_end, t_end):
                matched_texts.append({
                    "text": seg["text"],
                    "t_start": t_start,
                    "t_end": t_end,
                    "overlap": True
                })

        # 即使没有匹配到文本，这个视觉事件也是有意义的
        aligned_record = {
            "visual_event": ve,
            "context_window": [ctx_start, ctx_end],
            "related_audio": matched_texts,
            "has_verification": len(matched_texts) > 0  # 是否有“双重验证”的潜力
        }
        aligned_results.append(aligned_record)

    # 按时间排序
    aligned_results.sort(key=lambda x: x["visual_event"]["start"])
    return aligned_results


# ===========================
# 3. 主入口
# ===========================

def run_alignment(
        case_dir: Path,
        fps: float = 25.0
):
    print(f"[Align] Processing {case_dir.name}...")

    # 1. 加载文件
    f_actions = case_dir / "actions.jsonl"  # Rule-based (Pose)
    f_det = case_dir / f"{case_dir.name}_behavior.jsonl"  # YOLO-based (如果你之前的脚本重命名了，这里要注意)
    # 如果找不到以 ID 命名的 jsonl，尝试找 case_det.jsonl
    if not f_det.exists():
        f_det = case_dir / "case_det.jsonl"

    f_trans = case_dir / "transcript.jsonl"

    # 2. 准备数据源
    actions_data = load_jsonl(f_actions)
    det_data = load_jsonl(f_det)
    trans_data = load_jsonl(f_trans)

    all_visual_events = []

    # A. 处理 Rule Actions (通常已经是离散事件，或者逐帧的)
    # 假设 actions.jsonl 是逐帧记录的 (e.g. {"frame": 10, "action": "hand_raise"})
    # 我们也需要像 YOLO 一样压缩
    # 如果 actions.jsonl 已经是 {start, end} 格式则直接用。这里假设是逐帧的：
    raw_action_rows = []
    for row in actions_data:
        # 兼容不同格式
        fidx = row.get("frame_idx", row.get("frame"))
        act = row.get("action")
        if fidx is not None and act and act != "stand":
            raw_action_rows.append({"frame_idx": fidx, "dets": [{"label": act}]})  # 伪造成 dets 格式以便复用压缩函数

    compressed_actions = compress_frame_dets(raw_action_rows, fps, min_duration=0.5)  # 动作稍微短点也可以
    for c in compressed_actions:
        c["type"] = "rule_pose"  # 标记来源
    all_visual_events.extend(compressed_actions)

    # B. 处理 YOLO Behaviors
    # 过滤掉空的 dets
    valid_det_rows = [r for r in det_data if r.get("dets")]
    compressed_dets = compress_frame_dets(valid_det_rows, fps, min_duration=0.8)
    all_visual_events.extend(compressed_dets)

    if not all_visual_events:
        print("  -> No visual events found to align.")
        # 依然生成一个空文件，防止前端 404
        with open(case_dir / "align.json", "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    # 3. 执行对齐
    alignment_result = align_events(all_visual_events, trans_data, window_sec=2.0)

    # 4. 输出
    out_file = case_dir / "align.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(alignment_result, f, ensure_ascii=False, indent=2)

    print(f"  -> Generated {out_file.name} with {len(alignment_result)} events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True, help="具体的 Case 目录，例如 .../rear__006")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    run_alignment(Path(args.case_dir), args.fps)