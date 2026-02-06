# scripts/10_visualize_timeline.py
import json
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ====== 行为颜色映射（0-8）======
COLOR_MAP = {
    0: "#2ecc71",  # Listen (绿)
    1: "#34495e",  # Distract (深蓝灰)
    2: "#e74c3c",  # Phone (红)
    3: "#3498db",  # Doze (蓝)
    4: "#e67e22",  # Chat (橙)
    5: "#9b59b6",  # Note (紫)
    6: "#f1c40f",  # Raise (黄)
    7: "#1abc9c",  # Stand (青)
    8: "#d35400",  # Read (深橘红)
}

LABELS = {
    0: "Listen",
    1: "Distract",
    2: "Phone",
    3: "Doze",
    4: "Chat",
    5: "Note",
    6: "Raise",
    7: "Stand",
    8: "Read",
}

# 群体行为颜色
GROUP_COLOR_MAP = {
    "lecture": "#A8E6CF",  # 浅绿
    "discussion": "#AEDFF7",  # 浅蓝
    "break": "#FFD3B6",  # 浅橙
    "individual_work": "#E2DBBE"  # 浅灰褐
}

# 这里的 Key 对应 04_complex_logic.py 输出的 return 字符串
ACTION_STR_TO_ID = {
    "listening": 0, "listen": 0,
    "distract": 1,
    "playing_phone": 2, "phone": 2, "cell phone": 2,
    "sleeping": 3, "doze": 3, "sleep": 3,
    "chatting": 4, "chat": 4,
    "writing": 5, "reading_writing": 5, "note": 5,
    "raising_hand": 6, "raise_hand": 6, "raise": 6, "hand_raise": 6,
    "standing": 7, "stand": 7,
    "reading": 8, "read": 8, "book": 8
}


# ===================== 工具函数 =====================

def _safe_int(x: Any, default: int = 10 ** 18) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _people_to_dict(people: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(people, dict):
        return {str(k): v for k, v in people.items() if isinstance(v, dict)}
    if isinstance(people, list):
        out: Dict[str, Dict[str, Any]] = {}
        for idx, p in enumerate(people):
            if not isinstance(p, dict):
                continue
            pid = p.get("person_id", p.get("track_id", p.get("id", idx)))
            out[str(pid)] = p
        return out
    return {}


def _get_fps(data: Dict[str, Any], user_fps: Optional[float]) -> Optional[float]:
    if user_fps and user_fps > 0:
        return float(user_fps)
    meta = data.get("meta") or {}
    fps = meta.get("fps")
    try:
        return float(fps) if fps else None
    except Exception:
        return None


def _extract_visual_seq(person: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("visual_sequence", "visual_actions", "visual_action_sequence", "actions"):
        v = person.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    v = person.get("visual")
    if isinstance(v, list) and v and isinstance(v[0], dict):
        return v
    return []


def _get_action_code(act: Dict[str, Any]) -> int:
    if "action" in act:
        val = str(act["action"]).lower().strip()
        if val in ACTION_STR_TO_ID:
            return ACTION_STR_TO_ID[val]
    for key in ("action_code", "code", "behavior", "label", "cls"):
        if key in act:
            return _safe_int(act.get(key), default=0)
    return 0


def _time_range_from_action(
        act: Dict[str, Any],
        fps: Optional[float],
        default_dur: float = 0.2
) -> Optional[Tuple[float, float]]:
    # 1) start_time/end_time
    for a, b in (("start_time", "end_time"), ("start", "end")):
        if a in act or b in act:
            st = act.get(a, None)
            ed = act.get(b, None)
            try:
                st = float(st) if st is not None else None
                ed = float(ed) if ed is not None else None
            except:
                st, ed = None, None
            if st is not None and ed is not None and ed > st: return st, ed
            if st is not None: return st, st + default_dur
            if ed is not None: return max(0.0, ed - default_dur), ed

    # 2) time + duration
    if "time" in act:
        try:
            t = float(act["time"])
            dur = float(act.get("duration", default_dur))
            return t, t + max(dur, default_dur)
        except:
            pass

    # 3) frame -> fps
    if fps and fps > 0:
        for a, b in (("start_frame", "end_frame"), ("frame_start", "frame_end")):
            if a in act or b in act:
                sf = act.get(a)
                ef = act.get(b)
                try:
                    sf = float(sf) if sf else None; ef = float(ef) if ef else None
                except:
                    continue
                if sf is not None and ef is not None: return sf / fps, ef / fps
                if sf is not None: return sf / fps, sf / fps + default_dur

    # 4) t
    for k in ("t", "timestamp"):
        if k in act:
            try:
                return float(act[k]), float(act[k]) + default_dur
            except:
                pass
    return None


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def merge_and_filter_segments(
        segs: List[Tuple[float, float, int]],
        gap_tol: float = 0.12,
        min_dur: float = 0.35,
) -> List[Tuple[float, float, int]]:
    if not segs: return []
    segs = sorted(segs, key=lambda x: x[0])
    merged: List[List[Any]] = [[segs[0][0], segs[0][1], segs[0][2]]]
    for st, ed, code in segs[1:]:
        last = merged[-1]
        if code == last[2] and st <= last[1] + gap_tol:
            last[1] = max(last[1], ed)
        else:
            merged.append([st, ed, code])
    return [(float(st), float(ed), int(code)) for st, ed, code in merged if (ed - st) >= min_dur]


def load_group_events(path: str) -> List[Dict[str, Any]]:
    events = []
    if not path or not os.path.exists(path):
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except:
                pass
    return events


# ===================== 主绘图 =====================

def draw_timeline(
        data: Dict[str, Any],
        out_path: str,
        group_events: List[Dict[str, Any]] = [],
        fps: Optional[float] = None,
        top_n: int = 50,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        min_total_time: float = 0.0,
        max_people: Optional[int] = None,
        gap_tol: float = 0.12,
        min_dur: float = 0.35,
) -> None:
    people_raw = data.get("people", {})
    people = _people_to_dict(people_raw)
    if not people:
        raise ValueError("No people found in input JSON.")

    fps = _get_fps(data, fps)
    per_person_segments: Dict[str, List[Tuple[float, float, int]]] = {}
    per_person_total: Dict[str, float] = {}
    global_max_t = 0.0

    # 1. 处理个人行为
    for pid, pobj in people.items():
        seq = _extract_visual_seq(pobj)
        segs = []
        for act in seq:
            if not isinstance(act, dict): continue
            tr = _time_range_from_action(act, fps=fps)
            if not tr: continue
            st, ed = tr
            code = _get_action_code(act)

            # Time filter
            if t_min is not None and ed < t_min: continue
            if t_max is not None and st > t_max: continue
            if t_min is not None: st = max(st, t_min)
            if t_max is not None: ed = min(ed, t_max)
            if ed <= st: continue

            segs.append((st, ed, code))
            global_max_t = max(global_max_t, ed)

        merged_segs = merge_and_filter_segments(segs, gap_tol=gap_tol, min_dur=min_dur)
        total = sum((ed - st) for st, ed, _ in merged_segs)
        if total >= min_total_time and merged_segs:
            per_person_segments[pid] = merged_segs
            per_person_total[pid] = total

    if not per_person_segments and not group_events:
        print("[Warning] No valid visual segments found and no group events. Writing empty outputs.")
        _ensure_dir(out_path)
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=240)
        plt.close()
        json_out_path = os.path.splitext(out_path)[0] + ".json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump({"items": []}, f, ensure_ascii=False, indent=2)
        return

    ranked = sorted(per_person_total.items(), key=lambda kv: (-kv[1], _safe_int(kv[0])))
    ranked = ranked[: (max_people if max_people is not None else top_n)]
    sorted_ids = [pid for pid, _ in ranked]

    # 2. 绘图设置
    # 如果有 Group Event，预留顶部一行
    has_group_events = len(group_events) > 0
    group_row_height = 2 if has_group_events else 0

    fig_h = max(4, min(0.35 * len(sorted_ids) + 2 + (1 if has_group_events else 0), 30))
    fig, ax = plt.subplots(figsize=(18, fig_h))

    x_end = t_max if t_max is not None else global_max_t
    if x_end <= 0: x_end = 10.0

    # 3. 绘制群体行为 (Top Row)
    if has_group_events:
        # Group Events 放在最上面，row = len(sorted_ids) + 1
        g_row_y = len(sorted_ids) + 0.5

        # 绘制背景条
        ax.text(-2, g_row_y, "Classroom\nState", va='center', ha='right', fontweight='bold', fontsize=9)

        for ev in group_events:
            gst = ev.get('start_time', 0)
            ged = ev.get('end_time', 0)
            glabel = ev.get('group_event', 'unknown')

            if ged <= gst: continue

            color = GROUP_COLOR_MAP.get(glabel, "#eeeeee")
            ax.barh(
                g_row_y,
                ged - gst,
                left=gst,
                height=0.8,
                color=color,
                edgecolor='none',
                alpha=0.8
            )
            # 标注文字
            if (ged - gst) > 2.0:  # 太短不写字
                ax.text(gst + (ged - gst) / 2, g_row_y, glabel.upper(),
                        ha='center', va='center', fontsize=7, color='#555')

        # 画一条分割线
        ax.axhline(y=len(sorted_ids), color='#ccc', linestyle='--', linewidth=1)

    # 4. 绘制个人行为
    for row, pid in enumerate(sorted_ids):
        segs = per_person_segments[pid]
        for (st, ed, code) in segs:
            ax.barh(
                row,
                ed - st,
                left=st,
                height=0.72,
                color=COLOR_MAP.get(code, "#333333"),
                linewidth=0,
                zorder=2,
            )

    # 坐标轴设置
    yticks = list(range(len(sorted_ids)))
    yticklabels = [f"ID {pid}" for pid in sorted_ids]

    if has_group_events:
        # [优化] 将群体标签加入 Y 轴刻度，替代 ax.text，防止被裁剪
        # 这里的 0.5 是为了让标签对齐到 Group 行的中间
        yticks.append(len(sorted_ids) + 0.5)
        yticklabels.append("Classroom\nState")
        # 既然用了 Y轴标签，就可以把上面的 ax.text(-2, ...) 删掉了


    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Classroom Behavior Timeline | merged gap={gap_tol}s", fontsize=12)

    pad = max(1.0, 0.02 * x_end)
    ax.set_xlim(0, x_end + pad)

    # 图例
    patches = [mpatches.Patch(color=COLOR_MAP[i], label=LABELS[i]) for i in range(9)]
    if has_group_events:
        patches.append(mpatches.Patch(color='none', label='| Group:'))
        for k, v in GROUP_COLOR_MAP.items():
            patches.append(mpatches.Patch(color=v, label=k.title()))

    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(14, len(patches)),
        fontsize=9,
        frameon=False,
    )

    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=240)
    plt.close(fig)

    # 5. 生成 JSON 数据 (含 Group)
    frontend_data = []

    # 5.1 个人数据
    for row_idx, pid in enumerate(sorted_ids):
        segs = per_person_segments[pid]
        for (st, ed, code) in segs:
            frontend_data.append({
                "type": "person",
                "track_id": int(pid),
                "row": row_idx,
                "start": float(f"{st:.2f}"),
                "end": float(f"{ed:.2f}"),
                "action_id": int(code),
                "action_label": LABELS.get(code, "Unknown"),
                "color": COLOR_MAP.get(code, "#000000")
            })

    # 5.2 群体数据
    for ev in group_events:
        frontend_data.append({
            "type": "group",
            "track_id": -1,  # 特殊 ID
            "start": ev.get('start_time'),
            "end": ev.get('end_time'),
            "label": ev.get('group_event'),
            "color": GROUP_COLOR_MAP.get(ev.get('group_event'), "#ccc")
        })

    json_out_path = os.path.splitext(out_path)[0] + ".json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump({"items": frontend_data}, f, ensure_ascii=False, indent=2)

    print(f"✅ Timeline PNG saved: {out_path}")
    print(f"✅ Timeline JSON saved: {json_out_path}")


# ===================== CLI =====================

def main():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="per_person_sequences.json path")
    parser.add_argument("--out", required=True, help="output png path")
    parser.add_argument("--group_src", default="", help="group_events.jsonl path (optional)")

    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--max_people", type=int, default=None)
    parser.add_argument("--t_min", type=float, default=None)
    parser.add_argument("--t_max", type=float, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--min_total_time", type=float, default=0.0)
    parser.add_argument("--gap_tol", type=float, default=0.12)
    parser.add_argument("--min_dur", type=float, default=0.35)

    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.is_absolute():
        src_path = (base_dir / src_path).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    group_src = Path(args.group_src) if args.group_src else None
    if group_src and not group_src.is_absolute():
        group_src = (base_dir / group_src).resolve()

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    group_events = []
    if group_src:
        group_events = load_group_events(str(group_src))

    draw_timeline(
        data=data,
        out_path=str(out_path),
        group_events=group_events,  # Pass loaded group events
        fps=args.fps,
        top_n=args.top_n,
        t_min=args.t_min,
        t_max=args.t_max,
        min_total_time=args.min_total_time,
        max_people=args.max_people,
        gap_tol=args.gap_tol,
        min_dur=args.min_dur,
    )


if __name__ == "__main__":
    main()
