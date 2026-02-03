# scripts/intelligence_class/tools/xx_generate_timeline_viz.py
import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

try:
    from scripts.intelligence_class._utils.action_map import ACTION_MAP, LABEL_NORMALIZE
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.intelligence_class._utils.action_map import ACTION_MAP, LABEL_NORMALIZE


def load_jsonl(path: Path):
    data = []
    if not path or not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return data


# =========================================================
# 绑定 track_id：用 pose_tracks_smooth.jsonl 做身份锚点
# =========================================================

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _index_tracks_by_frame(tracks_jsonl: Path):
    """返回: {frame_idx: [(track_id, bbox_xyxy), ...]}"""
    idx = {}
    if not tracks_jsonl or not tracks_jsonl.exists():
        return idx

    for row in load_jsonl(tracks_jsonl):
        f = row.get("frame_idx", row.get("frame"))
        if f is None:
            continue
        f = int(f)

        people = []
        for p in row.get("persons", []) or []:
            tid = p.get("track_id")
            bbox = p.get("bbox")
            if tid is None or bbox is None:
                continue
            people.append((int(tid), bbox))

        idx[f] = people

    return idx


def _match_tid(det_bbox, people, iou_thr=0.15):
    if det_bbox is None or not people:
        return 0
    best_tid, best_iou = 0, 0.0
    for tid, tb in people:
        v = _iou_xyxy(det_bbox, tb)
        if v > best_iou:
            best_iou, best_tid = v, tid
    return best_tid if best_iou >= iou_thr else 0


def _norm_action(label: str) -> str:
    if not label:
        return ""
    s = str(label).strip()
    s_low = s.lower()
    # 有的label是中文或其他格式，就先按原样；有映射的就映射
    return LABEL_NORMALIZE.get(s_low, LABEL_NORMALIZE.get(s, s))


def compress_timeline(
    frames_data: list,
    fps: float = 25.0,
    gap_sec: float = 0.2,
    min_event_sec: float = 0.2,
) -> list:
    """核心算法：将离散的帧压缩为连续的时间段 (Gantt Blocks)"""
    if not frames_data:
        return []

    gap_frames = max(1, math.ceil(gap_sec * fps))
    min_event_frames = max(1, math.ceil(min_event_sec * fps))

    # 1. 按 track_id 分组
    by_track = defaultdict(list)
    for item in frames_data:
        tid = item.get("track_id")
        action = item.get("action") or item.get("label")
        fidx = item.get("frame_idx")
        if fidx is None:
            fidx = item.get("frame")

        if tid is not None and action and fidx is not None:
            by_track[int(tid)].append({"frame": int(fidx), "action": action})

    compressed_events = []

    # 2. 对每个人的轨迹进行压缩
    for tid, trace in by_track.items():
        trace.sort(key=lambda x: x["frame"])
        if not trace:
            continue

        current_event = None
        for point in trace:
            f = point["frame"]
            act = point["action"]

            if current_event is None:
                current_event = {"track_id": tid, "action": act, "start_frame": f, "end_frame": f}
                continue

            is_same_action = (act == current_event["action"])
            is_continuous = (f - current_event["end_frame"] <= gap_frames)

            if is_same_action and is_continuous:
                current_event["end_frame"] = f
            else:
                if (current_event["end_frame"] - current_event["start_frame"]) >= min_event_frames:
                    compressed_events.append(current_event)
                current_event = {"track_id": tid, "action": act, "start_frame": f, "end_frame": f}

        if current_event and (current_event["end_frame"] - current_event["start_frame"]) >= min_event_frames:
            compressed_events.append(current_event)

    # 3. 格式化输出
    final_output = []
    for evt in compressed_events:
        act = evt["action"]
        final_output.append({
            "track_id": evt["track_id"],
            "action": act,
            "action_id": ACTION_MAP.get(act, 0),
            "start": round(evt["start_frame"] / fps, 2),
            "end": round(evt["end_frame"] / fps, 2),
            "frame_idx": evt["start_frame"],
            "duration": round((evt["end_frame"] - evt["start_frame"]) / fps, 2),
        })

    final_output.sort(key=lambda x: x["start"])
    return final_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True, help="案例输出目录, e.g. output/rear__001")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--short_video", type=int, default=0, help="use shorter gap/min durations for short clips")
    parser.add_argument("--gap_sec", type=float, default=None, help="max gap (seconds) to merge actions")
    parser.add_argument("--min_event_sec", type=float, default=None, help="min duration (seconds) to keep an event")

    # 新增：强制选择数据源
    # auto: 保持原逻辑（优先 actions.jsonl，找不到才用 *_behavior.jsonl）
    # actions: 只用 actions.jsonl
    # behavior: 只用 *_behavior.jsonl
    parser.add_argument("--source", choices=["auto", "actions", "behavior"], default="auto")

    # 新增：当 source=behavior 时，用 tracks 把 det bbox 绑定到稳定 track_id
    parser.add_argument(
        "--tracks",
        type=str,
        default="",
        help="用于分配 track_id 的 tracks.jsonl（例如 pose_tracks_smooth.jsonl）。为空则不绑定。",
    )

    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        print(f"❌ 目录不存在: {case_dir}")
        return

    # 1. 选择数据源
    source_file = None
    source_type = None

    actions_file = case_dir / "actions.jsonl"
    behavior_cands = list(case_dir.glob("*_behavior.jsonl"))
    behavior_file = behavior_cands[0] if behavior_cands else None

    if args.source == "actions":
        source_file, source_type = actions_file, "actions"
    elif args.source == "behavior":
        source_file, source_type = behavior_file, "yolo_behavior"
    else:
        # auto
        if actions_file.exists():
            source_file, source_type = actions_file, "actions"
        elif behavior_file and behavior_file.exists():
            source_file, source_type = behavior_file, "yolo_behavior"

    if not source_file or not source_file.exists():
        print(f"⚠️  在 {case_dir} 中未找到可用数据源：actions.jsonl 或 *_behavior.jsonl")
        with open(case_dir / "timeline_viz.json", "w", encoding="utf-8") as f:
            json.dump({"items": [], "fps": args.fps}, f)
        return

    print(f"[Timeline] 处理: {source_file.name} ({source_type})")

    # 2. 加载
    raw_data = load_jsonl(source_file)

    # 3. 如为 behavior：展平 +（可选）绑定 track_id
    if source_type == "yolo_behavior":
        tracks_idx = {}
        tracks_path = Path(args.tracks) if args.tracks else None
        if tracks_path and not tracks_path.is_absolute():
            # 相对路径默认从 case_dir 里找
            tracks_path = (case_dir / tracks_path).resolve()

        if tracks_path and tracks_path.exists():
            tracks_idx = _index_tracks_by_frame(tracks_path)
            print(f"[Timeline] tracks 绑定启用: {tracks_path.name} (frames={len(tracks_idx)})")
        else:
            if args.tracks:
                    print(f"[Timeline] tracks 文件不存在，跳过绑定: {tracks_path}")

        flat_data = []
        for row in raw_data:
            fidx = row.get("frame_idx", row.get("frame"))
            if fidx is None:
                continue
            fidx = int(fidx)

            # 兼容结构1：{frame_idx, dets:[{label, xyxy, ...}]}
            if "dets" in row:
                people = tracks_idx.get(fidx, []) if tracks_idx else []
                for d in row.get("dets", []) or []:
                    lbl = _norm_action(d.get("label") or d.get("action") or d.get("cls"))
                    if not lbl:
                        continue

                    # 取 bbox：优先 xyxy（case_det/jsonl 是这个字段），其次 bbox
                    det_bbox = d.get("xyxy") or d.get("bbox")

                    # 优先用已有 track_id；没有就用 tracks 绑定
                    tid = d.get("track_id")
                    if tid is None:
                        tid = d.get("id")
                    if tid is None:
                        tid = _match_tid(det_bbox, people) if tracks_idx else 0

                    # 最终 action 写成前端动作名（保持前端颜色逻辑）
                    flat_data.append({"frame_idx": fidx, "track_id": int(tid), "action": lbl})

            # 兼容结构2：{frame_idx, persons:[{track_id, action, ...}]}
            elif "persons" in row:
                for p in row.get("persons", []) or []:
                    lbl = _norm_action(p.get("action") or p.get("label"))
                    if not lbl:
                        continue
                    flat_data.append({"frame_idx": fidx, "track_id": int(p.get("track_id", 0)), "action": lbl})

        raw_data = flat_data

    # 4. 压缩生成 Gantt 数据
    gap_sec = args.gap_sec
    min_event_sec = args.min_event_sec
    if gap_sec is None:
        gap_sec = 0.12 if int(args.short_video) == 1 else 0.2
    if min_event_sec is None:
        min_event_sec = 0.12 if int(args.short_video) == 1 else 0.2

    timeline_items = compress_timeline(raw_data, args.fps, gap_sec=gap_sec, min_event_sec=min_event_sec)

    # 5. 保存
    out_file = case_dir / "timeline_viz.json"
    output_obj = {
        "meta": {
            "source": source_file.name,
            "source_type": source_type,
            "fps": args.fps,
            "total_events": len(timeline_items),
            "track_ids": sorted(list(set(d["track_id"] for d in timeline_items))) if timeline_items else [],
        },
        "items": timeline_items,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

    print(f"[Timeline] 已生成: {out_file} (事件数={len(timeline_items)})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        case_dir = None
        for i, arg in enumerate(sys.argv):
            if arg == "--case_dir" and i + 1 < len(sys.argv):
                case_dir = Path(sys.argv[i + 1])
                break
        if case_dir and case_dir.exists():
            try:
                fallback = {
                    "meta": {"error": str(exc), "source": None, "source_type": None, "fps": None},
                    "items": [],
                }
                with open(case_dir / "timeline_viz.json", "w", encoding="utf-8") as f:
                    json.dump(fallback, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        print(f"[Timeline] generation failed: {exc}")
