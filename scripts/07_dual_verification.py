# scripts/07_dual_verification.py
import json
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ================= 配置区 =================
DEFAULT_VIDEO_FPS = None          # 如果 actions 里没有 start_time/end_time 只有 frame，就需要 fps
DEFAULT_VISUAL_TIME_MODE = "start"  # start / end / mid：动作时间戳用开始/结束/中点
SCHEMA_VERSION = "1.1.0"
# ==========================================


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    data.append(obj)
            except Exception:
                # 允许少量脏行，但不直接崩
                continue
    return data


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def normalize_visual_actions(
    raw_actions: List[Dict[str, Any]],
    fps: Optional[float] = None,
    time_mode: str = "start",
) -> List[Dict[str, Any]]:
    """
    统一 actions.jsonl -> 标准动作片段字段（科研复现核心：统一 schema）
    输出字段：
      track_id, action, confidence, time, start_time, end_time, duration,
      (optional) side, start_frame, end_frame
    """
    out: List[Dict[str, Any]] = []

    if time_mode not in ("start", "end", "mid"):
        time_mode = "start"

    for a in raw_actions:
        if not isinstance(a, dict):
            continue

        track_id = a.get("track_id", a.get("id", a.get("student_id", a.get("tid"))))
        action = a.get("action", a.get("label", a.get("class_name", a.get("class"))))
        conf = a.get("confidence", a.get("conf", a.get("score", 1.0)))

        track_id_i = _to_int(track_id)
        if track_id_i is None or action is None:
            continue

        conf_f = _to_float(conf)
        if conf_f is None:
            conf_f = 1.0

        # 优先用 start_time/end_time（你当前 actions.jsonl 已包含）
        st = _to_float(a.get("start_time"))
        et = _to_float(a.get("end_time"))

        # fallback：frame + fps
        if st is None and et is None:
            sf = _to_float(a.get("start_frame"))
            ef = _to_float(a.get("end_frame"))
            if fps and (sf is not None or ef is not None):
                if sf is not None:
                    st = sf / float(fps)
                if ef is not None:
                    et = ef / float(fps)

        # time：动作代表点（start/end/mid）
        t: Optional[float] = None
        if st is not None and et is not None:
            if time_mode == "start":
                t = st
            elif time_mode == "end":
                t = et
            else:
                t = (st + et) / 2.0
        elif st is not None:
            t = st
        elif et is not None:
            t = et

        # duration
        dur = _to_float(a.get("duration"))
        if dur is None and st is not None and et is not None:
            dur = max(0.0, et - st)

        item: Dict[str, Any] = {
            "track_id": track_id_i,
            "action": str(action).lower().strip(),
            "confidence": float(conf_f),
            "time": float(t) if t is not None else None,
            "start_time": float(st) if st is not None else None,
            "end_time": float(et) if et is not None else None,
            "duration": float(dur) if dur is not None else None,
        }

        # 保留你可能用到的字段（但不强依赖）
        if "side" in a:
            item["side"] = a["side"]
        if "start_frame" in a:
            item["start_frame"] = a["start_frame"]
        if "end_frame" in a:
            item["end_frame"] = a["end_frame"]
        if "objects_found" in a:
            item["objects_found"] = a["objects_found"]

        out.append(item)

    # 科研级一致性：先按 track_id 再按 time 排
    out.sort(key=lambda x: (x["track_id"], x["time"] if x["time"] is not None else 1e18))
    return out


def normalize_transcripts(raw_transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    transcript.jsonl 标准化：start, end, text
    """
    out: List[Dict[str, Any]] = []
    for r in raw_transcripts:
        if not isinstance(r, dict):
            continue
        text = r.get("text")
        st = _to_float(r.get("start", r.get("time_start", r.get("ts_start"))))
        et = _to_float(r.get("end", r.get("time_end", r.get("ts_end"))))
        if text is None or st is None or et is None:
            continue
        if et < st:
            # 防御性修正：避免上游异常造成的倒序
            st, et = et, st
        out.append({"start": float(st), "end": float(et), "text": str(text)})
    out.sort(key=lambda x: x["start"])
    return out


def _validate_visual_actions(visual_actions: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    返回：valid_count, invalid_count
    """
    valid = 0
    invalid = 0
    for a in visual_actions:
        ok = True
        if _to_int(a.get("track_id")) is None:
            ok = False
        if not a.get("action"):
            ok = False
        # time 可以 None（但尽量不要），这里不强制
        if ok:
            valid += 1
        else:
            invalid += 1
    return valid, invalid


def build_per_person_sequences(
    visual_actions: List[Dict[str, Any]],
    transcripts: List[Dict[str, Any]],
    duplicate_speech_per_person: bool = True,
) -> List[Dict[str, Any]]:
    """
    输出 people: List[person]
    person:
      {
        "track_id": int,
        "person_id": int,   # alias，方便论文/前端
        "visual_sequence": [...],
        "speech_sequence": [...],  # 默认复制全局 transcript（兼容旧下游）
      }
    """
    per_map: Dict[int, Dict[str, Any]] = {}

    # 收集所有出现过的 track_id
    ids = sorted({a["track_id"] for a in visual_actions if isinstance(a, dict) and "track_id" in a})
    for tid in ids:
        per_map[tid] = {
            "track_id": int(tid),
            "person_id": int(tid),
            "visual_sequence": [],
        }
        if duplicate_speech_per_person:
            per_map[tid]["speech_sequence"] = transcripts

    # 分发动作
    for a in visual_actions:
        tid = _to_int(a.get("track_id"))
        if tid is None:
            continue
        if tid not in per_map:
            per_map[tid] = {
                "track_id": int(tid),
                "person_id": int(tid),
                "visual_sequence": [],
            }
            if duplicate_speech_per_person:
                per_map[tid]["speech_sequence"] = transcripts

        per_map[tid]["visual_sequence"].append(a)

    # 每个人的 visual_sequence 再按时间排序一次（防御性）
    people: List[Dict[str, Any]] = []
    for tid in sorted(per_map.keys()):
        p = per_map[tid]
        vs = p.get("visual_sequence", [])
        if isinstance(vs, list):
            vs.sort(key=lambda x: x.get("time", 1e18) if x.get("time") is not None else 1e18)
        people.append(p)

    return people


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Step07: Dual Verification Merge (research-grade schema)")
    parser.add_argument("--actions", type=str, required=True, help="actions.jsonl path")
    parser.add_argument("--transcript", type=str, required=True, help="transcript.jsonl path")
    parser.add_argument("--out", type=str, required=True, help="per_person_sequences.json output path")

    parser.add_argument("--fps", type=float, default=DEFAULT_VIDEO_FPS, help="fallback fps if actions only have frames")
    parser.add_argument("--time_mode", type=str, default=DEFAULT_VISUAL_TIME_MODE, choices=["start", "end", "mid"])
    parser.add_argument(
        "--duplicate_speech",
        type=int,
        default=1,
        help="1=copy transcript into each person (compat), 0=only keep top-level transcript",
    )

    args = parser.parse_args()

    action_path = Path(args.actions)
    transcript_path = Path(args.transcript)
    out_path = Path(args.out)

    if not action_path.is_absolute():
        action_path = (base_dir / action_path).resolve()
    if not transcript_path.is_absolute():
        transcript_path = (base_dir / transcript_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    if not action_path.exists():
        raise FileNotFoundError(f"找不到 actions.jsonl: {action_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"找不到 transcript.jsonl: {transcript_path}")

    raw_actions = load_jsonl(action_path)
    raw_transcripts = load_jsonl(transcript_path)

    visual_actions = normalize_visual_actions(raw_actions, fps=args.fps, time_mode=args.time_mode)
    transcripts = normalize_transcripts(raw_transcripts)

    valid_cnt, invalid_cnt = _validate_visual_actions(visual_actions)

    duplicate = bool(int(args.duplicate_speech) == 1)
    people = build_per_person_sequences(visual_actions, transcripts, duplicate_speech_per_person=duplicate)

    result: Dict[str, Any] = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "visual_time_mode": args.time_mode,
            "fps": args.fps,
            "total_people": len(people),
            "total_visual_actions": len(visual_actions),
            "total_visual_actions_valid": valid_cnt,
            "total_visual_actions_invalid": invalid_cnt,
            "total_speech_segments": len(transcripts),
            "duplicate_speech_per_person": duplicate,
            "inputs": {
                "actions": str(action_path),
                "transcript": str(transcript_path),
            },
        },
        # 顶层保留全局 transcript，后续你想做“每个人绑定说话人”也从这里扩展
        "speech_sequence": transcripts,
        # ✅ 关键修复：people 必须是 list（给 10/可视化/统计统一）
        "people": people,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ Step07 输出完成:", out_path)
    print("meta:", result["meta"])
    # 额外提示：如果 people 是 list，下游就不会再出现 “people not list” 这种灾难
    print(f"[CHECK] people type = {type(result['people']).__name__}, len={len(result['people'])}")


if __name__ == "__main__":
    main()
