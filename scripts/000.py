import json
import os
from pathlib import Path



# ===== 配置：不想改其他脚本就用固定位置 =====
ACTIONS_PATH = Path("output/actions.jsonl")
TRANSCRIPT_PATH = Path("output/transcript.jsonl")
OUT_PATH = Path("output/per_person_sequences.json")


def load_jsonl(path: Path):
    data = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def normalize_actions(raw_actions):
    """
    统一动作片段字段为：
    track_id, action, start_time, end_time, confidence, side(optional)
    兼容你现有 actions.jsonl（start_time/end_time/start_frame/end_frame 都不影响）
    """
    out = []
    for a in raw_actions:
        if not isinstance(a, dict):
            continue
        tid = a.get("track_id", a.get("id", a.get("student_id", a.get("tid"))))
        act = a.get("action", a.get("label", a.get("class_name", a.get("class"))))
        st = a.get("start_time", a.get("start"))
        et = a.get("end_time", a.get("end"))
        conf = a.get("confidence", a.get("conf", a.get("score", 1.0)))

        if tid is None or act is None or st is None or et is None:
            continue

        try:
            tid = int(tid)
            st = float(st)
            et = float(et)
            conf = float(conf)
        except Exception:
            continue

        item = {
            "track_id": tid,
            "action": str(act).lower().strip(),
            "start_time": st,
            "end_time": et,
            "confidence": conf,
        }
        if "side" in a:
            item["side"] = a["side"]
        if "start_frame" in a:
            item["start_frame"] = a["start_frame"]
        if "end_frame" in a:
            item["end_frame"] = a["end_frame"]
        if "duration" in a:
            item["duration"] = a["duration"]

        out.append(item)

    out.sort(key=lambda x: (x["track_id"], x["start_time"]))
    return out


def normalize_transcript(raw_transcript):
    """
    transcript.jsonl: 每条至少包含 start/end/text
    """
    out = []
    for r in raw_transcript:
        if not isinstance(r, dict):
            continue
        text = r.get("text")
        st = r.get("start", r.get("time_start", r.get("ts_start")))
        et = r.get("end", r.get("time_end", r.get("ts_end")))
        if text is None or st is None or et is None:
            continue
        st = _to_float(st)
        et = _to_float(et)
        if st is None or et is None:
            continue
        out.append({"start": st, "end": et, "text": str(text)})
    out.sort(key=lambda x: x["start"])
    return out


def main():
    base_dir = Path(__file__).resolve().parents[1]  # runs/ 上一层是项目根
    actions_path = base_dir / ACTIONS_PATH
    transcript_path = base_dir / TRANSCRIPT_PATH
    out_path = base_dir / OUT_PATH

    if not actions_path.exists():
        raise FileNotFoundError(f"找不到 {actions_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"找不到 {transcript_path}")

    raw_actions = load_jsonl(actions_path)
    raw_transcript = load_jsonl(transcript_path)

    actions = normalize_actions(raw_actions)
    transcript = normalize_transcript(raw_transcript)

    track_ids = sorted({a["track_id"] for a in actions})
    people = {}

    for tid in track_ids:
        people[str(tid)] = {
            "person_id": tid,
            "visual_sequence": [],
            "speech_sequence": transcript,  # 全局音频：每个人一份
        }

    for a in actions:
        people[str(a["track_id"])]["visual_sequence"].append(a)

    result = {
        "meta": {
            "total_people": len(track_ids),
            "total_visual_segments": len(actions),
            "total_speech_segments": len(transcript),
        },
        "people": people,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("✅ per_person_sequences.json 已生成:", out_path)
    print("meta:", result["meta"])


if __name__ == "__main__":
    main()
