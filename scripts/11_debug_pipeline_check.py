# scripts/11_debug_pipeline_check.py
import argparse
import json
from pathlib import Path
from collections import Counter


def iter_jsonl(path: Path, max_lines: int | None = None):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_lines is not None and line_no > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                yield {"__bad_line__": line[:200], "__line_no__": line_no}


def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"__error__": str(e)}


def pick_key(d: dict, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def sample_keys_jsonl(path: Path, n=3, max_lines: int | None = None):
    samples = []
    for i, obj in enumerate(iter_jsonl(path, max_lines=max_lines)):
        samples.append(list(obj.keys())[:50])
        if i + 1 >= n:
            break
    return samples


def main():
    parser = argparse.ArgumentParser(description="检查输出目录中的 JSON/JSONL 结构与字段")
    parser.add_argument("--out_dir", type=str, required=True, help="例如 output/demo5")
    parser.add_argument("--max_lines", type=int, default=2000, help="每个 JSONL 文件最多读取行数")
    parser.add_argument("--sample_size", type=int, default=3, help="采样展示的 JSONL 记录数")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    print("=" * 90)
    print("[DEBUG] out_dir =", out_dir)
    print("=" * 90)
    if not out_dir.exists():
        print("  -> MISSING:", out_dir)
        return

    objects_path = out_dir / "objects.jsonl"
    actions_path = out_dir / "actions.jsonl"
    transcript_path = out_dir / "transcript.jsonl"
    per_person_path = out_dir / "per_person_sequences.json"

    # -------- 1) objects.jsonl quick stats --------
    print("\n[1] objects.jsonl stats")
    if not objects_path.exists():
        print("  -> MISSING:", objects_path)
    else:
        cls_counter = Counter()
        name_counter = Counter()
        lines = 0
        dets = 0
        bad = 0
        for obj in iter_jsonl(objects_path, max_lines=args.max_lines):
            lines += 1
            if "__bad_line__" in obj:
                bad += 1
                continue
            objs = obj.get("objects") or []
            if not isinstance(objs, list):
                continue
            for o in objs:
                dets += 1
                cls_id = o.get("cls_id")
                name = o.get("name") or o.get("cls") or "unknown"
                if name is None:
                    name = "unknown"
                if cls_id is None:
                    cls_counter["NA"] += 1
                else:
                    cls_counter[str(cls_id)] += 1
                name_counter[str(name)] += 1

        print(f"  file={objects_path.name} lines={lines} dets={dets} bad_lines={bad}")
        print("  top names:", name_counter.most_common(10))

    # -------- 2) actions.jsonl schema check --------
    print("\n[2] actions.jsonl schema")
    if not actions_path.exists():
        print("  -> MISSING:", actions_path)
    else:
        print("  samples keys:", sample_keys_jsonl(actions_path, n=args.sample_size, max_lines=args.max_lines))

        # detect probable keys
        # time/frame keys
        t_keys = ["t", "time", "timestamp", "sec", "seconds"]
        f_keys = ["frame", "frame_idx", "fid"]
        pid_keys = ["pid", "person_id", "track_id", "id"]
        act_keys = ["action_code", "action", "label", "behavior", "cls", "class_id"]

        lines = 0
        has_time = 0
        has_pid = 0
        has_act = 0

        first = None
        for obj in iter_jsonl(actions_path, max_lines=args.max_lines):
            if "__bad_line__" in obj:
                continue
            if first is None:
                first = obj
            lines += 1

            if pick_key(obj, t_keys) or pick_key(obj, f_keys):
                has_time += 1
            if pick_key(obj, pid_keys):
                has_pid += 1
            if pick_key(obj, act_keys):
                has_act += 1

        print(f"  checked_lines={lines}")
        print(f"  time/frame present in {has_time}/{lines}")
        print(f"  person id present in {has_pid}/{lines}")
        print(f"  action field present in {has_act}/{lines}")

        if first:
            print("  first example (filtered):")
            keep = {}
            for k in list(first.keys()):
                if k in ["t", "time", "timestamp", "frame", "frame_idx", "pid", "person_id", "track_id",
                         "action_code", "action", "label", "behavior", "cls", "class_id"]:
                    keep[k] = first.get(k)
            print(" ", keep if keep else {k: first.get(k) for k in list(first.keys())[:12]})

    # -------- 3) transcript.jsonl check --------
    print("\n[3] transcript.jsonl schema")
    if not transcript_path.exists():
        print("  -> MISSING:", transcript_path)
    else:
        print("  samples keys:", sample_keys_jsonl(transcript_path, n=args.sample_size, max_lines=args.max_lines))
        lines = 0
        ok = 0
        for obj in iter_jsonl(transcript_path, max_lines=args.max_lines):
            if "__bad_line__" in obj:
                continue
            lines += 1
            if all(k in obj for k in ("start", "end", "text")):
                ok += 1
        print(f"  checked_lines={lines} valid(start/end/text)={ok}/{lines}")

    # -------- 4) per_person_sequences.json check (C confirm) --------
    print("\n[4] per_person_sequences.json (C check)")
    if not per_person_path.exists():
        print("  -> MISSING:", per_person_path)
    else:
        data = safe_read_json(per_person_path)
        if "__error__" in data:
            print("  -> JSON load error:", data["__error__"])
            return

        # support dict or list
        people = None
        if isinstance(data, dict) and "people" in data:
            people = data["people"]
            print("  top-level keys:", list(data.keys())[:30])
        elif isinstance(data, list):
            people = data
            print("  top-level is list")
        else:
            print("  unknown structure type:", type(data))
            print("  keys:", list(data.keys())[:30] if isinstance(data, dict) else "NA")
            return

        if not isinstance(people, list) or len(people) == 0:
            print("  people is empty or not list. len=", len(people) if hasattr(people, "__len__") else "NA")
            return

        # scan a few people for action fields
        found_action = 0
        found_visual = 0
        checked_people = min(8, len(people))
        action_like_keys = set()

        for i in range(checked_people):
            p = people[i] if isinstance(people[i], dict) else {}
            vs = p.get("visual_sequence") or p.get("visual_seq") or p.get("visual") or []
            if isinstance(vs, list) and len(vs) > 0:
                found_visual += 1
                # scan first 50 items
                for item in vs[:50]:
                    if isinstance(item, dict):
                        for k in item.keys():
                            if k in ("action_code", "action", "label", "behavior", "cls", "class_id"):
                                action_like_keys.add(k)
                                found_action += 1
                                break
                    if found_action:
                        break

        print(f"  people total={len(people)} checked={checked_people}")
        print(f"  visual_sequence present in {found_visual}/{checked_people}")
        print(f"  action-like keys found: {sorted(action_like_keys) if action_like_keys else 'NONE'}")

        if not action_like_keys:
            print("\n  >>> CONCLUSION: per_person_sequences.json 里没有 action 字段 -> C 成立：10 没法画对")
            print("  下一步应该查：07_dual_verification.py 是否 join 失败 / 字段名不匹配")
        else:
            print("\n  >>> CONCLUSION: per_person_sequences.json 有 action 字段 -> 10 还画不对就看 10_visualize_timeline.py 的解析逻辑")

    print("\n[DONE] debug check finished.")


if __name__ == "__main__":
    main()
