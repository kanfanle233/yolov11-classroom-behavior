# debug_pose_tracks.py
# -*- coding: utf-8 -*-
"""
Debug script for pose_tracks_smooth.jsonl schema / keypoints quality.

It helps you answer:
1) records use frame or frame_idx? persons or something else?
2) per-person: keypoints is None? list length? contains None items?
3) required keypoints (nose, shoulders, wrists, hips) are missing how often?
4) does record contain 't' / 'time_sec'? (for action rules timeline)
5) quick per-track coverage stats
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple


# COCO keypoint indices (Ultralytics pose usually follows COCO-17)
NOSE = 0
LS, RS = 5, 6
LW, RW = 9, 10
LH, RH = 11, 12

REQUIRED_KPTS = [NOSE, LS, RS, LW, RW, LH, RH]


def _load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except Exception as e:
                yield i, {"__parse_error__": str(e), "__raw__": line}


def _kpt_is_missing(k: Any) -> bool:
    """k should be dict with x/y or list/tuple, but many pipelines use dicts."""
    if k is None:
        return True
    if isinstance(k, dict):
        # some pipelines store {"x":..,"y":..,"conf":..}
        return ("x" not in k) or ("y" not in k) or (k["x"] is None) or (k["y"] is None)
    if isinstance(k, (list, tuple)):
        # could be [x,y,conf] or [x,y]
        if len(k) < 2:
            return True
        return (k[0] is None) or (k[1] is None)
    # unknown type
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="pose_tracks_smooth.jsonl path")
    ap.add_argument("--max_bad_examples", type=int, default=10, help="how many bad samples to print")
    ap.add_argument("--max_track_examples", type=int, default=5, help="how many track summaries to print")
    args = ap.parse_args()

    in_path = Path(args.in_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"not found: {in_path}")

    # ====== global stats ======
    rec_count = 0
    parse_errors = 0

    key_counter = Counter()
    frame_key_counter = Counter()
    time_key_counter = Counter()

    persons_total = 0
    persons_missing_trackid = 0
    persons_missing_bbox = 0

    kpts_none = 0
    kpts_not_list = 0
    kpts_len_counter = Counter()

    req_missing_counter = Counter()  # which required kpt index missing
    person_bad_reason = Counter()

    # per track stats
    track_frames = defaultdict(int)
    track_person_records = defaultdict(int)
    track_req_ok = defaultdict(int)  # records where all required keypoints exist

    bad_examples: List[Dict[str, Any]] = []

    for line_no, rec in _load_jsonl(in_path):
        rec_count += 1

        # parse error line
        if "__parse_error__" in rec:
            parse_errors += 1
            if len(bad_examples) < args.max_bad_examples:
                bad_examples.append({"line": line_no, "reason": "json_parse_error", "detail": rec.get("__parse_error__")})
            continue

        if not isinstance(rec, dict):
            if len(bad_examples) < args.max_bad_examples:
                bad_examples.append({"line": line_no, "reason": "record_not_dict", "type": str(type(rec))})
            continue

        for k in rec.keys():
            key_counter[k] += 1

        # frame key detection
        frame_val = None
        if "frame" in rec:
            frame_key_counter["frame"] += 1
            frame_val = rec.get("frame")
        elif "frame_idx" in rec:
            frame_key_counter["frame_idx"] += 1
            frame_val = rec.get("frame_idx")
        else:
            frame_key_counter["no_frame_key"] += 1

        # time key detection
        if "t" in rec:
            time_key_counter["t"] += 1
        if "time_sec" in rec:
            time_key_counter["time_sec"] += 1
        if ("t" not in rec) and ("time_sec" not in rec):
            time_key_counter["no_time_key"] += 1

        persons = rec.get("persons", None)
        if not isinstance(persons, list):
            if len(bad_examples) < args.max_bad_examples:
                bad_examples.append({"line": line_no, "reason": "persons_not_list", "keys": list(rec.keys())[:30]})
            continue

        persons_total += len(persons)

        for p in persons:
            if not isinstance(p, dict):
                person_bad_reason["person_not_dict"] += 1
                continue

            tid = p.get("track_id", None)
            if tid is None:
                persons_missing_trackid += 1
                person_bad_reason["missing_track_id"] += 1
                # still continue to inspect kpts

            bbox = p.get("bbox", None)
            if bbox is None:
                persons_missing_bbox += 1
                person_bad_reason["missing_bbox"] += 1

            kpts = p.get("keypoints", None)

            # keypoints none
            if kpts is None:
                kpts_none += 1
                person_bad_reason["kpts_none"] += 1
                if len(bad_examples) < args.max_bad_examples:
                    bad_examples.append({
                        "line": line_no,
                        "reason": "kpts_none",
                        "frame": frame_val,
                        "track_id": tid,
                        "person_keys": list(p.keys()),
                    })
                continue

            # keypoints type/len
            if not isinstance(kpts, list):
                kpts_not_list += 1
                person_bad_reason["kpts_not_list"] += 1
                if len(bad_examples) < args.max_bad_examples:
                    bad_examples.append({
                        "line": line_no,
                        "reason": "kpts_not_list",
                        "frame": frame_val,
                        "track_id": tid,
                        "kpts_type": str(type(kpts)),
                    })
                continue

            kpts_len_counter[len(kpts)] += 1

            # required keypoints check
            missing_req = []
            if len(kpts) <= max(REQUIRED_KPTS):
                # too short
                missing_req = REQUIRED_KPTS[:]  # all missing
            else:
                for idx in REQUIRED_KPTS:
                    if _kpt_is_missing(kpts[idx]):
                        missing_req.append(idx)

            for idx in missing_req:
                req_missing_counter[idx] += 1

            if tid is not None:
                track_frames[tid] += 1
                track_person_records[tid] += 1
                if not missing_req:
                    track_req_ok[tid] += 1

            if missing_req and len(bad_examples) < args.max_bad_examples:
                bad_examples.append({
                    "line": line_no,
                    "reason": "required_kpt_missing",
                    "frame": frame_val,
                    "track_id": tid,
                    "missing_idx": missing_req,
                    "kpts_len": len(kpts),
                })

    # ====== print report ======
    print("\n" + "=" * 90)
    print("[DEBUG REPORT] pose_tracks_smooth.jsonl")
    print(f"File: {in_path}")
    print("=" * 90)

    print(f"Records: {rec_count}")
    print(f"Parse errors: {parse_errors}")
    print(f"Persons total: {persons_total}")
    print(f"Missing track_id: {persons_missing_trackid}")
    print(f"Missing bbox: {persons_missing_bbox}")
    print("-" * 90)

    print("[Frame key distribution]:", dict(frame_key_counter))
    print("[Time key distribution ]:", dict(time_key_counter))
    print("-" * 90)

    print(f"keypoints None: {kpts_none}")
    print(f"keypoints not list: {kpts_not_list}")
    print(f"keypoints length distribution (top 10): {kpts_len_counter.most_common(10)}")
    print("-" * 90)

    idx_name = {
        NOSE: "NOSE",
        LS: "L_SHOULDER",
        RS: "R_SHOULDER",
        LW: "L_WRIST",
        RW: "R_WRIST",
        LH: "L_HIP",
        RH: "R_HIP",
    }
    req_missing_named = {idx_name.get(k, str(k)): v for k, v in req_missing_counter.items()}
    print("[Required keypoints missing count]:")
    for k, v in sorted(req_missing_named.items(), key=lambda x: -x[1]):
        print(f"  {k:>11}: {v}")
    print("-" * 90)

    # per-track coverage
    tids = sorted(track_frames.keys(), key=lambda t: track_frames[t], reverse=True)
    print(f"[Tracks] unique tracks: {len(tids)}")
    print(f"[Tracks] showing top {min(len(tids), args.max_track_examples)} by frames:")
    for tid in tids[: args.max_track_examples]:
        fcnt = track_frames[tid]
        ok = track_req_ok.get(tid, 0)
        ratio = (ok / fcnt) if fcnt else 0.0
        print(f"  track_id={tid} frames={fcnt} req_ok={ok} req_ok_ratio={ratio:.3f}")
    print("-" * 90)

    print("[Bad reason counter] (top 10):", person_bad_reason.most_common(10))
    print("-" * 90)

    if bad_examples:
        print(f"[Bad examples] showing up to {len(bad_examples)}:")
        for ex in bad_examples:
            print(json.dumps(ex, ensure_ascii=False))
    else:
        print("[Bad examples] none found (nice).")

    print("=" * 90)
    print("[NEXT] If required_kpt_missing is high -> your 04_action_rules must skip bad frames.")
    print("[NEXT] If persons_total is tiny -> pose step or tracking is failing for that view/video.")
    print("=" * 90)


if __name__ == "__main__":
    main()
