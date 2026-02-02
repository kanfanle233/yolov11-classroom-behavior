# debug_pose_chain.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
import argparse
from pathlib import Path


def read_jsonl(path: Path, max_lines: int | None = None):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if max_lines is not None and len(out) >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            out.append((i, json.loads(line)))
    return out


def summarize_pose_keypoints(pose_jsonl: Path, sample_frames: int = 3):
    rows = read_jsonl(pose_jsonl, max_lines=200)

    total_frames = 0
    persons_total = 0
    kpts_none = 0
    kpts_list = 0
    kpts_len_hist = {}

    sample_printed = 0

    for _, rec in rows:
        total_frames += 1
        persons = rec.get("persons", [])
        if not isinstance(persons, list):
            continue
        persons_total += len(persons)
        for p in persons:
            kpts = p.get("keypoints", None)
            if kpts is None:
                kpts_none += 1
            elif isinstance(kpts, list):
                kpts_list += 1
                kpts_len_hist[len(kpts)] = kpts_len_hist.get(len(kpts), 0) + 1
            else:
                # weird type counts as none-ish
                kpts_none += 1

        # print a few frames for eyeballing
        if sample_printed < sample_frames:
            print("\n[POSE SAMPLE FRAME]")
            print("frame/frame_idx:", rec.get("frame", rec.get("frame_idx")))
            ps = rec.get("persons", [])
            if isinstance(ps, list) and ps:
                one = ps[0]
                print("person keys:", list(one.keys()))
                print("track_id:", one.get("track_id"))
                print("bbox:", one.get("bbox"))
                k = one.get("keypoints")
                if isinstance(k, list):
                    print("keypoints len:", len(k))
                    print("keypoints[0..2]:", k[:3])
                else:
                    print("keypoints:", k)
            sample_printed += 1

    print("\n" + "=" * 90)
    print("[SUMMARY] pose_keypoints_v2.jsonl (first 200 lines)")
    print(f"frames scanned: {total_frames}")
    print(f"persons total : {persons_total}")
    print(f"kpts None     : {kpts_none}")
    print(f"kpts list     : {kpts_list}")
    print("kpts len hist :", dict(sorted(kpts_len_hist.items(), key=lambda x: -x[1])[:10]))
    print("=" * 90)


def summarize_tracks(tracks_jsonl: Path, sample_frames: int = 3):
    rows = read_jsonl(tracks_jsonl, max_lines=50)

    persons_total = 0
    kpts_none = 0
    kpts_list = 0

    sample_printed = 0
    for _, rec in rows:
        persons = rec.get("persons", [])
        if not isinstance(persons, list):
            continue
        persons_total += len(persons)
        for p in persons:
            kpts = p.get("keypoints", None)
            if kpts is None:
                kpts_none += 1
            elif isinstance(kpts, list):
                kpts_list += 1
            else:
                kpts_none += 1

        if sample_printed < sample_frames:
            print("\n[TRACK SAMPLE FRAME]")
            print("frame/frame_idx:", rec.get("frame", rec.get("frame_idx")))
            ps = rec.get("persons", [])
            if isinstance(ps, list) and ps:
                one = ps[0]
                print("person keys:", list(one.keys()))
                print("track_id:", one.get("track_id"))
                print("bbox:", one.get("bbox"))
                print("keypoints:", one.get("keypoints"))
            sample_printed += 1

    print("\n" + "=" * 90)
    print("[SUMMARY] pose_tracks_smooth.jsonl (first 50 lines)")
    print(f"persons total : {persons_total}")
    print(f"kpts None     : {kpts_none}")
    print(f"kpts list     : {kpts_list}")
    print("=" * 90)


def cross_check_one_frame(pose_jsonl: Path, tracks_jsonl: Path, frame_id: int = 0):
    # build dict: frame -> first person keypoints
    pose_rows = read_jsonl(pose_jsonl, max_lines=500)
    track_rows = read_jsonl(tracks_jsonl, max_lines=500)

    def frame_key(rec):
        return rec.get("frame", rec.get("frame_idx"))

    pose_map = {frame_key(r): r for _, r in pose_rows}
    track_map = {frame_key(r): r for _, r in track_rows}

    pr = pose_map.get(frame_id)
    tr = track_map.get(frame_id)

    print("\n" + "=" * 90)
    print(f"[CROSS CHECK] frame={frame_id}")
    if pr is None:
        print("pose_keypoints_v2.jsonl: frame not found in first 500 lines")
    else:
        ps = pr.get("persons", [])
        print("pose persons:", len(ps) if isinstance(ps, list) else type(ps))
        if isinstance(ps, list) and ps:
            k = ps[0].get("keypoints")
            print("pose first keypoints type:", type(k), "len:", len(k) if isinstance(k, list) else None)
            print("pose first keypoints head:", k[:3] if isinstance(k, list) else k)

    if tr is None:
        print("pose_tracks_smooth.jsonl: frame not found in first 500 lines")
    else:
        ts = tr.get("persons", [])
        print("track persons:", len(ts) if isinstance(ts, list) else type(ts))
        if isinstance(ts, list) and ts:
            k2 = ts[0].get("keypoints")
            print("track first keypoints:", k2)

    print("=" * 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help="pose_keypoints_v2.jsonl")
    ap.add_argument("--tracks", required=True, help="pose_tracks_smooth.jsonl")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    pose = Path(args.pose).resolve()
    tracks = Path(args.tracks).resolve()

    if not pose.exists():
        raise FileNotFoundError(f"pose not found: {pose}")
    if not tracks.exists():
        raise FileNotFoundError(f"tracks not found: {tracks}")

    summarize_pose_keypoints(pose)
    summarize_tracks(tracks)
    cross_check_one_frame(pose, tracks, frame_id=int(args.frame))


if __name__ == "__main__":
    main()
