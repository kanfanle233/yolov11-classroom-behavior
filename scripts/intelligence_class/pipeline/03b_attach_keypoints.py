import argparse
import json
import sys
from pathlib import Path


def calculate_iou(box1, box2):
    """
    计算两个框的 IoU
    box: [x1, y1, x2, y2]
    """
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]

    x_inter1 = max(x1_min, x2_min)
    y_inter1 = max(y1_min, y2_min)
    x_inter2 = min(x1_max, x2_max)
    y_inter2 = min(y1_max, y2_max)

    width_inter = max(0, x_inter2 - x_inter1)
    height_inter = max(0, y_inter2 - y_inter1)
    area_inter = width_inter * height_inter

    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area_box1 + area_box2 - area_inter + 1e-6

    return area_inter / union


def process_frame(track_persons, raw_persons, iou_thr):
    """
    将 raw_persons 中的 keypoints 匹配并附加到 track_persons 上
    """
    matched_count = 0

    for tp in track_persons:
        t_box = tp.get("bbox")
        if not t_box:
            tp["keypoints"] = None
            continue

        best_iou = -1.0
        best_raw_p = None

        # 在原始检测中寻找与当前轨迹框重合度最高的
        for rp in raw_persons:
            r_box = rp.get("bbox")
            if not r_box:
                continue

            iou = calculate_iou(t_box, r_box)
            if iou > best_iou:
                best_iou = iou
                best_raw_p = rp

        # 如果 IoU 达标，就认为匹配上了，把 keypoints 拿过来
        if best_iou >= iou_thr and best_raw_p is not None:
            tp["keypoints"] = best_raw_p.get("keypoints", [])
            if "keypoints_conf" in best_raw_p:
                tp["keypoints_conf"] = best_raw_p["keypoints_conf"]
            matched_count += 1
        else:
            tp["keypoints"] = None

    return track_persons, matched_count


def main():
    parser = argparse.ArgumentParser(description="Attach raw keypoints to smoothed tracks via IoU matching")
    parser.add_argument("--pose", required=True, help="Raw pose jsonl (source, with keypoints)")
    parser.add_argument("--tracks", required=True, help="Smoothed tracks jsonl (target, without keypoints)")
    parser.add_argument("--out", required=True, help="Output jsonl path")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for matching")

    args = parser.parse_args()

    path_pose = Path(args.pose)
    path_tracks = Path(args.tracks)
    path_out = Path(args.out)

    if not path_pose.exists():
        print(f"[ERROR] Pose file not found: {path_pose}")
        sys.exit(1)

    if not path_tracks.exists():
        print(f"[ERROR] Tracks file not found: {path_tracks}")
        sys.exit(1)

    # 1. Load Raw Pose into memory (keyed by frame_idx)
    raw_map = {}
    print(f"[03b] Loading raw pose: {path_pose.name}")
    try:
        with open(path_pose, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    # 兼容 frame 或 frame_idx
                    fi = rec.get("frame_idx")
                    if fi is None:
                        fi = rec.get("frame")

                    if fi is not None:
                        fi = int(fi)
                        # 兼容 'persons' 或 'dets' 字段
                        persons = rec.get("persons") or rec.get("dets") or []
                        raw_map[fi] = persons
                except Exception:
                    continue
    except Exception as e:
        print(f"[ERROR] Failed reading pose file: {e}")
        sys.exit(1)

    print(f"      -> Loaded {len(raw_map)} frames with raw detections.")

    # 2. Stream Tracks, match, and write
    print(f"[03b] Processing tracks: {path_tracks.name} -> {path_out.name}")
    path_out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    total_matched_persons = 0

    with open(path_tracks, "r", encoding="utf-8") as f_in, \
            open(path_out, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line: continue

            try:
                rec = json.loads(line)
                fi = rec.get("frame_idx")
                if fi is None:
                    fi = rec.get("frame")

                track_persons = rec.get("persons", [])

                if fi is not None:
                    fi = int(fi)
                    if fi in raw_map:
                        raw_persons = raw_map[fi]
                        # 执行匹配逻辑
                        track_persons, matched_n = process_frame(track_persons, raw_persons, args.iou_thr)
                        total_matched_persons += matched_n
                    else:
                        # 这一帧没有原始检测数据
                        for p in track_persons:
                            p["keypoints"] = None

                rec["persons"] = track_persons
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"[WARN] Error processing frame line: {e}")
                continue

    print(f"[03b] Done. Processed {count} frames. Total persons matched with keypoints: {total_matched_persons}")


if __name__ == "__main__":
    main()