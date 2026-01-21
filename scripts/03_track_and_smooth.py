import os
import json
import math
import argparse
from collections import defaultdict
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]



def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ema_update(prev, cur, alpha=0.75):
    out = []
    for p, c in zip(prev, cur):
        if c["c"] is None or c["c"] < 0.30:
            out.append({"x": p["x"], "y": p["y"], "c": c["c"]})
        else:
            out.append({
                "x": alpha * c["x"] + (1 - alpha) * p["x"],
                "y": alpha * c["y"] + (1 - alpha) * p["y"],
                "c": c["c"],
            })
    return out


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ===== 命令行参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, default=os.path.join("output", "pose_keypoints_v2.jsonl"))
    parser.add_argument("--out", dest="out_path", type=str, default=os.path.join("output", "pose_tracks_smooth.jsonl"))
    parser.add_argument("--video", dest="video_path", type=str, default=os.path.join("data", "videos", "demo1.mp4"))
    args = parser.parse_args()

    in_path = os.path.join(base_dir, args.in_path)
    out_path = os.path.join(base_dir, args.out_path)
    video_path = os.path.join(base_dir, args.video_path)

    # ===== 用视频拿分辨率 =====
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        w, h = 1920, 1080

    frame_diag = math.hypot(w, h)

    # ====== 核心优化参数 (针对 ID 爆炸问题) ======
    person_conf_thres = 0.45
    alpha = 0.75

    # [修改点 1] 允许更长遮挡：从 150 改为 500
    # 500帧 / 25fps = 20秒。
    # 只要学生在 20 秒内重新被检测到，且位置没大变，就算同一个人。
    max_lost_frames = 500

    # 匹配更依赖位置：pose bbox 抖动大，IOU 要放宽
    iou_thres = 0.05

    # [修改点 2] 稍微放宽匹配半径，从 0.10 改为 0.15
    # 防止学生大幅度动作（如站起）导致中心点漂移出匹配范围
    max_center_dist = 0.15 * frame_diag

    # ★课堂先验：同一学生横向位置基本不动（强约束）
    # x 方向允许漂移：画面宽度的 5% (保持不变，防止把同桌认错)
    max_dx = 0.05 * w

    # [修改点 3] 过滤短暂出现的人：从 30 改为 75
    # 75帧 = 3秒。只有由于持续存在超过3秒的检测才会被记录。
    track_min_frames = 75

    next_id = 1
    tracks = {}  # tid -> state
    track_lengths = defaultdict(int)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input jsonl not found: {in_path}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ====== [新增] 临时文件：第一遍写盘，避免 temp_records 内存爆炸 ======
    tmp_path = out_path + ".tmp.jsonl"

    print(f"[INFO] Tracking started using strict logic...")
    print(f"       max_lost_frames={max_lost_frames} (Memory ~20s)")
    print(f"       track_min_frames={track_min_frames} (Filter <3s noise)")
    print(f"       tmp_path={tmp_path}")

    # ====== 第一遍：跑 tracking + 直接写临时 jsonl ======
    with open(in_path, "r", encoding="utf-8") as f_in, open(tmp_path, "w", encoding="utf-8") as f_tmp:
        for line_idx, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # 输入 jsonl 某一行坏了就跳过，避免整条视频重跑
                continue

            frame = rec["frame"]
            t = rec.get("t", None)

            dets = []
            for p in rec.get("persons", []):
                if p.get("conf", 1.0) < person_conf_thres:
                    continue
                if "bbox" not in p:
                    continue
                dets.append(p)

            det_to_tid = [-1] * len(dets)

            # ====== 匹配已有 track ======
            for di, d in enumerate(dets):
                bc = bbox_center(d["bbox"])
                best_tid, best_score = None, -1e9

                for tid, st in tracks.items():
                    # 检查是否过期
                    if frame - st["last_frame"] > max_lost_frames:
                        continue

                    prev_c = bbox_center(st["bbox"])

                    # ★课堂先验：横向漂移过大直接拒绝匹配 (最重要的防串ID逻辑)
                    if abs(bc[0] - prev_c[0]) > max_dx:
                        continue

                    dc = dist(bc, prev_c)
                    if dc > max_center_dist:
                        continue

                    iou = iou_xyxy(d["bbox"], st["bbox"])

                    # bbox 高度稳定性（课堂强特征）
                    h_cur = d["bbox"][3] - d["bbox"][1]
                    h_prev = st["bbox"][3] - st["bbox"][1]
                    h_ratio = abs(h_cur - h_prev) / max(h_prev, 1.0)

                    # 打分：位置优先，其次 IOU，再惩罚高度波动
                    score = 1.5 * iou - 1.2 * (dc / max_center_dist) - 0.6 * h_ratio

                    # 如果 IOU 很低也不要直接否决（pose bbox 会跳）
                    # 但位置和高度要过关
                    if iou < iou_thres and dc > (0.06 * frame_diag):
                        continue

                    if score > best_score:
                        best_score, best_tid = score, tid

                if best_tid is not None:
                    det_to_tid[di] = best_tid

            # ====== 冲突解决：一个 track 只能匹配一个 det ======
            used = {}
            for di, tid in enumerate(det_to_tid):
                if tid == -1:
                    continue
                bc = bbox_center(dets[di]["bbox"])
                prev_c = bbox_center(tracks[tid]["bbox"])
                dc = dist(bc, prev_c)
                iou = iou_xyxy(dets[di]["bbox"], tracks[tid]["bbox"])
                score = 1.5 * iou - 1.2 * (dc / max_center_dist)
                if tid not in used or score > used[tid][0]:
                    used[tid] = (score, di)

            det_to_tid = [-1] * len(dets)
            for tid, (_, di) in used.items():
                det_to_tid[di] = tid

            # ====== 新建 track ======
            for di, d in enumerate(dets):
                if det_to_tid[di] == -1:
                    tid = next_id
                    next_id += 1
                    det_to_tid[di] = tid
                    tracks[tid] = {
                        "bbox": d["bbox"],
                        "kpts": d["keypoints"],
                        "last_frame": frame,
                        "len": 0
                    }

            # ====== 更新状态 + EMA ======
            frame_out = []
            for di, d in enumerate(dets):
                tid = det_to_tid[di]
                st = tracks[tid]

                if st["len"] == 0:
                    smooth = d["keypoints"]
                else:
                    smooth = ema_update(st["kpts"], d["keypoints"], alpha)

                st["bbox"] = d["bbox"]
                st["kpts"] = smooth
                st["last_frame"] = frame
                st["len"] += 1
                track_lengths[tid] += 1

                frame_out.append({
                    "track_id": tid,
                    "bbox": d["bbox"],
                    "conf": d.get("conf", None),
                    "keypoints": smooth
                })

            # ====== [修改核心] 不再 append 到 temp_records，直接写盘 ======
            if frame_out:
                f_tmp.write(json.dumps({
                    "frame": frame,
                    "t": t,
                    "persons": frame_out
                }, ensure_ascii=False) + "\n")

            # 可选：定期 flush，避免中途崩溃丢太多
            if line_idx % 200 == 0:
                f_tmp.flush()

    # ====== 过滤短轨迹 (关键步骤) ======
    valid_ids = {tid for tid, ln in track_lengths.items() if ln >= track_min_frames}

    # ====== 第二遍：读临时文件 -> 过滤 -> 写最终输出 ======
    with open(tmp_path, "r", encoding="utf-8") as fi, open(out_path, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ps = [p for p in rec.get("persons", []) if p.get("track_id") in valid_ids]
            if ps:
                rec["persons"] = ps
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 清理临时文件（失败也不影响主结果）
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    print("[DONE] pose_tracks_smooth.jsonl:", out_path)
    print(f"[INFO] Final valid IDs count: {len(valid_ids)} (Filtered out {next_id - 1 - len(valid_ids)} short tracks)")


if __name__ == "__main__":
    main()
