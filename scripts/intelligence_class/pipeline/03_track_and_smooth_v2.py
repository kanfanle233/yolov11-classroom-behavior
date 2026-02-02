import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Utils: geometry & smoothing
# =========================
def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def center_xy(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def as_floats(x: Any) -> Optional[List[float]]:
    if x is None:
        return None
    if not isinstance(x, list):
        return None
    try:
        return [float(v) for v in x]
    except Exception:
        return None


def ema_list(prev: Optional[List[float]], cur: Optional[List[float]], alpha: float) -> Optional[List[float]]:
    """
    EMA: out = (1-alpha)*prev + alpha*cur
    alpha 越大越“跟手”，越小越平滑
    """
    if cur is None:
        return prev
    if prev is None:
        return cur
    if len(prev) != len(cur):
        return cur

    out = []
    for p, c in zip(prev, cur):
        # cur 里如果出现 None，就沿用 prev
        if c is None:
            out.append(p)
        else:
            out.append((1.0 - alpha) * p + alpha * c)
    return out


# =========================
# Input schema (intelligence_class)
# =========================
"""
期望输入 JSONL 每行一帧，结构建议（你自己在 intelligence_class 里统一即可）：

{
  "frame": 12,                 # 或 frame_idx
  "time_sec": 0.48,            # 可选
  "persons": [
    {
      "bbox": [x1,y1,x2,y2],   # 必须
      "keypoints": [...],      # 可选（17*2 或 17*3 展平）
      "conf": 0.87             # 可选
    },
    ...
  ]
}

兼容字段名：
- frame: frame / frame_idx
- persons: persons / detections
- bbox: bbox / xyxy / box
- keypoints: keypoints / kpts / pose
- conf: conf / score
"""


def parse_frame_obj(obj: Dict[str, Any], default_frame: int) -> Tuple[int, Optional[float], List[Dict[str, Any]]]:
    # frame
    frame = obj.get("frame", obj.get("frame_idx", default_frame))
    try:
        frame_i = int(frame)
    except Exception:
        frame_i = default_frame

    # time
    t = obj.get("time_sec", obj.get("t", None))
    try:
        t_f = float(t) if t is not None else None
    except Exception:
        t_f = None

    persons_raw = obj.get("persons", obj.get("detections", []))
    persons: List[Dict[str, Any]] = []
    if isinstance(persons_raw, list):
        for p in persons_raw:
            if not isinstance(p, dict):
                continue

            bbox = p.get("bbox", p.get("xyxy", p.get("box", None)))
            bbox = as_floats(bbox)
            if bbox is None or len(bbox) != 4:
                continue

            kps = p.get("keypoints", p.get("kpts", p.get("pose", None)))
            kps = as_floats(kps)  # 允许 None

            conf = p.get("conf", p.get("score", None))
            try:
                conf_f = float(conf) if conf is not None else None
            except Exception:
                conf_f = None

            persons.append({"bbox": bbox, "keypoints": kps, "conf": conf_f})
    return frame_i, t_f, persons


# =========================
# Tracking state
# =========================
class Track:
    def __init__(self, tid: int, start_frame: int):
        self.tid = tid
        self.start_frame = start_frame
        self.last_frame = start_frame
        self.hits = 0
        self.miss = 0

        self.bbox_s: Optional[List[float]] = None
        self.kps_s: Optional[List[float]] = None
        self.conf_s: Optional[float] = None

    def update(self, frame: int, bbox: List[float], kps: Optional[List[float]], conf: Optional[float],
               alpha_bbox: float, alpha_kps: float, alpha_conf: float):
        self.last_frame = frame
        self.hits += 1
        self.miss = 0

        self.bbox_s = ema_list(self.bbox_s, bbox, alpha_bbox)

        if kps is not None:
            self.kps_s = ema_list(self.kps_s, kps, alpha_kps)

        if conf is not None:
            if self.conf_s is None:
                self.conf_s = conf
            else:
                self.conf_s = (1.0 - alpha_conf) * self.conf_s + alpha_conf * conf


def greedy_match(costs: List[Tuple[float, int, int]]) -> List[Tuple[int, int]]:
    """
    costs: (cost, track_index, det_index), cost 越小越好
    """
    used_t = set()
    used_d = set()
    pairs: List[Tuple[int, int]] = []
    for c, ti, di in sorted(costs, key=lambda x: x[0]):
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        pairs.append((ti, di))
    return pairs


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser("intelligence_class track & smooth (standalone)")
    ap.add_argument("--in", dest="in_path", required=True, help="input pose jsonl (per-frame persons with bbox/keypoints)")
    ap.add_argument("--out", dest="out_path", required=True, help="output pose_tracks_smooth.jsonl")

    # match gating
    ap.add_argument("--iou_thr", type=float, default=0.20, help="min IoU to consider a match")
    ap.add_argument("--dist_thr", type=float, default=90.0, help="max center distance (pixels) to consider a match")

    # cost weights
    ap.add_argument("--w_iou", type=float, default=0.70, help="weight for (1-iou)")
    ap.add_argument("--w_dist", type=float, default=0.30, help="weight for normalized distance")

    # lifecycle
    ap.add_argument("--max_miss", type=int, default=10, help="max missed frames before a track is dropped")
    ap.add_argument("--min_frames", type=int, default=30, help="min hits to keep a track (filter short tracks)")
    ap.add_argument("--write_all", action="store_true", help="write all tracks without min_frames filtering (debug)")

    # EMA
    ap.add_argument("--alpha_bbox", type=float, default=0.45, help="EMA alpha for bbox")
    ap.add_argument("--alpha_kps", type=float, default=0.35, help="EMA alpha for keypoints")
    ap.add_argument("--alpha_conf", type=float, default=0.30, help="EMA alpha for confidence")

    args = ap.parse_args()

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # 读入所有帧（简单可靠；如需极大文件可再改为流式+二次过滤）
    frames: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                frames.append(json.loads(s))
            except Exception:
                continue

    tracks: List[Track] = []
    next_tid = 1

    # 先缓存所有输出，最后做 min_frames 过滤
    out_records: List[Dict[str, Any]] = []

    for idx, obj in enumerate(frames):
        frame_i, t_f, dets = parse_frame_obj(obj, default_frame=idx)

        # build cost list
        costs: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(tracks):
            if tr.bbox_s is None:
                continue
            tc = center_xy(tr.bbox_s)
            for di, det in enumerate(dets):
                dc = center_xy(det["bbox"])
                d = dist(tc, dc)
                if d > args.dist_thr:
                    continue
                iou = iou_xyxy(tr.bbox_s, det["bbox"])
                if iou < args.iou_thr:
                    continue

                d_norm = min(1.0, d / max(args.dist_thr, 1e-6))
                cost = args.w_iou * (1.0 - iou) + args.w_dist * d_norm
                costs.append((cost, ti, di))

        pairs = greedy_match(costs)
        matched_t = set(ti for ti, _ in pairs)
        matched_d = set(di for _, di in pairs)

        # update matched
        for ti, di in pairs:
            tr = tracks[ti]
            det = dets[di]
            tr.update(
                frame=frame_i,
                bbox=det["bbox"],
                kps=det["keypoints"],
                conf=det["conf"],
                alpha_bbox=args.alpha_bbox,
                alpha_kps=args.alpha_kps,
                alpha_conf=args.alpha_conf,
            )

        # miss for unmatched tracks + drop dead
        alive: List[Track] = []
        for ti, tr in enumerate(tracks):
            if ti not in matched_t:
                tr.miss += 1
            if tr.miss <= args.max_miss:
                alive.append(tr)
        tracks = alive

        # create new tracks for unmatched dets
        for di, det in enumerate(dets):
            if di in matched_d:
                continue
            tr = Track(next_tid, start_frame=frame_i)
            next_tid += 1
            tr.update(
                frame=frame_i,
                bbox=det["bbox"],
                kps=det["keypoints"],
                conf=det["conf"],
                alpha_bbox=args.alpha_bbox,
                alpha_kps=args.alpha_kps,
                alpha_conf=args.alpha_conf,
            )
            tracks.append(tr)

        # record current frame snapshot
        persons_out = []
        for tr in tracks:
            if tr.bbox_s is None:
                continue
            persons_out.append({
                "track_id": tr.tid,
                "bbox": [float(x) for x in tr.bbox_s],
                "keypoints": tr.kps_s,  # may be None
                "conf": float(tr.conf_s) if tr.conf_s is not None else None,
                "hits": tr.hits,
                "miss": tr.miss,
                "start_frame": tr.start_frame,
            })

        out_records.append({
            "frame": frame_i,
            "time_sec": t_f,
            "persons": persons_out
        })

    # compute max hits per track
    hits_map: Dict[int, int] = {}
    for rec in out_records:
        for p in rec["persons"]:
            tid = int(p["track_id"])
            hits_map[tid] = max(hits_map.get(tid, 0), int(p.get("hits", 0)))

    if args.write_all:
        keep = set(hits_map.keys())
    else:
        keep = set([tid for tid, h in hits_map.items() if h >= args.min_frames])

    # write filtered jsonl
    kept_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out_records:
            persons = [p for p in rec["persons"] if int(p["track_id"]) in keep]
            if persons:
                kept_count += 1
            f.write(json.dumps({
                "frame": rec["frame"],
                "time_sec": rec["time_sec"],
                "persons": persons
            }, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {out_path}")
    print(f"[INFO] frames={len(out_records)} tracks_total={len(hits_map)} tracks_kept={len(keep)} min_frames={args.min_frames}")
    print(f"[INFO] nonempty_frames_after_filter={kept_count}")


if __name__ == "__main__":
    main()
