import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

def iter_dets(obj):
    """
    兼容两种常见格式：
    A) 每行一个 detection: {"name":"person","conf":0.5,"bbox":[x1,y1,x2,y2],...}
    B) 每行一帧多个 dets: {"frame":12,"detections":[{...},{...}]}
    """
    if isinstance(obj, dict) and "detections" in obj and isinstance(obj["detections"], list):
        for d in obj["detections"]:
            if isinstance(d, dict):
                yield d
    elif isinstance(obj, dict):
        yield obj

def bbox_area_ratio(det):
    # bbox 可能是 [x1,y1,x2,y2] 或 [x,y,w,h]
    b = det.get("bbox") or det.get("box")
    if not b or not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    x1, y1, x2, y2 = b
    # 如果是 xywh，转成 x1y1x2y2
    if x2 >= 0 and y2 >= 0 and (x2 <= 1 or y2 <= 1) and (x1 <= 1 and y1 <= 1):
        # 这种情况可能是归一化坐标，先不处理
        pass
    # 粗略判断 xywh：w/h 通常比 x2/y2 小很多且 x2,y2 为宽高
    if x2 > 0 and y2 > 0 and x2 < 50 and y2 < 50 and x1 >= 0 and y1 >= 0:
        # 很小不一定是 xywh，所以不强转
        pass
    # 尝试按 x1y1x2y2 计算
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    if w == 0 or h == 0:
        # 再尝试按 xywh
        w = max(0.0, float(x2))
        h = max(0.0, float(y2))
    return w * h

def bucket_conf(c):
    if c is None: return "NA"
    if c < 0.1: return "<0.1"
    if c < 0.2: return "0.1-0.2"
    if c < 0.3: return "0.2-0.3"
    if c < 0.5: return "0.3-0.5"
    if c < 0.7: return "0.5-0.7"
    return ">=0.7"

def bucket_area(a):
    if a is None: return "NA"
    if a < 400: return "<400px"
    if a < 1600: return "400-1600"
    if a < 6400: return "1600-6400"
    if a < 25600: return "6400-25600"
    return ">=25600"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()

    p = Path(args.jsonl)
    if not p.exists():
        raise FileNotFoundError(p)

    cls_cnt = Counter()
    conf_hist = Counter()
    area_hist = Counter()
    per_cls_conf = defaultdict(list)

    lines = 0
    dets_total = 0
    bad_lines = 0

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            for det in iter_dets(obj):
                name = det.get("name") or det.get("cls_name") or str(det.get("cls", "unknown"))
                conf = det.get("conf") or det.get("score")
                try:
                    conf = float(conf) if conf is not None else None
                except Exception:
                    conf = None

                cls_cnt[name] += 1
                conf_hist[bucket_conf(conf)] += 1
                per_cls_conf[name].append(conf if conf is not None else -1)

                a = bbox_area_ratio(det)
                area_hist[bucket_area(a)] += 1

                dets_total += 1

    print(f"\n[STATS] file={p} lines={lines} dets={dets_total} bad_lines={bad_lines}\n")

    print("Top classes:")
    for k, v in cls_cnt.most_common(20):
        print(f"  {k:12s}  {v}")

    print("\nConf histogram:")
    for k in ["<0.1","0.1-0.2","0.2-0.3","0.3-0.5","0.5-0.7",">=0.7","NA"]:
        if k in conf_hist:
            print(f"  {k:8s}  {conf_hist[k]}")

    print("\nBBox area histogram (rough, px^2):")
    for k in ["<400px","400-1600","1600-6400","6400-25600",">=25600","NA"]:
        if k in area_hist:
            print(f"  {k:10s}  {area_hist[k]}")

    # 重点看 phone/book/laptop/tablet 等是否为 0
    watch = ["person","cell phone","phone","book","laptop","tablet","backpack","handbag","pen","pencil"]
    print("\nWatch list:")
    for w in watch:
        print(f"  {w:10s}  {cls_cnt.get(w,0)}")

if __name__ == "__main__":
    main()
