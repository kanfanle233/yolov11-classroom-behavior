# scripts/intelligence_class/xx_aggregate_dataset_report.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict


# -------------------------
# Helpers
# -------------------------
def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile(values: List[float], q: float) -> float:
    """
    q in [0,1]
    使用线性插值的分位数（稳定且够用）
    """
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 1:
        return float(max(values))
    vs = sorted(values)
    n = len(vs)
    if n == 1:
        return float(vs[0])
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vs[lo])
    w = pos - lo
    return float(vs[lo] * (1 - w) + vs[hi] * w)


def safe_get(d: Dict[str, Any], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


# -------------------------
# Recommend scoring
# -------------------------
def recommend_score(summary: Dict[str, Any]) -> float:
    """
    目标：挑“可展示/可诊断”的样本
    - 优先有 behavior_overlay / overlay
    - missing_rate 不要极端：太低=没挑战，太高=几乎全空
    - 有少量 flags（比如 LOW_CONF / LONG_EMPTY_SEGMENT）反而更值得展示
    """
    best_video = str(summary.get("best_video", "none"))
    miss = float(safe_get(summary, "quality", "missing_rate", default=1.0) or 1.0)
    flags = safe_get(summary, "quality", "flags", default=[]) or []

    # 基础：视频可播性
    if best_video == "behavior_overlay":
        s = 5.0
    elif best_video == "overlay":
        s = 4.0
    elif best_video == "raw":
        s = 2.0
    else:
        s = 0.0

    # missing_rate：目标区间 0.2~0.8 最适合展示“真实问题”
    # 用一个“倒 U 型”奖励：离 0.5 越近越高
    # 最高 +3，最差接近 0
    s += max(0.0, 3.0 * (1.0 - abs(miss - 0.5) / 0.5))  # miss=0.5 -> +3, miss=0 or 1 -> +0

    # flags：少量“可解释问题”加分，但 MOSTLY_EMPTY 减分（太空没意义）
    flag_set = set(map(str, flags))
    if "MOSTLY_EMPTY" in flag_set:
        s -= 2.0
    if "LONG_EMPTY_SEGMENT" in flag_set:
        s += 0.6
    if "LOW_CONF" in flag_set:
        s += 0.6
    if "MANY_EMPTY_FRAMES" in flag_set:
        s += 0.3

    # detect 质量：avg_conf 太低扣一点
    avg_conf = float(safe_get(summary, "detect", "avg_conf", default=0.0) or 0.0)
    if avg_conf > 0:
        if avg_conf < 0.15:
            s -= 0.6
        elif avg_conf < 0.25:
            s -= 0.3

    return float(s)


# -------------------------
# Main aggregation
# -------------------------
def aggregate_view(view_dir: Path, top_n: int, min_missing: float, max_missing: float) -> Dict[str, Any]:
    summaries = list(view_dir.glob("*_summary.json"))
    total = len(summaries)

    kind_counter = Counter()
    flags_counter = Counter()
    missing_rates: List[float] = []

    # 推荐样本候选：(score, case_id)
    rec_candidates: List[Tuple[float, str]] = []

    # 额外：你可能想看“最差/最好”的样本
    best_cases: List[Tuple[float, str]] = []
    worst_cases: List[Tuple[float, str]] = []

    for sp in summaries:
        try:
            s = read_json(sp)
        except Exception:
            continue

        case_id = str(s.get("case_id", sp.name.replace("_summary.json", "")))
        best_video = str(s.get("best_video", "none"))
        kind_counter[best_video] += 1

        miss = float(safe_get(s, "quality", "missing_rate", default=1.0) or 1.0)
        missing_rates.append(miss)

        flags = safe_get(s, "quality", "flags", default=[]) or []
        for f in flags:
            flags_counter[str(f)] += 1

        # 推荐样本：missing_rate 在区间内 + 视频可播优先
        if min_missing <= miss <= max_missing:
            score = recommend_score(s)
            rec_candidates.append((score, case_id))

        # 也顺便产出 top/bottom missing_rate（给论文写“视角难度”）
        best_cases.append((1.0 - miss, case_id))   # missing 越低越好
        worst_cases.append((miss, case_id))        # missing 越高越差

    # missing_rate stats
    mr_mean = float(sum(missing_rates) / len(missing_rates)) if missing_rates else 0.0
    mr_p50 = percentile(missing_rates, 0.50)
    mr_p90 = percentile(missing_rates, 0.90)

    # flags top
    flags_top = [[k, int(v)] for k, v in flags_counter.most_common(20)]

    # recommend
    rec_candidates.sort(key=lambda x: -x[0])
    recommend_samples = [cid for _, cid in rec_candidates[:top_n]]

    # best/worst by missing_rate
    best_cases.sort(key=lambda x: -x[0])
    worst_cases.sort(key=lambda x: -x[0])
    best_missing_samples = [cid for _, cid in best_cases[:min(10, len(best_cases))]]
    worst_missing_samples = [cid for _, cid in worst_cases[:min(10, len(worst_cases))]]

    return {
        "view": view_dir.name,
        "total": total,
        "video_kind": dict(kind_counter),
        "missing_rate": {
            "mean": round(mr_mean, 4),
            "p50": round(mr_p50, 4),
            "p90": round(mr_p90, 4),
        },
        "flags_top": flags_top,
        "recommend_samples": recommend_samples,
        "best_missing_samples": best_missing_samples,
        "worst_missing_samples": worst_missing_samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_root", type=str, default="output/智慧课堂学生行为数据集", help="数据集输出根目录")
    ap.add_argument("--out", type=str, default="", help="输出路径（默认写到 ds_root/summary_report_v2.json）")
    ap.add_argument("--top_n", type=int, default=50, help="每个视角推荐样本数量")
    ap.add_argument("--min_missing", type=float, default=0.2, help="推荐样本 missing_rate 下限")
    ap.add_argument("--max_missing", type=float, default=0.8, help="推荐样本 missing_rate 上限")
    ap.add_argument("--views", type=str, default="", help="只处理指定视角，逗号分隔，如：正方视角,教师视角")
    args = ap.parse_args()

    ds_root = Path(args.ds_root).resolve()
    if not ds_root.exists():
        raise FileNotFoundError(f"ds_root not found: {ds_root}")

    out_path = Path(args.out).resolve() if args.out else (ds_root / "summary_report_v2.json")

    view_dirs = [d for d in ds_root.iterdir() if d.is_dir()]
    if args.views.strip():
        wanted = set([x.strip() for x in args.views.split(",") if x.strip()])
        view_dirs = [d for d in view_dirs if d.name in wanted]

    views_payload: Dict[str, Any] = {}
    total_cases_all = 0

    for vd in sorted(view_dirs, key=lambda p: p.name):
        payload = aggregate_view(vd, top_n=args.top_n, min_missing=args.min_missing, max_missing=args.max_missing)
        views_payload[vd.name] = payload
        total_cases_all += int(payload.get("total", 0))
        print(f"[OK] view={vd.name} total={payload['total']} recommend={len(payload['recommend_samples'])}")

    report = {
        "dataset": "智慧课堂学生行为数据集",
        "ds_root": str(ds_root),
        "total_cases_all_views": total_cases_all,
        "params": {
            "top_n": args.top_n,
            "min_missing": args.min_missing,
            "max_missing": args.max_missing,
        },
        "views": views_payload,
    }

    write_json(out_path, report)
    print("======================================")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
