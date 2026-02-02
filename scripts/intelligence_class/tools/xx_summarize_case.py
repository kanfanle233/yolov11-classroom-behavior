# scripts/intelligence_class/xx_summarize_case.py
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


# -------------------------
# IO helpers
# -------------------------
def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def safe_write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# File locate helpers
# -------------------------
def find_meta(view_dir: Path, prefix: str) -> Optional[Path]:
    # 001.meta.json / 0001.meta.json / 001_behavior.meta.json
    cands = []
    cands += list(view_dir.glob(f"{prefix}.meta.json"))
    cands += list(view_dir.glob(f"{prefix}_meta.json"))
    if not cands:
        cands += list(view_dir.glob(f"{prefix}*.meta.json"))
        cands += list(view_dir.glob(f"{prefix}*_meta.json"))
    if not cands:
        return None
    return sorted(cands, key=lambda x: x.stat().st_size, reverse=True)[0]


def find_main_jsonl(view_dir: Path, prefix: str) -> Optional[Path]:
    # 主 jsonl：优先 prefix.jsonl；排除 behavior
    p = view_dir / f"{prefix}.jsonl"
    if p.exists():
        return p
    cands = [x for x in view_dir.glob(f"{prefix}*.jsonl") if "behavior" not in x.name.lower()]
    if not cands:
        return None
    return sorted(cands, key=lambda x: x.stat().st_size, reverse=True)[0]


def find_overlay(view_dir: Path, prefix: str) -> Optional[Path]:
    # prefix_overlay.mp4（排除 behavior_overlay）
    cands = [x for x in view_dir.glob(f"{prefix}*_overlay.mp4") if "behavior_overlay" not in x.name.lower()]
    if not cands:
        return None
    return sorted(cands, key=lambda x: x.stat().st_size, reverse=True)[0]


def find_behavior_overlay(view_dir: Path, prefix: str) -> Optional[Path]:
    cands = list(view_dir.glob(f"{prefix}*_behavior_overlay.mp4"))
    if not cands:
        return None
    return sorted(cands, key=lambda x: x.stat().st_size, reverse=True)[0]


# -------------------------
# Core stats
# -------------------------
def _extract_conf_from_boxes(frame_obj: Any) -> List[float]:
    """
    你的 jsonl 每行大概是：
      {"pred":{"boxes":[{"conf":0.8,...}, ...], "keypoints":...}}
    也可能 boxes 在顶层。
    """
    confs: List[float] = []
    if not isinstance(frame_obj, dict):
        return confs

    boxes = None
    if "boxes" in frame_obj and isinstance(frame_obj["boxes"], list):
        boxes = frame_obj["boxes"]
    elif "pred" in frame_obj and isinstance(frame_obj["pred"], dict):
        pb = frame_obj["pred"].get("boxes")
        if isinstance(pb, list):
            boxes = pb

    if not boxes:
        return confs

    for b in boxes:
        if not isinstance(b, dict):
            continue
        c = b.get("conf", b.get("score", b.get("confidence")))
        if c is None:
            continue
        try:
            confs.append(float(c))
        except Exception:
            pass
    return confs


def _count_boxes(frame_obj: Any) -> int:
    if not isinstance(frame_obj, dict):
        return 0
    if "boxes" in frame_obj and isinstance(frame_obj["boxes"], list):
        return len(frame_obj["boxes"])
    if "pred" in frame_obj and isinstance(frame_obj["pred"], dict):
        pb = frame_obj["pred"].get("boxes")
        if isinstance(pb, list):
            return len(pb)
    return 0


def _get_frame_idx(frame_obj: Any) -> Optional[int]:
    if not isinstance(frame_obj, dict):
        return None
    for k in ("frame_idx", "frame", "frame_id"):
        if k in frame_obj:
            try:
                return int(frame_obj[k])
            except Exception:
                continue
    return None


def summarize_case(view: str, view_dir: Path, prefix: str, max_lines: int = 200000) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta_path = find_meta(view_dir, prefix)
    main_path = find_main_jsonl(view_dir, prefix)

    meta: Dict[str, Any] = {}
    if meta_path and meta_path.exists():
        try:
            meta = read_json(meta_path)
        except Exception:
            meta = {}

    fps = float(meta.get("fps", 25.0)) if meta else 25.0
    width = meta.get("width")
    height = meta.get("height")
    input_video = meta.get("input_video") or meta.get("source_video") or meta.get("video_path")

    # 主 jsonl 统计
    frame_count = 0
    frames_with_person = 0
    total_boxes = 0
    conf_sum = 0.0
    conf_n = 0
    track_ids = set()

    # 异常/缺失：连续无人帧（用于“异常片段”提示）
    max_consecutive_empty = 0
    cur_empty = 0

    last_frame_idx = None

    if main_path and main_path.exists():
        for i, rec in enumerate(iter_jsonl(main_path)):
            if i >= max_lines:
                break

            frame_count += 1
            n_boxes = _count_boxes(rec)
            total_boxes += n_boxes

            if n_boxes > 0:
                frames_with_person += 1
                cur_empty = 0
            else:
                cur_empty += 1
                max_consecutive_empty = max(max_consecutive_empty, cur_empty)

            confs = _extract_conf_from_boxes(rec)
            for c in confs:
                conf_sum += c
                conf_n += 1

            # track_id 可能在 rec 顶层，也可能在 pred 里（你项目里不一定有）
            tid = rec.get("track_id")
            if tid is None and isinstance(rec.get("pred"), dict):
                tid = rec["pred"].get("track_id")
            if tid is not None:
                try:
                    track_ids.add(int(tid))
                except Exception:
                    pass

            fi = _get_frame_idx(rec)
            if fi is not None:
                last_frame_idx = fi

    # 如果 meta 没 frame_count，用 last_frame_idx 兜底
    meta_frames = meta.get("frame_count") or meta.get("frames") or meta.get("n_frames")
    if meta_frames is not None:
        try:
            meta_frames = int(meta_frames)
        except Exception:
            meta_frames = None

    est_frames = meta_frames if meta_frames else (last_frame_idx + 1 if last_frame_idx is not None else frame_count)
    duration = float(meta.get("duration", (est_frames / fps if fps > 0 else 0.0)))

    avg_boxes = (total_boxes / frame_count) if frame_count > 0 else 0.0
    avg_conf = (conf_sum / conf_n) if conf_n > 0 else 0.0

    missing_rate = 1.0 - (frames_with_person / frame_count) if frame_count > 0 else 1.0

    # flags：让前端“立刻可用”
    flags = []
    if frame_count == 0:
        flags.append("EMPTY_JSONL")
    if missing_rate >= 0.9:
        flags.append("MOSTLY_EMPTY")
    elif missing_rate >= 0.5:
        flags.append("MANY_EMPTY_FRAMES")
    if max_consecutive_empty >= int(max(1, fps * 3)):  # 连续 >=3 秒无人
        flags.append("LONG_EMPTY_SEGMENT")
    if avg_conf < 0.2 and conf_n > 0:
        flags.append("LOW_CONF")

    # best video kind
    beh_ov = find_behavior_overlay(view_dir, prefix)
    ov = find_overlay(view_dir, prefix)
    best_video = "none"
    if beh_ov and beh_ov.exists():
        best_video = "behavior_overlay"
    elif ov and ov.exists():
        best_video = "overlay"
    elif input_video:
        best_video = "raw"

    summary = {
        "case_id": prefix,
        "view": view,
        "video": {
            "fps": fps,
            "duration": round(duration, 3),
            "frames": int(est_frames) if est_frames is not None else frame_count,
            "width": width,
            "height": height,
            "input_video": input_video,
        },
        "detect": {
            "jsonl_path": str(main_path) if main_path else None,
            "frames_in_jsonl": frame_count,
            "frames_with_person": frames_with_person,
            "avg_boxes": round(float(avg_boxes), 4),
            "avg_conf": round(float(avg_conf), 4),
            "track_count": len(track_ids),
        },
        "quality": {
            "missing_rate": round(float(missing_rate), 4),
            "max_consecutive_empty_frames": int(max_consecutive_empty),
            "flags": flags or ["OK"],
        },
        "best_video": best_video,
        "artifacts": {
            "meta": str(meta_path) if meta_path else None,
            "overlay": str(ov) if ov else None,
            "behavior_overlay": str(beh_ov) if beh_ov else None,
        },
    }

    debug = {
        "meta_path": str(meta_path) if meta_path else None,
        "main_path": str(main_path) if main_path else None,
    }
    return summary, debug


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="output/智慧课堂学生行为数据集", help="数据集输出根目录")
    ap.add_argument("--view", type=str, default="", help="只处理某个视角（可选）")
    ap.add_argument("--id", type=str, default="", help="只处理某个 case 前缀（可选，如 0001/001/1）")
    ap.add_argument("--max_lines", type=int, default=200000, help="单个 jsonl 最多读取多少行（防止超大文件拖死）")
    ap.add_argument("--overwrite", type=int, default=0, help="1=覆盖已有 _summary.json")
    args = ap.parse_args()

    ds_root = Path(args.out_root).resolve()
    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset output root not found: {ds_root}")

    views = [d for d in ds_root.iterdir() if d.is_dir()]
    if args.view:
        views = [ds_root / args.view]
        if not views[0].exists():
            raise FileNotFoundError(f"View not found: {views[0]}")

    total_cases = 0
    written = 0
    skipped = 0

    for vd in views:
        view_name = vd.name
        # ✅ 新增：每个视角自己的计数器
        view_written = 0
        view_skipped = 0

        # 找所有主 jsonl 的 prefix（排除 behavior）
        jsonls = [p for p in vd.glob("*.jsonl") if "behavior" not in p.name.lower()]
        prefixes = sorted({p.stem for p in jsonls})

        # 允许只跑一个 id
        if args.id:
            # 归一化：把 "1"->"0001" 这种不强制，直接按“包含数字前缀”匹配
            wanted_digits = "".join([c for c in args.id if c.isdigit()])
            if wanted_digits:
                prefixes = [p for p in prefixes if "".join([c for c in p if c.isdigit()]) == wanted_digits]
            else:
                prefixes = [p for p in prefixes if p == args.id]

        for pref in prefixes:
            total_cases += 1
            out_path = vd / f"{pref}_summary.json"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                view_skipped += 1  # ✅ 新增
                continue

            summary, _ = summarize_case(view_name, vd, pref, max_lines=args.max_lines)
            safe_write_json(out_path, summary)
            written += 1
            view_written += 1  # ✅ 新增

            # ✅ 改打印：只显示当前 view 的 written/skipped
        print(f"[DONE] view={view_name} total={len(prefixes)} written={view_written} skipped={view_skipped}")

    print("======================================")
    print(f"Views processed: {len(views)}")
    print(f"Cases seen      : {total_cases}")
    print(f"Summaries written: {written}")
    print(f"Skipped existing : {skipped}")
    print("Output: <view>/<case>_summary.json")


if __name__ == "__main__":
    main()
