import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


# -----------------------------
# 1) 读 index/report：兼容 dataset_index.json 和 batch_report.json
# -----------------------------
def load_index(index_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(index_path.read_text(encoding="utf-8"))

    # A) dataset_index.json: 直接 list
    if isinstance(data, list):
        return data

    # B) {"items":[...]}
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]

    # C) batch_report.json 常见：{"records":[...]} 或 {"jobs":[...]} 或 {"results":[...]}
    for key in ("records", "jobs", "results"):
        if isinstance(data, dict) and key in data and isinstance(data[key], list):
            return data[key]

    raise ValueError(f"Unknown index/report schema in {index_path}")


def is_valid_jsonl(p: Path, min_bytes: int = 64) -> bool:
    """存在 + 非空(>=min_bytes) 视为有效，防止空文件假跳过。"""
    return p.exists() and p.is_file() and p.stat().st_size >= min_bytes


# -----------------------------
# 2) 从 record 中抽取字段：最大兼容
# -----------------------------
def pick_str(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def pick_num(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def record_to_job(
    it: Dict[str, Any],
    fallback_output_root: Path
) -> Optional[Dict[str, Any]]:
    """
    统一成 job:
      {
        video_id: str,
        view: str,
        video_path: Path,
        out_dir: Path,
        fps: Optional[float],
      }
    """
    video_id = pick_str(it, ["video_id", "id", "case_id", "name"])
    if not video_id:
        return None

    view = pick_str(it, ["view", "view_name", "camera", "camera_view"])

    video_path_s = pick_str(it, ["video_path", "video", "video_file", "input_video", "src_video"])
    video_path = Path(video_path_s).resolve() if video_path_s else Path()

    # batch_report 里通常直接有 out_dir（你贴的 dry-run cmd 就有）
    out_dir_s = pick_str(it, ["out_dir", "output_dir", "outpath"])
    if out_dir_s:
        out_dir = Path(out_dir_s).resolve()
    else:
        # 回退：output_root/video_id（不建议长期依赖，但保证能跑）
        out_dir = (fallback_output_root / video_id).resolve()

    fps = pick_num(it, ["fps", "video_fps"])

    return {
        "video_id": video_id,
        "view": view,
        "video_path": video_path,
        "out_dir": out_dir,
        "fps": fps,
    }


# -----------------------------
# 3) 子进程跑 step03
# -----------------------------
def run_one(
    py: str,
    step03_script: Path,
    video_path: Path,
    out_dir: Path,
    fps: Optional[float],
    extra: List[str],
) -> Tuple[int, str]:
    cmd = [py, str(step03_script), "--video", str(video_path), "--out_dir", str(out_dir)]
    if fps is not None:
        cmd += ["--fps", str(fps)]
    cmd += extra

    p = subprocess.run(cmd, capture_output=True, text=True)
    msg = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index",
        type=str,
        default="YOLOv11/output/_batch/batch_report.json",
        help="batch_report.json or dataset_index.json"
    )
    ap.add_argument("--python", type=str, default=sys.executable, help="python executable")
    ap.add_argument("--step03", type=str, required=True, help="FULL smooth script path (step03)")

    # ⚠️ 重要：默认 output_root 要对齐 YOLOv11/output（你现在项目实际输出在这里）
    ap.add_argument("--output_root", type=str, default="YOLOv11/output", help="fallback output root")
    ap.add_argument("--limit", type=int, default=0, help="limit N videos (0 = all)")
    ap.add_argument("--views", type=str, default="", help="comma views filter, e.g. front,rear,teacher")
    ap.add_argument("--force", action="store_true", help="force rerun even if outputs exist")
    ap.add_argument("--min_bytes", type=int, default=64, help="min bytes to consider output valid")
    ap.add_argument("--log", type=str, default="YOLOv11/output/_tracks_smooth_run.log", help="log file")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra args passed to step03 script")
    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    step03_script = Path(args.step03).resolve()
    output_root = Path(args.output_root).resolve()
    log_path = Path(args.log).resolve()

    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not step03_script.exists():
        raise FileNotFoundError(step03_script)

    output_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_index(index_path)

    view_filter = set()
    if args.views.strip():
        view_filter = set([v.strip() for v in args.views.split(",") if v.strip()])

    ok = skip = fail = 0
    total = 0

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 90 + "\n")
        log.write(f"[RUN] index={index_path}\n")
        log.write(f"[RUN] step03={step03_script}\n")
        log.write(f"[RUN] python={args.python}\n")
        log.write(f"[RUN] output_root(fallback)={output_root}\n")
        log.write("=" * 90 + "\n")

        for it in items:
            if not isinstance(it, dict):
                continue

            job = record_to_job(it, fallback_output_root=output_root)
            if job is None:
                continue

            video_id = job["video_id"]
            view = job["view"]
            video_path: Path = job["video_path"]
            out_dir: Path = job["out_dir"]
            fps = job["fps"]

            if not video_path.exists():
                # report 里有记录但视频不存在 -> 记录一下
                log.write(f"[MISS] {video_id} view={view} video_not_found={video_path}\n")
                continue

            if view_filter and view not in view_filter:
                continue

            total += 1
            if args.limit and total > args.limit:
                break

            out_dir.mkdir(parents=True, exist_ok=True)

            target = out_dir / "pose_tracks_smooth.jsonl"

            if (not args.force) and is_valid_jsonl(target, min_bytes=args.min_bytes):
                skip += 1
                log.write(f"[SKIP] {video_id} view={view} target_exists={target}\n")
                continue

            code, msg = run_one(
                py=args.python,
                step03_script=step03_script,
                video_path=video_path,
                out_dir=out_dir,
                fps=fps,
                extra=args.extra,
            )

            if code == 0 and is_valid_jsonl(target, min_bytes=args.min_bytes):
                ok += 1
                log.write(f"[OK]   {video_id} view={view} -> {target}\n")
            else:
                fail += 1
                log.write(f"[FAIL] {video_id} view={view} code={code}\n")
                log.write(msg[:8000] + "\n")  # 防止日志爆炸

        log.write(f"\n[SUMMARY] total={total} ok={ok} skip={skip} fail={fail}\n")

    print(f"[DONE] total={total} ok={ok} skip={skip} fail={fail}")
    print(f"[LOG]  {log_path}")


if __name__ == "__main__":
    main()
