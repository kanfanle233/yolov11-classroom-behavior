import argparse
import json
import subprocess
import sys
import time
import datetime
import shutil
from pathlib import Path


# ==========================================
# 0. 让脚本在任何目录运行都能 import scripts.intelligence_class._utils.pathing
#    （仅用于路径工具导入；不影响你原有逻辑）
# ==========================================
_this = Path(__file__).resolve()
for p in [_this] + list(_this.parents):
    if (p / "data").exists() and (p / "scripts").exists():
        sys.path.insert(0, str(p))
        break

from scripts.intelligence_class._utils.pathing import find_project_root, find_sibling_script


# ==========================================
# 1. 路径与环境配置
# ==========================================

def resolve_paths(custom_index_path=None, custom_out_root=None):
    """
    解析项目核心路径
    基于当前脚本位置：YOLOv11/scripts/intelligence_class/02_batch_run.py
    """
    current_file = Path(__file__).resolve()

    # ✅ 统一：向上寻找项目根目录 (YOLOv11) —— 不再依赖 parents[2]
    project_root = find_project_root(current_file)

    # ✅ 统一：单视频执行器：01_run_single_video.py —— 不再要求在同一目录
    target_script = find_sibling_script(
        "01_run_single_video.py",
        start_file=current_file,
        project_root=project_root
    )

    index_file = project_root / "output" / "dataset_index.json"
    if custom_index_path:
        index_file = Path(custom_index_path).resolve()

    output_root = project_root / "output"
    if custom_out_root:
        output_root = Path(custom_out_root).resolve()

    batch_dir = output_root / "_batch"
    paths = {
        "root": project_root,
        "target_script": target_script,
        "index_file": index_file,
        "output_root": output_root,
        "batch_dir": batch_dir,
        "failure_log": batch_dir / "batch_failures.jsonl",
        "report_file": batch_dir / "batch_report.json",
    }

    if not paths["target_script"].exists():
        print(f"[FATAL] 找不到单视频执行器: {paths['target_script']}")
        sys.exit(1)

    return paths


# ==========================================
# 2. 核心逻辑函数
# ==========================================

def load_index(index_path: Path):
    """加载并校验索引文件"""
    if not index_path.exists():
        print(f"[FATAL] 索引文件不存在: {index_path}")
        print("请先运行 000.py (或 01_scan_dataset.py) 生成索引。")
        sys.exit(1)

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("videos", [])
    except Exception as e:
        print(f"[FATAL] 索引文件损坏: {e}")
        sys.exit(1)


def is_complete_outdir(out_dir: Path, min_bytes: int = 256, marker_name: str = "actions.jsonl") -> bool:
    """
    判定输出目录是否完成：需存在 marker 文件且大小达到阈值。
    """
    if not out_dir.exists():
        return False

    marker_path = out_dir / marker_name
    if not marker_path.exists():
        return False

    if min_bytes <= 0:
        return True

    try:
        return marker_path.stat().st_size >= min_bytes
    except OSError:
        return False


def compute_out_dirs(output_root: Path, view_code: str, video_id: str):
    """
    新结构：output/<view_code>/<video_id>/
    旧结构：output/<video_id>/
    """
    new_dir = output_root / view_code / video_id
    legacy_dir = output_root / video_id
    return new_dir, legacy_dir


def maybe_migrate_legacy(legacy_dir: Path, new_dir: Path, dry_run: bool, on_conflict: str = "skip"):
    """
    可选：把旧结构目录移动到新结构目录
    """
    if not legacy_dir.exists() or not legacy_dir.is_dir():
        return False

    new_dir.parent.mkdir(parents=True, exist_ok=True)

    if new_dir.exists():
        if on_conflict == "error":
            raise FileExistsError(f"Dest exists: {new_dir}")
        # skip
        return False

    if dry_run:
        print(f"[DRY] MIGRATE {legacy_dir} -> {new_dir}")
        return True

    shutil.move(str(legacy_dir), str(new_dir))
    print(f"[MIGRATE] {legacy_dir.name} -> {new_dir}")
    return True


def append_failure_log(log_path: Path, entry: dict, error_msg: str):
    """将失败信息追加写入 JSONL"""
    record = {
        "time": datetime.datetime.now().isoformat(),
        "video_id": entry.get("video_id"),
        "view_code": entry.get("view_code") or entry.get("view"),
        "video_path": entry.get("video_path"),
        "error": str(error_msg).strip()
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] 无法写入失败日志: {e}")


def save_report(report_path: Path, stats: dict, params: dict, failed_examples: list, start_time_iso: str,
                end_time_iso: str):
    """保存最终运行报告"""
    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "counts": stats,
        "params": params,
        "failed_examples": failed_examples
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 无法保存报告: {e}")


# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="大规模视频分析批处理调度器（按视角分层输出）")

    parser.add_argument("--index", type=str, default=None, help="索引文件路径 (默认 output/dataset_index.json)")
    parser.add_argument("--out_root", type=str, default=None, help="输出根目录 (默认 output/)")

    parser.add_argument("--views", type=str, default=None, help="只处理指定视角 (如 rear,front,top1)")
    parser.add_argument("--limit", type=int, default=0, help="处理数量限制 (0=不限)")
    parser.add_argument("--start", type=int, default=0, help="从筛选后列表第 N 个开始 (用于断点)")
    parser.add_argument("--end", type=int, default=None, help="直到筛选后列表第 N 个结束(不含)")

    parser.add_argument("--skip_existing", type=int, default=1, help="跳过已完成 (1=Yes, 0=No)")
    parser.add_argument("--min_bytes", type=int, default=256, help="判定完成的 actions.jsonl 最小字节数阈值")
    parser.add_argument("--short_video", type=int, default=0, help="短视频模式：降低 min_bytes 与轨迹门槛")
    parser.add_argument("--migrate_legacy", type=int, default=0,
                        help="发现旧结构已完成时，是否自动搬迁到新结构 (1=Yes,0=No)")
    parser.add_argument("--dry_run", action="store_true", help="仅打印计划，不执行")
    parser.add_argument("--stream_output", type=int, default=0,
                        help="实时输出子进程日志 (1=Yes, 0=No)")

    args = parser.parse_args()

    paths = resolve_paths(args.index, args.out_root)
    python_exe = sys.executable
    log_dir = paths["batch_dir"] / "logs"
    if int(args.short_video) == 1 and args.min_bytes == 256:
        args.min_bytes = 128

    print("=" * 60)
    print("🚀 批处理调度器启动 (Batch Scheduler)")
    print(f"📂 项目根目录: {paths['root']}")
    print(f"📁 输出根目录: {paths['output_root']}")
    print("=" * 60)

    all_videos = load_index(paths["index_file"])

    # --- 1. 视角筛选 ---
    allowed_views = args.views.split(",") if args.views else None
    target_videos = []
    for v in all_videos:
        v_code = v.get("view_code") or v.get("view") or "unknown"
        if allowed_views and v_code not in allowed_views:
            continue
        target_videos.append(v)

    total_filtered = len(target_videos)
    slice_start = max(0, args.start)
    slice_end = args.end if args.end is not None else total_filtered
    slice_end = min(slice_end, total_filtered)
    if args.limit > 0:
        slice_end = min(slice_end, slice_start + args.limit)

    if slice_start >= total_filtered:
        print(f"[WARN] 起始索引 {slice_start} 超出任务总数 {total_filtered}，无任务可做。")
        sys.exit(0)

    # 初始任务列表（基于索引范围）
    initial_tasks = target_videos[slice_start:slice_end]

    stats = {"total_in_range": len(initial_tasks), "success": 0, "failed": 0, "skipped": 0, "migrated": 0}

    # --- 2. 智能跳过逻辑 (修改点: 预处理过滤) ---
    final_tasks = []
    if args.skip_existing == 1:
        print(f"🔍 [智能过滤] 正在扫描已存在的输出文件...")
        skipped_count = 0

        for entry in initial_tasks:
            video_id = entry.get("video_id")
            view_code = entry.get("view_code") or entry.get("view") or "unknown"

            # 计算路径
            out_dir_new, out_dir_legacy = compute_out_dirs(paths["output_root"], view_code, video_id)

            # 检查是否已存在 (同时检查新旧结构)
            is_done_new = is_complete_outdir(out_dir_new, min_bytes=args.min_bytes)
            is_done_legacy = is_complete_outdir(out_dir_legacy, min_bytes=args.min_bytes)

            # 如果存在且不迁移，则跳过
            should_skip = False
            if is_done_new:
                should_skip = True
            elif is_done_legacy:
                if args.migrate_legacy == 1:
                    # 如果需要迁移，则不能从任务列表移除，需要进入主循环处理迁移
                    should_skip = False
                else:
                    should_skip = True

            if should_skip:
                skipped_count += 1
            else:
                final_tasks.append(entry)

        stats["skipped"] = skipped_count
        print(f"⏩ 已自动跳过: {skipped_count} 个已完成任务")
        print(f"▶️ 剩余待执行: {len(final_tasks)} 个任务")
    else:
        final_tasks = initial_tasks

    tasks = final_tasks

    print("-" * 60)
    failed_examples = []
    t0 = time.time()
    start_iso = datetime.datetime.now().isoformat()

    # --- 3. 执行循环 ---
    for i, entry in enumerate(tasks):
        video_id = entry.get("video_id")
        view_code = entry.get("view_code") or entry.get("view") or "unknown"

        if not video_id:
            stats["failed"] += 1
            append_failure_log(paths["failure_log"], entry, "Missing video_id")
            continue

        raw_path = entry.get("video_path")
        if raw_path is None:
            stats["failed"] += 1
            append_failure_log(paths["failure_log"], entry, "Missing video_path")
            continue

        video_abs_path = Path(raw_path) if Path(raw_path).is_absolute() else (paths["root"] / raw_path)
        if not video_abs_path.exists():
            stats["failed"] += 1
            append_failure_log(paths["failure_log"], entry, f"Missing video file: {video_abs_path}")
            continue
        out_dir_new, out_dir_legacy = compute_out_dirs(paths["output_root"], view_code, video_id)

        # 进度前缀 (显示当前剩余任务中的进度)
        prefix = f"[{i + 1}/{len(tasks)}][{video_id}]"

        # 3.1 再次检查 (防止边缘情况或处理迁移)
        if args.skip_existing == 1:
            if is_complete_outdir(out_dir_new, min_bytes=args.min_bytes):
                # 理论上不会进这里，除非预筛选后文件突然生成，但为了安全保留
                print(f"{prefix} SKIP (已存在)")
                continue

            if is_complete_outdir(out_dir_legacy, min_bytes=args.min_bytes):
                # 旧结构存在：可选迁移
                if args.migrate_legacy == 1:
                    moved = maybe_migrate_legacy(out_dir_legacy, out_dir_new, dry_run=args.dry_run, on_conflict="skip")
                    if moved:
                        stats["migrated"] += 1
                        print(f"{prefix} SKIP (已迁移)")
                        stats["skipped"] += 1  # 迁移也算处理完成
                        continue
                else:
                    print(f"{prefix} SKIP (旧结构已完成)")
                    continue

        # 3.2 创建新输出目录
        if not args.dry_run:
            out_dir_new.mkdir(parents=True, exist_ok=True)

        # 3.3 构造命令
        cmd = [
            python_exe, str(paths["target_script"]),
            "--video", str(video_abs_path),
            "--video_id", str(video_id),
            "--out_dir", str(out_dir_new),
        ]
        if int(args.short_video) == 1:
            cmd += ["--short_video", "1"]

        if args.dry_run:
            print(f"{prefix} [DRY-RUN] CMD: {' '.join(cmd)}")
            stats["success"] += 1
            continue

        # 3.4 实际执行
        loop_start = time.time()
        try:
            if args.stream_output == 1:
                subprocess.run(cmd, check=True)
            else:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{video_id}.log"
                with log_path.open("w", encoding="utf-8") as f:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )

            duration = time.time() - loop_start
            print(f"{prefix} SUCCESS ({duration:.1f}s)")
            stats["success"] += 1

        except subprocess.CalledProcessError as e:
            stats["failed"] += 1
            err_msg = e.stderr.strip() if e.stderr else "Unknown Subprocess Error"
            if args.stream_output == 1:
                err_msg = f"Exit {e.returncode} (see console logs)"
            short_err = err_msg[-300:].replace("\n", " ")
            print(f"{prefix} FAILED (Exit {e.returncode}) -> {short_err}")

            append_failure_log(paths["failure_log"], entry, err_msg)
            if len(failed_examples) < 20:
                failed_examples.append({"id": video_id, "error": short_err})

        except Exception as e:
            stats["failed"] += 1
            print(f"{prefix} ERROR (System) -> {str(e)}")

            append_failure_log(paths["failure_log"], entry, str(e))
            if len(failed_examples) < 20:
                failed_examples.append({"id": video_id, "error": str(e)})

    total_time = time.time() - t0
    end_iso = datetime.datetime.now().isoformat()

    params_log = {k: v for k, v in vars(args).items()}
    save_report(paths["report_file"], stats, params_log, failed_examples, start_iso, end_iso)

    print("=" * 60)
    print(f"🏁 批处理结束 (耗时: {total_time:.1f}s)")
    print(
        f"📊 统计: 范围总数 {stats['total_in_range']} | 跳过 {stats['skipped']} | 成功 {stats['success']} | 失败 {stats['failed']}")
    print(f"📄 报告已保存: {paths['report_file']}")
    if stats["failed"] > 0:
        print(f"⚠️ 失败日志: {paths['failure_log']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
