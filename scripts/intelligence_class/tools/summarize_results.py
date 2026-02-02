# summarize_results.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional


def project_root_from_this_file() -> Path:
    """
    约定脚本位置：YOLOv11/scripts/intelligence_class/summarize_results.py
    project_root = YOLOv11
    """
    return Path(__file__).resolve().parents[2]


def read_failed_list(failed_path: Path) -> List[str]:
    if not failed_path.exists():
        return []
    lines = []
    with open(failed_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def parse_one_jsonl(jsonl_path: Path, max_lines: int = 0) -> Dict[str, Any]:
    """
    解析单个 jsonl，统计：
    - frames: 行数（等于处理过的帧数，考虑 stride）
    - total_boxes: 总 box 数
    - total_persons_pose: 如果有 keypoints_xy，统计每帧人数并累加
    """
    frames = 0
    total_boxes = 0
    total_persons_pose = 0

    # 为了速度，允许采样：max_lines>0 就最多读这么多行
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            frames += 1
            try:
                obj = json.loads(line)
            except Exception:
                # 脏行直接跳过
                continue

            pred = obj.get("pred", {})
            boxes = pred.get("boxes", [])
            if isinstance(boxes, list):
                total_boxes += len(boxes)

            # pose: keypoints_xy: list[N][K][2]
            kpxy = pred.get("keypoints_xy", None)
            if isinstance(kpxy, list):
                # 每帧检测到的人数 = len(kpxy)
                total_persons_pose += len(kpxy)

            if max_lines > 0 and frames >= max_lines:
                break

    return {
        "frames": frames,
        "total_boxes": total_boxes,
        "total_persons_pose": total_persons_pose,
    }


def human_table(rows: List[Tuple[str, Any]]) -> str:
    """
    简单对齐输出
    """
    if not rows:
        return ""
    klen = max(len(k) for k, _ in rows)
    lines = []
    for k, v in rows:
        lines.append(f"{k:<{klen}} : {v}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="output/智慧课堂学生行为数据集", help="relative to project root")
    parser.add_argument("--report_name", type=str, default="summary_report", help="base name without extension")
    parser.add_argument("--max_lines_per_jsonl", type=int, default=0, help="0=read full; >0 = sample first N lines for speed")
    parser.add_argument("--include_failed_list", type=int, default=1, help="1/0 include failed file list in txt/json report")

    args = parser.parse_args()
    include_failed = bool(int(args.include_failed_list))

    project_root = project_root_from_this_file()
    out_root = (project_root / args.out_root).resolve()

    if not out_root.exists():
        raise FileNotFoundError(f"Output root not found: {out_root}")

    failed_path = out_root / "failed_videos.txt"
    failed_list = read_failed_list(failed_path)

    # 视角目录：out_root 下的子目录（排除文件）
    view_dirs = sorted([p for p in out_root.iterdir() if p.is_dir()], key=lambda x: x.name)

    # 全局统计
    global_videos = 0
    global_jsonl = 0
    global_overlay = 0

    global_frames = 0
    global_total_boxes = 0
    global_total_persons_pose = 0

    per_view: Dict[str, Dict[str, Any]] = {}

    # 遍历视角
    for vd in view_dirs:
        view = vd.name

        jsonls = sorted(vd.glob("*.jsonl"))
        overlays = sorted(vd.glob("*_overlay.mp4"))

        # 每个 jsonl 认为是一个视频输出
        video_count = len(jsonls)
        overlay_count = len(overlays)

        # 解析 jsonl（可能很多文件，这里按 max_lines_per_jsonl 控制速度）
        frames_sum = 0
        boxes_sum = 0
        persons_pose_sum = 0

        for j in jsonls:
            st = parse_one_jsonl(j, max_lines=args.max_lines_per_jsonl)
            frames_sum += safe_int(st.get("frames", 0))
            boxes_sum += safe_int(st.get("total_boxes", 0))
            persons_pose_sum += safe_int(st.get("total_persons_pose", 0))

        per_view[view] = {
            "videos_by_jsonl": video_count,
            "overlays": overlay_count,
            "frames": frames_sum,
            "total_boxes": boxes_sum,
            "total_persons_pose": persons_pose_sum,
            "avg_boxes_per_frame": (boxes_sum / frames_sum) if frames_sum > 0 else 0.0,
            "avg_persons_pose_per_frame": (persons_pose_sum / frames_sum) if frames_sum > 0 else 0.0,
            "avg_frames_per_video": (frames_sum / video_count) if video_count > 0 else 0.0,
        }

        global_videos += video_count
        global_jsonl += video_count
        global_overlay += overlay_count
        global_frames += frames_sum
        global_total_boxes += boxes_sum
        global_total_persons_pose += persons_pose_sum

    report = {
        "output_root": str(out_root),
        "views": per_view,
        "totals": {
            "videos_by_jsonl": global_videos,
            "overlays": global_overlay,
            "frames": global_frames,
            "total_boxes": global_total_boxes,
            "total_persons_pose": global_total_persons_pose,
            "avg_boxes_per_frame": (global_total_boxes / global_frames) if global_frames > 0 else 0.0,
            "avg_persons_pose_per_frame": (global_total_persons_pose / global_frames) if global_frames > 0 else 0.0,
        },
        "failed": {
            "failed_videos_txt": str(failed_path) if failed_path.exists() else "",
            "count": len(failed_list),
            "items": failed_list if include_failed else [],
        },
        "note": (
            "videos_by_jsonl 以 *.jsonl 文件数为准（每个 jsonl 对应一个输入视频片段的输出）。"
            "如果你采用 stride/采样，frames 为已处理帧数（不是原始总帧）。"
        ),
    }

    # ===== 打印到控制台（清晰一点）=====
    print("\n" + "=" * 90)
    print("[SUMMARY REPORT] 智慧课堂学生行为数据集（当前 output 状态）")
    print("=" * 90)

    # totals
    totals_rows = [
        ("Output Root", out_root),
        ("Views", len(view_dirs)),
        ("Videos (by jsonl)", report["totals"]["videos_by_jsonl"]),
        ("Overlays", report["totals"]["overlays"]),
        ("Frames (processed)", report["totals"]["frames"]),
        ("Total boxes", report["totals"]["total_boxes"]),
        ("Total pose persons", report["totals"]["total_persons_pose"]),
        ("Avg boxes / frame", f"{report['totals']['avg_boxes_per_frame']:.4f}"),
        ("Avg persons / frame", f"{report['totals']['avg_persons_pose_per_frame']:.4f}"),
        ("Failed count", report["failed"]["count"]),
    ]
    print(human_table(totals_rows))

    print("\n[PER VIEW]")
    # 视角逐行打印
    for view, st in per_view.items():
        print("-" * 90)
        rows = [
            ("View", view),
            ("Videos (jsonl)", st["videos_by_jsonl"]),
            ("Overlays", st["overlays"]),
            ("Frames (processed)", st["frames"]),
            ("Total boxes", st["total_boxes"]),
            ("Total pose persons", st["total_persons_pose"]),
            ("Avg boxes / frame", f"{st['avg_boxes_per_frame']:.4f}"),
            ("Avg persons / frame", f"{st['avg_persons_pose_per_frame']:.4f}"),
            ("Avg frames / video", f"{st['avg_frames_per_video']:.2f}"),
        ]
        print(human_table(rows))

    if include_failed and failed_list:
        print("\n[FAILED LIST] (first 50)")
        for x in failed_list[:50]:
            print(" -", x)
        if len(failed_list) > 50:
            print(f" ... and {len(failed_list) - 50} more")

    # ===== 保存报告文件 =====
    report_json = out_root / f"{args.report_name}.json"
    report_txt = out_root / f"{args.report_name}.txt"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # txt 版本更易读
    lines = []
    lines.append("[SUMMARY REPORT] 智慧课堂学生行为数据集（当前 output 状态）")
    lines.append("=" * 90)
    lines.append(human_table(totals_rows))
    lines.append("")
    lines.append("[PER VIEW]")
    for view, st in per_view.items():
        lines.append("-" * 90)
        lines.append(human_table([
            ("View", view),
            ("Videos (jsonl)", st["videos_by_jsonl"]),
            ("Overlays", st["overlays"]),
            ("Frames (processed)", st["frames"]),
            ("Total boxes", st["total_boxes"]),
            ("Total pose persons", st["total_persons_pose"]),
            ("Avg boxes / frame", f"{st['avg_boxes_per_frame']:.4f}"),
            ("Avg persons / frame", f"{st['avg_persons_pose_per_frame']:.4f}"),
            ("Avg frames / video", f"{st['avg_frames_per_video']:.2f}"),
        ]))
    if include_failed:
        lines.append("")
        lines.append(f"[FAILED] count={len(failed_list)}")
        if failed_list:
            lines.append("[FAILED LIST]")
            lines.extend([f"- {x}" for x in failed_list])

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 90)
    print(f"[SAVED] {report_txt}")
    print(f"[SAVED] {report_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()
