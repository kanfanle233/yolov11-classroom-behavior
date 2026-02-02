# 01_dataset_convert_case_to_yolo.py
# -*- coding: utf-8 -*-
"""
Convert "智慧课堂学生行为数据集案例（正方视角）" (jpg + json) to YOLO detection dataset.

Input:
  data/智慧课堂学生行为数据集/案例/智慧课堂学生行为数据集案例（正方视角）/
    ├─ 0.jpg
    ├─ 0.json
    ├─ 1.jpg
    ├─ 1.json
    ...

JSON format:
  {"labels":[{"name":"tt","x1":...,"y1":...,"x2":...,"y2":...}, ...]}
Pixel coordinates (x1,y1,x2,y2).

Output:
  output/case_yolo/
    ├─ images/train|val|test/*.jpg
    ├─ labels/train|val|test/*.txt
    └─ data.yaml

Split: train:val:test = 8:1:1 (random, reproducible with --seed)

Usage (from YOLOv11 project root):
  python "scripts/intelligence_classs/01_dataset_convert_case_to_yolo.py" --seed 42
"""

import os
import json
import shutil
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ✅ 让脚本在任何目录运行都能 import scripts.intelligence_class._utils.pathing
_this = Path(__file__).resolve()
for p in [_this] + list(_this.parents):
    if (p / "data").exists() and (p / "scripts").exists():
        sys.path.insert(0, str(p))
        break

# ✅ 统一使用路径工具（若你的 pathing.py 没有 resolve_under_project，会自动 fallback）
from scripts.intelligence_class._utils.pathing import find_project_root
try:
    from scripts.intelligence_class._utils.pathing import resolve_under_project  # type: ignore
except Exception:  # pragma: no cover
    resolve_under_project = None  # type: ignore


# ----------------------------
# Category mapping
# ----------------------------
CLASS_MAP: Dict[str, int] = {
    "dx": 0,  # 低头写字
    "dk": 1,  # 低头看书
    "tt": 2,  # 抬头听课
    "zt": 3,  # 转头
    "js": 4,  # 举手
    "zl": 5,  # 站立
    "xt": 6,  # 小组讨论
    "jz": 7,  # 教师指导
}
NAMES: List[str] = ["dx", "dk", "tt", "zt", "js", "zl", "xt", "jz"]


def get_project_root() -> Path:
    """
    This script is expected at:
      YOLOv11/scripts/intelligence_class/01_dataset_convert_case_to_yolo.py
    project_root = YOLOv11

    ✅ 改为：使用 _utils/pathing.py 的 find_project_root()，避免目录重构后 parents[...] 失效。
    """
    return find_project_root(Path(__file__).resolve())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def convert_box_to_yolo(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert pixel bbox (x1,y1,x2,y2) -> YOLO normalized (cx,cy,w,h).
    Return None if invalid.
    """
    if W <= 0 or H <= 0:
        return None

    # fix reversed coords
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # clamp to image boundary
    x1 = clamp(x1, 0.0, float(W - 1))
    x2 = clamp(x2, 0.0, float(W - 1))
    y1 = clamp(y1, 0.0, float(H - 1))
    y2 = clamp(y2, 0.0, float(H - 1))

    bw = x2 - x1
    bh = y2 - y1
    if bw < 2 or bh < 2:
        return None

    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = bw / W
    h = bh / H

    # safety clamp
    cx = clamp(cx, 0.0, 1.0)
    cy = clamp(cy, 0.0, 1.0)
    w = clamp(w, 0.0, 1.0)
    h = clamp(h, 0.0, 1.0)

    if w <= 0 or h <= 0:
        return None
    return (cx, cy, w, h)


def read_image_size_fast(img_path: Path) -> Tuple[int, int]:
    """
    Read image size without extra deps:
    - Use PIL if available
    - Fallback to OpenCV if available
    """
    # Try PIL first
    try:
        from PIL import Image  # type: ignore
        with Image.open(img_path) as im:
            W, H = im.size
            return int(W), int(H)
    except Exception:
        pass

    # Try OpenCV
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError("cv2.imread returned None")
        H, W = img.shape[:2]
        return int(W), int(H)
    except Exception as e:
        raise RuntimeError(f"Cannot read image size for {img_path}: {e}")


def write_yolo_label_txt(out_txt: Path, yolo_lines: List[str]) -> None:
    ensure_dir(out_txt.parent)
    with open(out_txt, "w", encoding="utf-8") as f:
        for line in yolo_lines:
            f.write(line + "\n")


def write_data_yaml(out_yaml: Path, dataset_root: Path) -> None:
    """
    Ultralytics expects:
      path: <dataset root>
      train: images/train
      val: images/val
      test: images/test
      names: [...]
    Use POSIX-like slashes for yaml (works on Windows too).
    """
    ensure_dir(out_yaml.parent)
    content = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for i, n in enumerate(NAMES):
        content.append(f"  {i}: {n}")
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(content) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default="data/智慧课堂学生行为数据集/案例/智慧课堂学生行为数据集案例（正方视角）",
        help="relative to project root",
    )
    parser.add_argument("--out_dir", type=str, default="output/case_yolo", help="relative to project root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", type=int, default=0, help="1 to delete existing output/case_yolo first")

    args = parser.parse_args()

    # validate ratios
    s = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {s}")

    random.seed(args.seed)

    project_root = get_project_root()

    # ✅ 改：in/out 参数统一相对 project_root 解析（支持用户传绝对路径）
    if resolve_under_project is not None:
        in_dir = resolve_under_project(project_root, args.in_dir)
        out_dir = resolve_under_project(project_root, args.out_dir)
    else:
        # fallback: 如果你的 pathing.py 没有 resolve_under_project
        def _resolve(p: str) -> Path:
            pp = Path(p)
            return pp.resolve() if pp.is_absolute() else (project_root / pp).resolve()

        in_dir = _resolve(args.in_dir)
        out_dir = _resolve(args.out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    if out_dir.exists() and int(args.overwrite) == 1:
        shutil.rmtree(out_dir)

    # Prepare output folders
    for split in ["train", "val", "test"]:
        ensure_dir(out_dir / "images" / split)
        ensure_dir(out_dir / "labels" / split)

    # Collect pairs: N.jpg + N.json
    jpgs = sorted(list(in_dir.glob("*.jpg")))
    # also accept jpeg/png just in case
    jpgs += sorted(list(in_dir.glob("*.jpeg")))
    jpgs += sorted(list(in_dir.glob("*.png")))

    items: List[Tuple[Path, Path]] = []
    missing_json = 0
    for img_path in jpgs:
        json_path = in_dir / (img_path.stem + ".json")
        if not json_path.exists():
            missing_json += 1
            continue
        items.append((img_path, json_path))

    if not items:
        raise RuntimeError(f"No (image,json) pairs found in: {in_dir}")

    # Shuffle & split
    random.shuffle(items)
    n = len(items)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    # remainder -> test
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    assert len(test_items) == n_test

    splits = {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

    # Convert
    stats = {
        "total_pairs": n,
        "missing_json": missing_json,
        "splits": {k: len(v) for k, v in splits.items()},
        "labels_total": 0,
        "labels_kept": 0,
        "labels_dropped": 0,
        "unknown_class": 0,
        "bad_box": 0,
    }

    for split, split_items in splits.items():
        for img_path, json_path in split_items:
            # read size
            W, H = read_image_size_fast(img_path)

            # read json labels
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to parse json: {json_path} ({e})")

            labels = ann.get("labels", [])
            if not isinstance(labels, list):
                labels = []

            yolo_lines: List[str] = []
            for obj in labels:
                stats["labels_total"] += 1
                if not isinstance(obj, dict):
                    stats["labels_dropped"] += 1
                    continue

                name = obj.get("name", None)
                if name not in CLASS_MAP:
                    stats["unknown_class"] += 1
                    stats["labels_dropped"] += 1
                    continue

                # coords
                try:
                    x1 = float(obj.get("x1"))
                    y1 = float(obj.get("y1"))
                    x2 = float(obj.get("x2"))
                    y2 = float(obj.get("y2"))
                except Exception:
                    stats["bad_box"] += 1
                    stats["labels_dropped"] += 1
                    continue

                yolo = convert_box_to_yolo(x1, y1, x2, y2, W=W, H=H)
                if yolo is None:
                    stats["bad_box"] += 1
                    stats["labels_dropped"] += 1
                    continue

                cls_id = CLASS_MAP[name]
                cx, cy, bw, bh = yolo
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                stats["labels_kept"] += 1

            # copy image
            out_img = out_dir / "images" / split / (img_path.stem + ".jpg")
            # normalize to jpg extension for consistency; if original is png/jpeg, still copy as .jpg name
            shutil.copy2(img_path, out_img)

            # write label txt (even empty file is acceptable for YOLO; indicates no objects)
            out_lbl = out_dir / "labels" / split / (img_path.stem + ".txt")
            write_yolo_label_txt(out_lbl, yolo_lines)

    # write data.yaml
    data_yaml = out_dir / "data.yaml"
    # write path relative or absolute? Ultralytics supports absolute. We'll write absolute posix for robustness.
    write_data_yaml(data_yaml, dataset_root=out_dir)

    # write a small conversion report
    report_path = out_dir / "convert_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("[DONE] Case annotations converted to YOLO dataset.")
    print(f"[IN ] {in_dir}")
    print(f"[OUT] {out_dir}")
    print(f"[YAML] {data_yaml}")
    print(f"[REPORT] {report_path}")
    print("-" * 90)
    print(f"Total pairs     : {stats['total_pairs']} (missing_json={stats['missing_json']})")
    print(f"Split train/val/test : {stats['splits']}")
    print(f"Labels total    : {stats['labels_total']}")
    print(f"Labels kept     : {stats['labels_kept']}")
    print(f"Labels dropped  : {stats['labels_dropped']} (unknown_class={stats['unknown_class']}, bad_box={stats['bad_box']})")
    print("=" * 90)


if __name__ == "__main__":
    main()
