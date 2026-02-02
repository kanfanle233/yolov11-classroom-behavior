# 02_dataset_augment_yolo_labels.py
# -*- coding: utf-8 -*-

import os
import cv2
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple


# =========================
# 基础工具
# =========================

def project_root_from_this_file() -> Path:
    # YOLOv11/scripts/intelligence_class/02_dataset_augment_yolo_labels.py
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_yolo_label(label_path: Path) -> List[List[float]]:
    """
    读取 YOLO 标签：cls cx cy w h
    """
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            labels.append([float(x) for x in parts])
    return labels


def write_yolo_label(label_path: Path, labels: List[List[float]]):
    with open(label_path, "w", encoding="utf-8") as f:
        for l in labels:
            f.write(" ".join(f"{x:.6f}" for x in l) + "\n")


# =========================
# 增强算子
# =========================

def random_brightness_contrast(img, brightness=0.2, contrast=0.2):
    """
    brightness / contrast ∈ [0, 1]
    """
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta = random.uniform(-brightness, brightness) * 255
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def horizontal_flip(img, labels):
    """
    水平翻转：cx -> 1 - cx
    """
    img_flipped = cv2.flip(img, 1)
    new_labels = []
    for cls, cx, cy, w, h in labels:
        new_labels.append([cls, 1.0 - cx, cy, w, h])
    return img_flipped, new_labels


def rotate_small_angle(img, labels, angle_deg=5):
    """
    小角度旋转（±angle_deg），并同步修正 bbox
    说明：这里只做轻量旋转，避免复杂裁剪
    """
    h, w = img.shape[:2]
    angle = random.uniform(-angle_deg, angle_deg)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    new_labels = []
    for cls, cx, cy, bw, bh in labels:
        # 转回像素坐标
        px = cx * w
        py = cy * h

        nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]

        ncx = nx / w
        ncy = ny / h

        # bbox 尺寸不变（小角度近似）
        if 0 <= ncx <= 1 and 0 <= ncy <= 1:
            new_labels.append([cls, ncx, ncy, bw, bh])

    return img_rot, new_labels


# =========================
# 主流程
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/标注数据集", help="relative to project root")
    parser.add_argument("--out_root", type=str, default="output/augmented", help="relative to project root")

    parser.add_argument("--aug_times", type=int, default=2, help="augment copies per image")
    parser.add_argument("--seed", type=int, default=42)

    # 增强开关
    parser.add_argument("--enable_flip", type=int, default=1)
    parser.add_argument("--enable_brightness", type=int, default=1)
    parser.add_argument("--enable_rotate", type=int, default=1)

    args = parser.parse_args()

    random.seed(args.seed)

    project_root = project_root_from_this_file()
    data_root = (project_root / args.data_root).resolve()
    out_root = (project_root / args.out_root).resolve()

    img_in = data_root / "images"
    lbl_in = data_root / "labels"

    img_out = out_root / "images"
    lbl_out = out_root / "labels"
    log_path = out_root / "augment_log.txt"

    ensure_dir(img_out)
    ensure_dir(lbl_out)

    images = sorted(img_in.glob("*.jpg")) + sorted(img_in.glob("*.png"))

    total_aug = 0

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"[INFO] Augment start, images={len(images)}\n")

        for img_path in images:
            label_path = lbl_in / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            labels = read_yolo_label(label_path)

            if img is None or not labels:
                continue

            for i in range(args.aug_times):
                aug_img = img.copy()
                aug_labels = [l.copy() for l in labels]
                ops = []

                if args.enable_flip and random.random() < 0.5:
                    aug_img, aug_labels = horizontal_flip(aug_img, aug_labels)
                    ops.append("flip")

                if args.enable_brightness:
                    aug_img = random_brightness_contrast(aug_img)
                    ops.append("bright")

                if args.enable_rotate and random.random() < 0.5:
                    aug_img, aug_labels = rotate_small_angle(aug_img, aug_labels)
                    ops.append("rotate")

                out_img_name = f"{img_path.stem}_aug{i}.jpg"
                out_lbl_name = f"{img_path.stem}_aug{i}.txt"

                cv2.imwrite(str(img_out / out_img_name), aug_img)
                write_yolo_label(lbl_out / out_lbl_name, aug_labels)

                log.write(f"{out_img_name} <- {img_path.name} ops={ops}\n")
                total_aug += 1

        log.write(f"[DONE] total_augmented={total_aug}\n")

    print("=" * 80)
    print("[AUGMENT DONE]")
    print(f"Input images : {len(images)}")
    print(f"Augmented    : {total_aug}")
    print(f"Output dir   : {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
