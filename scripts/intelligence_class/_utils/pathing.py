# -*- coding: utf-8 -*-
"""
pathing.py
统一解决两件事：
1) find_project_root(): 从任意脚本位置向上爬，找到 YOLOv11 项目根（同时包含 data/ 和 scripts/）
2) find_sibling_script(): 在 scripts/intelligence_class/ 内跨子目录找某个脚本（例如 01_run_single_video.py）
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def find_project_root(start: Path | None = None) -> Path:
    """
    向上查找项目根目录（同时包含 data/ 和 scripts/）。
    你从 intelligence_class 的任何子目录跑任何脚本都能稳住。
    """
    if start is None:
        # 默认从当前文件工作目录出发不靠谱，所以强烈建议调用者传 __file__
        start = Path.cwd()

    start = Path(start).resolve()
    candidates: Iterable[Path] = [start] + list(start.parents)

    for p in candidates:
        if (p / "data").exists() and (p / "scripts").exists():
            return p

    # 如果没找到，给一个明确错误
    raise FileNotFoundError(
        f"[pathing] Could not find project root from: {start}\n"
        f"Expected a directory containing both 'data/' and 'scripts/'."
    )


def find_intelligence_class_dir(project_root: Path) -> Path:
    """
    返回 scripts/intelligence_class 目录（你的目录名实际叫 intelligence class 或 intelligence_class 都兼容）。
    """
    scripts_dir = project_root / "scripts"
    # 兼容：有的人目录叫 intelligence class（带空格）
    candidates = [
        scripts_dir / "intelligence_class",
        scripts_dir / "intelligence class",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"[pathing] Could not find intelligence_class directory under: {scripts_dir}\n"
        f"Tried: {candidates}"
    )


def find_sibling_script(
    name: str,
    start_file: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """
    在 scripts/intelligence_class/ 下递归查找脚本文件（跨 pipeline/tools/training 等子目录）。
    - name: 例如 '01_run_single_video.py'
    - start_file: 建议传 Path(__file__)
    - project_root: 若已知可直接传，避免重复查找
    """
    if project_root is None:
        if start_file is None:
            project_root = find_project_root(Path.cwd())
        else:
            project_root = find_project_root(Path(start_file))

    ic_dir = find_intelligence_class_dir(project_root)

    matches = list(ic_dir.rglob(name))
    matches = [m for m in matches if m.is_file()]

    if not matches:
        raise FileNotFoundError(
            f"[pathing] Script '{name}' not found under: {ic_dir}"
        )

    # 如果找到多个，优先选择更“浅”的路径（通常是你想要的）
    matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return matches[0]


def resolve_under_project(project_root: Path, rel_or_abs: str) -> Path:
    """
    把一个路径参数解析为绝对路径：
    - 如果用户传的是绝对路径，直接返回
    - 如果是相对路径，默认相对于 project_root
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()
