# scripts/intelligence_class/dataset_service.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict
import json


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


class DatasetService:
    """
    只做“读取 + 聚合 + API 友好输出”，不跑模型、不写大文件。
    约定 output 根目录下已有：
      - dataset_index.json
      - batch_report.json (可选)
      - batch_failures.jsonl (可选)
    """

    def __init__(self, output_dir: Path, dataset_name: str = "智慧课堂学生行为数据集"):
        self.output_dir = Path(output_dir).resolve()
        self.dataset_name = dataset_name
        self.dataset_root = (self.output_dir / dataset_name).resolve()

        self.index_path = self.output_dir / "dataset_index.json"
        self.report_path = self.output_dir / "batch_report.json"
        self.failures_path = self.output_dir / "batch_failures.jsonl"

        self._index_cache: Optional[Dict[str, Any]] = None

    # -------------------------
    # dataset index
    # -------------------------
    def get_index(self, force_reload: bool = False) -> Dict[str, Any]:
        if self._index_cache is not None and not force_reload:
            return self._index_cache

        if not self.index_path.exists():
            # 允许空，但要把路径告诉前端/你自己方便排查
            self._index_cache = {
                "dataset": self.dataset_name,
                "dataset_root": str(self.dataset_root),
                "index_path": str(self.index_path),
                "views": {},
                "warning": "dataset_index.json not found",
            }
            return self._index_cache

        idx = read_json(self.index_path)
        # 兜底字段
        idx.setdefault("dataset", self.dataset_name)
        idx.setdefault("dataset_root", str(self.dataset_root))
        idx.setdefault("index_path", str(self.index_path))
        idx.setdefault("views", {})
        self._index_cache = idx
        return idx

    # -------------------------
    # view stats
    # -------------------------
    def get_views_stats(self) -> Dict[str, Any]:
        idx = self.get_index()
        views = idx.get("views", {})

        out: Dict[str, Any] = {}
        for view, info in views.items():
            cases = info.get("cases", {}) or {}
            total = len(cases)

            def _truthy(v):  # 有些 index 会用 0/1 或 True/False
                return bool(v) and str(v).lower() not in ("none", "null", "")

            with_overlay = sum(1 for c in cases.values() if _truthy(c.get("has_overlay")) or _truthy(c.get("overlay")))
            with_behavior = sum(1 for c in cases.values() if _truthy(c.get("has_behavior")) or _truthy(c.get("behavior")))
            with_meta = sum(1 for c in cases.values() if _truthy(c.get("has_meta")) or _truthy(c.get("meta")))
            with_main = sum(1 for c in cases.values() if _truthy(c.get("has_main")) or _truthy(c.get("main")))

            out[view] = {
                "total_cases": total,
                "with_meta": with_meta,
                "with_main": with_main,
                "with_overlay": with_overlay,
                "with_behavior": with_behavior,
            }

        return {
            "dataset": self.dataset_name,
            "dataset_root": str(self.dataset_root),
            "views_stats": out,
        }

    # -------------------------
    # failures digest
    # -------------------------
    def get_failures(self, view: Optional[str] = None, limit: int = 2000) -> Dict[str, Any]:
        """
        从 batch_failures.jsonl 聚合失败原因。
        你 batch_failures 的字段名可能不统一：reason/error/message/stage 等
        这里做容错抽取。
        """
        if not self.failures_path.exists():
            return {
                "dataset": self.dataset_name,
                "failures_path": str(self.failures_path),
                "view": view,
                "total_failures": 0,
                "by_reason": {},
                "warning": "batch_failures.jsonl not found",
            }

        by_reason = defaultdict(int)
        by_stage = defaultdict(int)
        total = 0
        items = []

        for rec in iter_jsonl(self.failures_path):
            if view and str(rec.get("view", "")).strip() != str(view).strip():
                continue

            total += 1
            # reason 容错
            reason = rec.get("reason") or rec.get("error") or rec.get("message") or "unknown"
            stage = rec.get("stage") or rec.get("step") or rec.get("phase") or "unknown"

            by_reason[str(reason)] += 1
            by_stage[str(stage)] += 1

            if len(items) < limit:
                items.append(rec)

        # 排序输出更像“报告”
        by_reason_sorted = dict(sorted(by_reason.items(), key=lambda kv: -kv[1]))
        by_stage_sorted = dict(sorted(by_stage.items(), key=lambda kv: -kv[1]))

        return {
            "dataset": self.dataset_name,
            "failures_path": str(self.failures_path),
            "view": view,
            "total_failures": total,
            "by_stage": by_stage_sorted,
            "by_reason": by_reason_sorted,
            "samples": items,  # 给前端/你自己 debug 用
        }

    # -------------------------
    # batch report (optional)
    # -------------------------
    def get_batch_report(self) -> Dict[str, Any]:
        if not self.report_path.exists():
            return {
                "dataset": self.dataset_name,
                "report_path": str(self.report_path),
                "warning": "batch_report.json not found",
            }
        rep = read_json(self.report_path)
        rep.setdefault("dataset", self.dataset_name)
        rep.setdefault("report_path", str(self.report_path))
        return rep
