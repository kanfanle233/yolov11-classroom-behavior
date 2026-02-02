# -*- coding: utf-8 -*-
"""
00_collect_code_for_ai.py
在 scripts/intelligence class/ 下运行：
  python 00_collect_code_for_ai.py --project_root ..\\.. --out_dir ..\\..\\output\\_code_audit

功能：
1) 递归扫描代码文件（py/js/ts/html/css/json/yaml/md 等）
2) 过滤常见垃圾目录（.git、venv、__pycache__、runs、output 等可配置）
3) 提取：
   - 文件树（tree）
   - 每个文件：行数/大小/最近修改时间/sha1
   - Python 额外提取：imports、函数/类定义、TODO/FIXME、疑似入口(main/app)、FastAPI/Flask/Uvicorn关键词、subprocess调用
4) 输出：
   - report.json（全量结构化信息）
   - report.md（给AI看的“导航摘要”）
   - all_code.txt（可选，把源码拼接成一个大文件，方便一次性喂给AI）
"""

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# 1) 默认配置
# -----------------------------
DEFAULT_INCLUDE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".css",
    ".json", ".yaml", ".yml",
    ".md", ".txt",
    ".toml", ".ini", ".cfg",
    ".sh", ".bat", ".ps1",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".idea", ".vscode",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", "node_modules",
    "runs", "output", "dist", "build",
    ".ipynb_checkpoints",
}

DEFAULT_EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
}

# 单文件合并时，每个文件最多保留多少行（防止巨大文件把 AI 撑爆）
MAX_LINES_PER_FILE_IN_ALLTXT = 2000

# 单文件合并时，单个文件最大字节（超了就跳过合并）
MAX_BYTES_PER_FILE_IN_ALLTXT = 2 * 1024 * 1024  # 2MB

# -----------------------------
# 2) 小工具
# -----------------------------
def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def read_text_safely(path: Path) -> Tuple[str, str]:
    """
    返回 (text, encoding_used)
    """
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp936", "latin1"):
        try:
            return path.read_text(encoding=enc, errors="strict"), enc
        except Exception:
            pass
    # 最后兜底：忽略错误
    return path.read_text(encoding="utf-8", errors="ignore"), "utf-8(ignore)"

def is_excluded_dir(dir_name: str, exclude_dirs: set) -> bool:
    return dir_name in exclude_dirs

def norm_rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")

# -----------------------------
# 3) Python 解析（轻量正则，不依赖 ast，避免奇葩编码/语法导致 crash）
# -----------------------------
RE_IMPORT = re.compile(r"^\s*(import\s+[\w\.,\s]+|from\s+[\w\.]+\s+import\s+[\w\.,\s\*]+)\s*$")
RE_DEF = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(")
RE_CLASS = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*(\(|:)")
RE_TODO = re.compile(r"(TODO|FIXME|BUG|HACK)\b[:：]?", re.IGNORECASE)

# 一些你项目里很可能关心的“架构/入口”关键词
ARCH_KEYWORDS = {
    "FastAPI": re.compile(r"\bFastAPI\b"),
    "Uvicorn": re.compile(r"\buvicorn\b"),
    "Flask": re.compile(r"\bFlask\b"),
    "Dash": re.compile(r"\bdash\b"),
    "Jinja2Templates": re.compile(r"\bJinja2Templates\b"),
    "StaticFiles": re.compile(r"\bStaticFiles\b"),
    "subprocess": re.compile(r"\bsubprocess\.(run|Popen|call)\b"),
    "argparse": re.compile(r"\bargparse\b"),
    "YOLO(Ultralytics)": re.compile(r"\bultralytics\b|\bYOLO\b"),
    "OpenCV": re.compile(r"\bcv2\b"),
}

def analyze_python(text: str) -> Dict:
    imports: List[str] = []
    funcs: List[str] = []
    classes: List[str] = []
    todos: List[str] = []
    hits: Dict[str, int] = {k: 0 for k in ARCH_KEYWORDS.keys()}

    lines = text.splitlines()
    for line in lines:
        m = RE_IMPORT.match(line)
        if m:
            imports.append(m.group(1).strip())

        m = RE_DEF.match(line)
        if m:
            funcs.append(m.group(1))

        m = RE_CLASS.match(line)
        if m:
            classes.append(m.group(1))

        if RE_TODO.search(line):
            # 截断避免太长
            todos.append(line.strip()[:200])

        for k, r in ARCH_KEYWORDS.items():
            if r.search(line):
                hits[k] += 1

    # 猜入口文件：包含 if __name__ == "__main__" 或 app = FastAPI()
    is_entry = ("__name__" in text and "__main__" in text) or ("app = FastAPI" in text) or ("FastAPI(" in text)

    return {
        "imports": imports[:200],
        "functions": funcs[:300],
        "classes": classes[:200],
        "todos": todos[:200],
        "keyword_hits": {k: v for k, v in hits.items() if v > 0},
        "is_entry_like": bool(is_entry),
    }

# -----------------------------
# 4) 数据结构
# -----------------------------
@dataclass
class FileInfo:
    rel_path: str
    abs_path: str
    ext: str
    size_bytes: int
    line_count: int
    mtime: str
    sha1: str
    encoding: Optional[str] = None
    python_meta: Optional[Dict] = None

# -----------------------------
# 5) 主扫描逻辑
# -----------------------------
def scan_project(
    project_root: Path,
    include_exts: set,
    exclude_dirs: set,
    exclude_files: set,
    max_files: int = 20000,
) -> List[FileInfo]:
    out: List[FileInfo] = []
    file_count = 0

    for root, dirs, files in os.walk(project_root):
        # 过滤目录
        dirs[:] = [d for d in dirs if not is_excluded_dir(d, exclude_dirs)]

        for fn in files:
            if fn in exclude_files:
                continue
            path = Path(root) / fn
            ext = path.suffix.lower()

            if ext not in include_exts:
                continue

            try:
                st = path.stat()
            except Exception:
                continue

            file_count += 1
            if file_count > max_files:
                break

            # 读取（只对文本类做）
            text, enc = read_text_safely(path)
            b = text.encode(enc if enc else "utf-8", errors="ignore")
            h = sha1_bytes(b)
            lc = text.count("\n") + 1 if text else 0

            py_meta = None
            if ext == ".py":
                py_meta = analyze_python(text)

            info = FileInfo(
                rel_path=norm_rel(path, project_root),
                abs_path=str(path.resolve()),
                ext=ext,
                size_bytes=int(st.st_size),
                line_count=int(lc),
                mtime=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                sha1=h,
                encoding=enc,
                python_meta=py_meta,
            )
            out.append(info)

        if file_count > max_files:
            break

    # 按路径排序
    out.sort(key=lambda x: x.rel_path.lower())
    return out

def build_tree(paths: List[str]) -> Dict:
    tree: Dict = {}
    for p in paths:
        parts = p.split("/")
        cur = tree
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur.setdefault(parts[-1], None)
    return tree

def tree_to_markdown(tree: Dict, indent: str = "") -> str:
    lines: List[str] = []
    for k in sorted(tree.keys()):
        v = tree[k]
        if v is None:
            lines.append(f"{indent}- {k}")
        else:
            lines.append(f"{indent}- {k}/")
            lines.append(tree_to_markdown(v, indent + "  "))
    return "\n".join(lines)

def write_all_code_txt(project_root: Path, files: List[FileInfo], out_path: Path) -> Dict:
    """
    拼接源码为单文件，方便一次性喂给AI。
    返回统计信息。
    """
    written_files = 0
    skipped_too_big = 0
    skipped_too_long = 0

    with out_path.open("w", encoding="utf-8", errors="ignore") as f:
        for info in files:
            src_path = Path(info.abs_path)
            try:
                if info.size_bytes > MAX_BYTES_PER_FILE_IN_ALLTXT:
                    skipped_too_big += 1
                    continue

                text, enc = read_text_safely(src_path)
                lines = text.splitlines()
                if len(lines) > MAX_LINES_PER_FILE_IN_ALLTXT:
                    lines = lines[:MAX_LINES_PER_FILE_IN_ALLTXT]
                    skipped_too_long += 1

                f.write("\n" + "=" * 90 + "\n")
                f.write(f"FILE: {info.rel_path}\n")
                f.write(f"SIZE: {info.size_bytes} bytes | LINES: {info.line_count} | MTIME: {info.mtime}\n")
                f.write("=" * 90 + "\n")
                f.write("\n".join(lines))
                f.write("\n")

                written_files += 1
            except Exception:
                continue

    return {
        "written_files": written_files,
        "skipped_too_big": skipped_too_big,
        "skipped_truncated_lines": skipped_too_long,
        "all_code_txt": str(out_path),
    }

def make_report_md(project_root: Path, files: List[FileInfo], tree_md: str) -> str:
    # 统计入口/重点文件
    entry_like = [fi for fi in files if fi.ext == ".py" and fi.python_meta and fi.python_meta.get("is_entry_like")]
    fastapi_like = [fi for fi in files if fi.ext == ".py" and fi.python_meta and "FastAPI" in (fi.python_meta.get("keyword_hits") or {})]
    subprocess_like = [fi for fi in files if fi.ext == ".py" and fi.python_meta and "subprocess" in (fi.python_meta.get("keyword_hits") or {})]
    yolo_like = [fi for fi in files if fi.ext == ".py" and fi.python_meta and "YOLO(Ultralytics)" in (fi.python_meta.get("keyword_hits") or {})]

    def top_list(items: List[FileInfo], limit=20) -> str:
        if not items:
            return "(none)"
        return "\n".join([f"- {x.rel_path}" for x in items[:limit]])

    total_files = len(files)
    total_lines = sum(f.line_count for f in files)
    total_bytes = sum(f.size_bytes for f in files)

    md = []
    md.append("# Code Audit Pack (for AI)\n")
    md.append(f"- project_root: `{project_root}`")
    md.append(f"- files: **{total_files}**")
    md.append(f"- total_lines: **{total_lines}**")
    md.append(f"- total_size: **{total_bytes/1024/1024:.2f} MB**\n")

    md.append("## 1) Project Tree (filtered)\n")
    md.append(tree_md + "\n")

    md.append("## 2) Suspected Entry / App Files\n")
    md.append(top_list(entry_like) + "\n")

    md.append("## 3) FastAPI-related Python Files\n")
    md.append(top_list(fastapi_like) + "\n")

    md.append("## 4) Files using subprocess (pipeline runners)\n")
    md.append(top_list(subprocess_like) + "\n")

    md.append("## 5) Files mentioning Ultralytics/YOLO\n")
    md.append(top_list(yolo_like) + "\n")

    md.append("## 6) TODO/FIXME/BUG hints (first 50 hits)\n")
    hits = []
    for fi in files:
        if fi.ext == ".py" and fi.python_meta and fi.python_meta.get("todos"):
            for t in fi.python_meta["todos"]:
                hits.append((fi.rel_path, t))
    if not hits:
        md.append("(none)\n")
    else:
        for rel, t in hits[:50]:
            md.append(f"- `{rel}`: {t}")
        md.append("")

    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=None,
                        help="项目根目录（YOLOv11）。默认：本脚本向上两级")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="输出目录，默认：<project_root>/output/_code_audit")
    parser.add_argument("--include_exts", type=str, default="",
                        help="额外包含扩展名，用逗号分隔，比如: .py,.html")
    parser.add_argument("--exclude_dirs", type=str, default="",
                        help="额外排除目录名，用逗号分隔，比如: data,runs,output")
    parser.add_argument("--exclude_output", action="store_true",
                        help="默认排除 output；如果你想包含 output 目录，就加这个参数（不建议）")
    parser.add_argument("--no_all_code", action="store_true",
                        help="不生成 all_code.txt（只生成 report.json + report.md）")
    parser.add_argument("--max_files", type=int, default=20000)

    args = parser.parse_args()

    here = Path(__file__).resolve()
    default_root = here.parents[2]  # YOLOv11/scripts/intelligence class/ -> YOLOv11
    project_root = Path(args.project_root).resolve() if args.project_root else default_root

    include_exts = set(DEFAULT_INCLUDE_EXTS)
    if args.include_exts.strip():
        for x in args.include_exts.split(","):
            x = x.strip().lower()
            if x and not x.startswith("."):
                x = "." + x
            if x:
                include_exts.add(x)

    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
    if args.exclude_output:
        # 用户要求包含 output，则从 exclude 中移除
        exclude_dirs.discard("output")

    if args.exclude_dirs.strip():
        for d in args.exclude_dirs.split(","):
            d = d.strip()
            if d:
                exclude_dirs.add(d)

    exclude_files = set(DEFAULT_EXCLUDE_FILES)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (project_root / "output" / "_code_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("🧠 Code Collector for AI")
    print(f"📂 project_root : {project_root}")
    print(f"📁 out_dir      : {out_dir}")
    print(f"✅ include_exts : {sorted(include_exts)}")
    print(f"🚫 exclude_dirs : {sorted(exclude_dirs)}")
    print("============================================================")

    files = scan_project(
        project_root=project_root,
        include_exts=include_exts,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files,
        max_files=args.max_files,
    )

    tree = build_tree([f.rel_path for f in files])
    tree_md = tree_to_markdown(tree)

    report = {
        "meta": {
            "project_root": str(project_root),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "file_count": len(files),
        },
        "files": [asdict(f) for f in files],
        "tree": tree,
    }

    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    all_code_txt = out_dir / "all_code.txt"

    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(make_report_md(project_root, files, tree_md), encoding="utf-8")

    alltxt_info = None
    if not args.no_all_code:
        alltxt_info = write_all_code_txt(project_root, files, all_code_txt)

    print("\n✅ DONE")
    print(f"- report.json : {report_json}")
    print(f"- report.md   : {report_md}")
    if alltxt_info:
        print(f"- all_code.txt: {all_code_txt}")
        print(f"  written_files={alltxt_info['written_files']}, "
              f"skipped_too_big={alltxt_info['skipped_too_big']}, "
              f"truncated_files={alltxt_info['skipped_truncated_lines']}")
    print("\n接下来：把 report.md + report.json（必要时再加 all_code.txt）发给 AI 让它提建议。")

if __name__ == "__main__":
    main()
