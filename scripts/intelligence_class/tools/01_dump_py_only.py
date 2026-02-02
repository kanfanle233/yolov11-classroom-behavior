# -*- coding: utf-8 -*-
"""
01_dump_py_and_html.py

功能：
- 锁定 intelligence_class 根目录
- 递归搜索所有 .py 和 .html 文件
- 【新增】自动排除无关文件夹 (如 venv, .git, runs, __pycache__)
- 按文件夹结构排序并汇总
"""

from pathlib import Path
from datetime import datetime

from scripts.intelligence_class._utils.pathing import find_project_root

# =========================
# 1. 路径配置
# =========================
# 向上两级，定位到 intelligence_class 根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = find_project_root(Path(__file__).resolve())
OUTPUT_DIR = ROOT_DIR / "output" / "_code_dump"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 搜索配置 (在这里定义你要什么，不要什么)
# =========================

# A. 想要的文件后缀
TARGET_EXTENSIONS = ["*.py", "*.html"]

# B. 必须要排除的文件夹名 (黑名单)
# 如果你有其他不想看的文件夹（比如 build, dist, temp），加在这里
IGNORE_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    "venv",
    ".venv",
    "runs",  # 既然你是跑模型，runs 里面通常是日志和权重，不需要给 AI 看
    "egg-info"
}

# =========================
# 3. 收集并过滤文件
# =========================
print(f"📂 扫描根目录: {PROJECT_ROOT}")

all_files = []

# 遍历所有指定的后缀
for ext in TARGET_EXTENSIONS:
    # rglob 是递归搜索
    all_files.extend(PROJECT_ROOT.rglob(ext))

# 过滤逻辑：只要路径中包含 IGNORE_DIRS 里的任意一个词，就剔除
valid_files = []
for p in all_files:
    # 拆分路径，检查每一层文件夹是否在黑名单里
    # 例如: path/to/venv/script.py -> 'venv' 在黑名单 -> 剔除
    parts = set(part.lower() for part in p.parts)
    if not parts.intersection(IGNORE_DIRS):
        valid_files.append(p)

# 去重并排序 (按文件夹+文件名排序)
valid_files = sorted(list(set(valid_files)), key=lambda x: (x.parent.name, x.name.lower()))

# 排除本脚本自身
current_script = Path(__file__).resolve()
if current_script in valid_files:
    valid_files.remove(current_script)

if not valid_files:
    raise RuntimeError(f"❌ 在 {PROJECT_ROOT} 下未找到任何 py 或 html 文件")

# =========================
# 4. 生成文件名与写入
# =========================
now = datetime.now().strftime("%Y%m%d_%H%M%S")
file_count = len(valid_files)

output_name = f"intelligence_class_CODE_{file_count}files_{now}.txt"
output_path = OUTPUT_DIR / output_name

with output_path.open("w", encoding="utf-8", errors="ignore") as f:
    f.write("#" * 100 + "\n")
    f.write(f"# 项目根目录: {PROJECT_ROOT}\n")
    f.write(f"# 包含文件类型: {TARGET_EXTENSIONS}\n")
    f.write(f"# 已排除目录: {IGNORE_DIRS}\n")
    f.write(f"# 文件总数 : {file_count}\n")
    f.write("#" * 100 + "\n\n")

    for idx, filepath in enumerate(valid_files, 1):
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            # 计算相对路径 (例如: web_ui/index.html)
            rel_path = filepath.relative_to(PROJECT_ROOT)
        except Exception as e:
            content = f"❌ 读取错误: {e}"
            rel_path = filepath.name

        # 写入分隔符和文件名
        f.write(f"\n{'=' * 80}\n")
        f.write(f"File [{idx}/{file_count}]: {rel_path}\n")
        f.write(f"{'=' * 80}\n")

        # 如果是 HTML，给个提示方便 AI 识别
        if filepath.suffix == '.html':
            f.write("\n")
            f.write(content)
            f.write("\n\n")
        else:
            f.write(content)

        f.write("\n")

print(f"✅ 汇总完成！")
print(f"   包含 .py 和 .html")
print(f"   共 {file_count} 个文件")
print(f"📄 保存位置: {output_path}")
