# ==================================================
# 文件路径: paths.py
# ==================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# [新增] 智慧课堂数据集根目录
# 请根据你的实际 D盘/E盘 路径修改这里
# 例如: DATASET_ROOT = Path("E:/智慧课堂学生行为数据集")
DATASET_ROOT = DATA_DIR / "智慧课堂学生行为数据集"