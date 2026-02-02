import os
import sys
import subprocess
from pathlib import Path

# ================= 配置区域 =================
# 1. 定义要处理的数字范围 (包含 1 和 6)
TARGET_RANGE = range(1, 7)  # 处理 01, 02, 03, 04, 05, 06

# 2. 定义所有视角及其 ID 前缀
VIEWS = {
    "后方视角": "rear",
    "教师视角": "teacher",
    "正方视角": "front",
    "斜上方视角1": "top1",
    "斜上方视角2": "top2",
    "上方视角": "top"
}


# ============================================

def find_project_root(current_path: Path) -> Path:
    """
    智能查找项目根目录（YOLOv11），兼容脚本放在根目录或子目录的情况。
    依据：是否存在 data 目录 或 scripts 目录
    """
    candidate = current_path.parent
    for _ in range(5):  # 最多向上查找 5 层
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate
        candidate = candidate.parent
    # 兜底：假设在 scripts/intelligence_class/pipeline/ 下
    return current_path.parents[3]


# 路径解析
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = find_project_root(CURRENT_SCRIPT_PATH)

# 核心目录与脚本
DATA_ROOT = PROJECT_ROOT / "data" / "智慧课堂学生行为数据集"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "智慧课堂学生行为数据集" / "_demo_web"
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "intelligence_class" / "pipeline" / "01_run_single_video.py"
PYTHON_EXE = sys.executable  # 使用当前环境的 Python 解释器


def main():
    print(f"📍 当前脚本路径: {CURRENT_SCRIPT_PATH}")
    print(f"🏠 项目根目录: {PROJECT_ROOT}")

    if not PIPELINE_SCRIPT.exists():
        print(f"❌ 找不到核心脚本: {PIPELINE_SCRIPT}")
        print("   请检查脚本位置或 PROJECT_ROOT 解析逻辑。")
        return

    print(f"🎯 目标范围: Case {min(TARGET_RANGE):03d} - {max(TARGET_RANGE):03d}")
    print(f"📂 数据目录: {DATA_ROOT}")
    print(f"🐍 Python解释器: {PYTHON_EXE}")

    # 遍历每个视角
    for view_name, prefix in VIEWS.items():
        view_dir = DATA_ROOT / view_name
        if not view_dir.exists():
            continue

        print(f"\n🚀 [视角] {view_name} ({prefix})")

        # 扫描该视角下的所有 mp4 文件
        videos = sorted(list(view_dir.glob("*.mp4")))

        if not videos:
            print(f"   ⚠️  该目录下无 MP4 文件")
            continue

        count = 0
        for video_path in videos:
            # 1. 智能解析 ID (0001 -> 1, 01 -> 1)
            try:
                raw_num = int(video_path.stem)
            except ValueError:
                continue

            # 2. 过滤：只处理 1-6
            if raw_num not in TARGET_RANGE:
                continue

            count += 1

            # 3. 构造标准参数
            video_id = f"{prefix}__{video_path.stem}"
            case_id = f"{raw_num:03d}"
            out_dir = OUTPUT_ROOT / view_name / video_id

            print(f"   ▶️  正在处理: {video_path.name} -> Case {case_id}")

            # 4. 构造并执行命令
            cmd = [
                PYTHON_EXE, str(PIPELINE_SCRIPT),
                "--video", str(video_path),
                "--video_id", video_id,
                "--out_dir", str(out_dir),
                "--case_id", case_id,
                "--view", view_name,

                "--skip_existing", "0",
                "--case_det", "1",
                "--run_pose", "1",
                "--run_track", "1",
                "--run_actions", "1",
                "--run_asr", "1",
                "--run_align", "1",
                "--export_behavior", "1",
                "--make_overlays", "1",
                "--run_summarize", "1",
                "--run_aggregate", "1",
                "--run_projection", "1"
            ]

            try:
                # 显式指定 cwd 为项目根目录，确保路径引用不乱
                subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            except subprocess.CalledProcessError:
                print(f"   ❌ 处理失败: {video_path.name}")
            except KeyboardInterrupt:
                print("\n🛑 用户终止")
                return

        if count == 0:
            print(f"   ℹ️  该视角下没有找到 ID 为 {list(TARGET_RANGE)} 的视频")

    print("\n✅ 所有指定范围的视频处理完毕！")


if __name__ == "__main__":
    main()