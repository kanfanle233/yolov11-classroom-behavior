import sys
import subprocess
from pathlib import Path

# ================= 配置区 (在这里修改测试参数) =================

# 1. 想要测试的视频 (根据你截图里的文件)
# 注意：这里我们使用相对路径，脚本会自动算出绝对路径
TEST_VIDEO_REL_PATH = "YOLOv11/data/智慧课堂学生行为数据集/后方视角/0015.mp4"

# 2. 给这次测试起个 ID
TEST_VIDEO_ID = "debug_rear_0015"

# 3. 是否只是打印命令而不运行？ (True=只检查路径, False=真跑)
IS_DRY_RUN = False


# ============================================================

def main():
    # 1. 解析路径
    current_file = Path(__file__).resolve()
    # 脚本所在目录 (scripts/intelligence class/)
    script_dir = current_file.parent
    # 项目根目录 (YOLOv11/)
    project_root = script_dir.parents[2]

    # 目标执行的脚本
    target_script = script_dir / "01_run_single_video.py"

    # 视频完整路径
    video_path = project_root / TEST_VIDEO_REL_PATH

    # 输出目录
    out_dir = project_root / "output" / TEST_VIDEO_ID

    # Python解释器 (使用当前环境的python)
    python_exe = sys.executable

    # 2. 检查文件是否存在
    if not video_path.exists():
        print(f"❌ 错误：找不到测试视频文件！")
        print(f"   路径: {video_path}")
        return

    print("=" * 60)
    print(" 🚀 调试启动器 (Debug Runner)")
    print("=" * 60)
    print(f"执行脚本: {target_script.name}")
    print(f"测试视频: {video_path.name}")
    print(f"输出目录: {out_dir}")
    print("-" * 60)

    # 3. 组装命令
    cmd = [
        python_exe, str(target_script),
        "--video", str(video_path),
        "--video_id", TEST_VIDEO_ID,
        "--out_dir", str(out_dir),
        "--fps", "25"
    ]

    if IS_DRY_RUN:
        cmd.append("--dry_run")

    # 4. 调用子进程运行
    try:
        # check=True 意味着如果脚本报错，这里也会抛出异常
        subprocess.run(cmd, check=True)
        print("\n✅ 测试运行完成！")
        print(f"请检查输出目录: {out_dir}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行失败，退出码: {e.returncode}")
    except KeyboardInterrupt:
        print("\n⚠️ 用户手动中断")


if __name__ == "__main__":
    main()