import argparse
import subprocess
import sys
from pathlib import Path

def resolve_paths() -> tuple[Path, Path]:
    current_file = Path(__file__).resolve()
    script_dir = current_file.parent
    project_root = script_dir.parents[2]
    return script_dir, project_root


def build_parser(default_video_rel: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单视频管线调试启动器")
    parser.add_argument(
        "--video",
        default=default_video_rel,
        help="测试视频路径（可为相对项目根目录的相对路径）",
    )
    parser.add_argument("--video_id", default="debug_rear_0015", help="测试视频 ID")
    parser.add_argument("--fps", default="25", help="帧率 (传给单视频脚本)")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不实际执行")
    return parser


def main():
    script_dir, project_root = resolve_paths()

    default_video_rel = "data/智慧课堂学生行为数据集/后方视角/0015.mp4"
    parser = build_parser(default_video_rel)
    args = parser.parse_args()

    target_script = script_dir / "01_run_single_video.py"
    if not target_script.exists():
        print(f"❌ 错误：找不到目标脚本！")
        print(f"   路径: {target_script}")
        return

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (project_root / video_path).resolve()

    out_dir = project_root / "output" / args.video_id
    python_exe = sys.executable

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

    cmd = [
        python_exe,
        str(target_script),
        "--video",
        str(video_path),
        "--video_id",
        args.video_id,
        "--out_dir",
        str(out_dir),
        "--fps",
        str(args.fps),
    ]

    if args.dry_run:
        cmd.append("--dry_run")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 测试运行完成！")
        print(f"请检查输出目录: {out_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行失败，退出码: {e.returncode}")
    except KeyboardInterrupt:
        print("\n⚠️ 用户手动中断")


if __name__ == "__main__":
    main()
