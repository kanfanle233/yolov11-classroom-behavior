import argparse
import subprocess
import sys
import json
from pathlib import Path


# =========================================================================
# 路径与环境解析
# =========================================================================

def resolve_paths():
    """解析项目根目录和脚本目录"""
    # 当前文件: .../YOLOv11/scripts/intelligence class/01_run_single_video.py
    current_file = Path(__file__).resolve()

    # 向两级找到 scripts 目录 (parents[0]=intelligence class, parents[1]=scripts)
    # 再向上一级找到 项目根目录 (parents[2]=YOLOv11)
    project_root = current_file.parents[2]

    # 核心脚本目录 (通常是 YOLOv11/scripts)
    scripts_dir = project_root / "scripts"

    return project_root, scripts_dir


def ensure_exists(file_path: Path, desc: str) -> bool:
    """检查文件是否存在，不存在则抛出清晰错误"""
    if not file_path.exists():
        print(f"\n[FATAL ERROR] 找不到{desc}！")
        print(f"  期望路径: {file_path}")
        return False
    return True


# =========================================================================
# 核心执行器 (修改重点：移除 capture_output，让错误直接喷出来)
# =========================================================================

def run_step(cmd, step_name, dry_run=False):
    """执行单个步骤的子进程封装"""
    print(f"\n" + "=" * 60)
    print(f"[RUN] {step_name}")
    print(f"      CMD: {' '.join(cmd)}")
    print("=" * 60)

    if dry_run:
        return True, "Dry Run"

    try:
        # 🔥 修改点：去掉 capture_output=True
        # 让子进程的输出直接显示在主控台，这样 tqdm 进度条、argparse 报错都能看到
        subprocess.run(
            cmd,
            check=True,  # 如果返回码非0，抛出 CalledProcessError
            # stdout=None, # 默认继承父进程
            # stderr=None, # 默认继承父进程
        )
        return True, "Success"

    except subprocess.CalledProcessError as e:
        # 因为没有 capture，错误信息已经在屏幕上了，这里只需记录状态
        error_msg = f"Step failed with exit code {e.returncode}."
        print(f"\n❌ [ERROR] {step_name} 失败！(Exit: {e.returncode})")
        print(f"   请向上翻看具体的报错日志 ^^^")
        return False, error_msg

    except Exception as e:
        print(f"\n❌ [EXCEPTION] {str(e)}")
        return False, str(e)


def run_single_video(video_path: str, video_id: str, out_dir: str, fps: float = 25.0, dry_run: bool = False):
    """
    顺序运行 pipeline 处理单个视频
    """
    project_root, scripts_dir = resolve_paths()

    # 路径标准化
    video_p = Path(video_path).resolve()
    out_p = Path(out_dir).resolve()

    if not ensure_exists(video_p, "输入视频"):
        return {"status": "failed", "error": "Missing input video"}

    if not dry_run:
        out_p.mkdir(parents=True, exist_ok=True)

    # 核心文件路径
    path_pose_jsonl = out_p / "pose_keypoints_v2.jsonl"
    path_track_jsonl = out_p / "pose_tracks_smooth.jsonl"
    path_actions_jsonl = out_p / "actions.jsonl"
    path_transcript_jsonl = out_p / "transcript.jsonl"

    # 模型路径
    model_pose = project_root / "yolo11s-pose.pt"

    # 脚本路径定义 (在此处统一定义，方便检查)
    script_02 = scripts_dir / "02_export_keypoints_jsonl.py"
    script_03 = scripts_dir / "03_track_and_smooth.py"
    script_04 = scripts_dir / "04_action_rules.py"
    script_06 = scripts_dir / "06_api_asr_realtime.py"  # 这个脚本经常变动，需小心

    result = {
        "video_id": video_id,
        "status": "pending",
        "out_dir": str(out_p),
        "actions": None,
        "transcript": None,
        "error": None
    }

    python_exe = sys.executable

    required_files = [
        (script_02, "脚本 02_export_keypoints_jsonl.py"),
        (script_03, "脚本 03_track_and_smooth.py"),
        (script_04, "脚本 04_action_rules.py"),
        (model_pose, "姿态模型 yolo11s-pose.pt"),
    ]
    for file_path, desc in required_files:
        if not ensure_exists(file_path, desc):
            return {"status": "failed", "error": f"Missing {desc}"}

    steps = [
        {
            "name": "Step 1: Pose Estimation",
            "cmd": [
                python_exe, str(script_02),
                "--video", str(video_p),
                "--out", str(path_pose_jsonl),
                "--model", str(model_pose),
            ],
        },
        {
            "name": "Step 2: Tracking & Smoothing",
            "cmd": [
                python_exe, str(script_03),
                "--video", str(video_p),
                "--in", str(path_pose_jsonl),
                "--out", str(path_track_jsonl),
            ],
        },
        {
            "name": "Step 3: Action Rules",
            "cmd": [
                python_exe, str(script_04),
                "--in", str(path_track_jsonl),
                "--out", str(path_actions_jsonl),
                "--fps", str(fps),
            ],
        },
    ]

    for step in steps:
        success, msg = run_step(step["cmd"], step["name"], dry_run)
        if not success:
            result["status"] = "failed"
            result["error"] = msg
            return result

    result["actions"] = str(path_actions_jsonl)

    # ----------------------------------------------------
    # Step 4: ASR (06) - 软执行
    # ----------------------------------------------------
    if script_06.exists():
        cmd_asr = [
            python_exe, str(script_06),
            "--video", str(video_p),
            "--out_dir", str(out_p),
        ]
        success, msg = run_step(cmd_asr, "Step 4: ASR (Optional)", dry_run)
        if success:
            result["transcript"] = str(path_transcript_jsonl)
        else:
            print(f"[WARN] ASR 运行失败，但这不影响主流程。Error: {msg}")
    else:
        print(f"\n[SKIP] Step 4: ASR 脚本未找到 ({script_06.name})，跳过语音识别。")

    # ----------------------------------------------------
    # Finalize
    # ----------------------------------------------------
    result["status"] = "success"
    return result


def main():
    parser = argparse.ArgumentParser(description="单视频 Pipeline 执行器 (Debug Mode)")
    parser.add_argument("--video", required=True, help="视频绝对路径")
    parser.add_argument("--video_id", required=True, help="视频 ID")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    try:
        res = run_single_video(args.video, args.video_id, args.out_dir, args.fps, args.dry_run)

        # 结果 JSON 打印到 stdout，方便上层捕获（虽然现在 stdout 混杂了日志，但在 debug 模式下这是可以接受的）
        print("\n" + "=" * 50)
        print(" PIPELINE FINAL RESULT JSON")
        print("=" * 50)
        print(json.dumps(res, indent=2, ensure_ascii=False))

        if res["status"] != "success":
            sys.exit(1)

    except Exception as e:
        print(f"[FATAL EXCEPTION] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
