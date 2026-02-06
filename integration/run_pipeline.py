import argparse
import subprocess
import sys
from pathlib import Path


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated pipeline launcher")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    project_root = resolve_project_root()
    pipeline = project_root / "scripts" / "09_run_pipeline.py"
    if not pipeline.exists():
        raise FileNotFoundError(f"pipeline script not found: {pipeline}")

    cmd = [
        args.python,
        str(pipeline),
        "--video",
        args.video,
    ]
    if args.name:
        cmd += ["--name", args.name]
    if args.out_dir:
        cmd += ["--out_dir", args.out_dir]
    cmd += list(args.extra)

    subprocess.run(cmd, check=False, cwd=str(project_root))


if __name__ == "__main__":
    main()
