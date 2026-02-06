import argparse
import subprocess
import sys
from pathlib import Path


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated server launcher")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    project_root = resolve_project_root()
    server_app = project_root / "server" / "app.py"
    if not server_app.exists():
        raise FileNotFoundError(f"server app not found: {server_app}")

    cmd = [
        args.python,
        "-m",
        "uvicorn",
        "server.app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    subprocess.run(cmd, check=False, cwd=str(project_root))


if __name__ == "__main__":
    main()
