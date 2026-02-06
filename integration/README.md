# Integration Layer (Isolated)

This folder is an **isolation layer** that keeps orchestration logic separate from:

- `scripts/` (direct scripts)
- `scripts/intelligence_class/`
- `server/`
- `web_viz/`

The goal is to avoid cross-directory coupling by placing any **orchestration/entry** logic
here instead of modifying existing modules.

## Entry Points

- `run_server.py`: start the FastAPI server without importing server modules directly.
- `run_pipeline.py`: run the `scripts/09_run_pipeline.py` pipeline via subprocess.
