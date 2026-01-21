import os
import sys
import json
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 尝试导入 Levenshtein，如果没有就用简化版
try:
    import Levenshtein  # type: ignore
except Exception:
    Levenshtein = None


# =========================================================
# 1) 路径：优先使用 paths.py（与你 scripts/ 中一致）
# =========================================================
def _try_import_paths():
    try:
        import paths  # type: ignore
        return paths
    except Exception:
        # 允许从 server 目录导入失败时，手动加路径
        server_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(server_dir)
        try:
            import paths  # type: ignore
            return paths
        except Exception:
            return None


paths_mod = _try_import_paths()

# 没有 paths.py 就按项目结构推断
BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(getattr(paths_mod, "PROJECT_ROOT", BASE_DIR)).resolve()
OUTPUT_DIR = Path(getattr(paths_mod, "OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
DATA_DIR = Path(getattr(paths_mod, "DATA_DIR", PROJECT_ROOT / "data")).resolve()
VIDEO_DIR = (DATA_DIR / "videos").resolve()

# 你的前端模板一般在 web_viz/templates；如果没有就回退到项目根目录
TEMPLATE_DIR = (PROJECT_ROOT / "web_viz" / "templates")
if not TEMPLATE_DIR.exists():
    TEMPLATE_DIR = PROJECT_ROOT

# 静态资源目录：web_viz/static 或项目根
STATIC_DIR = (PROJECT_ROOT / "web_viz" / "static")
if not STATIC_DIR.exists():
    STATIC_DIR = PROJECT_ROOT

app = FastAPI(title="Classroom Multi-Modal Viz Server")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2) 静态文件挂载（兼容旧路径：/output /data）
# =========================================================
# 兼容你 .txt Flask 版：/output/<path:filename> -> output目录 :contentReference[oaicite:2]{index=2}
if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
    # 也给一个别名：/outputs
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# 兼容你 .txt Flask 版：/data/<path:filename> -> data目录 :contentReference[oaicite:3]{index=3}
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# 视频别名（更直观）
if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")

# 静态资源
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# =========================================================
# 3) 小工具：简易 Levenshtein（无三方库时兜底）
# =========================================================
def simple_levenshtein(seq1: str, seq2: str) -> float:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.float32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return float(matrix[size_x - 1, size_y - 1])


# =========================================================
# 4) 缓存读取：timeline/stats/transcript/tracks
# =========================================================
@lru_cache(maxsize=32)
def load_timeline_data_cached(video_id: str) -> Optional[Dict[str, Any]]:
    # 先找 timeline_chart.json（你现在 step10 生成的是 *.json，结构 {"items": [...]}） :contentReference[oaicite:4]{index=4}
    p1 = OUTPUT_DIR / video_id / "timeline_chart.json"
    if p1.exists():
        return json.loads(p1.read_text(encoding="utf-8"))

    # 回退 timeline_data.json
    p2 = OUTPUT_DIR / video_id / "timeline_data.json"
    if p2.exists():
        return json.loads(p2.read_text(encoding="utf-8"))

    return None


@lru_cache(maxsize=32)
def load_stats_data_cached(video_id: str) -> Dict[str, Any]:
    p = OUTPUT_DIR / video_id / "timeline_chart_stats.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"class_pie_chart": []}


@lru_cache(maxsize=32)
def load_transcript_data_cached(video_id: str) -> List[Dict[str, Any]]:
    p = OUTPUT_DIR / video_id / "transcript.jsonl"
    data: List[Dict[str, Any]] = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    return data


@lru_cache(maxsize=16)
def load_tracks_data_cached(video_id: str) -> Dict[str, Any]:
    """
    返回结构:
    {
      "12": [{"id": 3, "box": [x1,y1,x2,y2]}, ...],
      "13": ...
    }
    """
    p = OUTPUT_DIR / video_id / "pose_tracks_smooth.jsonl"
    if not p.exists():
        return {}

    tracks: Dict[str, Any] = {}
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                frame = d.get("frame")
                persons = d.get("persons", [])
                tracks[str(frame)] = [{"id": pp.get("track_id"), "box": pp.get("bbox")} for pp in persons]
    except Exception:
        return {}

    return tracks


# =========================================================
# 5) 核心：Projection（支持 euclidean / levenshtein + pca/mds/tsne）
# =========================================================
@lru_cache(maxsize=64)
def compute_projection_cached(video_id: str, method: str, metric: str) -> Dict[str, Any]:
    raw = load_timeline_data_cached(video_id)
    if not raw:
        return {"method": method, "metric": metric, "points": []}

    items = raw.get("items", raw if isinstance(raw, list) else [])
    if not isinstance(items, list):
        items = []

    # --- 构建每个人的数据 ---
    person_data: Dict[int, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        # 只处理 person 行；group 行会用 track_id = -1（你生成 json 时就是这么干的） :contentReference[oaicite:5]{index=5}
        if item.get("type", "person") != "person":
            continue

        tid = int(item.get("track_id", -999))
        if tid < 0:
            continue

        if tid not in person_data:
            person_data[tid] = {"actions": [], "spatial": []}

        st = float(item.get("start", 0.0))
        ed = float(item.get("end", st))
        code = int(item.get("action_id", -1))
        row = int(item.get("row", 0))

        person_data[tid]["actions"].append({"code": code, "dur": max(0.0, ed - st), "start": st})
        person_data[tid]["spatial"].append(row)

    track_ids = sorted(person_data.keys())
    n_samples = len(track_ids)
    if n_samples < 2:
        return {"method": method, "metric": metric, "points": []}

    coords = None

    # -------- A) Vector-based: metric = euclidean --------
    if metric == "euclidean":
        # 9 actions + 1 spatial + 1 activity
        features = np.zeros((n_samples, 11), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            acts = sorted(person_data[tid]["actions"], key=lambda x: x["start"])
            total_dur = float(sum(a["dur"] for a in acts))

            counts = np.zeros(9, dtype=np.float32)
            for a in acts:
                c = int(a["code"])
                if 0 <= c < 9:
                    counts[c] += float(a["dur"])

            if total_dur > 1e-6:
                features[i, :9] = counts / total_dur

            ys = person_data[tid]["spatial"]
            features[i, 9] = float(sum(ys) / max(1, len(ys)))
            features[i, 10] = float(len(acts) / total_dur) if total_dur > 1e-6 else 0.0

        X = StandardScaler().fit_transform(features)

        if method == "pca":
            coords = PCA(n_components=2).fit_transform(X)
        elif method == "mds":
            coords = MDS(n_components=2, dissimilarity="euclidean", random_state=42, n_jobs=-1).fit_transform(X)
        elif method == "tsne":
            perp = min(30, max(5, n_samples - 1))
            coords = TSNE(
                n_components=2,
                perplexity=perp,
                metric="euclidean",
                random_state=42,
                n_jobs=-1,
            ).fit_transform(X)
        else:
            # 不认识就回退
            coords = PCA(n_components=2).fit_transform(X)

    # -------- B) Sequence-based: metric = levenshtein --------
    elif metric == "levenshtein":
        sequences: List[str] = []
        for tid in track_ids:
            acts = sorted(person_data[tid]["actions"], key=lambda x: x["start"])
            # 简单拼接 action_id 字符串
            seq_str = "".join([str(int(a["code"])) for a in acts])
            sequences.append(seq_str)

        dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if Levenshtein is not None:
                    d = float(Levenshtein.distance(sequences[i], sequences[j]))
                else:
                    d = simple_levenshtein(sequences[i], sequences[j])

                max_len = max(len(sequences[i]), len(sequences[j]), 1)
                dist_matrix[i, j] = dist_matrix[j, i] = d / float(max_len)

        # PCA 不能处理距离矩阵，强制回落到 MDS
        if method == "pca":
            method = "mds"

        if method == "mds":
            coords = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                n_jobs=-1
            ).fit_transform(dist_matrix)
        elif method == "tsne":
            perp = min(30, max(5, n_samples - 1))
            coords = TSNE(
                n_components=2,
                perplexity=perp,
                metric="precomputed",
                random_state=42,
                n_jobs=-1
            ).fit_transform(dist_matrix)
        else:
            coords = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                n_jobs=-1
            ).fit_transform(dist_matrix)

    else:
        # 未知 metric
        return {"method": method, "metric": metric, "points": []}

    # 归一化到 0-1
    coords = MinMaxScaler().fit_transform(coords)

    points = []
    for i, tid in enumerate(track_ids):
        points.append({"track_id": int(tid), "x": float(coords[i, 0]), "y": float(coords[i, 1])})

    return {"method": method, "metric": metric, "points": points}


# =========================================================
# 6) 路由：HTML + API
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    主页：默认渲染 templates/index.html
    如果你暂时没做模板，也会回退到 PROJECT_ROOT/index.html
    """
    # 优先 templates
    index_tpl = Path(TEMPLATE_DIR) / "index.html"
    if index_tpl.exists():
        return templates.TemplateResponse("index.html", {"request": request})

    # 回退到根目录 index.html
    index_file = PROJECT_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    raise HTTPException(404, "index.html not found in templates/ or project root")


@app.get("/api/list_cases")
def list_cases():
    """
    自动扫描 output 目录，告诉前端有哪些 demo 可用（兼容你 txt 的 Flask 接口） :contentReference[oaicite:6]{index=6}
    """
    if not OUTPUT_DIR.exists():
        return []
    cases = []
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir():
            cases.append(d.name)
    return sorted(cases)


@app.get("/api/config")
def get_config():
    """
    暴露部分配置（兼容你 txt 的 Flask 接口） :contentReference[oaicite:7]{index=7}
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "data_dir": str(DATA_DIR),
        "video_dir": str(VIDEO_DIR),
        "template_dir": str(TEMPLATE_DIR),
    }


@app.get("/api/timeline/{video_id}")
def get_timeline(video_id: str):
    data = load_timeline_data_cached(video_id)
    if not data:
        raise HTTPException(404, "Timeline data not found")
    # 统一输出 {"items": [...]}
    if isinstance(data, list):
        return {"items": data}
    return {"items": data.get("items", [])}


@app.get("/api/stats/{video_id}")
def get_stats(video_id: str):
    return load_stats_data_cached(video_id)


@app.get("/api/transcript/{video_id}")
def get_transcript(video_id: str):
    raw = load_transcript_data_cached(video_id)
    # 输出尽量前端友好
    out = []
    for l in raw:
        out.append({
            "start": l.get("start"),
            "end": l.get("end"),
            "text": l.get("text", l.get("sentence", "")),
            "verified": bool(l.get("verified", False)),
        })
    return out


@app.get("/api/tracks/{video_id}")
def get_tracks(video_id: str):
    return load_tracks_data_cached(video_id)


@app.get("/api/projection/{video_id}")
def get_projection(video_id: str, method: str = "pca", metric: str = "euclidean"):
    method = method.lower().strip()
    metric = metric.lower().strip()
    if method not in {"pca", "mds", "tsne"}:
        raise HTTPException(400, "method must be one of: pca, mds, tsne")
    if metric not in {"euclidean", "levenshtein"}:
        raise HTTPException(400, "metric must be one of: euclidean, levenshtein")
    return compute_projection_cached(video_id, method, metric)


# =========================================================
# 7) 启动
# =========================================================
if __name__ == "__main__":
    import uvicorn

    print("Server starting...")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print("Open http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000)
