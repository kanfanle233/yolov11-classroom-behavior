import os
import sys
import json
import shutil
import subprocess
import warnings
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from functools import lru_cache
from urllib.parse import quote
from collections import defaultdict
from enum import Enum

import numpy as np

# ---------------- MODIFIED IMPORTS ----------------
# Added Query
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# --------------------------------------------------

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

try:
    from scripts.intelligence_class._utils.action_map import ACTION_MAP as COMMON_ACTION_MAP
    from scripts.intelligence_class._utils.action_map import LABEL_NORMALIZE as COMMON_LABEL_NORMALIZE
except ModuleNotFoundError:
    PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))
    from scripts.intelligence_class._utils.action_map import ACTION_MAP as COMMON_ACTION_MAP
    from scripts.intelligence_class._utils.action_map import LABEL_NORMALIZE as COMMON_LABEL_NORMALIZE


# ================================
# 0) 全局配置
# ================================
CURRENT_FILE = Path(__file__).resolve()

# 尝试定位项目根目录：
# - 若 server/app.py 结构：PROJECT_ROOT = server 的上一层
# - 若 scripts/.../tools/app.py 结构：按 parents[3] 回退
PROJECT_ROOT = CURRENT_FILE.parents[1]
if not (PROJECT_ROOT / "data").exists() and len(CURRENT_FILE.parents) >= 4:
    cand = CURRENT_FILE.parents[3]
    if (cand / "data").exists() or (cand / "output").exists() or (cand / "scripts").exists():
        PROJECT_ROOT = cand

# 你的数据集路径（按你现在结构）
DATA_ROOT = PROJECT_ROOT / "data" / "智慧课堂学生行为数据集"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "智慧课堂学生行为数据集" / "_demo_web"

# 触发分析用的 pipeline（可选）
PIPELINE_SCRIPT = PROJECT_ROOT / "scripts" / "intelligence_class" / "pipeline" / "01_run_single_video.py"
PYTHON_EXE = sys.executable

app = FastAPI(title="Smart Classroom Analytics Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 自动部署 index.html（可选）
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
src_html = CURRENT_FILE.parent / "index.html"
dst_html = OUTPUT_ROOT / "index.html"
if src_html.exists():
    try:
        if (not dst_html.exists()) or src_html.stat().st_mtime > dst_html.stat().st_mtime:
            shutil.copy2(src_html, dst_html)
            print(f"[Auto-Deploy] Updated index.html -> {OUTPUT_ROOT}")
    except Exception as e:
        print(f"[Warn] Failed to deploy index.html: {e}")

# 静态挂载
if OUTPUT_ROOT.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="output")
if DATA_ROOT.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_ROOT)), name="data")


# ================================
# 1) 通用工具
# ================================

def _extract_num(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.findall(r"(\d+)", s)
    if not m:
        return None
    return int(m[-1])


def _url_segments(prefix: str, parts: List[str]) -> str:
    return prefix + "/" + "/".join(quote(p) for p in parts if p)


def _safe_case_id(case_id: str) -> str:
    # 支持传入 "view/case" 这种形式
    if "/" in case_id:
        case_id = case_id.split("/", 1)[-1]
    if "\\" in case_id:
        case_id = case_id.split("\\", 1)[-1]
    return case_id.strip()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not path.exists():
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return data


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_case_dir(case_id: str, view: Optional[str] = None) -> Optional[Path]:
    """在 OUTPUT_ROOT 下按 view/case_id 查找目录；支持数字模糊匹配。"""
    if not OUTPUT_ROOT.exists():
        return None

    case_id = _safe_case_id(case_id)
    want_num = _extract_num(case_id)

    view_dirs = []
    if view:
        cand = OUTPUT_ROOT / view
        if cand.exists() and cand.is_dir():
            view_dirs = [cand]
        else:
            view_dirs = [p for p in OUTPUT_ROOT.iterdir() if p.is_dir()]
    else:
        view_dirs = [p for p in OUTPUT_ROOT.iterdir() if p.is_dir()]

    for view_dir in view_dirs:
        if not view_dir.is_dir() or view_dir.name.startswith((".", "_")):
            continue

        # 1) 精确匹配
        direct = view_dir / case_id
        if direct.exists() and direct.is_dir():
            return direct

        # 2) 数字匹配
        if want_num is not None:
            for sub in view_dir.iterdir():
                if not sub.is_dir():
                    continue
                sub_num = _extract_num(sub.name)
                if sub_num == want_num:
                    return sub

    return None


def _find_base_id(case_dir: Path) -> str:
    """找到该 case 的基础编号。"""
    metas = list(case_dir.glob("*.meta.json"))
    for p in metas:
        m = re.match(r"^(\d+)\.meta\.json$", p.name)
        if m:
            return m.group(1).zfill(3)

    for p in case_dir.glob("*.jsonl"):
        m = re.match(r"^(\d+)\.jsonl$", p.name)
        if m:
            return m.group(1).zfill(3)

    n = _extract_num(case_dir.name)
    if n is not None:
        return f"{n:03d}"
    return "001"


def _ffprobe_codec(path: Path) -> str:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=5)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _ffprobe_duration_fps(path: Path) -> dict:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,avg_frame_rate,duration", "-of", "json", str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=5)
        data = json.loads(r.stdout)
        stream = data["streams"][0]
        frames = int(stream.get("nb_frames", 0) or 0)
        fps_str = stream.get("avg_frame_rate", "0/0")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 0.0
        else:
            fps = float(fps_str)
        return {"frames": frames, "fps": fps}
    except Exception:
        return {"frames": 0, "fps": 0.0}


def _pick_overlay_files(mp4s: List[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    pose_file = None
    behavior_file = None
    for p in mp4s:
        name = p.name.lower()
        if name.endswith("_behavior_overlay.mp4"):
            behavior_file = p
            continue
        if name.endswith("_overlay.mp4") and "behavior_overlay" not in name:
            pose_file = p
    return pose_file, behavior_file


# ================================
# 2) Timeline 读取/兜底生成
# ================================

ACTION_MAP = COMMON_ACTION_MAP
LABEL_NORMALIZE = COMMON_LABEL_NORMALIZE


def _normalize_action(label: str) -> str:
    if not label:
        return "sit"
    label = str(label).strip()
    if label in ACTION_MAP:
        return label
    if label in LABEL_NORMALIZE:
        return LABEL_NORMALIZE[label]
    alias = {
        "认真听讲": "listen", "听讲": "listen", "做笔记": "writing", "写字": "writing",
        "看书": "reading", "玩手机": "phone", "打瞌睡": "sleep", "交头接耳": "interact",
        "站立": "stand", "举手": "hand_raise", "低头": "bow_head",
    }
    return alias.get(label, label)


def _iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-6 else 0.0


def _xyxy_from_bbox(b: Any) -> Optional[List[float]]:
    if not isinstance(b, list) or len(b) < 4:
        return None
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
    try:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return [x1, y1, x1 + max(0.0, x2), y1 + max(0.0, y2)]
    return [x1, y1, x2, y2]


def _index_tracks_by_frame(tracks_jsonl: Path) -> Dict[int, List[Dict[str, Any]]]:
    idx: Dict[int, List[Dict[str, Any]]] = {}
    for row in load_jsonl(tracks_jsonl):
        f = row.get("frame") or row.get("frame_idx")
        if f is None:
            continue
        persons = row.get("persons", []) or []
        packed = []
        for p in persons:
            tid = p.get("track_id")
            bb = _xyxy_from_bbox(p.get("bbox"))
            if tid is None or bb is None:
                continue
            packed.append({"track_id": int(tid), "bbox": bb})
        idx[int(f)] = packed
    return idx


def _match_track_id(det_xyxy: List[float], persons: List[Dict[str, Any]], iou_thr: float = 0.30) -> Optional[int]:
    best_tid = None
    best_iou = 0.0
    for p in persons:
        bb = p.get("bbox")
        if bb is None:
            continue
        iou = _iou_xyxy(det_xyxy, bb)
        if iou > best_iou:
            best_iou = iou
            best_tid = int(p.get("track_id"))
    if best_tid is not None and best_iou >= iou_thr:
        return best_tid
    return None


def _bucket_track_id(base: int, action: str) -> int:
    if base != 0:
        return base
    act_id = int(ACTION_MAP.get(action, 0))
    return 10000 + act_id


def compress_timeline(frames_data: List[Dict[str, Any]], fps: float = 25.0) -> List[Dict[str, Any]]:
    if not frames_data:
        return []
    frames = []
    by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in frames_data:
        tid = item.get("track_id")
        action = item.get("action") or item.get("label")
        fidx = item.get("frame_idx") or item.get("frame")
        if tid is None or action is None or fidx is None:
            continue
        try:
            tid_int = int(tid)
        except Exception:
            continue
        if tid_int < 0:
            continue
        act_norm = _normalize_action(str(action))
        fidx_int = int(fidx)
        frames.append(fidx_int)
        tid_bucket = _bucket_track_id(tid_int, act_norm)
        by_track[tid_bucket].append({"frame": fidx_int, "action": act_norm})

    compressed: List[Dict[str, Any]] = []
    if frames:
        duration_sec = (max(frames) - min(frames)) / max(fps, 1e-6)
    else:
        duration_sec = 0.0
    gap_sec = 0.12 if duration_sec <= 20.0 else 0.2
    min_event_sec = 0.12 if duration_sec <= 20.0 else 0.2
    gap_frames = max(1, int(round(gap_sec * fps)))
    min_event_frames = max(1, int(round(min_event_sec * fps)))
    for tid, trace in by_track.items():
        trace.sort(key=lambda x: x["frame"])
        if not trace:
            continue
        cur = None
        for p in trace:
            f = p["frame"]
            act = p["action"]
            if cur is None:
                cur = {"track_id": tid, "action": act, "start_frame": f, "end_frame": f}
                continue
            is_same = (act == cur["action"])
            is_cont = (f - cur["end_frame"] <= gap_frames)
            if is_same and is_cont:
                cur["end_frame"] = f
            else:
                if (cur["end_frame"] - cur["start_frame"]) >= min_event_frames:
                    compressed.append(cur)
                cur = {"track_id": tid, "action": act, "start_frame": f, "end_frame": f}
        if cur and (cur["end_frame"] - cur["start_frame"]) >= min_event_frames:
            compressed.append(cur)

    out: List[Dict[str, Any]] = []
    for evt in compressed:
        act = evt["action"]
        out.append(
            {
                "track_id": int(evt["track_id"]),
                "action": act,
                "action_id": int(ACTION_MAP.get(act, 0)),
                "start": round(evt["start_frame"] / max(fps, 1e-6), 2),
                "end": round(evt["end_frame"] / max(fps, 1e-6), 2),
                "frame_idx": int(evt["start_frame"]),
                "duration": round((evt["end_frame"] - evt["start_frame"]) / max(fps, 1e-6), 2),
            }
        )
    out.sort(key=lambda x: x["start"])
    return out


def _find_best_fps(case_dir: Path) -> float:
    for p in case_dir.glob("*_summary.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            video = raw.get("video", {}) if isinstance(raw, dict) else {}
            meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
            fps = video.get("fps", meta.get("fps", 0)) or 0
            if fps:
                return float(fps)
        except Exception:
            pass
    base_id = _find_base_id(case_dir)
    for meta_name in [f"{base_id}.meta.json", f"{base_id}_behavior.meta.json"]:
        p = case_dir / meta_name
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                fps = raw.get("fps", 0) or 0
                if fps:
                    return float(fps)
            except Exception:
                pass
    mp4s = list(case_dir.glob("*.mp4"))
    if mp4s:
        v = _ffprobe_duration_fps(mp4s[0])
        if v.get("fps", 0) > 0:
            return float(v["fps"])
    return 25.0


def _pick_behavior_jsonl(case_dir: Path) -> Optional[Path]:
    base_id = _find_base_id(case_dir)
    cands = [case_dir / f"{base_id}_behavior.jsonl", case_dir / "case_det.jsonl"]
    for p in cands:
        if p.exists():
            return p
    gs = list(case_dir.glob("*_behavior.jsonl"))
    return gs[0] if gs else None


def _pick_tracks_jsonl(case_dir: Path) -> Optional[Path]:
    for name in ["pose_tracks_smooth.jsonl", "pose_tracks_smooth_kpts.jsonl"]:
        p = case_dir / name
        if p.exists():
            return p
    base_id = _find_base_id(case_dir)
    p = case_dir / f"{base_id}.jsonl"
    return p if p.exists() else None


def _build_timeline_viz(case_dir: Path) -> Dict[str, Any]:
    fps = _find_best_fps(case_dir)
    actions_file = case_dir / "actions.jsonl"
    if actions_file.exists():
        raw = load_jsonl(actions_file)
        items = compress_timeline(raw, fps)
        return {
            "meta": {
                "source": actions_file.name, "fps": fps,
                "total_events": len(items),
                "track_ids": sorted(list({it["track_id"] for it in items})),
            },
            "items": items,
        }
    beh_file = _pick_behavior_jsonl(case_dir)
    if not beh_file:
        return {"meta": {"source": "none", "fps": fps, "total_events": 0, "track_ids": []}, "items": []}
    tracks_file = _pick_tracks_jsonl(case_dir)
    tracks_idx = _index_tracks_by_frame(tracks_file) if tracks_file and tracks_file.exists() else {}
    raw = load_jsonl(beh_file)
    flat: List[Dict[str, Any]] = []
    for row in raw:
        fidx = row.get("frame_idx", row.get("frame"))
        if fidx is None:
            continue
        fidx = int(fidx)
        if "dets" in row:
            persons = tracks_idx.get(fidx, [])
            for d in row.get("dets", []) or []:
                lbl = d.get("label")
                if lbl is None:
                    continue
                act = _normalize_action(lbl)
                tid = d.get("track_id")
                if tid is None:
                    bb = _xyxy_from_bbox(d.get("xyxy"))
                    if bb is not None and persons:
                        tid = _match_track_id(bb, persons, iou_thr=0.30)
                if tid is None:
                    tid = 0
                flat.append({"frame_idx": fidx, "track_id": int(tid), "action": act})
        elif "persons" in row:
            for p in row.get("persons", []) or []:
                tid = p.get("track_id")
                act = p.get("action")
                if tid is None or act is None:
                    continue
                flat.append({"frame_idx": fidx, "track_id": int(tid), "action": _normalize_action(act)})
    items = compress_timeline(flat, fps)
    return {
        "meta": {
            "source": beh_file.name, "fps": fps, "total_events": len(items),
            "track_ids": sorted(list({it["track_id"] for it in items})),
            "tracks": tracks_file.name if tracks_file else None,
        },
        "items": items,
    }


# ================================
# 3) 特征/投影
# ================================

class ProjectionMethod(str, Enum):
    PCA = "pca"
    MDS = "mds"
    TSNE = "tsne"


class ProjectionMetric(str, Enum):
    EUCLIDEAN = "euclidean"
    LEVENSHTEIN = "levenshtein"


# ---------------- NEW HELPERS for Sklearn Compatibility ----------------
def _get_sklearn_components():
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, TSNE
    return StandardScaler, MinMaxScaler, PCA, MDS, TSNE


def _make_mds_precomputed(mds_cls, n_components=2, random_state=0):
    """Compatible wrapper for MDS with precomputed dissimilarity."""
    try:
        return mds_cls(
            n_components=n_components,
            dissimilarity="precomputed",
            normalized_stress="auto",
            random_state=random_state,
        )
    except TypeError:
        # Fallback for older sklearn without normalized_stress
        return mds_cls(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
        )


def _fit_tsne_2d(tsne_cls, X_std):
    """Compatible wrapper for TSNE fit_transform."""
    perplexity = max(1, min(30, len(X_std) - 1))
    try:
        tsne = tsne_cls(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=0,
        )
    except TypeError:
        # Fallback for older sklearn without learning_rate="auto"
        tsne = tsne_cls(
            n_components=2,
            init="pca",
            learning_rate=200.0,
            perplexity=perplexity,
            random_state=0,
        )
    return tsne.fit_transform(X_std)
# -----------------------------------------------------------------------


def _empty_features(all_actions: List[str]) -> Tuple[np.ndarray, List[int], Dict[int, Dict[str, Any]]]:
    dim = len(all_actions) + 3
    return np.zeros((0, dim), dtype=float), [], {}


def _build_features_from_tracks_stats(
    tracks_stats: Dict[int, Dict[str, Any]],
    all_actions: List[str],
) -> Tuple[np.ndarray, List[int], Dict[int, Dict[str, Any]]]:
    X: List[List[float]] = []
    ids: List[int] = []
    for tid, stats in tracks_stats.items():
        actions = stats.get("actions", []) or []
        # --- FIX 4.3: Allow tracks without actions to be projected ---
        if not actions:
            actions = ["unknown"]  # Placeholder to ensure point exists
        # -------------------------------------------------------------
        total = len(actions)
        act_counts = {a: 0 for a in all_actions}
        for a in actions:
            if a in act_counts:
                act_counts[a] += 1
        vec = [act_counts[a] / total for a in all_actions]
        positions = stats.get("positions", []) or []
        avg_pos = float(sum(positions) / len(positions)) if positions else 0.0
        vec.append(avg_pos)
        active_count = sum(1 for a in actions if a not in ["stand", "sit"])
        vec.append(active_count / total)
        vec.append(total / max(total, 1))
        X.append(vec)
        ids.append(int(tid))
    return np.array(X, dtype=float), ids, tracks_stats


def _tracks_stats_from_actions_jsonl(case_dir: Path) -> Dict[int, Dict[str, Any]]:
    f_actions = case_dir / "actions.jsonl"
    if not f_actions.exists():
        return {}
    data = load_jsonl(f_actions)
    if not data:
        return {}
    tracks_stats: Dict[int, Dict[str, Any]] = {}
    for row in data:
        tid = row.get("track_id", -1)
        if tid is None:
            tid = -1
        try:
            tid = int(tid)
        except Exception:
            tid = -1
        if tid <= 0:
            continue
        if tid not in tracks_stats:
            tracks_stats[tid] = {"actions": [], "positions": []}
        act = _normalize_action(row.get("action", row.get("label", "sit")))
        tracks_stats[tid]["actions"].append(act)
        bbox = row.get("bbox", row.get("xyxy", None))
        bb = _xyxy_from_bbox(bbox) if bbox is not None else None
        if bb is not None:
            cx = (bb[0] + bb[2]) / 2.0
            tracks_stats[tid]["positions"].append(float(cx))
    return tracks_stats


def _load_or_build_timeline(case_dir: Path) -> Dict[str, Any]:
    p = case_dir / "timeline_viz.json"
    if p.exists():
        data = _read_json(p)
        if isinstance(data, dict) and "items" in data:
            return data
    return _build_timeline_viz(case_dir)


def _tracks_stats_from_timeline(case_dir: Path) -> Dict[int, Dict[str, Any]]:
    tl = _load_or_build_timeline(case_dir)
    items = tl.get("items", []) if isinstance(tl, dict) else []
    if not items:
        return {}
    tracks_stats: Dict[int, Dict[str, Any]] = {}
    for it in items:
        tid = it.get("track_id")
        act = it.get("action") or it.get("label")
        if tid is None or act is None:
            continue
        try:
            tid = int(tid)
        except Exception:
            continue
        if tid <= 0:
            continue
        act = _normalize_action(str(act))
        dur = it.get("duration", None)
        try:
            dur_f = float(dur) if dur is not None else 0.0
        except Exception:
            dur_f = 0.0
        rep = int(round(dur_f))
        rep = max(1, min(rep, 20))
        if tid not in tracks_stats:
            tracks_stats[tid] = {"actions": [], "positions": []}
        tracks_stats[tid]["actions"].extend([act] * rep)
    return tracks_stats


def _is_features_good(X: np.ndarray, ids: List[int], min_points: int = 3) -> bool:
    if X is None or len(ids) < min_points or not isinstance(X, np.ndarray) or X.shape[0] < min_points:
        return False
    return True


# ---------------- FIX 4.4: Cache Invalidation Logic ----------------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p and p.exists() else 0.0


def _sig(case_dir: Path):
    """Generate signature based on file modification times/sizes to invalidate cache."""
    files = ["actions.jsonl", "timeline_viz.json", "pose_tracks_smooth.jsonl", "001_behavior.jsonl"]
    sig = []
    for fn in files:
        p = case_dir / fn
        if p.exists():
            st = p.stat()
            sig.append((fn, st.st_mtime, st.st_size))
        else:
            sig.append((fn, 0, 0))
    return tuple(sig)


def compute_action_features(case_dir: Path):
    """Wrapper to compute features with proper cache invalidation."""
    return compute_action_features_cached(str(case_dir), _sig(case_dir))


@lru_cache(maxsize=32)
def compute_action_features_cached(case_dir_str: str, sig: Tuple):
    """Actual computation, cached by signature."""
    case_dir = Path(case_dir_str)
    all_actions = [
        "hand_raise", "stand", "sit", "reading", "writing",
        "phone", "sleep", "bow_head", "listen", "interact"
    ]
    try:
        ts_actions = _tracks_stats_from_actions_jsonl(case_dir)
        X1, ids1, ts1 = _build_features_from_tracks_stats(ts_actions, all_actions)
        if _is_features_good(X1, ids1, min_points=3):
            return X1, ids1, ts1
    except Exception as e:
        print(f"[features] actions.jsonl failed: {case_dir.name} -> {e}")
    try:
        ts_tl = _tracks_stats_from_timeline(case_dir)
        X2, ids2, ts2 = _build_features_from_tracks_stats(ts_tl, all_actions)
        if _is_features_good(X2, ids2, min_points=3):
            return X2, ids2, ts2
    except Exception as e:
        print(f"[features] timeline fallback failed: {case_dir.name} -> {e}")
    return _empty_features(all_actions)
# -------------------------------------------------------------------


def compute_levenshtein_matrix(tracks_stats: Dict[int, Any], ids: List[int]):
    n = len(ids)
    dist_matrix = np.zeros((n, n))
    unique_acts = sorted(list(set(a for t in tracks_stats.values() for a in t["actions"])))
    act_map = {a: chr(65 + i) for i, a in enumerate(unique_acts)}
    seqs = []
    for tid in ids:
        s = "".join([act_map.get(a, " ") for a in tracks_stats[tid]["actions"]])
        seqs.append(s)
    for i in range(n):
        for j in range(i + 1, n):
            d = Levenshtein.distance(seqs[i], seqs[j]) if Levenshtein else abs(len(seqs[i]) - len(seqs[j]))
            max_len = max(len(seqs[i]), len(seqs[j]), 1)
            dist_matrix[i, j] = dist_matrix[j, i] = d / max_len
    return dist_matrix


# ================================
# 4) 路由
# ================================

# ---------------- FIXED HOME ROUTE (FIX 4.1) ----------------
@app.get("/", include_in_schema=False)
def home():
    index_path = OUTPUT_ROOT / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    # 关键：不要走 Jinja2，直接当静态 html 发回去
    return FileResponse(str(index_path), media_type="text/html; charset=utf-8")

# 可选：别让 favicon 404 刷屏
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)
# --------------------------------------------------


@app.get("/api/list_views")
def list_views():
    if not OUTPUT_ROOT.exists():
        return []
    views = [p.name for p in OUTPUT_ROOT.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))]
    return sorted(views)


@app.get("/api/list_cases")
def list_cases():
    cases = []
    if OUTPUT_ROOT.exists():
        for view_dir in OUTPUT_ROOT.iterdir():
            if not (view_dir.is_dir() and not view_dir.name.startswith((".", "_"))):
                continue
            for case_dir in view_dir.iterdir():
                if not case_dir.is_dir():
                    continue
                base_id = _find_base_id(case_dir)
                num4 = _extract_num(case_dir.name)
                num4_str = f"{num4:04d}" if num4 is not None else None
                fps = 0.0
                frames = 0
                meta_path = case_dir / f"{base_id}.meta.json"
                if not meta_path.exists():
                    meta_path = case_dir / f"{base_id}_behavior.meta.json"
                if meta_path.exists():
                    meta = _read_json(meta_path) or {}
                    fps = float(meta.get("fps", 0) or 0)
                    frames = int(meta.get("frames", 0) or 0)
                if fps <= 0 or frames <= 0:
                    mp4s = list(case_dir.glob("*.mp4"))
                    if mp4s:
                        v = _ffprobe_duration_fps(mp4s[0])
                        fps = fps or float(v.get("fps", 0) or 0)
                        frames = frames or int(v.get("frames", 0) or 0)
                cases.append(
                    {
                        "id": case_dir.name,
                        "view": view_dir.name,
                        "case_base": base_id,
                        "case_num4": num4_str,
                        "path": f"{view_dir.name}/{case_dir.name}",
                        "frames": frames,
                        "fps": fps,
                        "loaded": True,
                    }
                )
    return cases


@app.get("/api/summary/{case_id}")
def get_summary(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        raise HTTPException(404, "case not found")
    base_id = _find_base_id(case_dir)
    candidates = [
        case_dir / f"{base_id}_summary.json",
        case_dir / f"{case_dir.name}_summary.json",
        case_dir / f"{case_dir.name.split('__')[-1]}_summary.json" if "__" in case_dir.name else None,
    ]
    candidates = [p for p in candidates if p is not None]
    summary_path = None
    for p in candidates:
        if p.exists():
            summary_path = p
            break
    if summary_path is None:
        g = list(case_dir.glob("*_summary.json"))
        if g:
            summary_path = g[0]
    base_resp = {
        "case_id": case_dir.name,
        "case_base": base_id,
        "view": case_dir.parent.name,
        "frames": 0, "duration": 0, "fps": 0, "raw": None,
    }
    if summary_path is None:
        mp4s = list(case_dir.glob("*.mp4"))
        if mp4s:
            v = _ffprobe_duration_fps(mp4s[0])
            base_resp["frames"] = v["frames"]
            base_resp["fps"] = v["fps"]
        return base_resp
    raw = _read_json(summary_path) or {}
    video = raw.get("video", {}) if isinstance(raw, dict) else {}
    meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    frames = int(video.get("frames", meta.get("frames", 0)) or 0)
    duration = float(video.get("duration", meta.get("duration", 0)) or 0)
    fps = float(video.get("fps", meta.get("fps", 0)) or 0)
    if frames == 0 or fps == 0:
        mp4s = list(case_dir.glob("*.mp4"))
        if mp4s:
            v = _ffprobe_duration_fps(mp4s[0])
            if frames == 0:
                frames = int(v["frames"])
            if fps == 0:
                fps = float(v["fps"])
    return {
        "case_id": case_dir.name,
        "case_base": base_id,
        "view": case_dir.parent.name,
        "frames": frames, "duration": duration, "fps": fps,
        "raw": raw,
        "summary_url": _url_segments("/output", [case_dir.parent.name, case_dir.name, summary_path.name]),
    }


@app.get("/api/media/{case_id}")
def api_media(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    resp = {
        "final_url": "",
        "pose_url": "",
        "behavior_url": "",
        "original_url": "",
        "audio_url": "",
        "debug": {
            "case_id_req": case_id, "view_req": view, "final_codec": "unknown",
            "candidates_checked": [], "all_candidates": [],
        },
    }
    if not case_dir:
        return resp
    view_name = case_dir.parent.name
    base_id = _find_base_id(case_dir)
    mp4s = [p for p in case_dir.glob("*.mp4") if p.is_file()]
    pose_file, behavior_file = _pick_overlay_files(mp4s)

    def score(p: Path) -> int:
        n = p.name.lower()
        s = 0
        if "behavior_overlay" in n:
            s += 2000
        if n.endswith("_behavior_overlay.mp4"):
            s += 1000
        if "overlay" in n:
            s += 200
        if "final" in n:
            s += 50
        if n.endswith("_h264.mp4"):
            s += 500
        return s

    sorted_mp4s = sorted(mp4s, key=score, reverse=True)
    resp["debug"]["all_candidates"] = [p.name for p in sorted_mp4s]
    final_file = ""
    final_codec = ""
    for p in sorted_mp4s:
        codec = _ffprobe_codec(p)
        resp["debug"]["candidates_checked"].append({"name": p.name, "codec": codec})
        if codec == "h264":
            final_file = p.name
            final_codec = codec
            break
        if not final_file:
            final_file = p.name
            final_codec = codec
    if final_file:
        resp["final_url"] = _url_segments("/output", [view_name, case_dir.name, final_file])
        resp["debug"]["final_pick"] = final_file
        resp["debug"]["final_codec"] = final_codec
    if pose_file:
        resp["pose_url"] = _url_segments("/output", [view_name, case_dir.name, pose_file.name])
    if behavior_file:
        resp["behavior_url"] = _url_segments("/output", [view_name, case_dir.name, behavior_file.name])
    want_num = _extract_num(case_dir.name) or _extract_num(case_id)
    original_file = ""
    if want_num is not None:
        view_dir = DATA_ROOT / view_name
        if view_dir.exists():
            for w in (4, 3, 2, 1):
                cand = f"{want_num:0{w}d}.mp4"
                if (view_dir / cand).exists():
                    original_file = cand
                    break
            if not original_file:
                for p in view_dir.glob("*.mp4"):
                    if _extract_num(p.stem) == want_num:
                        original_file = p.name
                        break
    if original_file:
        resp["original_url"] = _url_segments("/data", [view_name, original_file])
    wav = case_dir / f"{base_id}.wav"
    if not wav.exists():
        wavs = list(case_dir.glob("*.wav"))
        wav = wavs[0] if wavs else None
    if wav and wav.exists():
        resp["audio_url"] = _url_segments("/output", [view_name, case_dir.name, wav.name])
    if not resp["final_url"] and resp["original_url"]:
        resp["final_url"] = resp["original_url"]
    return resp


@app.get("/api/files/{case_id}")
def list_case_files(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        return []
    out = []
    for p in sorted(case_dir.iterdir()):
        if p.is_file():
            stat = p.stat()
            out.append({"name": p.name, "size": stat.st_size, "mtime": stat.st_mtime, "suffix": p.suffix.lower()})
    return out


def _safe_filename(filename: str, allowed_suffixes: set) -> Path:
    if not filename or "\x00" in filename:
        raise HTTPException(400, "bad filename")
    if any(x in filename for x in ["..", "/", "\\"]):
        raise HTTPException(400, "bad filename")
    if Path(filename).name != filename:
        raise HTTPException(400, "bad filename")
    suf = Path(filename).suffix.lower()
    if suf not in allowed_suffixes:
        raise HTTPException(400, "unsupported file type")
    return Path(filename)


@app.get("/api/raw/{case_id}/{filename}")
def get_raw_file(case_id: str, filename: str, view: Optional[str] = None):
    allowed_suffixes = {".json", ".jsonl", ".mp4", ".wav", ".txt", ".log"}
    fname = _safe_filename(filename, allowed_suffixes)
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        raise HTTPException(404, "case not found")
    base = case_dir.resolve(strict=False)
    p = (case_dir / fname).resolve(strict=False)
    try:
        p.relative_to(base)
    except Exception:
        raise HTTPException(400, "bad filename")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    suf = p.suffix.lower()
    media_map = {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".mp4": "video/mp4",
        ".wav": "audio/wav",
        ".txt": "text/plain; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
    }
    media = media_map.get(suf, "application/octet-stream")
    return FileResponse(str(p), media_type=media, filename=p.name)


@app.get("/api/audio/{case_id}")
def get_audio(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        raise HTTPException(404, "Case directory not found")
    wavs = list(case_dir.glob("*.wav"))
    if not wavs:
        raise HTTPException(404, "No audio (.wav) found")
    target = wavs[0]
    for w in wavs:
        if w.name.lower() in ["output.wav", "audio.wav"]:
            target = w
            break
    return FileResponse(str(target), media_type="audio/wav", filename=target.name)


@app.get("/api/actions/{case_id}")
def get_actions(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        return []
    return load_jsonl(case_dir / "actions.jsonl")


@app.get("/api/transcript/{case_id}")
def get_transcript(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        return []
    rows = load_jsonl(case_dir / "transcript.jsonl")
    return [{"start": r.get("start", 0), "end": r.get("end", 0), "text": r.get("text", "")} for r in rows]


@app.get("/api/align/{case_id}")
def get_align(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        return {}
    p = case_dir / "align.json"
    if not p.exists():
        return {}
    return _read_json(p) or {}


@app.get("/api/tracks/{case_id}")
def get_tracks(case_id: str, view: Optional[str] = None, full: int = 0, source: str = "tracks"):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        return {}
    base_id = _find_base_id(case_dir)
    f_tracks: Optional[Path] = None
    if source == "legacy":
        p = case_dir / f"{base_id}.jsonl"
        if p.exists():
            f_tracks = p
    elif source == "behavior":
        p = case_dir / f"{base_id}_behavior.jsonl"
        if not p.exists():
            p = case_dir / "case_det.jsonl"
        if p.exists():
            f_tracks = p
    else:
        for name in ["pose_tracks_smooth.jsonl", "pose_tracks_smooth_kpts.jsonl"]:
            p = case_dir / name
            if p.exists():
                f_tracks = p
                break
    if not f_tracks:
        return {}
    data = load_jsonl(f_tracks)
    frame_dict: Dict[str, Any] = {}
    for row in data:
        f_idx = row.get("frame_idx", row.get("frame"))
        if f_idx is None:
            continue
        f_key = str(int(f_idx))
        if full:
            frame_dict[f_key] = row
            continue
        objs = []
        if "persons" in row:
            for p in row.get("persons", []) or []:
                bb = p.get("bbox")
                objs.append({"id": p.get("track_id", -1), "box": bb})
        elif "dets" in row:
            for d in row.get("dets", []) or []:
                objs.append({"id": d.get("track_id", -1), "box": d.get("xyxy"), "label": d.get("label")})
        frame_dict[f_key] = objs
    return frame_dict


@app.get("/api/timeline/{case_id}")
def get_timeline(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        return {"meta": {"source": "case_not_found", "fps": 0, "total_events": 0, "track_ids": []}, "items": []}
    tl_path = case_dir / "timeline_viz.json"
    raw = None
    if tl_path.exists():
        raw = _read_json(tl_path)
    if not isinstance(raw, dict):
        raw = _build_timeline_viz(case_dir)
    meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    items = raw.get("items", []) if isinstance(raw, dict) else []
    out_items = []
    for i, it in enumerate(items or []):
        content_raw = it.get("action") or it.get("label") or ""
        content = _normalize_action(content_raw)
        action_id = it.get("action_id")
        if action_id is None or content_raw != content:
            action_id = ACTION_MAP.get(content, 0)
        out_items.append(
            {
                "id": i,
                "start": it.get("start", 0),
                "end": it.get("end", 0),
                "content": content,
                "track_id": it.get("track_id"),
                "action_id": int(action_id or 0),
                "frame_idx": it.get("frame_idx", 0),
                "duration": it.get("duration", 0),
            }
        )
    return {"meta": meta, "items": out_items}


@app.get("/api/timeline_raw/{case_id}")
def get_timeline_raw(case_id: str, view: Optional[str] = None):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        raise HTTPException(404, "case not found")
    p = case_dir / "timeline_viz.json"
    if not p.exists():
        return _build_timeline_viz(case_dir)
    return _read_json(p) or {}


@app.get("/api/projection/{case_id}")
def get_projection(
    case_id: str,
    view: Optional[str] = None,
    method: ProjectionMethod = ProjectionMethod.PCA,
    metric: ProjectionMetric = ProjectionMetric.EUCLIDEAN,
    force: int = Query(0, ge=0, le=1),  # Added force param
):
    case_dir = get_case_dir(case_id, view=view)
    if not case_dir:
        return {"points": []}

    # ---------------- FIX 4.2: Smart Static Cache ----------------
    static_file = case_dir / "static_projection.json"
    actions_file = case_dir / "actions.jsonl"
    tl_file = case_dir / "timeline_viz.json"

    newest_input = max(_mtime(actions_file), _mtime(tl_file))

    # Only use static if NOT forced AND static is newer than inputs
    if not force and static_file.exists():
         if _mtime(static_file) >= newest_input:
            data = _read_json(static_file)
            if (
                isinstance(data, dict)
                and isinstance(data.get("points"), list)
                and len(data["points"]) >= 3
            ):
                return data
    # -------------------------------------------------------------

    try:
        StandardScaler, MinMaxScaler, PCA, MDS, TSNE = _get_sklearn_components()
        # Use wrapper with signature
        X, ids, tracks_stats = compute_action_features(case_dir)
        if not _is_features_good(X, ids, min_points=3):
            return {"points": []}

        X = np.nan_to_num(X, nan=0.0)

        # ---------------- MODIFIED PROJECTION LOGIC ----------------
        if metric == ProjectionMetric.LEVENSHTEIN:
            dist_matrix = compute_levenshtein_matrix(tracks_stats, ids)
            # Safe call using helper
            mds = _make_mds_precomputed(MDS, n_components=2, random_state=0)
            points_2d = mds.fit_transform(dist_matrix)

        else:
            scaler = StandardScaler()
            X_std = np.nan_to_num(scaler.fit_transform(X), nan=0.0)

            if method == ProjectionMethod.PCA:
                points_2d = (
                    np.zeros((len(X_std), 2))
                    if np.sum(np.var(X_std, axis=0)) == 0
                    else PCA(n_components=2).fit_transform(X_std)
                )
            elif method == ProjectionMethod.TSNE:
                # Safe call using helper
                points_2d = _fit_tsne_2d(TSNE, X_std)
            else:  # MDS
                points_2d = MDS(n_components=2).fit_transform(X_std)
        # -----------------------------------------------------------

        points_norm = MinMaxScaler().fit_transform(points_2d)

        result = []
        for i, (x, y) in enumerate(points_norm):
            result.append(
                {
                    "track_id": ids[i],
                    "x": float(0.5 if np.isnan(x) else x),
                    "y": float(0.5 if np.isnan(y) else y),
                    "info": f"Track {ids[i]}",
                }
            )
        return {"points": result}

    except Exception as e:
        print(f"[Projection Error] {case_id}: {e}")
        return {"points": []}


# ================================
# 5) 触发分析（可选）
# ================================

class RunParams(BaseModel):
    video_rel_path: str
    case_id: str


def _view_key_from_view_name(view_name: str) -> str:
    if "正方" in view_name:
        return "front"
    if "后方" in view_name:
        return "rear"
    if "斜上方视角1" in view_name or view_name.endswith("视角1"):
        return "top1"
    if "斜上方视角2" in view_name or view_name.endswith("视角2"):
        return "top2"
    if "教师" in view_name:
        return "teacher"
    return "view"


@app.post("/api/run_analysis")
def trigger_analysis(params: RunParams):
    if not DATA_ROOT.exists():
        raise HTTPException(500, f"DATA_ROOT not exists: {DATA_ROOT}")

    video_rel_path = params.video_rel_path.replace("\\", "/").lstrip("/")
    video_full_path = (DATA_ROOT / video_rel_path).resolve()
    if not video_full_path.exists():
        raise HTTPException(404, "Video not found")

    view_name = Path(video_rel_path).parent.name
    view_key = _view_key_from_view_name(view_name)
    num = _extract_num(Path(video_rel_path).stem)
    if num is None:
        num = _extract_num(params.case_id)
    if num is None:
        raise HTTPException(400, "Cannot infer numeric id from video_rel_path/case_id")

    base3 = f"{num:03d}"
    base4 = f"{num:04d}"
    case_dir_name = params.case_id.strip()
    if "__" not in case_dir_name:
        case_dir_name = f"{view_key}__{base4}"

    out_dir = OUTPUT_ROOT / view_name / case_dir_name
    video_id_param = f"{view_key}__{base4}"

    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(500, f"PIPELINE_SCRIPT not found: {PIPELINE_SCRIPT}")

    cmd = [
        PYTHON_EXE,
        str(PIPELINE_SCRIPT),
        "--video", str(video_full_path),
        "--video_id", video_id_param,
        "--out_dir", str(out_dir),
        "--case_id", base3,
        "--view", view_name,
        "--run_summarize", "1",
        "--run_aggregate", "1",
        "--run_projection", "1",
        "--run_timeline", "1",
        "--skip_existing", "0",
    ]

    try:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8")
        if res.returncode == 0:
            return {
                "status": "success",
                "case_dir": f"{view_name}/{case_dir_name}",
                "case_base": base3,
                "logs": "Pipeline finished.",
            }
        return JSONResponse(
            {
                "status": "error",
                "message": "Pipeline failed",
                "stdout": res.stdout,
                "stderr": res.stderr,
                "cmd": " ".join(cmd),
            },
            status_code=500,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    print("Server starting at http://127.0.0.1:8000")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
