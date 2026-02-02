import json
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

# 尝试引入 Levenshtein
try:
    import Levenshtein
except ImportError:
    Levenshtein = None


def load_jsonl(path: Path):
    data = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
    return data


def compute_features_and_save(case_dir: Path):
    """
    计算单个案例的投影数据并保存为 static_projection.json
    """
    f_actions = case_dir / "actions.jsonl"
    if not f_actions.exists():
        return

    data = load_jsonl(f_actions)
    tracks_stats = {}

    # 1. 提取特征 (与 app.py 逻辑一致)
    for row in data:
        tid = row.get("track_id", -1)
        if tid == -1: continue

        if tid not in tracks_stats:
            tracks_stats[tid] = {"actions": [], "positions": []}

        act = row.get("action", "stand")
        tracks_stats[tid]["actions"].append(act)

        bbox = row.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2 if len(bbox) == 4 else 0
        tracks_stats[tid]["positions"].append(cx)

    all_possible_actions = ["hand_raise", "stand", "sit", "reading", "writing", "phone", "sleep", "bow_head",
                            "lean_table"]
    X = []
    ids = []

    for tid, stats in tracks_stats.items():
        total = len(stats["actions"])
        if total == 0: continue

        act_counts = {a: 0 for a in all_possible_actions}
        for a in stats["actions"]:
            if a in act_counts: act_counts[a] += 1

        vec = [act_counts[a] / total for a in all_possible_actions]
        vec.append(sum(stats["positions"]) / len(stats["positions"]))  # spatial
        vec.append(sum(1 for a in stats["actions"] if a not in ["stand", "sit"]) / total)  # activity

        X.append(vec)
        ids.append(tid)

    if len(X) < 2:
        return

    # 2. 计算 PCA (默认使用 PCA + Euclidean，最稳健)
    try:
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(X_std)
        points_norm = MinMaxScaler().fit_transform(points_2d)

        result = []
        for i, (x, y) in enumerate(points_norm):
            result.append({
                "track_id": ids[i],
                "x": float(x),
                "y": float(y),
                "info": f"Track {ids[i]}"
            })

        # 3. 保存为静态文件
        out_file = case_dir / "static_projection.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"points": result, "method": "pca_static"}, f, indent=2)
        print(f"  [OK] Generated static projection for {case_dir.name}")

    except Exception as e:
        print(f"  [Err] Failed {case_dir.name}: {e}")


def main(demo_web_root: str):
    root = Path(demo_web_root)
    if not root.exists():
        print("Demo Web root not found")
        return

    print(f"Scanning {root} for cases...")
    # 遍历所有视角和案例
    for view_dir in root.iterdir():
        if view_dir.is_dir() and not view_dir.name.startswith("_"):
            for case_dir in view_dir.iterdir():
                if case_dir.is_dir():
                    compute_features_and_save(case_dir)


if __name__ == "__main__":
    # 默认路径适配您的项目结构
    # YOLOv11/scripts/intelligence_class/tools/xx... -> YOLOv11/output/.../_demo_web
    default_root = Path(__file__).parents[3] / "output" / "智慧课堂学生行为数据集" / "_demo_web"
    main(str(default_root))