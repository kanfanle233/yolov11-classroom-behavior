import argparse
import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    print("Error: 'umap-learn' is not installed. Please run: pip install umap-learn")
    exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="Path to embeddings.pkl")
    parser.add_argument("--meta", required=True, help="Path to student_features.json")
    parser.add_argument("--out", required=True, help="Path to student_projection.json")
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--n_clusters", type=int, default=3)

    args = parser.parse_args()

    emb_path = Path(args.emb)
    if not emb_path.exists():
        print(f"Embedding file not found: {emb_path}")
        return

    print(f"[1/4] Loading embeddings from {emb_path}...")
    with open(emb_path, "rb") as f:
        raw_embeddings = pickle.load(f)

    track_ids = []
    X = []

    # === 数据清洗与降维核心逻辑 ===
    for tid, vec_list in raw_embeddings.items():
        if len(vec_list) == 0: continue

        # 1. 处理列表中的每个张量
        processed_vecs = []
        for v in vec_list:
            v = np.array(v)
            # 如果特征向量维度 > 1 (比如 1x2x2x2304)，进行平均池化
            if v.ndim > 1:
                # 计算需要压缩的维度：除了最后一个维度(Channel)外的所有维度
                reduce_axes = tuple(range(v.ndim - 1))
                v = np.mean(v, axis=reduce_axes)
            processed_vecs.append(v)

        # 2. 对该学生的所有时间片段取平均
        # Stack shape: (Time, 2304) -> Mean -> (2304,)
        mean_vec = np.mean(np.array(processed_vecs), axis=0)

        track_ids.append(tid)
        X.append(mean_vec)

    X = np.array(X)
    print(f"      Aggregated features shape: {X.shape} (Expected: N x 2304)")

    # 双重保险：如果聚合后依然有多余维度 (比如 batch 维没去掉)
    if X.ndim > 2:
        print(f"      [Auto-Fix] Still has extra dims {X.shape}, flattening...")
        # 保留第一维(Samples)和最后一维(Features)，平均中间所有
        reduce_axes = tuple(range(1, X.ndim - 1))
        X = np.mean(X, axis=reduce_axes)
        print(f"      Fixed shape: {X.shape}")

    if len(X) < 3:
        print("Not enough data for UMAP (need > 3 samples). Writing empty result.")
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    # === UMAP 降维 ===
    print(f"[2/4] Running UMAP (metric={args.metric})...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        n_components=2,
        random_state=42
    )
    embedding_2d = reducer.fit_transform(X)

    # === 聚类 ===
    print(f"[3/4] Running Clustering (KMeans, k={args.n_clusters})...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # === 输出 ===
    print(f"[4/4] Exporting to {args.out}...")
    meta_map = {}
    if Path(args.meta).exists():
        with open(args.meta, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            for m in meta_data:
                meta_map[m['track_id']] = m

    output_data = []
    for i, tid in enumerate(track_ids):
        info = meta_map.get(tid, {})
        output_data.append({
            "track_id": tid,
            "x": float(embedding_2d[i, 0]),
            "y": float(embedding_2d[i, 1]),
            "cluster": int(labels[i]),
            "avg_attention": info.get("Avg Attention", 0),
            "activity": info.get("Activity Lvl", 0)
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"[Done] Semantic projection saved. Clusters: {len(set(labels))}")


if __name__ == "__main__":
    main()