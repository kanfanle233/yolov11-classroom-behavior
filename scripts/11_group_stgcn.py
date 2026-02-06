import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict


# === 1. 轻量级 ST-GCN 模型定义 ===
class GraphConv(nn.Module):
    """简单的图卷积层: X' = D^-1 * A * X * W"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, A):
        # x: (N, C, T, V)
        # A: (N, V, V)  <-- 每一帧或每个窗口的邻接矩阵
        # 这里简化处理：假设一个窗口内 A 是静态的或取平均

        # 矩阵乘法: A * x
        # x reshaped -> (N, V, C*T) for matmul?
        # 为了简化，我们使用爱因斯坦求和约定
        # n: batch, c: channels, t: time, v: vertices, u: neighbors
        x_out = torch.einsum('nvu,nctu->nctv', A, x)
        return self.conv(x_out)


class ClassroomSTGCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, max_nodes=50):
        super().__init__()
        self.max_nodes = max_nodes

        # 1. 空间图卷积 (Spatial)
        self.gcn1 = GraphConv(in_channels, 64)
        self.gcn2 = GraphConv(64, 128)

        # 2. 时间卷积 (Temporal) - 也就是 1D Conv over time
        self.tcn1 = nn.Conv2d(64, 64, kernel_size=(9, 1), padding=(4, 0))
        self.tcn2 = nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0))

        # 3. 分类头
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, A):
        # x: (Batch, Channels, Time, Nodes)
        # A: (Batch, Nodes, Nodes)

        # Block 1
        x = F.relu(self.tcn1(self.gcn1(x, A)))

        # Block 2
        x = F.relu(self.tcn2(self.gcn2(x, A)))

        # Global Pooling (Time + Nodes)
        x = F.avg_pool2d(x, x.size()[2:])  # -> (N, 128, 1, 1)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)
        return x


# === 2. 辅助函数：构建交互图 ===

def build_graph_from_frame(persons, img_width=1920, img_height=1080, max_nodes=50, dist_thres=0.15):
    """
    基于距离构建邻接矩阵。
    注意：这里的坐标需要归一化 (0-1)
    """
    # 初始化特征矩阵 (C, V) -> (x, y, conf)
    features = np.zeros((3, max_nodes), dtype=np.float32)
    # 初始化邻接矩阵 (V, V)
    adj = np.eye(max_nodes, dtype=np.float32)  # 自环

    # 填充存在的节点
    valid_count = min(len(persons), max_nodes)
    coords = []

    for i in range(valid_count):
        p = persons[i]
        bbox = p['bbox']  # [x1, y1, x2, y2]
        cx = (bbox[0] + bbox[2]) / 2 / img_width
        cy = (bbox[1] + bbox[3]) / 2 / img_height
        conf = p.get('conf', 1.0)

        features[0, i] = cx
        features[1, i] = cy
        features[2, i] = conf
        coords.append((cx, cy))

    # 计算邻接关系
    for i in range(valid_count):
        for j in range(i + 1, valid_count):
            dist = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            if dist < dist_thres:  # 归一化后的距离阈值
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # 归一化邻接矩阵 D^-1 A (简单版)
    row_sum = np.sum(adj, axis=1)
    row_sum[row_sum == 0] = 1
    norm_adj = adj / row_sum[:, np.newaxis]

    return features, norm_adj


# === 3. 主逻辑 ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True, help="pose_tracks_smooth.jsonl")
    parser.add_argument("--out", required=True, help="group_events.jsonl")
    parser.add_argument("--model_weight", default="", help="Path to trained ST-GCN weights")
    parser.add_argument("--window_size", type=int, default=50, help="Frame window size for classification")
    parser.add_argument("--width", type=int, default=1920, help="frame width for normalization")
    parser.add_argument("--height", type=int, default=1080, help="frame height for normalization")
    parser.add_argument("--fps", type=float, default=25.0, help="video fps for timestamps")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    pose_path = Path(args.pose)
    if not pose_path.is_absolute():
        pose_path = (base_dir / pose_path).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    model_weight = Path(args.model_weight) if args.model_weight else None
    if model_weight and not model_weight.is_absolute():
        model_weight = (base_dir / model_weight).resolve()

    # 加载数据
    data_by_frame = defaultdict(list)
    max_frame = 0
    with open(pose_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            data_by_frame[d['frame']] = d['persons']
            max_frame = max(max_frame, d['frame'])

    print(f"[Info] Loaded data up to frame {max_frame}. Analyzing group interactions...")

    # 准备模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 类别定义：0: Lecture (听讲), 1: Discussion (讨论), 2: Break/Chaos (课间/混乱)
    model = ClassroomSTGCN(num_classes=3).to(device)

    has_weights = False
    if model_weight and model_weight.exists():
        model.load_state_dict(torch.load(model_weight))
        has_weights = True
        model.eval()
        print(f"[Model] Loaded weights from {model_weight}")
    else:
        print("[Model] No weights provided. Using heuristic-based pseudo-prediction.")

    # 滑动窗口处理
    window_step = args.window_size // 2
    events = []

    for start_f in range(0, max_frame, window_step):
        end_f = start_f + args.window_size

        # 收集窗口内的数据构建 Tensor
        # Shape: (1, C, T, V)
        clip_features = []
        clip_adjs = []

        for f_idx in range(start_f, end_f):
            persons = data_by_frame.get(f_idx, [])
            feat, adj = build_graph_from_frame(persons, img_width=args.width, img_height=args.height)
            clip_features.append(feat)
            clip_adjs.append(adj)

        # 转换为 Tensor
        # feature: list of (C, V) -> (T, C, V) -> (1, C, T, V)
        tx = np.stack(clip_features, axis=0)
        tx = np.transpose(tx, (1, 0, 2))
        input_x = torch.tensor(tx, dtype=torch.float32).unsqueeze(0).to(device)

        # adj: list of (V, V) -> Use the average adj for the window?
        # ST-GCN typically uses fixed adj, but here it's dynamic.
        # Approximation: Use the adj of the middle frame for graph structure
        mid_idx = len(clip_adjs) // 2
        input_a = torch.tensor(clip_adjs[mid_idx], dtype=torch.float32).unsqueeze(0).to(device)

        # === 推理或规则判定 ===
        label = "unknown"
        confidence = 0.0

        if has_weights:
            with torch.no_grad():
                logits = model(input_x, input_a)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)

                mapping = {0: "lecture", 1: "discussion", 2: "break"}
                label = mapping.get(pred_idx.item(), "unknown")
                confidence = conf.item()
        else:
            # === 启发式规则 (用于无模型时的 Demo 展示) ===
            # 计算这一段时间内的平均交互密度
            # 密度 = 总边数 / (节点数 * (节点数-1))
            avg_adj = np.mean(np.array(clip_adjs), axis=0)  # (V, V)
            # 去掉对角线
            np.fill_diagonal(avg_adj, 0)

            num_people = np.sum(input_x[0, 2, 0, :].cpu().numpy() > 0)  # Count valid nodes based on conf > 0
            if num_people > 1:
                num_edges = np.sum(avg_adj > 0.5) / 2  # 无向图
                interaction_ratio = num_edges / num_people

                if interaction_ratio > 0.6:
                    label = "discussion"
                    confidence = 0.85
                elif interaction_ratio > 0.1:
                    label = "lecture"  # 少量交互，主要是并排坐
                    confidence = 0.6
                else:
                    label = "individual_work"  # 几乎无交互
                    confidence = 0.5
            else:
                label = "empty"
                confidence = 1.0

        if label != "empty":
            events.append({
                "start_frame": start_f,
                "end_frame": end_f,
                "start_time": start_f / float(args.fps),
                "end_time": end_f / float(args.fps),
                "group_event": label,
                "confidence": float(f"{confidence:.2f}")
            })

    # 保存结果
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[Done] Detected {len(events)} group interaction segments.")


if __name__ == "__main__":
    main()
