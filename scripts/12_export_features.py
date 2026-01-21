import json
import argparse
import numpy as np
from pathlib import Path

# 定义行为映射权重 (用于计算注意力得分)
# 0: Listen (High), 1: Distract (Low), 2: Phone (Low), 3: Doze (Low)
# 4: Chat (Mid), 5: Note (High), 6: Raise (High), 7: Stand (Mid), 8: Read (High)
ATTENTION_WEIGHTS = {
    0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0,
    4: 0.5, 5: 1.0, 6: 1.0, 7: 0.5, 8: 1.0
}


def calculate_features(person_data, fps=25.0):
    """计算单个学生的高维特征"""
    items = person_data.get("visual_sequence", [])
    if not items:
        return None

    total_frames = 0
    attention_score_sum = 0.0
    action_switch_count = 0
    last_action = -1

    # 统计各类时长
    action_counts = {k: 0 for k in ATTENTION_WEIGHTS.keys()}

    for item in items:
        # 获取持续帧数
        start_f = item.get("start_frame") or (item.get("start_time", 0) * fps)
        end_f = item.get("end_frame") or (item.get("end_time", 0) * fps)
        duration = max(1, end_f - start_f)

        code = item.get("action_code", 0)

        # 1. 累加总帧数
        total_frames += duration

        # 2. 累加注意力得分
        weight = ATTENTION_WEIGHTS.get(code, 0.5)
        attention_score_sum += (duration * weight)

        # 3. 记录动作切换 (衡量多动程度/活跃度)
        if code != last_action:
            action_switch_count += 1
            last_action = code

        # 4. 统计具体动作
        if code in action_counts:
            action_counts[code] += duration

    if total_frames == 0:
        return None

    # === 特征工程 ===
    # 1. 平均注意力 (0~1)
    avg_attention = attention_score_sum / total_frames

    # 2. 活跃度 (每分钟动作切换次数) - 代理“多动”指标
    total_seconds = total_frames / fps
    activity_freq = (action_switch_count / total_seconds) * 60 if total_seconds > 0 else 0

    # 3. 互动/发言倾向 (举手 + 聊天 + 站立 占比)
    interaction_ratio = (action_counts[4] + action_counts[6] + action_counts[7]) / total_frames

    # 4. 玩手机/分心占比
    distract_ratio = (action_counts[1] + action_counts[2]) / total_frames

    return {
        "track_id": person_data.get("track_id"),
        "Avg Attention": round(avg_attention, 3),
        "Activity Lvl": round(activity_freq, 1),
        "Interaction": round(interaction_ratio, 3),
        "Distraction": round(distract_ratio, 3)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="per_person_sequences.json")
    parser.add_argument("--out", required=True, help="student_features.json")
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        print("Source file not found.")
        return

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理每个人
    people = data.get("people", {})
    if isinstance(people, list):  # 兼容列表格式
        people_dict = {}
        for p in people:
            pid = p.get("track_id") or p.get("id")
            people_dict[str(pid)] = p
        people = people_dict

    features_list = []
    for pid, p_data in people.items():
        feat = calculate_features(p_data)
        if feat:
            features_list.append(feat)

    # 输出
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(features_list, f, indent=2)

    print(f"[Done] Exported features for {len(features_list)} students to {args.out}")


if __name__ == "__main__":
    main()