import argparse
import json
import sys
import datetime
from pathlib import Path
from collections import defaultdict

# ==========================================
# 配置与常量
# ==========================================

# 视角名称到英文短码的映射
VIEW_MAPPING = {
    "正方视角": "front",
    "斜上方视角1": "top1",
    "斜上方视角2": "top2",
    "后方视角": "rear",
    "教师视角": "teacher",
    "案例": "case"  # 以防开启 include_case
}


def resolve_project_paths():
    """解析项目根目录和默认路径"""
    # 当前脚本在 scripts/intelligence class/ 下，所以根目录是往上两级
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # YOLOv11 目录

    # 默认数据集路径 (根据提示词要求)
    default_dataset_root = project_root / "data" / "智慧课堂学生行为数据集"

    # 默认输出路径
    default_output_dir = project_root / "output"

    return project_root, default_dataset_root, default_output_dir


# ==========================================
# 核心功能函数
# ==========================================

def resolve_views(dataset_root: Path, user_views: list, include_case: bool):
    """
    解析需要扫描的视角文件夹列表
    返回: list of (folder_name, view_code, folder_path)
    """
    target_views = []

    # 确定要扫描的中文名列表
    if user_views:
        names_to_scan = user_views
    else:
        names_to_scan = ["正方视角", "斜上方视角1", "斜上方视角2", "后方视角", "教师视角"]

    if include_case and "案例" not in names_to_scan:
        names_to_scan.append("案例")

    print(f"[INFO] 计划扫描的视角: {names_to_scan}")

    # 验证文件夹是否存在
    for name in names_to_scan:
        folder_path = dataset_root / name
        if folder_path.exists() and folder_path.is_dir():
            code = VIEW_MAPPING.get(name, "unknown")
            target_views.append({
                "name": name,
                "code": code,
                "path": folder_path
            })
        else:
            print(f"[WARN] 目录不存在，跳过: {folder_path}")

    return target_views


def scan_videos(target_views, recursive: bool):
    """
    扫描目录下的 mp4 文件
    返回: list of (view_info_dict, file_path_obj)
    """
    found_files = []

    for view in target_views:
        folder_path = view["path"]
        print(f"[SCAN] 正在扫描: {folder_path.name} ...")

        pattern = "**/*.mp4" if recursive else "*.mp4"

        # 使用生成器进行扫描
        files = list(folder_path.glob(pattern)) if not recursive else list(folder_path.rglob("*.mp4"))

        # 再次过滤后缀（防止大小写问题，虽Windows不敏感但严谨）
        valid_files = [f for f in files if f.suffix.lower() == '.mp4']

        for f in valid_files:
            found_files.append((view, f))

    return found_files


def build_entries(found_files, project_root: Path, limit: int):
    """
    构建 VideoEntry 结构化数据
    """
    entries = []

    # 用于排序
    found_files.sort(key=lambda x: (x[0]['code'], x[1].stem))

    count = 0
    for view_info, file_path in found_files:
        if limit > 0 and count >= limit:
            break

        stem = file_path.stem
        view_code = view_info['code']
        view_display = view_info['name']

        # 生成全局唯一ID
        video_id = f"{view_code}__{stem}"

        # 计算相对路径
        try:
            rel_path = str(file_path.relative_to(project_root))
        except ValueError:
            # 如果不在项目目录下（比如跨盘符），则使用绝对路径
            rel_path = str(file_path)

        entry = {
            "view_display": view_display,
            "view_code": view_code,
            "video_path": str(file_path.resolve()),
            "rel_video_path": rel_path,
            "stem": stem,
            "video_id": video_id
        }
        entries.append(entry)
        count += 1

    return entries


def write_index(entries, output_dir: Path, dataset_root: Path, counts_per_view: dict):
    """
    写入 output/dataset_index.json
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dataset_index.json"

    meta = {
        "scan_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_root": str(dataset_root),
        "total_videos": len(entries),
        "view_counts": counts_per_view
    }

    data = {
        "meta": meta,
        "videos": entries
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] 索引文件已生成: {json_path}")
    return json_path


def print_summary(entries):
    """
    控制台打印摘要
    """
    print("\n" + "=" * 50)
    print(" 扫描结果摘要 (Scan Summary)")
    print("=" * 50)

    counts = defaultdict(int)
    for e in entries:
        counts[e['view_code']] += 1

    # 打印各视角统计
    # 按照预定义顺序打印比较美观
    order = ["front", "top1", "top2", "rear", "teacher", "case"]
    for code in order:
        if counts[code] > 0:
            print(f" {code:<10}: {counts[code]} videos")

    # 打印其他的（如果有 unknown）
    for code, count in counts.items():
        if code not in order:
            print(f" {code:<10}: {count} videos")

    print("-" * 50)
    print(f" TOTAL     : {len(entries)} videos")
    print("=" * 50)

    print("\n[Preview] 前 10 个 VideoEntry ID:")
    for i, e in enumerate(entries[:10]):
        print(f" {i + 1}. {e['video_id']}  ->  {e['rel_video_path']}")
    if len(entries) > 10:
        print(" ...")

    return counts


# ==========================================
# 主程序
# ==========================================

def main():
    project_root, default_data_root, default_out_dir = resolve_project_paths()

    parser = argparse.ArgumentParser(description="扫描智慧课堂多视角数据集并生成索引")

    parser.add_argument("--dataset_root", type=str, default=str(default_data_root),
                        help="数据集根目录路径")
    parser.add_argument("--output_dir", type=str, default=str(default_out_dir),
                        help="输出目录路径")
    parser.add_argument("--views", type=str, default=None,
                        help="指定扫描的视角，逗号分隔，例如：正方视角,后方视角")
    parser.add_argument("--include_case", action="store_true", default=False,
                        help="是否包含'案例'文件夹")
    parser.add_argument("--recursive", type=bool, default=True,
                        help="是否递归扫描子目录")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理视频数量，0为不限制")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    # 1. 检查根目录
    if not dataset_root.exists():
        print(f"[ERROR] 数据集根目录不存在: {dataset_root}")
        sys.exit(1)

    print(f"项目根目录: {project_root}")
    print(f"数据集目录: {dataset_root}")

    # 2. 解析视角
    user_views_list = args.views.split(',') if args.views else None
    target_views = resolve_views(dataset_root, user_views_list, args.include_case)

    if not target_views:
        print("[ERROR] 没有找到有效的视角文件夹，请检查目录结构或参数。")
        sys.exit(1)

    # 3. 扫描视频
    raw_files = scan_videos(target_views, args.recursive)

    if not raw_files:
        print("[WARN] 未扫描到任何 .mp4 文件。")
        sys.exit(0)

    # 4. 构建结构化数据
    entries = build_entries(raw_files, project_root, args.limit)

    # 5. 打印摘要并统计
    counts_map = print_summary(entries)

    # 6. 输出文件
    write_index(entries, output_dir, dataset_root, dict(counts_map))


if __name__ == "__main__":
    main()