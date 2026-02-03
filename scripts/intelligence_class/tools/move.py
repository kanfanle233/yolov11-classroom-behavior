import shutil
from pathlib import Path

from scripts.intelligence_class._utils.pathing import find_project_root

# 1. 动态获取项目 output 目录
# 假设本脚本在 .../scripts/intelligence_class/tools/move.py
current_file = Path(__file__).resolve()
project_root = find_project_root(current_file)
base_path = project_root / "output"

# 2. 定义需要分类的前缀及其对应的目标文件夹名
categories = ['front', 'top1', 'rear']


def organize_output_folders(root_dir):
    root_dir = Path(root_dir)
    # 检查路径是否存在
    if not root_dir.exists():
        print(f"错误：路径 {root_dir} 不存在。")
        return

    print(f"正在整理目录: {root_dir}")

    # 首先创建目标分类文件夹（front, top1, rear）
    for category in categories:
        target_folder = root_dir / category
        if not target_folder.exists():
            target_folder.mkdir(exist_ok=True)
            print(f"创建分类文件夹: {category}")

    # 遍历根目录下的所有内容
    count = 0
    # 使用 iterdir 获取所有顶层目录
    for item_path in root_dir.iterdir():
        if not item_path.is_dir():
            continue

        item_name = item_path.name

        # 跳过刚刚创建的三个目标文件夹本身
        if item_name in categories:
            continue

        for category in categories:
            # 检查文件夹名称是否以指定的前缀开头 (e.g., rear__0001 -> rear)
            if item_name.startswith(category):
                destination = root_dir / category / item_name
                try:
                    # 如果目标已存在，shutil.move 可能会报错，这里简单处理
                    if destination.exists():
                        print(f"跳过（目标已存在）: {item_name}")
                    else:
                        shutil.move(str(item_path), str(destination))
                        print(f"已移动: {item_name} -> {category}/")
                        count += 1
                except Exception as e:
                    print(f"移动 {item_name} 时出错: {e}")
                break  # 匹配到一个分类后即停止搜索

    print(f"\n整理完成！共移动了 {count} 个文件夹。")


if __name__ == "__main__":
    organize_output_folders(base_path)
