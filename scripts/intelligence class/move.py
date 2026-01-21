import os
import shutil

# 1. 设置目标根目录路径（使用原始字符串 r'' 避免反斜杠转义问题）
base_path = r'F:\PythonProject\pythonProject\YOLOv11\output'

# 2. 定义需要分类的前缀及其对应的目标文件夹名
categories = ['front', 'top1', 'rear']


def organize_output_folders(root_dir):
    # 检查路径是否存在
    if not os.path.exists(root_dir):
        print(f"错误：路径 {root_dir} 不存在。")
        return

    # 首先创建目标分类文件夹（front, top1, rear）
    for category in categories:
        target_folder = os.path.join(root_dir, category)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print(f"创建分类文件夹: {category}")

    # 遍历根目录下的所有内容
    items = os.listdir(root_dir)

    count = 0
    for item in items:
        item_path = os.path.join(root_dir, item)

        # 只处理文件夹，且跳过刚刚创建的三个目标文件夹
        if os.path.isdir(item_path) and item not in categories:
            for category in categories:
                # 检查文件夹名称是否以指定的前缀开头
                if item.startswith(category):
                    destination = os.path.join(root_dir, category, item)
                    try:
                        shutil.move(item_path, destination)
                        print(f"已移动: {item} -> {category}/")
                        count += 1
                        break  # 匹配到一个分类后即停止搜索
                    except Exception as e:
                        print(f"移动 {item} 时出错: {e}")

    print(f"\n整理完成！共移动了 {count} 个文件夹。")


if __name__ == "__main__":
    organize_output_folders(base_path)