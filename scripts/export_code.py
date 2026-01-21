import os

# ====== 配置区域 ======
SOURCE_DIR = r"F:\PythonProject\pythonProject\YOLOv11"  # 你的代码源目录
DEST_DIR = r"C:\Users\Lenovo\Desktop\yolo"  # 你的目标目录
OUTPUT_FILENAME = "all04.txt"  # 目标文件名

# 不需要导出的文件夹 (避免导出虚拟环境或缓存代码)
IGNORE_DIRS = {'.git', '.idea', '__pycache__', 'venv', 'env', '.vscode'}


# =====================

def merge_py_files():
    # 1. 确保目标文件夹存在
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"创建目录: {DEST_DIR}")

    output_path = os.path.join(DEST_DIR, OUTPUT_FILENAME)
    file_count = 0

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # 遍历目录
            for root, dirs, files in os.walk(SOURCE_DIR):
                # 过滤掉不需要的文件夹（原地修改 dirs 列表）
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

                for file in files:
                    if file.endswith('.py'):
                        # 排除掉这个脚本自己，防止死循环
                        if file == "export_code.py":
                            continue

                        file_path = os.path.join(root, file)
                        # 获取相对路径 (例如: server/app.py)
                        rel_path = os.path.relpath(file_path, SOURCE_DIR)

                        try:
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                content = infile.read()

                            # 写入分隔符和文件名
                            outfile.write("=" * 50 + "\n")
                            outfile.write(f"文件路径: {rel_path}\n")
                            outfile.write("=" * 50 + "\n\n")
                            outfile.write(content)
                            outfile.write("\n\n")  # 文件之间空两行

                            print(f"已合并: {rel_path}")
                            file_count += 1

                        except Exception as e:
                            print(f"无法读取文件 {rel_path}: {e}")

        print("-" * 30)
        print(f"成功! 共合并了 {file_count} 个 .py 文件。")
        print(f"输出文件位于: {output_path}")

    except Exception as e:
        print(f"写入目标文件失败: {e}")


if __name__ == "__main__":
    merge_py_files()