import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按前缀整理 output 目录下的视频结果")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="输出根目录 (默认使用项目根目录下的 output/)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="front,top1,rear",
        help="需要整理的前缀列表，用逗号分隔",
    )
    parser.add_argument("--dry_run", action="store_true", help="只打印移动计划，不执行")
    return parser.parse_args()


def resolve_root_dir(root_dir: str | None) -> Path:
    if root_dir:
        return Path(root_dir).resolve()

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    return project_root / "output"


def organize_output_folders(root_dir: Path, categories: list[str], dry_run: bool) -> int:
    if not root_dir.exists():
        print(f"错误：路径 {root_dir} 不存在。")
        return 0

    for category in categories:
        target_folder = root_dir / category
        if not target_folder.exists():
            if dry_run:
                print(f"[DRY] 将创建分类文件夹: {target_folder}")
            else:
                target_folder.mkdir(parents=True, exist_ok=True)
                print(f"创建分类文件夹: {category}")

    count = 0
    for item in root_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name in categories:
            continue

        for category in categories:
            if item.name.startswith(category):
                destination = root_dir / category / item.name
                if dry_run:
                    print(f"[DRY] 已移动: {item.name} -> {category}/")
                    count += 1
                    break

                try:
                    shutil.move(str(item), str(destination))
                    print(f"已移动: {item.name} -> {category}/")
                    count += 1
                    break
                except Exception as e:
                    print(f"移动 {item.name} 时出错: {e}")
                break

    print(f"\n整理完成！共移动了 {count} 个文件夹。")
    return count


def main() -> None:
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    root_dir = resolve_root_dir(args.root_dir)
    organize_output_folders(root_dir, categories, args.dry_run)


if __name__ == "__main__":
    main()
