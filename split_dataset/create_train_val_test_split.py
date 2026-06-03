import argparse
import csv
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]


def find_dataset_root(source: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if all((source / class_name).is_dir() for class_name in DEFAULT_CLASSES):
        return source

    candidates = [p for p in source.rglob("*") if p.is_dir() and p.name == "Integrated_Dataset_384"]
    for candidate in [source, *candidates]:
        if all((candidate / class_name).is_dir() for class_name in DEFAULT_CLASSES):
            return candidate

    raise FileNotFoundError(
        "Could not find a dataset folder containing the 10 expected class directories."
    )


def list_images(dataset_root: Path, class_name: str) -> list[Path]:
    class_dir = dataset_root / class_name
    return sorted(
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    n_total = len(items)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val :]
    return train_items, val_items, test_items


def ensure_empty_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite if you want to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_split(split_name: str, class_name: str, images: list[Path], output_dir: Path) -> list[dict[str, str]]:
    target_class_dir = output_dir / split_name / class_name
    target_class_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in images:
        dst = target_class_dir / src.name
        shutil.copy2(src, dst)
        rows.append(
            {
                "split": split_name,
                "class_name": class_name,
                "source_path": str(src),
                "target_path": str(dst),
            }
        )
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "class_name", "source_path", "target_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_class_names(output_dir: Path, class_names: list[str]) -> None:
    class_path = output_dir / "class_names.json"
    with class_path.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test folders.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw_data/Integrated_Dataset_384"),
        help="Dataset folder or parent folder containing Integrated_Dataset_384.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/split_dataset"),
        help="Output folder for train/val/test splits.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train, val, and test ratios must all be positive.")

    dataset_root = find_dataset_root(args.source)
    ensure_empty_output(args.output, args.overwrite)
    write_class_names(args.output, DEFAULT_CLASSES)

    random.seed(args.seed)
    all_rows = []
    summary = []

    for class_name in DEFAULT_CLASSES:
        images = list_images(dataset_root, class_name)
        if not images:
            raise ValueError(f"No images found for class: {class_name}")

        random.shuffle(images)
        train_items, val_items, test_items = split_items(
            images,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        all_rows.extend(copy_split("train", class_name, train_items, args.output))
        all_rows.extend(copy_split("val", class_name, val_items, args.output))
        all_rows.extend(copy_split("test", class_name, test_items, args.output))

        summary.append(
            {
                "class_name": class_name,
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
                "total": len(images),
            }
        )

    write_manifest(args.output, all_rows)

    print(f"Dataset root: {dataset_root}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print("class_name,train,val,test,total")
    for row in summary:
        print(
            f"{row['class_name']},{row['train']},{row['val']},"
            f"{row['test']},{row['total']}"
        )


if __name__ == "__main__":
    main()
