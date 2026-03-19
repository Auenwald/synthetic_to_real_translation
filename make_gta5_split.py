#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import random
import argparse


def make_split(
    root_dir: str,
    val_ratio: float = 0.15,
    seed: int = 1337,
    split_tag: str | None = None,
    check_files: bool = True,
):
    root = Path(root_dir)
    image_dir = root / "images"
    label_dir = root / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    image_files = {p.name: p for p in image_dir.glob("*.png")}
    label_files = {p.name: p for p in label_dir.glob("*.png")}

    common_names = sorted(set(image_files.keys()) & set(label_files.keys()))
    only_images = sorted(set(image_files.keys()) - set(label_files.keys()))
    only_labels = sorted(set(label_files.keys()) - set(image_files.keys()))

    if len(common_names) == 0:
        raise RuntimeError("No matched image-label pairs found.")

    print("=== GTA5 pair scan ===")
    print(f"Images found     : {len(image_files)}")
    print(f"Labels found     : {len(label_files)}")
    print(f"Matched pairs    : {len(common_names)}")
    print(f"Images only      : {len(only_images)}")
    print(f"Labels only      : {len(only_labels)}")

    valid_names = []
    skipped_corrupt = 0
    skipped_size = 0

    if check_files:
        for name in common_names:
            img_path = image_files[name]
            mask_path = label_files[name]

            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size != mask.size:
                        skipped_size += 1
                        continue
            except (OSError, UnidentifiedImageError) as e:
                skipped_corrupt += 1
                print(f"[SKIP] {name}: {e}")
                continue

            valid_names.append(name)
    else:
        valid_names = common_names

    if len(valid_names) == 0:
        raise RuntimeError("No valid pairs left after filtering.")

    rng = random.Random(seed)
    rng.shuffle(valid_names)

    n = len(valid_names)
    val_n = int(round(val_ratio * n))

    val_names = sorted(valid_names[:val_n])
    train_names = sorted(valid_names[val_n:])

    if split_tag is None:
        train_pct = int(round((1 - val_ratio) * 100))
        val_pct = int(round(val_ratio * 100))
        split_tag = f"seed{seed}_{train_pct}-{val_pct}"

    train_file = root / f"gta5_train_{split_tag}.txt"
    val_file = root / f"gta5_val_{split_tag}.txt"

    train_file.write_text("\n".join(train_names) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(val_names) + "\n", encoding="utf-8")

    print("\n=== GTA5 split created ===")
    print(f"Valid pairs      : {len(valid_names)}")
    print(f"Skipped corrupt  : {skipped_corrupt}")
    print(f"Skipped mismatch : {skipped_size}")
    print(f"Train images     : {len(train_names)}")
    print(f"Val images       : {len(val_names)}")
    print(f"Seed             : {seed}")
    print(f"Val ratio        : {val_ratio}")
    print(f"Train list       : {train_file}")
    print(f"Val list         : {val_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create reproducible train/val split for GTA5"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to GTA5 root (contains images/ and labels/)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used once for the split"
    )
    parser.add_argument(
        "--split_tag",
        type=str,
        default=None,
        help="Optional custom tag for split filenames"
    )
    parser.add_argument(
        "--no_check",
        action="store_true",
        help="Skip file-open and size checks"
    )

    args = parser.parse_args()
    make_split(
        root_dir=args.root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_tag=args.split_tag,
        check_files=not args.no_check,
    )