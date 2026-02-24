#!/usr/bin/env python3
from pathlib import Path
import random
import argparse


def make_split(
    root_dir: str,
    val_ratio: float = 0.15,
    seed: int = 1337,
    split_tag: str | None = None,
):
    root = Path(root_dir)
    rgb_dir = root / "RGB"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    # Basenames, z.B. 000123.png
    names = sorted([p.name for p in rgb_dir.glob("*.png")])
    if len(names) == 0:
        raise RuntimeError(f"No PNG images found in {rgb_dir}")

    rng = random.Random(seed)
    rng.shuffle(names)

    n = len(names)
    val_n = int(round(val_ratio * n))

    val_names = sorted(names[:val_n])
    train_names = sorted(names[val_n:])

    if split_tag is None:
        split_tag = f"seed{seed}_{int((1 - val_ratio) * 100)}-{int(val_ratio * 100)}"

    train_file = root / f"synthia_train_{split_tag}.txt"
    val_file   = root / f"synthia_val_{split_tag}.txt"

    train_file.write_text("\n".join(train_names) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(val_names) + "\n", encoding="utf-8")

    print("=== SYNTHIA split created ===")
    print(f"Total images : {n}")
    print(f"Train images : {len(train_names)}")
    print(f"Val images   : {len(val_names)}")
    print(f"Seed         : {seed}")
    print(f"Val ratio    : {val_ratio}")
    print(f"Train list   : {train_file}")
    print(f"Val list     : {val_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create reproducible train/val split for SYNTHIA-RAND-CITYSCAPES")
    parser.add_argument("--root", required=True, help="Path to synthia root (contains RGB/, GT/...)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed used ONCE for the split")
    parser.add_argument("--split_tag", type=str, default=None, help="Optional custom tag for split filenames")

    args = parser.parse_args()
    make_split(
        root_dir=args.root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_tag=args.split_tag,
    )