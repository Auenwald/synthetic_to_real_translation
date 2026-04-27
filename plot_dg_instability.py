
#!/usr/bin/env python3
"""
Plot training instability / source-target mismatch from DG segmentation logs.

Expected log directory:
    ./logs_diss/

Example files:
    synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_0_new.json
    ...
    synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_4_new.json

    gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_0.json
    ...
    gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_4.json

The script:
- loads all matching seed files
- extracts epoch-wise mean IoU for source validation and both target domains
- computes mean/std across seeds
- creates a 2-panel figure:
    (a) target-domain fluctuations across epochs
    (b) best source validation != best target checkpoint
- optionally overlays the seed-0 trajectories faintly in the background

Usage examples:
    python plot_dg_instability.py \
        --log_dir ./logs_diss \
        --prefix synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval \
        --suffix _new.json \
        --source_key synthia \
        --out synthia_instability.png

    python plot_dg_instability.py \
        --log_dir ./logs_diss \
        --prefix gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval \
        --suffix .json \
        --source_key gta5 \
        --out gta5_instability.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot DG training instability from epoch-wise JSON logs.")
    p.add_argument("--log_dir", type=str, default="./logs_diss", help="Directory containing the JSON logs.")
    p.add_argument(
        "--prefix",
        type=str,
        required=True,
        help=(
            "Filename prefix before the numeric interval and seed, e.g. "
            "'synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval'"
        ),
    )
    p.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Averaging interval encoded in the filename."
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds to load."
    )
    p.add_argument(
        "--suffix",
        type=str,
        default=".json",
        help="Filename suffix after the seed, e.g. '.json' or '_new.json'."
    )
    p.add_argument(
        "--source_key",
        type=str,
        required=True,
        choices=["synthia", "gta5"],
        help="Top-level JSON key for the source validation curve."
    )
    p.add_argument("--city_key", type=str, default="cityscapes", help="Top-level JSON key for Cityscapes.")
    p.add_argument("--bdd_key", type=str, default="bdd", help="Top-level JSON key for BDD100K.")
    p.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="First epoch to plot."
    )
    p.add_argument(
        "--end_epoch",
        type=int,
        default=None,
        help="Last epoch to plot. If omitted, inferred from the logs."
    )
    p.add_argument(
        "--show_seed0",
        action="store_true",
        help="Overlay seed-0 trajectories as faint lines."
    )
    p.add_argument(
        "--title_prefix",
        type=str,
        default="",
        help="Optional prefix for subplot titles, e.g. 'SYNTHIA→'."
    )
    p.add_argument(
        "--out",
        type=str,
        default="dg_instability_figure.png",
        help="Output image path."
    )
    return p.parse_args()


def build_file_path(log_dir: Path, prefix: str, interval: int, seed: int, suffix: str) -> Path:
    name = f"{prefix}_{interval}_seed_{seed}{suffix}"
    return log_dir / name


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_curve(data: dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expected format:
        data[key][epoch_str]["mean_iou"]

    Returns:
        epochs: np.ndarray shape [E]
        values: np.ndarray shape [E]
    """
    if key not in data:
        raise KeyError(f"Key '{key}' not found. Available top-level keys: {list(data.keys())}")

    epoch_map = data[key]
    epochs = sorted(int(k) for k in epoch_map.keys())
    values = np.array([float(epoch_map[str(ep)]["mean_iou"]) for ep in epochs], dtype=np.float64)
    return np.array(epochs, dtype=np.int64), values


def align_curves(curves: List[Tuple[np.ndarray, np.ndarray]], start_epoch: int, end_epoch: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align multiple (epochs, values) curves onto the common epoch range.
    Returns:
        common_epochs: [E]
        stacked_values: [N, E]
    """
    all_epoch_sets = [set(map(int, ep.tolist())) for ep, _ in curves]
    common = sorted(set.intersection(*all_epoch_sets))
    common = [ep for ep in common if ep >= start_epoch and (end_epoch is None or ep <= end_epoch)]

    if not common:
        raise ValueError("No common epoch range found across logs.")

    common_epochs = np.array(common, dtype=np.int64)
    stacked = []

    for epochs, values in curves:
        lookup = {int(ep): float(v) for ep, v in zip(epochs, values)}
        stacked.append([lookup[ep] for ep in common])

    return common_epochs, np.array(stacked, dtype=np.float64)


def mean_std(stacked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)


def find_peak_epoch(epochs: np.ndarray, values: np.ndarray) -> Tuple[int, float]:
    idx = int(np.argmax(values))
    return int(epochs[idx]), float(values[idx])


def format_arrow_text(name: str, epoch: int, value: float) -> str:
    return f"Best {name}\nEpoch {epoch}: {value:.1f}"


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)

    seed_data: Dict[int, dict] = {}
    missing: List[Path] = []

    for seed in args.seeds:
        path = build_file_path(log_dir, args.prefix, args.interval, seed, args.suffix)
        if not path.exists():
            missing.append(path)
            continue
        seed_data[seed] = load_json(path)

    if not seed_data:
        raise FileNotFoundError(
            "No matching log files found.\n"
            + "\n".join(str(p) for p in missing)
        )

    if missing:
        print("Warning: some requested seed files were not found:")
        for p in missing:
            print(f"  - {p}")

    source_curves = []
    city_curves = []
    bdd_curves = []

    for seed in sorted(seed_data):
        data = seed_data[seed]
        source_curves.append(extract_curve(data, args.source_key))
        city_curves.append(extract_curve(data, args.city_key))
        bdd_curves.append(extract_curve(data, args.bdd_key))

    epochs, source_stack = align_curves(source_curves, args.start_epoch, args.end_epoch)
    _, city_stack = align_curves(city_curves, args.start_epoch, args.end_epoch)
    _, bdd_stack = align_curves(bdd_curves, args.start_epoch, args.end_epoch)

    source_mean, source_std = mean_std(source_stack)
    city_mean, city_std = mean_std(city_stack)
    bdd_mean, bdd_std = mean_std(bdd_stack)

    source_peak_epoch, source_peak_value = find_peak_epoch(epochs, source_mean)
    city_peak_epoch, city_peak_value = find_peak_epoch(epochs, city_mean)
    bdd_peak_epoch, bdd_peak_value = find_peak_epoch(epochs, bdd_mean)

    corr_city = float(np.corrcoef(source_mean, city_mean)[0, 1])
    corr_bdd = float(np.corrcoef(source_mean, bdd_mean)[0, 1])

    # Styling
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
    })

    c_source = "#1f77b4"
    c_city = "#ff7f0e"
    c_bdd = "#2ca02c"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)

    # -------------------------------
    # Panel (a): Target fluctuations
    # -------------------------------
    ax = axes[0]

    if args.show_seed0 and 0 in seed_data:
        _, city0 = extract_curve(seed_data[0], args.city_key)
        _, bdd0 = extract_curve(seed_data[0], args.bdd_key)
        ax.plot(epochs, city0[:len(epochs)], color=c_city, alpha=0.20, linewidth=1.2)
        ax.plot(epochs, bdd0[:len(epochs)], color=c_bdd, alpha=0.20, linewidth=1.2)

    ax.plot(epochs, city_mean, color=c_city, marker="o", markersize=3.5, linewidth=2.2, label="Cityscapes (target)")
    ax.fill_between(epochs, city_mean - city_std, city_mean + city_std, color=c_city, alpha=0.14)

    ax.plot(epochs, bdd_mean, color=c_bdd, marker="o", markersize=3.5, linewidth=2.2, label="BDD100K (target)")
    ax.fill_between(epochs, bdd_mean - bdd_std, bdd_mean + bdd_std, color=c_bdd, alpha=0.14)

    # Light fluctuation region: later training
    fluct_start = max(args.start_epoch + 7, int(epochs[0] + 7))
    ax.axvspan(fluct_start, int(epochs[-1]), color="#4c78a8", alpha=0.06)
    ax.text((fluct_start + int(epochs[-1])) / 2, max(city_mean.max(), bdd_mean.max()) + 1.0,
            "Fluctuation region", ha="center", va="bottom", color="#2f5f8f")

    ax.scatter([city_peak_epoch], [city_peak_value], color=c_city, s=45, zorder=5)
    ax.scatter([bdd_peak_epoch], [bdd_peak_value], color=c_bdd, s=45, zorder=5)

    ax.annotate(
        format_arrow_text("Cityscapes", city_peak_epoch, city_peak_value),
        xy=(city_peak_epoch, city_peak_value),
        xytext=(city_peak_epoch + 2.0, city_peak_value + 2.8),
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )
    ax.annotate(
        format_arrow_text("BDD100K", bdd_peak_epoch, bdd_peak_value),
        xy=(bdd_peak_epoch, bdd_peak_value),
        xytext=(bdd_peak_epoch + 2.0, bdd_peak_value - 5.5),
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )

    ax.set_title("(a) Target-domain mIoU fluctuates across checkpoints")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("mIoU (%)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(int(epochs[0]), int(epochs[-1]))

    ymin = min(bdd_mean.min(), city_mean.min()) - 2.5
    ymax = max(bdd_mean.max(), city_mean.max()) + 4.5
    ax.set_ylim(ymin, ymax)

    # -------------------------------
    # Panel (b): Source vs target mismatch
    # -------------------------------
    ax = axes[1]

    if args.show_seed0 and 0 in seed_data:
        _, source0 = extract_curve(seed_data[0], args.source_key)
        _, city0 = extract_curve(seed_data[0], args.city_key)
        _, bdd0 = extract_curve(seed_data[0], args.bdd_key)
        ax.plot(epochs, source0[:len(epochs)], color=c_source, alpha=0.18, linewidth=1.2)
        ax.plot(epochs, city0[:len(epochs)], color=c_city, alpha=0.18, linewidth=1.2)
        ax.plot(epochs, bdd0[:len(epochs)], color=c_bdd, alpha=0.18, linewidth=1.2)

    ax.plot(epochs, source_mean, color=c_source, marker="o", markersize=3.5, linewidth=2.2, label=f"{args.source_key.upper()} validation (source)")
    ax.fill_between(epochs, source_mean - source_std, source_mean + source_std, color=c_source, alpha=0.12)

    ax.plot(epochs, city_mean, color=c_city, marker="o", markersize=3.5, linewidth=2.0, label="Cityscapes (target)")
    ax.fill_between(epochs, city_mean - city_std, city_mean + city_std, color=c_city, alpha=0.12)

    ax.plot(epochs, bdd_mean, color=c_bdd, marker="o", markersize=3.5, linewidth=2.0, label="BDD100K (target)")
    ax.fill_between(epochs, bdd_mean - bdd_std, bdd_mean + bdd_std, color=c_bdd, alpha=0.12)

    ax.axvline(city_peak_epoch, color=c_city, linestyle="--", linewidth=1.3, alpha=0.9)
    ax.axvline(source_peak_epoch, color=c_source, linestyle="--", linewidth=1.3, alpha=0.9)

    ax.scatter([source_peak_epoch], [source_peak_value], color=c_source, s=55, zorder=5)
    ax.scatter([city_peak_epoch], [city_peak_value], color=c_city, s=45, zorder=5)
    ax.scatter([bdd_peak_epoch], [bdd_peak_value], color=c_bdd, s=45, zorder=5)

    ax.annotate(
        f"Best target checkpoint\nEpoch {city_peak_epoch}",
        xy=(city_peak_epoch, city_peak_value),
        xytext=(max(int(epochs[0]) + 1, city_peak_epoch - 6), city_peak_value + 10.0),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        color=c_city,
    )

    ax.annotate(
        f"Best source validation\nEpoch {source_peak_epoch}: {source_peak_value:.1f}",
        xy=(source_peak_epoch, source_peak_value),
        xytext=(max(int(epochs[0]) + 10, source_peak_epoch - 8), source_peak_value - 8.0),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        color="black",
    )

    box_text = (
        "A checkpoint that looks best on the\n"
        "source domain is not necessarily\n"
        "best on the target domain.\n\n"
        f"Pearson r(source,target):\n"
        f"Cityscapes = {corr_city:.2f}\n"
        f"BDD100K = {corr_bdd:.2f}"
    )
    ax.text(
        0.05, 0.06, box_text,
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.5"),
        va="bottom",
    )

    ax.set_title("(b) Best source validation ≠ best target checkpoint")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("mIoU (%)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(int(epochs[0]), int(epochs[-1]))

    ymin2 = min(bdd_mean.min(), city_mean.min(), source_mean.min()) - 3.0
    ymax2 = max(bdd_mean.max(), city_mean.max(), source_mean.max()) + 6.0
    ax.set_ylim(ymin2, ymax2)

    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"Saved figure to: {args.out}")


if __name__ == "__main__":
    main()
