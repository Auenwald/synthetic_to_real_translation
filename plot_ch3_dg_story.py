
#!/usr/bin/env python3
"""
Create Chapter-3 figures for DG segmentation instability from epoch-wise JSON logs.

Story-oriented output:
1) Figure A: target-domain fluctuation
   - left: one representative seed (default: seed 0)
   - right: mean ± std across seeds
   - curves: Cityscapes and BDD100K target-domain mIoU across epochs

2) Figure B: proxy validation failure
   - source validation vs target-domain curves
   - shows that best source validation checkpoint does not necessarily match
     the best target-domain checkpoint
   - default: mean curves across seeds, optional seed-0 overlay

Supported source domains:
- SYNTHIA
- GTA5

Expected log layout:
./logs_diss/
    synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_0_new.json
    ...
    synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_4_new.json

    gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_0.json
    ...
    gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_4.json

Example:
python plot_ch3_dg_story.py \
  --log_dir ./logs_diss \
  --prefix synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval \
  --interval 20 \
  --suffix _new.json \
  --source_key synthia \
  --out_prefix synthia_ch3

This writes:
- synthia_ch3_target_fluctuation.png
- synthia_ch3_proxy_failure.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Chapter-3 DG instability figures from JSON logs.")
    p.add_argument("--log_dir", type=str, default="./logs_diss", help="Directory with JSON logs.")
    p.add_argument("--prefix", type=str, required=True, help="Filename prefix before '_<interval>_seed_<seed><suffix>'.")
    p.add_argument("--interval", type=int, default=20, help="Averaging interval encoded in the filename.")
    p.add_argument("--suffix", type=str, default=".json", help="Suffix after the seed, e.g. '.json' or '_new.json'.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds to use.")
    p.add_argument("--rep_seed", type=int, default=0, help="Representative seed for the left panel of Figure A.")
    p.add_argument("--source_key", type=str, required=True, choices=["synthia", "gta5"], help="Top-level JSON key for source validation.")
    p.add_argument("--city_key", type=str, default="cityscapes", help="Top-level JSON key for Cityscapes.")
    p.add_argument("--bdd_key", type=str, default="bdd", help="Top-level JSON key for BDD100K.")
    p.add_argument("--start_epoch", type=int, default=1, help="First epoch to include.")
    p.add_argument("--end_epoch", type=int, default=None, help="Last epoch to include.")
    p.add_argument("--overlay_seed0_in_proxy", action="store_true", help="Overlay representative seed in Figure B.")
    p.add_argument("--out_prefix", type=str, default="ch3_dg", help="Output prefix.")
    return p.parse_args()


def build_path(log_dir: Path, prefix: str, interval: int, seed: int, suffix: str) -> Path:
    return log_dir / f"{prefix}_{interval}_seed_{seed}{suffix}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_curve(data: dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
    if key not in data:
        raise KeyError(f"Missing key '{key}'. Available keys: {list(data.keys())}")
    epoch_map = data[key]
    epochs = sorted(int(k) for k in epoch_map.keys())
    values = np.array([float(epoch_map[str(ep)]["mean_iou"]) for ep in epochs], dtype=np.float64)
    return np.array(epochs, dtype=np.int64), values


def align_curves(curves: List[Tuple[np.ndarray, np.ndarray]], start_epoch: int, end_epoch: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    common = sorted(set.intersection(*[set(map(int, ep.tolist())) for ep, _ in curves]))
    common = [ep for ep in common if ep >= start_epoch and (end_epoch is None or ep <= end_epoch)]
    if not common:
        raise ValueError("No common epochs found across the selected logs.")

    common_epochs = np.array(common, dtype=np.int64)
    stacked = []
    for epochs, values in curves:
        lookup = {int(ep): float(v) for ep, v in zip(epochs, values)}
        stacked.append([lookup[ep] for ep in common_epochs])
    return common_epochs, np.array(stacked, dtype=np.float64)


def compute_mean_std(stacked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)


def find_peak(epochs: np.ndarray, values: np.ndarray) -> Tuple[int, float]:
    idx = int(np.argmax(values))
    return int(epochs[idx]), float(values[idx])


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
    })


def safe_y_limits(*arrays: np.ndarray, pad_low: float = 2.0, pad_high: float = 3.0) -> Tuple[float, float]:
    ymin = min(float(np.min(a)) for a in arrays) - pad_low
    ymax = max(float(np.max(a)) for a in arrays) + pad_high
    return ymin, ymax


def load_all_seed_data(args: argparse.Namespace) -> Dict[int, dict]:
    log_dir = Path(args.log_dir)
    seed_data: Dict[int, dict] = {}
    missing = []

    for seed in args.seeds:
        path = build_path(log_dir, args.prefix, args.interval, seed, args.suffix)
        if not path.exists():
            missing.append(path)
            continue
        seed_data[seed] = load_json(path)

    if not seed_data:
        raise FileNotFoundError(
            "No matching log files found.\n" + "\n".join(str(p) for p in missing)
        )

    if missing:
        print("Warning: missing files:")
        for p in missing:
            print(f"  - {p}")

    return seed_data


def get_aligned_stacks(seed_data: Dict[int, dict], args: argparse.Namespace):
    source_curves, city_curves, bdd_curves = [], [], []

    for seed in sorted(seed_data):
        source_curves.append(extract_curve(seed_data[seed], args.source_key))
        city_curves.append(extract_curve(seed_data[seed], args.city_key))
        bdd_curves.append(extract_curve(seed_data[seed], args.bdd_key))

    epochs, source_stack = align_curves(source_curves, args.start_epoch, args.end_epoch)
    _, city_stack = align_curves(city_curves, args.start_epoch, args.end_epoch)
    _, bdd_stack = align_curves(bdd_curves, args.start_epoch, args.end_epoch)

    return epochs, source_stack, city_stack, bdd_stack


def plot_target_fluctuation(seed_data: Dict[int, dict], args: argparse.Namespace, epochs: np.ndarray, city_stack: np.ndarray, bdd_stack: np.ndarray) -> Path:
    setup_style()

    city_mean, city_std = compute_mean_std(city_stack)
    bdd_mean, bdd_std = compute_mean_std(bdd_stack)

    rep_seed = args.rep_seed if args.rep_seed in seed_data else sorted(seed_data.keys())[0]
    rep_city_epochs, rep_city_vals = extract_curve(seed_data[rep_seed], args.city_key)
    rep_bdd_epochs, rep_bdd_vals = extract_curve(seed_data[rep_seed], args.bdd_key)

    # align rep seed to common epochs
    rep_city_lookup = {int(ep): float(v) for ep, v in zip(rep_city_epochs, rep_city_vals)}
    rep_bdd_lookup = {int(ep): float(v) for ep, v in zip(rep_bdd_epochs, rep_bdd_vals)}
    rep_city = np.array([rep_city_lookup[int(ep)] for ep in epochs], dtype=np.float64)
    rep_bdd = np.array([rep_bdd_lookup[int(ep)] for ep in epochs], dtype=np.float64)

    city_peak_rep_ep, city_peak_rep_val = find_peak(epochs, rep_city)
    bdd_peak_rep_ep, bdd_peak_rep_val = find_peak(epochs, rep_bdd)

    city_peak_mean_ep, city_peak_mean_val = find_peak(epochs, city_mean)
    bdd_peak_mean_ep, bdd_peak_mean_val = find_peak(epochs, bdd_mean)

    c_city = "#ff7f0e"
    c_bdd = "#2ca02c"

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)

    # left: representative seed
    ax = axes[0]
    ax.plot(epochs, rep_city, color=c_city, marker="o", markersize=3.5, linewidth=2.1, label="Cityscapes")
    ax.plot(epochs, rep_bdd, color=c_bdd, marker="o", markersize=3.5, linewidth=2.1, label="BDD100K")

    fluct_start = max(int(epochs[0]) + 7, 8)
    ax.axvspan(fluct_start, int(epochs[-1]), color="#4c78a8", alpha=0.06)
    ax.text((fluct_start + int(epochs[-1])) / 2, max(rep_city.max(), rep_bdd.max()) + 0.8,
            "Fluctuation region", ha="center", va="bottom", color="#2f5f8f")

    ax.scatter([city_peak_rep_ep], [city_peak_rep_val], color=c_city, s=45, zorder=5)
    ax.scatter([bdd_peak_rep_ep], [bdd_peak_rep_val], color=c_bdd, s=45, zorder=5)

    ax.annotate(
        f"Best Cityscapes\nEpoch {city_peak_rep_ep}: {city_peak_rep_val:.1f}",
        xy=(city_peak_rep_ep, city_peak_rep_val),
        xytext=(city_peak_rep_ep + 2.0, city_peak_rep_val + 2.7),
        arrowprops=dict(arrowstyle="->", lw=1.1),
    )
    ax.annotate(
        f"Best BDD100K\nEpoch {bdd_peak_rep_ep}: {bdd_peak_rep_val:.1f}",
        xy=(bdd_peak_rep_ep, bdd_peak_rep_val),
        xytext=(bdd_peak_rep_ep + 2.0, bdd_peak_rep_val - 5.2),
        arrowprops=dict(arrowstyle="->", lw=1.1),
    )

    ax.set_title(f"(a) One representative run (seed {rep_seed})")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("mIoU (%)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(int(epochs[0]), int(epochs[-1]))
    ax.set_ylim(*safe_y_limits(rep_city, rep_bdd, pad_low=2.0, pad_high=4.0))

    # right: mean ± std
    ax = axes[1]
    ax.plot(epochs, city_mean, color=c_city, marker="o", markersize=3.5, linewidth=2.2, label="Cityscapes")
    ax.fill_between(epochs, city_mean - city_std, city_mean + city_std, color=c_city, alpha=0.14)

    ax.plot(epochs, bdd_mean, color=c_bdd, marker="o", markersize=3.5, linewidth=2.2, label="BDD100K")
    ax.fill_between(epochs, bdd_mean - bdd_std, bdd_mean + bdd_std, color=c_bdd, alpha=0.14)

    ax.scatter([city_peak_mean_ep], [city_peak_mean_val], color=c_city, s=45, zorder=5)
    ax.scatter([bdd_peak_mean_ep], [bdd_peak_mean_val], color=c_bdd, s=45, zorder=5)

    ax.annotate(
        f"Peak mean Cityscapes\nEpoch {city_peak_mean_ep}: {city_peak_mean_val:.1f}",
        xy=(city_peak_mean_ep, city_peak_mean_val),
        xytext=(city_peak_mean_ep + 2.0, city_peak_mean_val + 2.8),
        arrowprops=dict(arrowstyle="->", lw=1.1),
    )
    ax.annotate(
        f"Peak mean BDD100K\nEpoch {bdd_peak_mean_ep}: {bdd_peak_mean_val:.1f}",
        xy=(bdd_peak_mean_ep, bdd_peak_mean_val),
        xytext=(bdd_peak_mean_ep + 2.0, bdd_peak_mean_val - 5.0),
        arrowprops=dict(arrowstyle="->", lw=1.1),
    )

    ax.set_title("(b) Mean ± std across seeds")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("mIoU (%)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(int(epochs[0]), int(epochs[-1]))
    ax.set_ylim(*safe_y_limits(city_mean + city_std, bdd_mean + bdd_std, city_mean - city_std, bdd_mean - bdd_std, pad_low=2.0, pad_high=4.0))

    fig.suptitle("Target-domain performance fluctuates across checkpoints", y=1.02)
    out = Path(f"{args.out_prefix}_target_fluctuation.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_proxy_failure(seed_data: Dict[int, dict], args: argparse.Namespace, epochs: np.ndarray, source_stack: np.ndarray, city_stack: np.ndarray, bdd_stack: np.ndarray) -> Path:
    setup_style()

    source_mean, source_std = compute_mean_std(source_stack)
    city_mean, city_std = compute_mean_std(city_stack)
    bdd_mean, bdd_std = compute_mean_std(bdd_stack)

    source_peak_ep, source_peak_val = find_peak(epochs, source_mean)
    city_peak_ep, city_peak_val = find_peak(epochs, city_mean)
    bdd_peak_ep, bdd_peak_val = find_peak(epochs, bdd_mean)

    corr_city = float(np.corrcoef(source_mean, city_mean)[0, 1])
    corr_bdd = float(np.corrcoef(source_mean, bdd_mean)[0, 1])

    c_source = "#1f77b4"
    c_city = "#ff7f0e"
    c_bdd = "#2ca02c"

    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)

    if args.overlay_seed0_in_proxy:
        rep_seed = args.rep_seed if args.rep_seed in seed_data else sorted(seed_data.keys())[0]
        s_ep, s_val = extract_curve(seed_data[rep_seed], args.source_key)
        c_ep, c_val = extract_curve(seed_data[rep_seed], args.city_key)
        b_ep, b_val = extract_curve(seed_data[rep_seed], args.bdd_key)

        s_lookup = {int(ep): float(v) for ep, v in zip(s_ep, s_val)}
        c_lookup = {int(ep): float(v) for ep, v in zip(c_ep, c_val)}
        b_lookup = {int(ep): float(v) for ep, v in zip(b_ep, b_val)}

        s_rep = np.array([s_lookup[int(ep)] for ep in epochs], dtype=np.float64)
        c_rep = np.array([c_lookup[int(ep)] for ep in epochs], dtype=np.float64)
        b_rep = np.array([b_lookup[int(ep)] for ep in epochs], dtype=np.float64)

        ax.plot(epochs, s_rep, color=c_source, alpha=0.18, linewidth=1.2)
        ax.plot(epochs, c_rep, color=c_city, alpha=0.18, linewidth=1.2)
        ax.plot(epochs, b_rep, color=c_bdd, alpha=0.18, linewidth=1.2)

    ax.plot(epochs, source_mean, color=c_source, marker="o", markersize=3.5, linewidth=2.2, label=f"{args.source_key.upper()} validation (source)")
    ax.fill_between(epochs, source_mean - source_std, source_mean + source_std, color=c_source, alpha=0.12)

    ax.plot(epochs, city_mean, color=c_city, marker="o", markersize=3.5, linewidth=2.0, label="Cityscapes (target)")
    ax.fill_between(epochs, city_mean - city_std, city_mean + city_std, color=c_city, alpha=0.12)

    ax.plot(epochs, bdd_mean, color=c_bdd, marker="o", markersize=3.5, linewidth=2.0, label="BDD100K (target)")
    ax.fill_between(epochs, bdd_mean - bdd_std, bdd_mean + bdd_std, color=c_bdd, alpha=0.12)

    ax.axvline(source_peak_ep, color=c_source, linestyle="--", linewidth=1.3, alpha=0.9)
    ax.axvline(city_peak_ep, color=c_city, linestyle="--", linewidth=1.3, alpha=0.9)

    ax.annotate(
        f"Best source validation\nEpoch {source_peak_ep}: {source_peak_val:.1f}",
        xy=(source_peak_ep, source_peak_val),
        xytext=(max(int(epochs[0]) + 10, source_peak_ep - 8), source_peak_val - 8.0),
        arrowprops=dict(arrowstyle="->", lw=1.1),
    )
    ax.annotate(
        f"Best target checkpoint\nEpoch {city_peak_ep}",
        xy=(city_peak_ep, city_peak_val),
        xytext=(max(int(epochs[0]) + 1, city_peak_ep - 5), city_peak_val + 10.0),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        color=c_city,
    )

    text = (
        "A checkpoint that looks best on the\n"
        "source domain is not necessarily best\n"
        "on the target domain.\n\n"
        f"Pearson r(source,target):\n"
        f"Cityscapes = {corr_city:.2f}\n"
        f"BDD100K = {corr_bdd:.2f}"
    )
    ax.text(
        0.05, 0.05, text,
        transform=ax.transAxes,
        fontsize=9.5,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.86, edgecolor="0.5"),
    )

    ax.set_title("Source-domain proxy validation does not reliably predict target performance")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("mIoU (%)")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(int(epochs[0]), int(epochs[-1]))
    ax.set_ylim(*safe_y_limits(source_mean + source_std, city_mean + city_std, bdd_mean + bdd_std, source_mean - source_std, city_mean - city_std, bdd_mean - bdd_std, pad_low=3.0, pad_high=6.0))

    out = Path(f"{args.out_prefix}_proxy_failure.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main() -> None:
    args = parse_args()
    seed_data = load_all_seed_data(args)
    epochs, source_stack, city_stack, bdd_stack = get_aligned_stacks(seed_data, args)

    plot_target_fluctuation(seed_data, args, epochs, city_stack, bdd_stack)
    plot_proxy_failure(seed_data, args, epochs, source_stack, city_stack, bdd_stack)


if __name__ == "__main__":
    main()
