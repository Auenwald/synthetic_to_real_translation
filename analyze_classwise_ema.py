#!/usr/bin/env python3
"""
Class-wise EMA stability analysis for semantic segmentation logs.

This script compares No EMA vs. EMA on a class-wise level.

It computes, per class and target domain:
- temporal mean IoU over epochs
- temporal std over epochs
- TFR = Target Fluctuation Range = max IoU - min IoU over epochs
- TFR reduction = TFR_NoEMA - TFR_EMA

Positive TFR reduction means EMA reduced temporal fluctuation.

Example usage for your SYNTHIA logs:

python analyze_classwise_ema.py \
  --log-dir ./logs_diss \
  --log-pattern "synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_*_new.json" \
  --outdir ./classwise_analysis_synthia_ema0995_interval20 \
  --pairs cityscapes:cityscapes-ema bdd:bdd-ema \
  --epoch-start 6 \
  --epoch-end 30
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Final 16-class train-ID order used in the logs.
CLASS_NAMES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "sky",
    10: "person",
    11: "rider",
    12: "car",
    13: "bus",
    14: "motorcycle",
    15: "bicycle",
}


def extract_seed_from_filename(path: Path) -> int:
    """Extracts seed from filenames containing '_seed_<number>'."""
    match = re.search(r"_seed_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not extract seed from filename: {path.name}")
    return int(match.group(1))


def find_logs(log_dir: Path, log_pattern: str) -> List[Path]:
    """Finds and sorts log files by seed."""
    logs = sorted(log_dir.glob(log_pattern), key=extract_seed_from_filename)

    if not logs:
        raise FileNotFoundError(
            f"No logs found in '{log_dir}' with pattern '{log_pattern}'."
        )

    return logs


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_pairs(pair_args: List[str]) -> List[Tuple[str, str]]:
    """
    Parse pairs such as:
        cityscapes:cityscapes-ema
        bdd:bdd-ema
    """
    pairs = []

    for item in pair_args:
        if ":" not in item:
            raise ValueError(
                f"Invalid pair '{item}'. Expected format: noema_key:ema_key"
            )

        noema, ema = item.split(":", maxsplit=1)
        pairs.append((noema.strip(), ema.strip()))

    return pairs


def get_epochs(data: dict, key: str, epoch_start: int, epoch_end: int) -> List[int]:
    """Returns sorted epochs available for a given key in the selected range."""
    if key not in data:
        raise KeyError(
            f"Key '{key}' not found in log. Available keys: {list(data.keys())}"
        )

    epochs = []
    for e in data[key].keys():
        try:
            e_int = int(e)
        except ValueError:
            continue

        if epoch_start <= e_int <= epoch_end:
            epochs.append(e_int)

    epochs = sorted(epochs)

    if not epochs:
        raise ValueError(
            f"No epochs found for key '{key}' in range {epoch_start}--{epoch_end}."
        )

    return epochs


def extract_class_series(
    data: dict,
    key: str,
    class_id: int,
    epoch_start: int,
    epoch_end: int,
) -> np.ndarray:
    """
    Extracts a class IoU series over epochs for one config/target/class.
    """
    epochs = get_epochs(data, key, epoch_start, epoch_end)
    values = []

    for e in epochs:
        epoch_entry = data[key][str(e)]
        per_class = epoch_entry.get("per_class_iou")

        if per_class is None:
            raise KeyError(f"Missing 'per_class_iou' for key '{key}', epoch {e}.")

        class_key = str(class_id)

        if class_key not in per_class:
            raise KeyError(
                f"Missing class '{class_key}' for key '{key}', epoch {e}."
            )

        values.append(float(per_class[class_key]))

    return np.asarray(values, dtype=float)


def compute_temporal_metrics(values: np.ndarray) -> Dict[str, float]:
    """
    Computes temporal metrics over epochs for one seed/config/target/class.
    """
    return {
        "epoch_mean": float(np.mean(values)),
        "epoch_std": float(np.std(values, ddof=0)),
        "tfr": float(np.max(values) - np.min(values)),
        "epoch_min": float(np.min(values)),
        "epoch_max": float(np.max(values)),
    }


def build_per_seed_metrics(
    logs: List[Path],
    pairs: List[Tuple[str, str]],
    epoch_start: int,
    epoch_end: int,
) -> pd.DataFrame:
    """
    Computes temporal metrics for every seed, target, config, and class.
    """
    rows = []

    for path in logs:
        seed = extract_seed_from_filename(path)
        data = load_json(path)

        for noema_key, ema_key in pairs:
            for config_name, key in [("No EMA", noema_key), ("EMA", ema_key)]:
                for class_id, class_name in CLASS_NAMES.items():
                    values = extract_class_series(
                        data=data,
                        key=key,
                        class_id=class_id,
                        epoch_start=epoch_start,
                        epoch_end=epoch_end,
                    )

                    metrics = compute_temporal_metrics(values)

                    rows.append(
                        {
                            "seed": seed,
                            "log_file": path.name,
                            "target": noema_key,
                            "config": config_name,
                            "class_id": class_id,
                            "class_name": class_name,
                            **metrics,
                        }
                    )

    return pd.DataFrame(rows)


def aggregate_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates temporal metrics across seeds.
    """
    metric_cols = ["epoch_mean", "epoch_std", "tfr", "epoch_min", "epoch_max"]

    agg = (
        df.groupby(["target", "config", "class_id", "class_name"], as_index=False)
        .agg(
            **{f"{metric}_mean": (metric, "mean") for metric in metric_cols},
            **{f"{metric}_seed_std": (metric, "std") for metric in metric_cols},
        )
    )

    return agg


def compute_ema_deltas(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Computes EMA - No EMA deltas based on already aggregated metrics.

    Note:
        delta_tfr = TFR_EMA - TFR_NoEMA
        Negative delta_tfr means EMA reduced fluctuation.
    """
    noema = agg[agg["config"] == "No EMA"].copy()
    ema = agg[agg["config"] == "EMA"].copy()

    merge_cols = ["target", "class_id", "class_name"]

    merged = noema.merge(
        ema,
        on=merge_cols,
        suffixes=("_noema", "_ema"),
    )

    rows = []

    for _, r in merged.iterrows():
        rows.append(
            {
                "target": r["target"],
                "class_id": r["class_id"],
                "class_name": r["class_name"],
                "epoch_mean_noema": r["epoch_mean_mean_noema"],
                "epoch_mean_ema": r["epoch_mean_mean_ema"],
                "delta_epoch_mean": (
                    r["epoch_mean_mean_ema"] - r["epoch_mean_mean_noema"]
                ),
                "epoch_std_noema": r["epoch_std_mean_noema"],
                "epoch_std_ema": r["epoch_std_mean_ema"],
                "delta_epoch_std": (
                    r["epoch_std_mean_ema"] - r["epoch_std_mean_noema"]
                ),
                "tfr_noema": r["tfr_mean_noema"],
                "tfr_ema": r["tfr_mean_ema"],
                "delta_tfr": r["tfr_mean_ema"] - r["tfr_mean_noema"],
                "tfr_noema_seed_std": r["tfr_seed_std_noema"],
                "tfr_ema_seed_std": r["tfr_seed_std_ema"],
            }
        )

    return pd.DataFrame(rows)


def compute_ema_reductions_per_seed(per_seed: pd.DataFrame) -> pd.DataFrame:
    """
    Computes EMA reductions per seed.

    TFR reduction:
        TFR_NoEMA - TFR_EMA

    Positive values indicate stabilization by EMA.
    """
    noema = per_seed[per_seed["config"] == "No EMA"].copy()
    ema = per_seed[per_seed["config"] == "EMA"].copy()

    merge_cols = ["seed", "target", "class_id", "class_name"]

    merged = noema.merge(
        ema,
        on=merge_cols,
        suffixes=("_noema", "_ema"),
    )

    rows = []

    for _, r in merged.iterrows():
        rows.append(
            {
                "seed": r["seed"],
                "target": r["target"],
                "class_id": r["class_id"],
                "class_name": r["class_name"],
                "tfr_noema": r["tfr_noema"],
                "tfr_ema": r["tfr_ema"],
                "tfr_reduction": r["tfr_noema"] - r["tfr_ema"],
                "epoch_std_noema": r["epoch_std_noema"],
                "epoch_std_ema": r["epoch_std_ema"],
                "epoch_std_reduction": r["epoch_std_noema"] - r["epoch_std_ema"],
                "epoch_mean_noema": r["epoch_mean_noema"],
                "epoch_mean_ema": r["epoch_mean_ema"],
                "epoch_mean_change": r["epoch_mean_ema"] - r["epoch_mean_noema"],
            }
        )

    return pd.DataFrame(rows)


def aggregate_reductions_for_table(reductions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates class-wise reductions over seeds.

    This table is the most useful one for reporting:
    - mean ± std TFR reduction over seeds
    - mean ± std epoch-std reduction over seeds
    """
    agg = (
        reductions
        .groupby(["target", "class_id", "class_name"], as_index=False)
        .agg(
            tfr_reduction_mean=("tfr_reduction", "mean"),
            tfr_reduction_std=("tfr_reduction", "std"),
            epoch_std_reduction_mean=("epoch_std_reduction", "mean"),
            epoch_std_reduction_std=("epoch_std_reduction", "std"),
            epoch_mean_change_mean=("epoch_mean_change", "mean"),
            epoch_mean_change_std=("epoch_mean_change", "std"),
            tfr_noema_mean=("tfr_noema", "mean"),
            tfr_noema_std=("tfr_noema", "std"),
            tfr_ema_mean=("tfr_ema", "mean"),
            tfr_ema_std=("tfr_ema", "std"),
            epoch_std_noema_mean=("epoch_std_noema", "mean"),
            epoch_std_noema_std=("epoch_std_noema", "std"),
            epoch_std_ema_mean=("epoch_std_ema", "mean"),
            epoch_std_ema_std=("epoch_std_ema", "std"),
        )
    )

    return agg.sort_values(["target", "class_id"])


def fmt_mean_std(mean: float, std: float, digits: int = 2) -> str:
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def escape_latex_text(text: str) -> str:
    """Small helper for class names in LaTeX tables."""
    return text.replace("_", "\\_")


def export_target_classwise_latex_table(
    agg_reductions: pd.DataFrame,
    target: str,
    outpath: Path,
    caption: str,
    label: str,
) -> None:
    """
    Exports a detailed class-wise LaTeX table for one target domain.

    Columns:
        Class
        TFR No EMA
        TFR EMA
        TFR reduction
        epoch-std reduction
    """
    df = agg_reductions[agg_reductions["target"] == target].copy()
    df = df.sort_values("class_id")

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(f"\\caption{{\\textbf{{{caption}}}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Class & TFR No EMA & TFR EMA & TFR red. & $\\sigma_e$ red. \\\\")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        class_name = escape_latex_text(r["class_name"])

        tfr_noema = fmt_mean_std(r["tfr_noema_mean"], r["tfr_noema_std"])
        tfr_ema = fmt_mean_std(r["tfr_ema_mean"], r["tfr_ema_std"])
        tfr_red = fmt_mean_std(r["tfr_reduction_mean"], r["tfr_reduction_std"])
        std_red = fmt_mean_std(
            r["epoch_std_reduction_mean"], r["epoch_std_reduction_std"]
        )

        lines.append(
            f"{class_name} & {tfr_noema} & {tfr_ema} & {tfr_red} & {std_red} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    outpath.write_text("\n".join(lines), encoding="utf-8")


def export_compact_reduction_latex_table(
    agg_reductions: pd.DataFrame,
    outpath: Path,
    caption: str,
    label: str,
) -> None:
    """
    Exports compact LaTeX table:

        Class | CS TFR red. | BDD TFR red. | Mean

    Each target value is mean ± std over seeds.
    Mean is the mean over target-domain means.
    """
    df = agg_reductions.copy()

    df["tfr_red_str"] = df.apply(
        lambda r: fmt_mean_std(r["tfr_reduction_mean"], r["tfr_reduction_std"]),
        axis=1,
    )

    wide_str = df.pivot(
        index=["class_id", "class_name"],
        columns="target",
        values="tfr_red_str",
    ).reset_index()

    mean_df = (
        df.groupby(["class_id", "class_name"], as_index=False)
        .agg(
            mean_tfr_reduction=("tfr_reduction_mean", "mean"),
        )
    )

    # This is not seed std; it is the spread between target-domain means.
    target_spread = (
        df.groupby(["class_id", "class_name"])["tfr_reduction_mean"]
        .std()
        .reset_index(name="target_spread")
    )

    merged = wide_str.merge(mean_df, on=["class_id", "class_name"])
    merged = merged.merge(target_spread, on=["class_id", "class_name"])
    merged = merged.sort_values("class_id")

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(f"\\caption{{\\textbf{{{caption}}}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Class & CS TFR red. & BDD TFR red. & Mean \\\\")
    lines.append("\\midrule")

    for _, r in merged.iterrows():
        class_name = escape_latex_text(r["class_name"])

        cs = r.get("cityscapes", "--")
        bdd = r.get("bdd", "--")

        mean_val = r["mean_tfr_reduction"]
        spread = r["target_spread"]

        if pd.isna(spread):
            mean_str = f"{mean_val:.2f}"
        else:
            mean_str = f"{mean_val:.2f}$\\pm${spread:.2f}"

        lines.append(f"{class_name} & {cs} & {bdd} & {mean_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    outpath.write_text("\n".join(lines), encoding="utf-8")


def export_summary_latex_table(
    reductions: pd.DataFrame,
    outpath: Path,
    caption: str,
    label: str,
) -> None:
    """
    Exports a small summary table per target domain:
    - number of classes with lower TFR under EMA
    - mean TFR reduction
    - number of classes with lower epoch std under EMA
    """
    rows = []

    for target, group in reductions.groupby("target"):
        # First average per class over seeds
        class_avg = (
            group.groupby(["class_id", "class_name"], as_index=False)
            .agg(
                tfr_reduction=("tfr_reduction", "mean"),
                epoch_std_reduction=("epoch_std_reduction", "mean"),
            )
        )

        num_classes = len(class_avg)
        lower_tfr = int((class_avg["tfr_reduction"] > 0).sum())
        lower_std = int((class_avg["epoch_std_reduction"] > 0).sum())

        rows.append(
            {
                "target": target,
                "lower_tfr": lower_tfr,
                "lower_std": lower_std,
                "num_classes": num_classes,
                "mean_tfr_reduction": class_avg["tfr_reduction"].mean(),
                "mean_epoch_std_reduction": class_avg["epoch_std_reduction"].mean(),
            }
        )

    summary = pd.DataFrame(rows)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(f"\\caption{{\\textbf{{{caption}}}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Target & Classes with lower TFR & Mean TFR red. & Classes with lower $\\sigma_e$ \\\\")
    lines.append("\\midrule")

    for _, r in summary.iterrows():
        target = r["target"]
        lower_tfr = f"{int(r['lower_tfr'])}/{int(r['num_classes'])}"
        lower_std = f"{int(r['lower_std'])}/{int(r['num_classes'])}"
        mean_red = f"{r['mean_tfr_reduction']:.2f}"

        lines.append(f"{target} & {lower_tfr} & {mean_red} & {lower_std} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    outpath.write_text("\n".join(lines), encoding="utf-8")

    return summary


def plot_delta_tfr_heatmap(deltas: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap for delta_tfr = TFR_EMA - TFR_NoEMA.

    Negative values mean EMA reduced fluctuation.
    """
    pivot = deltas.pivot(index="class_name", columns="target", values="delta_tfr")

    ordered_classes = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    pivot = pivot.reindex(ordered_classes)

    fig, ax = plt.subplots(figsize=(7, 6))

    data = pivot.to_numpy()
    vmax = np.nanmax(np.abs(data))

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(r"Class-wise EMA effect: $\Delta$TFR = TFR$_{EMA}$ - TFR$_{NoEMA}$")
    ax.set_xlabel("Target domain")
    ax.set_ylabel("Class")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta$TFR")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_tfr_reduction_heatmap(agg_reductions: pd.DataFrame, outpath: Path) -> None:
    """
    Heatmap for TFR reduction = TFR_NoEMA - TFR_EMA.

    Positive values mean EMA reduced fluctuation.
    This is easier to interpret for thesis figures.
    """
    pivot = agg_reductions.pivot(
        index="class_name",
        columns="target",
        values="tfr_reduction_mean",
    )

    ordered_classes = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    pivot = pivot.reindex(ordered_classes)

    fig, ax = plt.subplots(figsize=(7, 6))

    data = pivot.to_numpy()
    vmax = np.nanmax(np.abs(data))

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = ax.imshow(data, aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(r"Class-wise TFR reduction by EMA")
    ax.set_xlabel("Target domain")
    ax.set_ylabel("Class")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("TFR reduction")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_mean_tfr_reduction(agg_reductions: pd.DataFrame, outpath: Path) -> None:
    """
    Bar plot:
        mean TFR reduction per class over target domains.

    Positive value means EMA reduced TFR.
    """
    summary = (
        agg_reductions.groupby(["class_id", "class_name"], as_index=False)
        .agg(mean_tfr_reduction=("tfr_reduction_mean", "mean"))
        .sort_values("class_id")
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(summary))
    y = summary["mean_tfr_reduction"].to_numpy()

    ax.bar(x, y)
    ax.axhline(0, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["class_name"], rotation=45, ha="right")

    ax.set_ylabel("Mean TFR reduction")
    ax.set_title("Average class-wise TFR reduction by EMA")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_tfr_noema_vs_ema(deltas: pd.DataFrame, outpath: Path) -> None:
    """
    Scatter plot:
        x = TFR without EMA
        y = TFR with EMA

    Points below diagonal mean EMA reduced fluctuation.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    x = deltas["tfr_noema"].to_numpy()
    y = deltas["tfr_ema"].to_numpy()

    ax.scatter(x, y)

    max_val = max(np.max(x), np.max(y))
    ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)

    ax.set_xlabel("TFR without EMA")
    ax.set_ylabel("TFR with EMA")
    ax.set_title("Class-wise TFR: No EMA vs. EMA")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def print_summary(reductions: pd.DataFrame) -> pd.DataFrame:
    """
    Prints human-readable class-wise EMA summary.
    """
    print("\n=== Class-wise EMA stability summary ===\n")

    summary_rows = []

    for target, group in reductions.groupby("target"):
        class_avg = (
            group.groupby(["class_id", "class_name"], as_index=False)
            .agg(
                tfr_reduction=("tfr_reduction", "mean"),
                epoch_std_reduction=("epoch_std_reduction", "mean"),
                epoch_mean_change=("epoch_mean_change", "mean"),
            )
        )

        n_classes = len(class_avg)
        n_reduced_tfr = int((class_avg["tfr_reduction"] > 0).sum())
        n_reduced_std = int((class_avg["epoch_std_reduction"] > 0).sum())

        mean_tfr_reduction = class_avg["tfr_reduction"].mean()
        mean_std_reduction = class_avg["epoch_std_reduction"].mean()
        mean_epoch_change = class_avg["epoch_mean_change"].mean()

        summary_rows.append(
            {
                "target": target,
                "classes_with_lower_tfr": n_reduced_tfr,
                "classes_with_lower_epoch_std": n_reduced_std,
                "num_classes": n_classes,
                "mean_tfr_reduction": mean_tfr_reduction,
                "mean_epoch_std_reduction": mean_std_reduction,
                "mean_epoch_mean_change": mean_epoch_change,
            }
        )

        print(f"Target: {target}")
        print(f"  Classes with lower TFR under EMA: {n_reduced_tfr}/{n_classes}")
        print(f"  Mean TFR reduction: {mean_tfr_reduction:.3f}")
        print(f"  Classes with lower epoch-std under EMA: {n_reduced_std}/{n_classes}")
        print(f"  Mean epoch-std reduction: {mean_std_reduction:.3f}")
        print(f"  Mean epoch-mean change: {mean_epoch_change:.3f}")

        best = class_avg.sort_values("tfr_reduction", ascending=False).head(5)
        worst = class_avg.sort_values("tfr_reduction", ascending=True).head(5)

        print("  Strongest TFR reductions:")
        for _, r in best.iterrows():
            print(f"    {r['class_name']}: TFR reduction={r['tfr_reduction']:.3f}")

        print("  Weakest / negative TFR reductions:")
        for _, r in worst.iterrows():
            print(f"    {r['class_name']}: TFR reduction={r['tfr_reduction']:.3f}")

        print()

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs_diss",
        help="Directory containing JSON logs.",
    )

    parser.add_argument(
        "--log-pattern",
        type=str,
        required=True,
        help="Glob pattern for log files.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./classwise_ema_analysis",
        help="Output directory.",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["cityscapes:cityscapes-ema", "bdd:bdd-ema"],
        help="Pairs of keys to compare: noema_key:ema_key.",
    )

    parser.add_argument(
        "--epoch-start",
        type=int,
        default=6,
        help="First epoch used for temporal stability analysis.",
    )

    parser.add_argument(
        "--epoch-end",
        type=int,
        default=30,
        help="Last epoch used for temporal stability analysis.",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logs = find_logs(log_dir, args.log_pattern)
    pairs = parse_pairs(args.pairs)

    print("Found logs:")
    for p in logs:
        print(f"  seed {extract_seed_from_filename(p)}: {p}")

    per_seed = build_per_seed_metrics(
        logs=logs,
        pairs=pairs,
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
    )

    agg = aggregate_across_seeds(per_seed)
    deltas = compute_ema_deltas(agg)

    reductions_per_seed = compute_ema_reductions_per_seed(per_seed)
    agg_reductions = aggregate_reductions_for_table(reductions_per_seed)

    # Save CSV outputs
    per_seed.to_csv(outdir / "classwise_metrics_per_seed.csv", index=False)
    agg.to_csv(outdir / "classwise_metrics_aggregated.csv", index=False)
    deltas.to_csv(outdir / "classwise_ema_deltas.csv", index=False)
    reductions_per_seed.to_csv(
        outdir / "classwise_ema_reductions_per_seed.csv",
        index=False,
    )
    agg_reductions.to_csv(
        outdir / "classwise_ema_reductions_aggregated.csv",
        index=False,
    )

    # Summary
    summary = print_summary(reductions_per_seed)
    summary.to_csv(outdir / "classwise_ema_summary.csv", index=False)

    # Figures
    plot_delta_tfr_heatmap(
        deltas,
        outdir / "classwise_delta_tfr_heatmap.png",
    )

    plot_tfr_reduction_heatmap(
        agg_reductions,
        outdir / "classwise_tfr_reduction_heatmap.png",
    )

    plot_mean_tfr_reduction(
        agg_reductions,
        outdir / "classwise_mean_tfr_reduction.png",
    )

    plot_tfr_noema_vs_ema(
        deltas,
        outdir / "classwise_tfr_noema_vs_ema.png",
    )

    # LaTeX tables
    export_compact_reduction_latex_table(
        agg_reductions=agg_reductions,
        outpath=outdir / "table_classwise_compact.tex",
        caption=(
            "Class-wise TFR reduction induced by EMA. "
            "Positive values indicate reduced temporal fluctuation under EMA."
        ),
        label="tab:classwise-ema-compact",
    )

    export_summary_latex_table(
        reductions=reductions_per_seed,
        outpath=outdir / "table_classwise_summary.tex",
        caption=(
            "Summary of class-wise EMA effects on temporal stability. "
            "A class is counted as improved if its mean TFR over seeds is lower under EMA."
        ),
        label="tab:classwise-ema-summary",
    )

    # Detailed per-target tables, if those targets exist
    available_targets = set(agg_reductions["target"].unique())

    if "cityscapes" in available_targets:
        export_target_classwise_latex_table(
            agg_reductions=agg_reductions,
            target="cityscapes",
            outpath=outdir / "table_classwise_cityscapes.tex",
            caption=(
                "Class-wise EMA effect on temporal stability for Cityscapes. "
                "TFR reduction is defined as TFR without EMA minus TFR with EMA."
            ),
            label="tab:classwise-ema-cityscapes",
        )

    if "bdd" in available_targets:
        export_target_classwise_latex_table(
            agg_reductions=agg_reductions,
            target="bdd",
            outpath=outdir / "table_classwise_bdd.tex",
            caption=(
                "Class-wise EMA effect on temporal stability for BDD100K. "
                "TFR reduction is defined as TFR without EMA minus TFR with EMA."
            ),
            label="tab:classwise-ema-bdd",
        )

    print(f"\nSaved results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()