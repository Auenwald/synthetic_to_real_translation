import json
import os
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

TABLE_DIR = "./tables_early_adaptation"
os.makedirs(TABLE_DIR, exist_ok=True)

START_EPOCH = 1
EARLY_WINDOWS = [5, 10]
STD_DDOF = 0

BASELINE_NAME = "Single Encoder"

TARGET_DATASETS = [
    ("cityscapes", "Cityscapes", "No"),
    ("cityscapes-ema", "Cityscapes", "Yes"),
    ("bdd", "BDD", "No"),
    ("bdd-ema", "BDD", "Yes"),
]


# ---------------------------------------------------------
# LOG PATHS
# ---------------------------------------------------------

single_encoder_paths = [
    "./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_0_new.json",
    "./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_1_new.json",
    "./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_2_new.json",
    "./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_3_new.json",
    "./logs_diss/synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0995_averaging_interval_20_seed_4_new.json",
]

dual_encoder_paths = [
    "./logs_diss/v2_edge_branched_ade_seed0.json",
    "./logs_diss/v2_edge_branched_ade_seed1.json",
    "./logs_diss/v2_edge_branched_ade_seed2.json",
]

METHODS = {
    "Single Encoder": single_encoder_paths,
    "Dual Encoder": dual_encoder_paths,
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def safe_std(values, ddof=0):
    values = np.asarray(values, dtype=float)
    if len(values) <= ddof:
        return 0.0
    return float(np.std(values, ddof=ddof))


def load_logs(paths):
    logs = []

    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: file does not exist: {p}")
            continue

        with open(p, "r") as f:
            logs.append(json.load(f))

    return logs


def load_all_methods(methods):
    loaded = {}

    for method_name, paths in methods.items():
        loaded[method_name] = load_logs(paths)
        print(f"{method_name}: loaded {len(loaded[method_name])} logs")

    return loaded


def get_epoch_series(log, dataset, start_epoch=1, end_epoch=10):
    if dataset not in log:
        return np.array([]), np.array([])

    items = []

    for epoch, values in log[dataset].items():
        ep = int(epoch)

        if ep < start_epoch or ep > end_epoch:
            continue

        if "mean_iou" not in values:
            continue

        items.append((ep, float(values["mean_iou"])))

    if len(items) == 0:
        return np.array([]), np.array([])

    items = sorted(items, key=lambda x: x[0])

    epochs = np.array([x[0] for x in items], dtype=int)
    values = np.array([x[1] for x in items], dtype=float)

    return epochs, values


def normalized_auc(epochs, values):
    if len(values) == 0:
        return np.nan

    if len(values) == 1:
        return float(values[0])

    area = np.trapz(values, epochs)
    duration = epochs[-1] - epochs[0]

    if duration == 0:
        return float(values[0])

    return float(area / duration)


def fmt(mean, std, decimals=3):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def compute_seed_metrics(log, dataset, start_epoch=1, early_end_epoch=10):
    epochs, values = get_epoch_series(
        log=log,
        dataset=dataset,
        start_epoch=start_epoch,
        end_epoch=early_end_epoch
    )

    if len(values) == 0:
        return None

    return {
        "early_mean": float(np.mean(values)),
        "early_auc": normalized_auc(epochs, values),
        "early_peak": float(np.max(values)),
    }


def aggregate_metrics(logs, dataset, start_epoch=1, early_end_epoch=10):
    per_seed = []

    for log in logs:
        metrics = compute_seed_metrics(
            log=log,
            dataset=dataset,
            start_epoch=start_epoch,
            early_end_epoch=early_end_epoch
        )

        if metrics is not None:
            per_seed.append(metrics)

    if len(per_seed) == 0:
        return None

    summary = {
        "num_seeds": len(per_seed),
    }

    for key in ["early_mean", "early_auc", "early_peak"]:
        vals = np.array([m[key] for m in per_seed], dtype=float)
        summary[key] = float(np.mean(vals))
        summary[f"{key}_std"] = safe_std(vals, ddof=STD_DDOF)

    return summary


# ---------------------------------------------------------
# TABLE BUILDING
# ---------------------------------------------------------

def build_comparison_rows(loaded_methods):
    rows = []

    for early_end_epoch in EARLY_WINDOWS:
        for dataset_key, dataset_name, ema_label in TARGET_DATASETS:

            summaries = {}

            for method_name, logs in loaded_methods.items():
                summaries[method_name] = aggregate_metrics(
                    logs=logs,
                    dataset=dataset_key,
                    start_epoch=START_EPOCH,
                    early_end_epoch=early_end_epoch
                )

            baseline = summaries.get(BASELINE_NAME)
            dual = summaries.get("Dual Encoder")

            if baseline is None or dual is None:
                print(f"Skipping {dataset_key}, window {START_EPOCH}-{early_end_epoch}: missing data")
                continue

            rows.append({
                "window": f"{START_EPOCH}-{early_end_epoch}",
                "dataset": dataset_name,
                "ema": ema_label,

                "single_n": baseline["num_seeds"],
                "dual_n": dual["num_seeds"],

                "single_early_mean": baseline["early_mean"],
                "single_early_mean_std": baseline["early_mean_std"],
                "dual_early_mean": dual["early_mean"],
                "dual_early_mean_std": dual["early_mean_std"],
                "delta_early_mean": dual["early_mean"] - baseline["early_mean"],

                "single_early_auc": baseline["early_auc"],
                "single_early_auc_std": baseline["early_auc_std"],
                "dual_early_auc": dual["early_auc"],
                "dual_early_auc_std": dual["early_auc_std"],
                "delta_early_auc": dual["early_auc"] - baseline["early_auc"],

                "single_peak": baseline["early_peak"],
                "single_peak_std": baseline["early_peak_std"],
                "dual_peak": dual["early_peak"],
                "dual_peak_std": dual["early_peak_std"],
                "delta_peak": dual["early_peak"] - baseline["early_peak"],
            })

    return rows


# ---------------------------------------------------------
# PRINT TABLES
# ---------------------------------------------------------

# ---------------------------------------------------------
# TERMINAL TABLE PRINTING
# ---------------------------------------------------------

def format_cell(value, width, align="left"):
    value = str(value)

    if len(value) > width:
        value = value[:width - 1] + "…"

    if align == "right":
        return value.rjust(width)

    return value.ljust(width)


def print_separator(widths):
    print("+-" + "-+-".join("-" * w for w in widths) + "-+")


def print_row(values, widths, aligns=None):
    if aligns is None:
        aligns = ["left"] * len(values)

    cells = [
        format_cell(v, w, a)
        for v, w, a in zip(values, widths, aligns)
    ]

    print("| " + " | ".join(cells) + " |")


def fmt_value(mean, std, decimals=2):
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def print_terminal_main_table(rows):
    print("\nMAIN TABLE")
    print("Single Encoder vs Dual Encoder\n")

    headers = [
        "Win",
        "Dataset",
        "EMA",
        "n",
        "Single Mean",
        "Dual Mean",
        "Single AUC",
        "Dual AUC",
        "Single Peak",
        "Dual Peak",
    ]

    widths = [5, 10, 5, 7, 13, 13, 13, 13, 13, 13]
    aligns = [
        "left", "left", "left", "left",
        "right", "right", "right", "right", "right", "right"
    ]

    print_separator(widths)
    print_row(headers, widths, aligns)
    print_separator(widths)

    for r in rows:
        values = [
            r["window"],
            r["dataset"],
            r["ema"],
            f"{r['single_n']}/{r['dual_n']}",
            fmt_value(r["single_early_mean"], r["single_early_mean_std"]),
            fmt_value(r["dual_early_mean"], r["dual_early_mean_std"]),
            fmt_value(r["single_early_auc"], r["single_early_auc_std"]),
            fmt_value(r["dual_early_auc"], r["dual_early_auc_std"]),
            fmt_value(r["single_peak"], r["single_peak_std"]),
            fmt_value(r["dual_peak"], r["dual_peak_std"]),
        ]

        print_row(values, widths, aligns)

    print_separator(widths)
    print("n = Single/Dual seeds")


def print_terminal_delta_table(rows):
    print("\nDELTA TABLE")
    print("Dual Encoder - Single Encoder\n")

    headers = [
        "Win",
        "Dataset",
        "EMA",
        "ΔMean",
        "ΔAUC",
        "ΔPeak",
    ]

    widths = [5, 10, 5, 10, 10, 10]
    aligns = ["left", "left", "left", "right", "right", "right"]

    print_separator(widths)
    print_row(headers, widths, aligns)
    print_separator(widths)

    for r in rows:
        values = [
            r["window"],
            r["dataset"],
            r["ema"],
            f"{r['delta_early_mean']:.3f}",
            f"{r['delta_early_auc']:.3f}",
            f"{r['delta_peak']:.3f}",
        ]

        print_row(values, widths, aligns)

    print_separator(widths)

def save_csv(rows, filename="early_adaptation_single_vs_dual.csv"):
    if len(rows) == 0:
        return

    import csv

    path = os.path.join(TABLE_DIR, filename)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {path}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    print("\nEarly target-domain generalization")
    print("Comparison: Single Encoder vs Dual Encoder")
    print("Metrics: EarlyMean@k, EarlyAUC@k, Peak@k\n")

    loaded_methods = load_all_methods(METHODS)

    rows = build_comparison_rows(loaded_methods)

    print_terminal_main_table(rows)
    print_terminal_delta_table(rows)

    save_csv(rows)

    print("\nDone.")