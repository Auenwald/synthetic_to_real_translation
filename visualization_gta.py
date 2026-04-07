import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

FIGURE_DIR = "./figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

paths = [
     "./logs_diss/gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0999_averaging_interval_20_seed_0.json",
     "./logs_diss/gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0999_averaging_interval_20_seed_1.json",
     "./logs_diss/gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0999_averaging_interval_20_seed_2.json",
     "./logs_diss/gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0999_averaging_interval_20_seed_3.json",
     "./logs_diss/gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_weight_decay_1e03_ema_decay_0999_averaging_interval_20_seed_4.json"
]


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

START_EPOCH = 6
END_EPOCH = 30
STD_DDOF = 0

# Optional: global plotting style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
})

# ---------------------------------------------------------
# LOAD LOGS
# ---------------------------------------------------------

logs = []
for p in paths:
    with open(p, "r") as f:
        logs.append(json.load(f))

# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------

def safe_std(values, ddof=0):
    values = np.asarray(values, dtype=float)
    if len(values) <= ddof:
        return 0.0
    return float(np.std(values, ddof=ddof))


def collect(dataset):
    """
    Collect mean_iou values per epoch across seeds.
    Returns:
        epochs: sorted list of epochs
        means: mean over seeds per epoch
        stds: std over seeds per epoch
        counts: number of seeds available per epoch
    """
    epoch_dict = defaultdict(list)

    for log in logs:
        if dataset not in log:
            continue

        for epoch, values in log[dataset].items():
            epoch_dict[int(epoch)].append(float(values["mean_iou"]))

    epochs = sorted(epoch_dict.keys())

    means = []
    stds = []
    counts = []

    for e in epochs:
        vals = epoch_dict[e]
        means.append(float(np.mean(vals)))
        stds.append(safe_std(vals, ddof=STD_DDOF))
        counts.append(len(vals))

    return np.array(epochs), np.array(means), np.array(stds), np.array(counts)


def get_epoch_value(log, dataset, epoch):
    epoch_str = str(epoch)
    if dataset not in log:
        return None
    if epoch_str not in log[dataset]:
        return None
    return float(log[dataset][epoch_str]["mean_iou"])


def get_values_in_window(log, dataset, start_epoch=5, end_epoch=30):
    values = []
    for e in range(start_epoch, end_epoch + 1):
        v = get_epoch_value(log, dataset, e)
        if v is not None:
            values.append(v)
    return values


def restrict_to_window(epochs, means, stds, start_epoch=None, end_epoch=None):
    mask = np.ones_like(epochs, dtype=bool)
    if start_epoch is not None:
        mask &= epochs >= start_epoch
    if end_epoch is not None:
        mask &= epochs <= end_epoch
    return epochs[mask], means[mask], stds[mask]


def compute_global_ylim(dataset_pairs, start_epoch=None, end_epoch=None, pad_ratio=0.08):
    """
    Compute a shared y-limit across several plots.
    dataset_pairs: list of (plain_dataset, ema_dataset)
    """
    ymins = []
    ymaxs = []

    for dataset_plain, dataset_ema in dataset_pairs:
        for ds in [dataset_plain, dataset_ema]:
            epochs, means, stds, _ = collect(ds)
            epochs, means, stds = restrict_to_window(
                epochs, means, stds, start_epoch, end_epoch
            )
            if len(epochs) == 0:
                continue
            ymins.append(np.min(means - stds))
            ymaxs.append(np.max(means + stds))

    if not ymins or not ymaxs:
        return None

    ymin = float(min(ymins))
    ymax = float(max(ymaxs))
    yrange = ymax - ymin
    pad = max(0.5, yrange * pad_ratio)

    return ymin - pad, ymax + pad


# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------

def plot_on_axis(
    ax,
    dataset_plain,
    dataset_ema,
    title,
    start_epoch=None,
    end_epoch=None,
    ylabel=True,
    show_legend=True,
    ylim=None
):
    epochs_plain, mean_plain, std_plain, _ = collect(dataset_plain)
    epochs_ema, mean_ema, std_ema, _ = collect(dataset_ema)

    epochs_plain, mean_plain, std_plain = restrict_to_window(
        epochs_plain, mean_plain, std_plain, start_epoch, end_epoch
    )
    epochs_ema, mean_ema, std_ema = restrict_to_window(
        epochs_ema, mean_ema, std_ema, start_epoch, end_epoch
    )

    # No EMA
    ax.plot(epochs_plain, mean_plain, linewidth=2.2, label="No EMA")
    ax.fill_between(
        epochs_plain,
        mean_plain - std_plain,
        mean_plain + std_plain,
        alpha=0.20
    )

    # EMA
    ax.plot(epochs_ema, mean_ema, linewidth=2.2, label="EMA")
    ax.fill_between(
        epochs_ema,
        mean_ema - std_ema,
        mean_ema + std_ema,
        alpha=0.20
    )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    if ylabel:
        ax.set_ylabel("mIoU")
    ax.grid(True, alpha=0.3)

    # x-axis: start at first real epoch, not artificially at 0
    if len(epochs_plain) > 0 and len(epochs_ema) > 0:
        xmin = min(np.min(epochs_plain), np.min(epochs_ema))
        xmax = max(np.max(epochs_plain), np.max(epochs_ema))
    elif len(epochs_plain) > 0:
        xmin = np.min(epochs_plain)
        xmax = np.max(epochs_plain)
    elif len(epochs_ema) > 0:
        xmin = np.min(epochs_ema)
        xmax = np.max(epochs_ema)
    else:
        xmin, xmax = 1, end_epoch if end_epoch is not None else 30

    ax.set_xlim(xmin, xmax)

    # nicer ticks
    tick_start = int(xmin)
    tick_end = int(xmax)
    if tick_end - tick_start <= 10:
        step = 1
    elif tick_end - tick_start <= 20:
        step = 2
    else:
        step = 5
    ax.set_xticks(np.arange(tick_start, tick_end + 1, step))

    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_legend:
        ax.legend(frameon=True)


def save_current_figure(filename):
    png_path = os.path.join(FIGURE_DIR, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_DIR, f"{filename}.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def plot_dataset_pair(dataset_plain, dataset_ema, title, filename, start_epoch=None, end_epoch=None):
    ylim = compute_global_ylim(
        [(dataset_plain, dataset_ema)],
        start_epoch=start_epoch,
        end_epoch=end_epoch
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_on_axis(
        ax=ax,
        dataset_plain=dataset_plain,
        dataset_ema=dataset_ema,
        title=title,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        ylabel=True,
        show_legend=True,
        ylim=ylim
    )

    fig.tight_layout()
    save_current_figure(filename)
    plt.show()


def plot_side_by_side(
    left_plain,
    left_ema,
    right_plain,
    right_ema,
    left_title,
    right_title,
    super_title,
    filename,
    start_epoch=None,
    end_epoch=None
):
    ylim = compute_global_ylim(
        [(left_plain, left_ema), (right_plain, right_ema)],
        start_epoch=start_epoch,
        end_epoch=end_epoch
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    plot_on_axis(
        ax=axes[0],
        dataset_plain=left_plain,
        dataset_ema=left_ema,
        title=left_title,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        ylabel=True,
        show_legend=True,
        ylim=ylim
    )

    plot_on_axis(
        ax=axes[1],
        dataset_plain=right_plain,
        dataset_ema=right_ema,
        title=right_title,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        ylabel=False,
        show_legend=True,
        ylim=ylim
    )

    fig.suptitle(super_title, y=1.02, fontsize=14)
    fig.tight_layout()
    save_current_figure(filename)
    plt.show()


# ---------------------------------------------------------
# CHECKPOINT SELECTION
# ---------------------------------------------------------

def select_checkpoint(selection_dataset):
    best_epochs = []

    for log in logs:
        if selection_dataset not in log:
            best_epochs.append(None)
            continue

        best_epoch = None
        best_val = -np.inf

        for epoch, values in log[selection_dataset].items():
            val = float(values["mean_iou"])
            ep = int(epoch)

            if val > best_val:
                best_val = val
                best_epoch = ep

        best_epochs.append(best_epoch)

    return best_epochs


def evaluate_target(best_epochs, target_dataset):
    results = []

    for log, epoch in zip(logs, best_epochs):
        if epoch is None:
            continue
        v = get_epoch_value(log, target_dataset, epoch)
        if v is not None:
            results.append(v)

    results = np.array(results, dtype=float)

    if len(results) == 0:
        return results, np.nan, np.nan

    return results, float(np.mean(results)), safe_std(results, ddof=STD_DDOF)


def print_checkpoint_selection(selection_dataset, target_datasets):
    best_epochs = select_checkpoint(selection_dataset)

    print(f"\nCheckpoint selection via {selection_dataset.upper()}")
    print(f"Selected epochs per seed: {best_epochs}")

    for target in target_datasets:
        per_seed, mean_val, std_val = evaluate_target(best_epochs, target)
        print(f"{target}: {mean_val:.4f} +- {std_val:.4f}")
        print(f"  per-seed values: {[round(x, 4) for x in per_seed]}")


# ---------------------------------------------------------
# TEMPORAL STABILITY METRICS
# ---------------------------------------------------------

def compute_temporal_metrics(log, dataset, start_epoch=5, end_epoch=30):
    values = get_values_in_window(log, dataset, start_epoch, end_epoch)

    if len(values) == 0:
        return None

    values = np.array(values, dtype=float)

    return {
        "mean": float(np.mean(values)),
        "std": safe_std(values, ddof=STD_DDOF),
        "delta": float(np.max(values) - np.min(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_epochs": int(len(values)),
    }


def aggregate_temporal_metrics(dataset, start_epoch=5, end_epoch=30):
    per_seed_metrics = []

    for seed_idx, log in enumerate(logs):
        metrics = compute_temporal_metrics(
            log=log,
            dataset=dataset,
            start_epoch=start_epoch,
            end_epoch=end_epoch
        )
        if metrics is not None:
            metrics["seed"] = seed_idx
            per_seed_metrics.append(metrics)

    if len(per_seed_metrics) == 0:
        return [], None

    summary = {}

    for key in ["mean", "std", "delta", "min", "max"]:
        vals = np.array([m[key] for m in per_seed_metrics], dtype=float)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = safe_std(vals, ddof=STD_DDOF)

    summary["num_seeds"] = len(per_seed_metrics)
    summary["start_epoch"] = start_epoch
    summary["end_epoch"] = end_epoch

    return per_seed_metrics, summary


def print_temporal_metrics(dataset, start_epoch=5, end_epoch=30):
    per_seed_metrics, summary = aggregate_temporal_metrics(
        dataset=dataset,
        start_epoch=start_epoch,
        end_epoch=end_epoch
    )

    print(f"\nTemporal stability metrics for {dataset} "
          f"(epochs {start_epoch}..{end_epoch})")

    if summary is None:
        print("No values available.")
        return

    for m in per_seed_metrics:
        print(
            f"  Seed {m['seed']}: "
            f"mean={m['mean']:.4f}, "
            f"std={m['std']:.4f}, "
            f"delta={m['delta']:.4f}, "
            f"min={m['min']:.4f}, "
            f"max={m['max']:.4f}, "
            f"n={m['n_epochs']}"
        )

    print("  --- aggregated over seeds ---")
    print(f"  Temporal mean: {summary['mean_mean']:.4f} +- {summary['mean_std']:.4f}")
    print(f"  Temporal std:  {summary['std_mean']:.4f} +- {summary['std_std']:.4f}")
    print(f"  Delta:         {summary['delta_mean']:.4f} +- {summary['delta_std']:.4f}")
    print(f"  Min:           {summary['min_mean']:.4f} +- {summary['min_std']:.4f}")
    print(f"  Max:           {summary['max_mean']:.4f} +- {summary['max_std']:.4f}")


# ---------------------------------------------------------
# OPTIONAL: COMPACT COMPARISON TABLE
# ---------------------------------------------------------

def print_temporal_comparison(datasets, start_epoch=5, end_epoch=30):
    print(f"\n=== Temporal comparison (epochs {start_epoch}..{end_epoch}) ===")
    print(f"{'Dataset':<18} {'Mean':<18} {'Std':<18} {'Delta':<18}")

    for ds in datasets:
        _, summary = aggregate_temporal_metrics(
            dataset=ds,
            start_epoch=start_epoch,
            end_epoch=end_epoch
        )

        if summary is None:
            print(f"{ds:<18} {'n/a':<18} {'n/a':<18} {'n/a':<18}")
            continue

        mean_str = f"{summary['mean_mean']:.3f} +- {summary['mean_std']:.3f}"
        std_str = f"{summary['std_mean']:.3f} +- {summary['std_std']:.3f}"
        delta_str = f"{summary['delta_mean']:.3f} +- {summary['delta_std']:.3f}"

        print(f"{ds:<18} {mean_str:<18} {std_str:<18} {delta_str:<18}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

# 1) Einzelplots - kompletter Trainingsverlauf
plot_dataset_pair(
    "cityscapes",
    "cityscapes-ema",
    "Cityscapes: Target-Domain mIoU During Training",
    "cityscapes_training_curve",
    start_epoch=None,
    end_epoch=END_EPOCH
)

plot_dataset_pair(
    "bdd",
    "bdd-ema",
    "BDD: Target-Domain mIoU During Training",
    "bdd_training_curve",
    start_epoch=None,
    end_epoch=END_EPOCH
)

# 2) Gemeinsame 2-Panel-Figure - kompletter Trainingsverlauf
plot_side_by_side(
    left_plain="cityscapes",
    left_ema="cityscapes-ema",
    right_plain="bdd",
    right_ema="bdd-ema",
    left_title="Cityscapes",
    right_title="BDD",
    super_title="[GTA5 -> Cityscapes, BDD] - EMA reduces temporal fluctuations on both target domains",
    filename="target_domain_training_curves_combined",
    start_epoch=None,
    end_epoch=END_EPOCH
)

# 3) Checkpoint selection
print_checkpoint_selection(
    selection_dataset="gta5",
    target_datasets=["cityscapes", "bdd"]
)

print_checkpoint_selection(
    selection_dataset="gta5-ema",
    target_datasets=["cityscapes-ema", "bdd-ema"]
)

# 4) Temporal stability metrics
for ds in ["cityscapes", "cityscapes-ema", "bdd", "bdd-ema"]:
    print_temporal_metrics(
        dataset=ds,
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH
    )

# 5) Compact overview
print_temporal_comparison(
    datasets=["cityscapes", "cityscapes-ema", "bdd", "bdd-ema"],
    start_epoch=START_EPOCH,
    end_epoch=END_EPOCH
)

print(f"\nFigures written to: {FIGURE_DIR}")


# 3) Checkpoint selection
print_checkpoint_selection(
    selection_dataset="gta5",
    target_datasets=["cityscapes", "bdd"]
)

print_checkpoint_selection(
    selection_dataset="gta5-ema",
    target_datasets=["cityscapes-ema", "bdd-ema"]
)