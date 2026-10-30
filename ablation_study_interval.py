import json
import os

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

LOG_DIR = "./logs_diss"
FIGURE_DIR = "./figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

SEEDS = range(5)

START_EPOCH = 6
END_EPOCH = 30

# Same convention as in the existing evaluation script
STD_DDOF = 0

# Fixed EMA decay
EMA_DECAY = "0995"

# EMA update intervals
AVERAGING_INTERVALS = [1, 20, 100, 500]

# Main configuration used in the dissertation
REFERENCE_INTERVAL = 20

# Source-domain keys used for checkpoint selection
SOURCE_ONLINE = "gta5"
SOURCE_EMA = "gta5-ema"

# Target-domain keys
TARGETS = {
    "Cityscapes": {
        "online": "cityscapes",
        "ema": "cityscapes-ema",
    },
    "BDD": {
        "online": "bdd",
        "ema": "bdd-ema",
    },
}

# Support both naming variants.
# "_new" is preferred if both exist.
# FILE_TEMPLATES = [
#     (
#         "synthia_to_cityscapes_bdd_segformer_b5_adamw_"
#         "lr_1e05_weight_decay_1e03_"
#         "ema_decay_{decay}_averaging_interval_{interval}_"
#         "seed_{seed}_new.json"
#     ),
#     (
#         "synthia_to_cityscapes_bdd_segformer_b5_adamw_"
#         "lr_1e05_weight_decay_1e03_"
#         "ema_decay_{decay}_averaging_interval_{interval}_"
#         "seed_{seed}.json"
#     ),
# ]

FILE_TEMPLATES = [
    (
        "gta5_to_cityscapes_bdd_segformer_b5_adamw_"
        "lr_1e05_weight_decay_1e03_"
        "ema_decay_{decay}_averaging_interval_{interval}_"
        "seed_{seed}_new.json"
    ),
    (
        "gta5_to_cityscapes_bdd_segformer_b5_adamw_"
        "lr_1e05_weight_decay_1e03_"
        "ema_decay_{decay}_averaging_interval_{interval}_"
        "seed_{seed}.json"
    ),
]



plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
})


# ============================================================
# FILE HANDLING
# ============================================================

def resolve_log_path(interval, seed):
    """
    Find the log file for one averaging interval and seed.

    Supported:
        ...seed_X_new.json
        ...seed_X.json

    "_new" is preferred if both exist.
    """

    attempted_paths = []

    for template in FILE_TEMPLATES:
        path = os.path.join(
            LOG_DIR,
            template.format(
                decay=EMA_DECAY,
                interval=interval,
                seed=seed,
            ),
        )

        attempted_paths.append(path)

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find log for averaging_interval={interval}, "
        f"seed={seed}.\n"
        "Tried:\n  "
        + "\n  ".join(attempted_paths)
    )


def load_logs(interval):
    logs = []

    for seed in SEEDS:
        path = resolve_log_path(
            interval=interval,
            seed=seed,
        )

        print(f"Loading: {path}")

        with open(path, "r") as f:
            logs.append(json.load(f))

    return logs


# ============================================================
# HELPERS
# ============================================================

def safe_std(values, ddof=0):
    values = np.asarray(values, dtype=float)

    if len(values) <= ddof:
        return 0.0

    return float(np.std(values, ddof=ddof))


def get_epoch_value(log, dataset, epoch):
    epoch = str(epoch)

    if dataset not in log:
        return None

    if epoch not in log[dataset]:
        return None

    return float(
        log[dataset][epoch]["mean_iou"]
    )


def get_values_in_window(
    log,
    dataset,
    start_epoch=6,
    end_epoch=30,
):
    values = []

    if dataset not in log:
        raise KeyError(
            f"Dataset '{dataset}' not found.\n"
            f"Available keys: {list(log.keys())}"
        )

    for epoch in range(start_epoch, end_epoch + 1):

        value = get_epoch_value(
            log,
            dataset,
            epoch,
        )

        if value is None:
            raise KeyError(
                f"Epoch {epoch} missing for dataset '{dataset}'."
            )

        values.append(value)

    return np.asarray(values, dtype=float)


# ============================================================
# TEMPORAL METRICS
# ============================================================

def compute_temporal_metrics(
    log,
    dataset,
    start_epoch=6,
    end_epoch=30,
):
    """
    Compute temporal metrics for one seed.
    """

    values = get_values_in_window(
        log=log,
        dataset=dataset,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )

    return {
        "mean": float(np.mean(values)),
        "std": safe_std(
            values,
            ddof=STD_DDOF,
        ),
        "delta": float(
            np.max(values) - np.min(values)
        ),
    }


def aggregate_temporal_metrics(
    logs,
    dataset,
    start_epoch=6,
    end_epoch=30,
):
    """
    Compute temporal metrics independently per seed and
    subsequently aggregate them across seeds.
    """

    per_seed = []

    for log in logs:

        metrics = compute_temporal_metrics(
            log=log,
            dataset=dataset,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
        )

        per_seed.append(metrics)

    result = {}

    for metric in ["mean", "std", "delta"]:

        values = np.asarray(
            [m[metric] for m in per_seed],
            dtype=float,
        )

        result[metric] = float(
            np.mean(values)
        )

        result[f"{metric}_seed_std"] = safe_std(
            values,
            ddof=STD_DDOF,
        )

    return result


# ============================================================
# CHECKPOINT SELECTION
# ============================================================

def select_checkpoint_epochs(
    logs,
    selection_dataset,
):
    """
    Select the best source-validation checkpoint independently
    for each seed.
    """

    best_epochs = []

    for log in logs:

        if selection_dataset not in log:
            raise KeyError(
                f"Selection dataset '{selection_dataset}' not found.\n"
                f"Available keys: {list(log.keys())}"
            )

        best_epoch = None
        best_value = -np.inf

        for epoch, values in log[selection_dataset].items():

            epoch_int = int(epoch)

            if epoch_int > END_EPOCH:
                continue

            value = float(
                values["mean_iou"]
            )

            if value > best_value:
                best_value = value
                best_epoch = epoch_int

        best_epochs.append(best_epoch)

    return best_epochs


def evaluate_selected_checkpoints(
    logs,
    selected_epochs,
    target_dataset,
):
    """
    Evaluate one target-domain checkpoint per seed.
    """

    values = []

    for log, epoch in zip(
        logs,
        selected_epochs,
    ):

        value = get_epoch_value(
            log,
            target_dataset,
            epoch,
        )

        if value is None:
            raise KeyError(
                f"Target value missing: "
                f"{target_dataset}, epoch={epoch}"
            )

        values.append(value)

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "values": values,
        "mean": float(np.mean(values)),
        "std": safe_std(
            values,
            ddof=STD_DDOF,
        ),
    }


# ============================================================
# TRAJECTORY EXTRACTION FOR FIGURE
# ============================================================

def collect_mean_trajectory(
    logs,
    dataset,
):
    """
    Collect epoch-wise mIoU across seeds.

    The figure uses the mean across all seeds.
    """

    epoch_values = {}

    for log in logs:

        if dataset not in log:
            raise KeyError(
                f"Dataset '{dataset}' missing.\n"
                f"Available keys: {list(log.keys())}"
            )

        for epoch, values in log[dataset].items():

            epoch = int(epoch)

            if epoch > END_EPOCH:
                continue

            if epoch not in epoch_values:
                epoch_values[epoch] = []

            epoch_values[epoch].append(
                float(values["mean_iou"])
            )

    epochs = np.asarray(
        sorted(epoch_values.keys())
    )

    means = []

    for epoch in epochs:

        values = np.asarray(
            epoch_values[epoch],
            dtype=float,
        )

        if len(values) != len(SEEDS):
            raise ValueError(
                f"Epoch {epoch}, dataset '{dataset}' has "
                f"{len(values)} seeds instead of {len(SEEDS)}."
            )

        means.append(
            float(np.mean(values))
        )

    return (
        epochs,
        np.asarray(means),
    )


# ============================================================
# LOAD ALL CONFIGURATIONS
# ============================================================

interval_logs = {}

for interval in AVERAGING_INTERVALS:

    print("\n" + "=" * 70)
    print(f"Loading averaging interval u = {interval}")
    print("=" * 70)

    interval_logs[interval] = load_logs(
        interval
    )


# Online training trajectory is independent of u.
# Use the reference configuration u=20.
reference_logs = interval_logs[
    REFERENCE_INTERVAL
]


# ============================================================
# PRINT TEMPORAL METRICS
# ============================================================

print("\n\n")
print("#" * 70)
print("TEMPORAL METRICS")
print("#" * 70)


for target_name, target_keys in TARGETS.items():

    print("\n")
    print("=" * 70)
    print(
        f"{target_name.upper()} "
        f"(epochs {START_EPOCH}..{END_EPOCH})"
    )
    print("=" * 70)

    # No EMA
    metrics = aggregate_temporal_metrics(
        logs=reference_logs,
        dataset=target_keys["online"],
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
    )

    print(
        "No EMA:"
        f"\n  mean  = "
        f"{metrics['mean']:.3f} "
        f"+- {metrics['mean_seed_std']:.3f}"
        f"\n  std   = "
        f"{metrics['std']:.3f} "
        f"+- {metrics['std_seed_std']:.3f}"
        f"\n  delta = "
        f"{metrics['delta']:.3f} "
        f"+- {metrics['delta_seed_std']:.3f}"
    )

    # EMA intervals
    for interval in AVERAGING_INTERVALS:

        metrics = aggregate_temporal_metrics(
            logs=interval_logs[interval],
            dataset=target_keys["ema"],
            start_epoch=START_EPOCH,
            end_epoch=END_EPOCH,
        )

        print(
            f"\nu = {interval}:"
            f"\n  mean  = "
            f"{metrics['mean']:.3f} "
            f"+- {metrics['mean_seed_std']:.3f}"
            f"\n  std   = "
            f"{metrics['std']:.3f} "
            f"+- {metrics['std_seed_std']:.3f}"
            f"\n  delta = "
            f"{metrics['delta']:.3f} "
            f"+- {metrics['delta_seed_std']:.3f}"
        )


# ============================================================
# PRINT SOURCE-SELECTED RESULTS
# ============================================================

print("\n\n")
print("#" * 70)
print("SOURCE-SELECTED CHECKPOINT PERFORMANCE")
print("#" * 70)


# ------------------------------------------------------------
# No EMA
# ------------------------------------------------------------

online_epochs = select_checkpoint_epochs(
    logs=reference_logs,
    selection_dataset=SOURCE_ONLINE,
)

print("\nNo EMA")
print(
    f"Selected epochs per seed: "
    f"{online_epochs}"
)

for target_name, target_keys in TARGETS.items():

    result = evaluate_selected_checkpoints(
        logs=reference_logs,
        selected_epochs=online_epochs,
        target_dataset=target_keys["online"],
    )

    print(
        f"{target_name}: "
        f"{result['mean']:.4f} "
        f"+- {result['std']:.4f}"
    )

    print(
        "  per-seed values: "
        f"{[round(v, 4) for v in result['values']]}"
    )


# ------------------------------------------------------------
# EMA intervals
# ------------------------------------------------------------

for interval in AVERAGING_INTERVALS:

    logs = interval_logs[interval]

    ema_epochs = select_checkpoint_epochs(
        logs=logs,
        selection_dataset=SOURCE_EMA,
    )

    print(
        f"\nEMA averaging interval u = {interval}"
    )

    print(
        f"Selected epochs per seed: "
        f"{ema_epochs}"
    )

    for target_name, target_keys in TARGETS.items():

        result = evaluate_selected_checkpoints(
            logs=logs,
            selected_epochs=ema_epochs,
            target_dataset=target_keys["ema"],
        )

        print(
            f"{target_name}: "
            f"{result['mean']:.4f} "
            f"+- {result['std']:.4f}"
        )

        print(
            "  per-seed values: "
            f"{[round(v, 4) for v in result['values']]}"
        )


# ============================================================
# PLOTTING
# ============================================================

def plot_target(
    ax,
    target_name,
    online_dataset,
    ema_dataset,
):
    """
    Plot No EMA and all averaging intervals.
    """

    # No EMA
    epochs, mean = collect_mean_trajectory(
        reference_logs,
        online_dataset,
    )

    ax.plot(
        epochs,
        mean,
        linewidth=2.2,
        linestyle="--",
        label="No EMA",
    )

    # EMA configurations
    for interval in AVERAGING_INTERVALS:

        epochs, mean = collect_mean_trajectory(
            interval_logs[interval],
            ema_dataset,
        )

        linewidth = (
            2.8
            if interval == REFERENCE_INTERVAL
            else 2.0
        )

        label = rf"$u={interval}$"

        if interval == REFERENCE_INTERVAL:
            label += " (reference)"

        ax.plot(
            epochs,
            mean,
            linewidth=linewidth,
            label=label,
        )

    ax.set_title(target_name)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)

    ax.set_xlim(
        min(epochs),
        END_EPOCH,
    )

    ax.set_xticks(
        np.arange(
            5,
            END_EPOCH + 1,
            5,
        )
    )


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.0, 4.8),
    sharex=True,
)


plot_target(
    ax=axes[0],
    target_name="Cityscapes",
    online_dataset="cityscapes",
    ema_dataset="cityscapes-ema",
)

axes[0].set_ylabel("mIoU (%)")


plot_target(
    ax=axes[1],
    target_name="BDD",
    online_dataset="bdd",
    ema_dataset="bdd-ema",
)


# Common legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03),
)

fig.tight_layout(
    rect=[0, 0, 1, 0.91]
)


# ============================================================
# SAVE
# ============================================================

png_path = os.path.join(
    FIGURE_DIR,
    "ema_averaging_interval_training_trajectories.png",
)

pdf_path = os.path.join(
    FIGURE_DIR,
    "ema_averaging_interval_training_trajectories.pdf",
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()

print(f"\nSaved: {png_path}")
print(f"Saved: {pdf_path}")