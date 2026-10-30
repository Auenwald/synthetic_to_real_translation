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

# Same convention as in the existing evaluation scripts
STD_DDOF = 0

# EMA decay -> filename representation
DECAY_CONFIGS = {
    0.9: "09",
    0.99: "099",
    0.995: "0995",
    0.999: "0999",
}

REFERENCE_DECAY = 0.995

# Source-domain keys for DG-compliant checkpoint selection
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

# Both filename variants are supported.
# "_new" is preferred if both exist.
FILE_TEMPLATES = [
    (
        "gta5_to_cityscapes_bdd_segformer_b5_adamw_"
        "lr_1e05_weight_decay_1e03_"
        "ema_decay_{decay}_averaging_interval_20_seed_{seed}_new.json"
    ),
    (
        "gta5_to_cityscapes_bdd_segformer_b5_adamw_"
        "lr_1e05_weight_decay_1e03_"
        "ema_decay_{decay}_averaging_interval_20_seed_{seed}.json"
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

def resolve_log_path(decay_string, seed):
    """
    Find the log file for a given decay and seed.

    Supported:
        ..._seed_X_new.json
        ..._seed_X.json

    If both exist, "_new" is preferred.
    """

    attempted_paths = []

    for template in FILE_TEMPLATES:
        path = os.path.join(
            LOG_DIR,
            template.format(
                decay=decay_string,
                seed=seed,
            ),
        )

        attempted_paths.append(path)

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find log for decay={decay_string}, seed={seed}.\n"
        "Tried:\n  "
        + "\n  ".join(attempted_paths)
    )


def load_logs(decay_string):
    logs = []

    for seed in SEEDS:
        path = resolve_log_path(
            decay_string=decay_string,
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
    epoch_str = str(epoch)

    if dataset not in log:
        return None

    if epoch_str not in log[dataset]:
        return None

    return float(
        log[dataset][epoch_str]["mean_iou"]
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
    Compute temporal metrics per seed and subsequently
    aggregate them across seeds.
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
    for every seed.
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
    Evaluate the source-selected checkpoint on one target
    domain for every seed.
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
# TRAJECTORY EXTRACTION
# ============================================================

def collect_mean_trajectory(
    logs,
    dataset,
):
    """
    Collect epoch-wise target-domain mIoU across seeds.

    Returns:
        epochs
        mean across seeds
        standard deviation across seeds
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
    stds = []

    for epoch in epochs:

        values = np.asarray(
            epoch_values[epoch],
            dtype=float,
        )

        if len(values) != len(SEEDS):
            raise ValueError(
                f"Epoch {epoch}, dataset '{dataset}' contains "
                f"{len(values)} seeds instead of {len(SEEDS)}."
            )

        means.append(
            float(np.mean(values))
        )

        stds.append(
            safe_std(values, ddof=STD_DDOF)
        )

    return (
        epochs,
        np.asarray(means),
        np.asarray(stds),
    )


# ============================================================
# LOAD ALL CONFIGURATIONS
# ============================================================

ema_logs = {}

for decay, decay_string in DECAY_CONFIGS.items():

    print("\n" + "=" * 70)
    print(f"Loading lambda = {decay}")
    print("=" * 70)

    ema_logs[decay] = load_logs(
        decay_string
    )


# Online trajectory does not depend on lambda.
# Use the reference configuration lambda=0.995.
reference_logs = ema_logs[
    REFERENCE_DECAY
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

    # --------------------------------------------------------
    # No EMA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EMA decay configurations
    # --------------------------------------------------------

    for decay in DECAY_CONFIGS:

        metrics = aggregate_temporal_metrics(
            logs=ema_logs[decay],
            dataset=target_keys["ema"],
            start_epoch=START_EPOCH,
            end_epoch=END_EPOCH,
        )

        print(
            f"\nlambda = {decay}:"
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
# EMA decay configurations
# ------------------------------------------------------------

for decay in DECAY_CONFIGS:

    logs = ema_logs[decay]

    ema_epochs = select_checkpoint_epochs(
        logs=logs,
        selection_dataset=SOURCE_EMA,
    )

    print(
        f"\nEMA decay lambda = {decay}"
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
# PLOTTING HELPER
# ============================================================

def plot_target(
    ax,
    target_name,
    online_dataset,
    ema_dataset,
):
    """
    Plot No EMA and all EMA decay configurations.
    """

    # --------------------------------------------------------
    # No EMA
    # --------------------------------------------------------

    epochs_online, mean_online, _ = collect_mean_trajectory(
        reference_logs,
        online_dataset,
    )

    ax.plot(
        epochs_online,
        mean_online,
        linewidth=2.2,
        linestyle="--",
        label="No EMA",
    )

    # --------------------------------------------------------
    # EMA configurations
    # --------------------------------------------------------

    for decay in DECAY_CONFIGS:

        epochs, mean, _ = collect_mean_trajectory(
            ema_logs[decay],
            ema_dataset,
        )

        linewidth = (
            2.8
            if decay == REFERENCE_DECAY
            else 2.0
        )

        label = rf"$\lambda={decay}$"

        if decay == REFERENCE_DECAY:
            label += r" (reference)"

        ax.plot(
            epochs,
            mean,
            linewidth=linewidth,
            label=label,
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(target_name)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)

    ax.set_xlim(
        min(epochs_online),
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


# ============================================================
# COMMON LEGEND
# ============================================================

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
    "gta5_ema_decay_training_trajectories.png",
)

pdf_path = os.path.join(
    FIGURE_DIR,
    "gta5_ema_decay_training_trajectories.pdf",
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