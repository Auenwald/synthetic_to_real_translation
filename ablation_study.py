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

# Same definition as used for the temporal metrics in the thesis.
STD_DDOF = 0

# Numerical decay value -> filename representation
DECAY_CONFIGS = {
    0.9: "09",
    0.99: "099",
    0.995: "0995",
    0.999: "0999",
}

# Main configuration used throughout the dissertation.
REFERENCE_DECAY = 0.995

# Both naming variants are supported.
# "_new" is preferred if both files exist.
FILE_TEMPLATES = [
    (
        "synthia_to_cityscapes_bdd_segformer_b5_adamw_"
        "lr_1e05_weight_decay_1e03_"
        "ema_decay_{decay}_averaging_interval_20_seed_{seed}_new.json"
    ),
    (
        "synthia_to_cityscapes_bdd_segformer_b5_adamw_"
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
    Resolve the log file for one EMA decay and seed.

    Supported variants:
        ..._seed_X_new.json
        ..._seed_X.json

    If both exist, the "_new" file is preferred.
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
        f"Could not find log file for decay={decay_string}, "
        f"seed={seed}.\n"
        "Tried:\n  "
        + "\n  ".join(attempted_paths)
    )


def load_logs(decay_string):
    """
    Load all five seeds for one EMA decay configuration.
    """

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


def get_values_in_window(
    log,
    dataset,
    start_epoch=6,
    end_epoch=30,
):
    """
    Extract target-domain mIoU for one trajectory within
    the requested epoch window.
    """

    if dataset not in log:
        raise KeyError(
            f"Dataset '{dataset}' not found.\n"
            f"Available keys: {list(log.keys())}"
        )

    values = []

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_str = str(epoch)

        if epoch_str not in log[dataset]:
            raise KeyError(
                f"Epoch {epoch} missing for dataset '{dataset}'."
            )

        values.append(
            float(log[dataset][epoch_str]["mean_iou"])
        )

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
    Compute the temporal metrics for one seed.
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
    Temporal metrics are first computed independently for
    each seed and then aggregated across seeds.

    Returns both:
        metric
        metric_seed_std
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
# COLLECT EMA DECAY RESULTS
# ============================================================

results = {
    "cityscapes": {},
    "bdd": {},
}


for decay, decay_string in DECAY_CONFIGS.items():

    print("\n" + "=" * 70)
    print(f"EMA decay lambda = {decay}")
    print("=" * 70)

    logs = load_logs(decay_string)

    results["cityscapes"][decay] = aggregate_temporal_metrics(
        logs=logs,
        dataset="cityscapes-ema",
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
    )

    results["bdd"][decay] = aggregate_temporal_metrics(
        logs=logs,
        dataset="bdd-ema",
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
    )


# ============================================================
# NO-EMA REFERENCE
# ============================================================

# The online trajectory does not depend on EMA decay.
# We take it from the main/reference configuration.
reference_logs = load_logs(
    DECAY_CONFIGS[REFERENCE_DECAY]
)

no_ema = {
    "cityscapes": aggregate_temporal_metrics(
        logs=reference_logs,
        dataset="cityscapes",
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
    ),

    "bdd": aggregate_temporal_metrics(
        logs=reference_logs,
        dataset="bdd",
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
    ),
}


# ============================================================
# PRINT RESULTS
# ============================================================

for target in ["cityscapes", "bdd"]:

    print("\n")
    print("=" * 70)
    print(
        f"{target.upper()} "
        f"(epochs {START_EPOCH}..{END_EPOCH})"
    )
    print("=" * 70)

    print(
        "No EMA:"
        f"\n  mean  = "
        f"{no_ema[target]['mean']:.3f} "
        f"+- {no_ema[target]['mean_seed_std']:.3f}"
        f"\n  std   = "
        f"{no_ema[target]['std']:.3f} "
        f"+- {no_ema[target]['std_seed_std']:.3f}"
        f"\n  delta = "
        f"{no_ema[target]['delta']:.3f} "
        f"+- {no_ema[target]['delta_seed_std']:.3f}"
    )

    for decay in DECAY_CONFIGS:

        metrics = results[target][decay]

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
# PLOT DATA
# ============================================================

decays = list(DECAY_CONFIGS.keys())

# Equal categorical spacing for the four tested configurations.
x = np.arange(len(decays))

decay_labels = [
    "0.9",
    "0.99",
    "0.995",
    "0.999",
]


city_std = np.asarray([
    results["cityscapes"][decay]["std"]
    for decay in decays
])

city_std_seed = np.asarray([
    results["cityscapes"][decay]["std_seed_std"]
    for decay in decays
])

bdd_std = np.asarray([
    results["bdd"][decay]["std"]
    for decay in decays
])

bdd_std_seed = np.asarray([
    results["bdd"][decay]["std_seed_std"]
    for decay in decays
])


city_delta = np.asarray([
    results["cityscapes"][decay]["delta"]
    for decay in decays
])

city_delta_seed = np.asarray([
    results["cityscapes"][decay]["delta_seed_std"]
    for decay in decays
])

bdd_delta = np.asarray([
    results["bdd"][decay]["delta"]
    for decay in decays
])

bdd_delta_seed = np.asarray([
    results["bdd"][decay]["delta_seed_std"]
    for decay in decays
])


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.2),
)


# ------------------------------------------------------------
# Left: Temporal standard deviation
# ------------------------------------------------------------

ax = axes[0]

city_line, = ax.plot(
    x,
    city_std,
    marker="o",
    linewidth=2.0,
    label="Cityscapes",
)

ax.fill_between(
    x,
    city_std - city_std_seed,
    city_std + city_std_seed,
    alpha=0.15,
    color=city_line.get_color(),
)

bdd_line, = ax.plot(
    x,
    bdd_std,
    marker="o",
    linewidth=2.0,
    label="BDD",
)

ax.fill_between(
    x,
    bdd_std - bdd_std_seed,
    bdd_std + bdd_std_seed,
    alpha=0.15,
    color=bdd_line.get_color(),
)


# No-EMA references
ax.axhline(
    no_ema["cityscapes"]["std"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.75,
    color=city_line.get_color(),
)

ax.axhline(
    no_ema["bdd"]["std"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.75,
    color=bdd_line.get_color(),
)


ax.set_title("Temporal Standard Deviation")
ax.set_xlabel(r"EMA Decay $\lambda$")
ax.set_ylabel(
    r"$\overline{\sigma}_{\mathcal{W}_{\mathrm{ep}}}$"
)

ax.set_xticks(x)
ax.set_xticklabels(decay_labels)

ax.grid(alpha=0.25)


# ------------------------------------------------------------
# Right: Temporal range
# ------------------------------------------------------------

ax = axes[1]

city_line_delta, = ax.plot(
    x,
    city_delta,
    marker="o",
    linewidth=2.0,
    label="Cityscapes",
)

ax.fill_between(
    x,
    city_delta - city_delta_seed,
    city_delta + city_delta_seed,
    alpha=0.15,
    color=city_line_delta.get_color(),
)

bdd_line_delta, = ax.plot(
    x,
    bdd_delta,
    marker="o",
    linewidth=2.0,
    label="BDD",
)

ax.fill_between(
    x,
    bdd_delta - bdd_delta_seed,
    bdd_delta + bdd_delta_seed,
    alpha=0.15,
    color=bdd_line_delta.get_color(),
)


# No-EMA references
ax.axhline(
    no_ema["cityscapes"]["delta"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.75,
    color=city_line_delta.get_color(),
)

ax.axhline(
    no_ema["bdd"]["delta"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.75,
    color=bdd_line_delta.get_color(),
)


ax.set_title("Temporal Range")
ax.set_xlabel(r"EMA Decay $\lambda$")
ax.set_ylabel(
    r"$\overline{\Delta}_{\mathcal{W}_{\mathrm{ep}}}$"
)

ax.set_xticks(x)
ax.set_xticklabels(decay_labels)

ax.grid(alpha=0.25)


# ============================================================
# COMMON LEGEND
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03),
)


fig.tight_layout(
    rect=[0, 0, 1, 0.93]
)


# ============================================================
# SAVE
# ============================================================

png_path = os.path.join(
    FIGURE_DIR,
    "ema_decay_stability_analysis.png",
)

pdf_path = os.path.join(
    FIGURE_DIR,
    "ema_decay_stability_analysis.pdf",
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

print(f"\nSaved figure: {png_path}")
print(f"Saved figure: {pdf_path}")