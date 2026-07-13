from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = Path("./logs_diss")

PLOT_DIR = Path("./plots_diss")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


SINGLE_ENCODER_PREFIX = (
    "gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_"
    "weight_decay_1e03_ema_decay_0995_averaging_interval_20"
)

BRANCHED_PREFIX = "gta5_v2_edge_branched_ade"

SINGLE_ENCODER_SEEDS = [0, 1, 2, 3, 4]
BRANCHED_SEEDS = [0, 1, 2]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def single_encoder_path(seed: int) -> Path:
    return LOG_DIR / f"{SINGLE_ENCODER_PREFIX}_seed_{seed}.json"


def branched_path(seed: int) -> Path:
    # Dateinamen laut Beispiel: gta5_v2_edge_branched_ade_seed0.json
    return LOG_DIR / f"{BRANCHED_PREFIX}_seed{seed}.json"


def extract_miou_curve(
    data: dict,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if key not in data:
        raise KeyError(
            f"Key '{key}' nicht in JSON gefunden. "
            f"Vorhandene Keys: {list(data.keys())}"
        )

    entries = data[key]
    steps = sorted(entries.keys(), key=lambda x: int(x))

    x = np.array([int(s) for s in steps])
    y = np.array([entries[s]["mean_iou"] for s in steps], dtype=float)

    return x, y


def collect_curves(
    paths: list[Path],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gibt x und Matrix [num_seeds, num_steps] zurück.
    Nutzt nur Steps, die in allen Seeds vorhanden sind.
    """
    curves = {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Log-Datei fehlt: {path}")

        data = load_json(path)
        x, y = extract_miou_curve(data, key)

        curves[path.name] = dict(zip(x, y))

    common_steps = sorted(
        set.intersection(*(set(curve.keys()) for curve in curves.values()))
    )

    values = np.array(
        [
            [curves[name][step] for step in common_steps]
            for name in curves.keys()
        ],
        dtype=float,
    )

    return np.array(common_steps), values


def plot_architecture_comparison(target_dataset: str) -> None:
    """
    Vergleicht:
    - Single Encoder, no EMA, 5 Seeds
    - Edge-Branched, EMA, 3 Seeds

    Für target_dataset:
    - "cityscapes"
    - "bdd"
    """
    target_label = (
        target_dataset.upper()
        if target_dataset == "bdd"
        else target_dataset.capitalize()
    )

    single_paths = [single_encoder_path(seed) for seed in SINGLE_ENCODER_SEEDS]
    branched_paths = [branched_path(seed) for seed in BRANCHED_SEEDS]

    # Single Encoder: ohne EMA
    x_single, values_single = collect_curves(
        paths=single_paths,
        key=target_dataset,
    )

    # Branched: mit EMA
    x_branched, values_branched = collect_curves(
        paths=branched_paths,
        key=f"{target_dataset}-ema",
    )

    single_mean = values_single.mean(axis=0)
    single_std = values_single.std(axis=0, ddof=1)

    branched_mean = values_branched.mean(axis=0)
    branched_std = values_branched.std(axis=0, ddof=1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_single,
        single_mean,
        marker="o",
        linewidth=2,
        linestyle="-",
        label="Single Encoder, no EMA, 5 seeds",
    )
    plt.fill_between(
        x_single,
        single_mean - single_std,
        single_mean + single_std,
        alpha=0.18,
    )

    plt.plot(
        x_branched,
        branched_mean,
        marker="o",
        linewidth=2,
        linestyle="--",
        label="FFT-Branched, EMA, 3 seeds",
    )
    plt.fill_between(
        x_branched,
        branched_mean - branched_std,
        branched_mean + branched_std,
        alpha=0.18,
    )

    plt.xlabel("Evaluation step / checkpoint")
    plt.ylabel("meanIoU")
    plt.title(f"{target_label} meanIoU — Single Encoder vs. FFT-Branched")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = (
        PLOT_DIR
        / f"gta5_single_encoder_no_ema_vs_edge_branched_ema_{target_dataset}.png"
    )

    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_architecture_comparison("cityscapes")
    plot_architecture_comparison("bdd")