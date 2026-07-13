from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = Path("./logs_diss")

PLOT_DIR = Path("./plots_diss")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# GTA5 -> Cityscapes/BDD
EXPERIMENT_PREFIX = (
    "gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_"
    "weight_decay_1e03_ema_decay_0995_averaging_interval_20"
)

SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["gta5", "cityscapes", "bdd"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_path(seed: int, prefix: str = EXPERIMENT_PREFIX) -> Path:
    # Wichtig: GTA5-Dateien haben KEIN "_new" am Ende
    return LOG_DIR / f"{prefix}_seed_{seed}.json"


def extract_miou_curve(data: dict, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Liest mean_iou für einen Datensatz ohne EMA.

    Erwartete Struktur:
    data["gta5"]["1"]["mean_iou"]
    data["cityscapes"]["1"]["mean_iou"]
    data["bdd"]["1"]["mean_iou"]
    """
    if dataset not in data:
        raise KeyError(
            f"Dataset-Key '{dataset}' nicht in JSON gefunden. "
            f"Vorhandene Keys: {list(data.keys())}"
        )

    entries = data[dataset]

    steps = sorted(entries.keys(), key=lambda x: int(x))

    x = np.array([int(s) for s in steps])
    y = np.array([entries[s]["mean_iou"] for s in steps], dtype=float)

    return x, y


def collect_seed_curves(
    seeds: list[int],
    dataset: str,
    prefix: str = EXPERIMENT_PREFIX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gibt x und eine Matrix [num_seeds, num_steps] zurück.
    Nutzt nur Steps, die in allen Seeds vorhanden sind.
    """
    curves = {}

    for seed in seeds:
        path = seed_path(seed, prefix)

        if not path.exists():
            raise FileNotFoundError(f"Log-Datei fehlt: {path}")

        data = load_json(path)
        x, y = extract_miou_curve(data, dataset)
        curves[seed] = dict(zip(x, y))

    common_steps = sorted(
        set.intersection(*(set(c.keys()) for c in curves.values()))
    )

    values = np.array(
        [[curves[seed][step] for step in common_steps] for seed in seeds],
        dtype=float,
    )

    return np.array(common_steps), values


def plot_seed_mean_std(
    seeds: list[int] = SEEDS,
    prefix: str = EXPERIMENT_PREFIX,
) -> None:
    plt.figure(figsize=(8, 5))

    for dataset in DATASETS:
        x, values = collect_seed_curves(seeds, dataset, prefix)

        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)

        plt.plot(x, mean, marker="o", linewidth=2, label=dataset.upper() if dataset == "gta5" else dataset.capitalize())
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Evaluation step / checkpoint")
    plt.ylabel("meanIoU")
    plt.title(f"meanIoU per dataset — GTA5 source — mean ± std over seeds {seeds} — no EMA")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    seed_str = "_".join(map(str, seeds))
    out_path = PLOT_DIR / f"gta5_miou_mean_std_seeds_{seed_str}_no_ema.png"

    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_seed_mean_std(seeds=[0, 1, 2, 3, 4])