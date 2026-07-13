from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = Path("./logs_diss")

PLOT_DIR = Path("./plots_diss")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

EXPERIMENTS = {
    "synthia": {
        "label": "SYNTHIA",
        "prefix": (
            "synthia_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_"
            "weight_decay_1e03_ema_decay_0999_averaging_interval_20"
        ),
        "suffix": "_new.json",
    },
    "gta5": {
        "label": "GTA5",
        "prefix": (
            "gta5_to_cityscapes_bdd_segformer_b5_adamw_lr_1e05_"
            "weight_decay_1e03_ema_decay_0999_averaging_interval_20"
        ),
        "suffix": ".json",
    },
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def seed_path(source: str, seed: int) -> Path:
    exp = EXPERIMENTS[source]
    return LOG_DIR / f"{exp['prefix']}_seed_{seed}{exp['suffix']}"


def metric_key(target_dataset: str, use_ema: bool) -> str:
    return f"{target_dataset}-ema" if use_ema else target_dataset


def extract_miou_curve(
    data: dict,
    target_dataset: str,
    use_ema: bool,
) -> tuple[np.ndarray, np.ndarray]:
    key = metric_key(target_dataset, use_ema)

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


def collect_seed_curves(
    source: str,
    target_dataset: str,
    use_ema: bool,
    seeds: list[int] = SEEDS,
) -> tuple[np.ndarray, np.ndarray]:
    curves = {}

    for seed in seeds:
        path = seed_path(source, seed)

        if not path.exists():
            raise FileNotFoundError(f"Log-Datei fehlt: {path}")

        data = load_json(path)
        x, y = extract_miou_curve(data, target_dataset, use_ema)
        curves[seed] = dict(zip(x, y))

    common_steps = sorted(
        set.intersection(*(set(curve.keys()) for curve in curves.values()))
    )

    values = np.array(
        [
            [curves[seed][step] for step in common_steps]
            for seed in seeds
        ],
        dtype=float,
    )

    return np.array(common_steps), values


def plot_ema_comparison_for_source_and_target(
    source: str,
    target_dataset: str,
    seeds: list[int] = SEEDS,
) -> None:
    """
    Eine Grafik pro Kombination:
    - Source: synthia oder gta5
    - Target: cityscapes oder bdd

    Zeigt:
    - target_dataset ohne EMA
    - target_dataset mit EMA

    Jeweils mean ± std über Seeds.
    """
    source_label = EXPERIMENTS[source]["label"]
    target_label = target_dataset.upper() if target_dataset == "bdd" else target_dataset.capitalize()

    plt.figure(figsize=(8, 5))

    configs = [
        (False, "No EMA", "-"),
        (True, "EMA", "--"),
    ]

    for use_ema, label, linestyle in configs:
        x, values = collect_seed_curves(
            source=source,
            target_dataset=target_dataset,
            use_ema=use_ema,
            seeds=seeds,
        )

        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)

        plt.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            label=label,
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.18,
        )

    plt.xlabel("Evaluation step / checkpoint")
    plt.ylabel("meanIoU")
    plt.title(f"{target_label} meanIoU — trained on {source_label} — EMA comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    seed_str = "_".join(map(str, seeds))
    out_path = (
        PLOT_DIR
        / f"{source}_{target_dataset}_ema_vs_no_ema_seeds_{seed_str}.png"
    )

    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for source in ["synthia", "gta5"]:
        for target_dataset in ["cityscapes", "bdd"]:
            plot_ema_comparison_for_source_and_target(
                source=source,
                target_dataset=target_dataset,
                seeds=[0, 1, 2, 3, 4],
            )