# --- Imports ---
from transformers import SegformerForSemanticSegmentation
from segformer_crossattention_wrapper import *
from segformer_crossattention_wrapperv2 import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

# --- Device ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Tee:
    """Hilfsklasse, um stdout gleichzeitig in Datei und Konsole zu schreiben."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # sofort schreiben

    def flush(self):
        for f in self.files:
            f.flush()


def extract_alphas(WrapperClass, mode, ckpt_path, device):
    """Lädt ein Modell und gibt die α-Werte (sigmoid) je Layer zurück."""
    model = WrapperClass(mode=mode).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[key], strict=False)
    alphas = [torch.sigmoid(p).item() for p in model.gating_weights]
    return alphas


def plot_adaptive_gating_with_seeds(cp_paths, WrapperClass=SegformerCrossAttentionWrapperV2,
                                    device=device, save_path="adaptive_gating_multi_seed.png",
                                    log_path="adaptive_gating_log.txt",
                                    show_individual=True):
    """
    Erwartet: cp_paths = [
        {'edge': [seed0, seed1, seed2]},
        {'fft': [seed0, seed1, seed2]},
        ...
    ]
    """
    colors = plt.cm.tab10.colors
    mode_stats = {}

    # --- Logging initialisieren ---
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        sys.stdout = tee  # alles doppelt ausgeben

        print("=" * 80)
        print("Adaptive Gating Logging gestartet")
        print("=" * 80, "\n")

        # --- Sammeln & Loggen ---
        for mode_dict in cp_paths:
            mode, paths = list(mode_dict.items())[0]
            mode_upper = mode.upper()

            all_seeds = []
            print(f"\n=== {mode_upper}: {len(paths)} Checkpoints ===")
            for ckpt in paths:
                alphas = extract_alphas(WrapperClass, mode, ckpt, device)
                all_seeds.append(alphas)
                print(f"{os.path.basename(ckpt)} -> {alphas}")

            all_seeds = np.array(all_seeds)
            n_seeds, n_layers = all_seeds.shape

            print(f"\n--- {mode_upper}: pro Layer über {n_seeds} Seeds ---")
            for li in range(n_layers):
                vals = all_seeds[:, li]
                print(f"Layer L{li}: {vals.tolist()}")

            mean = all_seeds.mean(axis=0)
            std = all_seeds.std(axis=0)

            print(f"\n--- {mode_upper}: Mean je Layer ---")
            print([round(x, 6) for x in mean.tolist()])
            print(f"--- {mode_upper}: Std je Layer ---")
            print([round(x, 6) for x in std.tolist()])

            mode_stats[mode_upper] = {"mean": mean, "std": std, "all": all_seeds}

        # --- Plot ---
        n_layers = next(iter(mode_stats.values()))["mean"].shape[0]
        layers = [f"L{i}" for i in range(n_layers)]
        x = np.arange(n_layers)

        plt.figure(figsize=(9, 5))
        for i, (mode, stats) in enumerate(mode_stats.items()):
            c = colors[i % len(colors)]

            if show_individual:
                for seed_vals in stats["all"]:
                    plt.plot(layers, seed_vals, alpha=0.25, linewidth=1)

            plt.plot(layers, stats["mean"], label=f"{mode} (mean)", color=c, marker="o")
            plt.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                             color=c, alpha=0.15)

        plt.ylim(0.475, 0.6)
        plt.title("Adaptive Gating per Layer (Mean ± Std über Seeds)")
        plt.ylabel("α (0 = Hybrid, 1 = RGB)")
        plt.xlabel("Layer")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\nFigure gespeichert unter: {save_path}")

        print(f"\nLogdatei gespeichert unter: {os.path.abspath(log_path)}")
        print("=" * 80)

        plt.show()

        sys.stdout = sys.__stdout__  # stdout zurücksetzen


# ----------- Deine Checkpoints -----------
cp_paths = [
    {'edge': [
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_crossattention_rgb_and_edges_conv_v2.pth',
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_rgb_and_edges_conv_v2_seed1.pth',
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_rgb_and_edges_conv_v2_seed2.pth',
    ]},
    {'fft': [
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_crossattention_rgb_and_fft_gating_conv_v2.pth',
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_rgb_and_fft_convv2_seed1.pth',
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_rgb_and_fft_convv2_seed2.pth',
    ]},
    {'dct': [
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_crossattention_rgb_and_dct_gating_conv_v2.pth',
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_rgb_and_dct_convv2_seed1.pth',
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_rgb_and_dct_convv2_seed2.pth',
    ]},
    {'hsv': [
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_crossattention_rgb_and_hsv_gating_conv_v2.pth',
        './checkpoints/synthia_to_cs_and_bdd_segformerb5_adamw_lr_1e05_hsv_convv2_seed1.pth',
        './checkpoints/synthia_to_cs_and_bdd_segformerb5_adamw_lr_1e05_hsv_convv2_seed2.pth',
    ]},
    {'lab': [
        './checkpoints/synthia_to_cs_and_bdd_lr1e5_and_lr1e4_rgb_and_lab_conv_v2.pth',
        './checkpoints/synthia_to_cs_and_bdd_segformerb5_adamw_lr_1e05_lab_convv2_seed1.pth',
        './checkpoints/synthia_to_cs_and_bdd_segformerb5_adamw_lr_1e05_lab_convv2_seed2.pth',
    ]}
]

# --- Aufruf ---
plot_adaptive_gating_with_seeds(
    cp_paths=cp_paths,
    WrapperClass=SegformerCrossAttentionWrapperV2,
    device=device,
    save_path="adaptive_gating_multi_seed.png",
    log_path="adaptive_gating_log.txt",
    show_individual=True
)
