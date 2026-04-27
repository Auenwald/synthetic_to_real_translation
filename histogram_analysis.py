"""
Color Histogram Analysis for Domain Gap Visualization
------------------------------------------------------
Computes, saves, and compares histograms across datasets.
Supports RGB and Edge (multiscale Scharr) modes.

Usage:
    # Compute RGB histogram:
    python histogram_analysis.py --compute --dataset bdd --path ./bdd/images/val --mode rgb

    # Compute Edge histogram:
    python histogram_analysis.py --compute --dataset bdd --path ./bdd/images/val --mode edge

    # Plot a single dataset histogram:
    python histogram_analysis.py --plot --dataset bdd_edge

    # Compare two datasets:
    python histogram_analysis.py --compare --datasets bdd_rgb cityscapes_rgb

    # Compare all saved datasets:
    python histogram_analysis.py --compare_all
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import jensenshannon
from scipy.ndimage import gaussian_filter, sobel
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

NUM_BINS      = 64          # bins per channel
COLOR_SPACE   = "rgb"       # "rgb", "lab", "hsv" (extendable)
LOG_DIR       = "./histogram_logs"
PLOT_DIR      = "./histogram_plots"
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CHANNEL_NAMES = {
    "rgb":  ["R", "G", "B"],
    "hsv":  ["H", "S", "V"],
    "lab":  ["L", "a", "b"],
    "edge": ["Edge"],
}

CHANNEL_COLORS = {
    "rgb":  ["#e74c3c", "#2ecc71", "#3498db"],
    "hsv":  ["#9b59b6", "#f39c12", "#1abc9c"],
    "lab":  ["#95a5a6", "#e74c3c", "#3498db"],
    "edge": ["#2c3e50"],
}

# ─────────────────────────────────────────────
# Core: Histogram computation
# ─────────────────────────────────────────────

def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as H×W×3 uint8 numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def compute_image_histogram(img: np.ndarray, num_bins: int = NUM_BINS) -> np.ndarray:
    """
    Compute normalized per-channel histogram for a single image.
    Returns h ∈ R^{3 × K}, normalized so each channel sums to 1.
    """
    histograms = []
    for c in range(img.shape[2]):
        h, _ = np.histogram(img[:, :, c], bins=num_bins, range=(0, 256))
        h = h.astype(np.float64) / (img.shape[0] * img.shape[1])  # normalize
        histograms.append(h)
    return np.stack(histograms, axis=0)  # (3, K)


def compute_edge_map(img_rgb: np.ndarray, sigmas=(0.5, 1.0, 2.0)) -> np.ndarray:
    """
    Multiscale Scharr edge detection (NumPy/SciPy port of multiscale_scharr_edges).
    Returns normalized edge map in [0, 1], shape (H, W).
    """
    # Luminance-weighted grayscale
    gray = (0.299 * img_rgb[:, :, 0] +
            0.587 * img_rgb[:, :, 1] +
            0.114 * img_rgb[:, :, 2]).astype(np.float32) / 255.0

    # Scharr kernels (matches PyTorch implementation)
    Kx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float32)
    Ky = Kx.T

    edges_multi = []
    for s in sigmas:
        blurred = gaussian_filter(gray, sigma=s)
        gx = sobel(blurred, axis=1)   # scipy sobel approximates Scharr well
        gy = sobel(blurred, axis=0)
        edges_multi.append(np.sqrt(gx**2 + gy**2))

    edges = np.max(np.stack(edges_multi, axis=0), axis=0)
    edges = edges / (edges.max() + 1e-6)
    return edges


def compute_edge_histogram(img_rgb: np.ndarray, num_bins: int = NUM_BINS) -> np.ndarray:
    """
    Compute normalized edge magnitude histogram.
    Returns h ∈ R^{1 × K}.
    """
    edges = compute_edge_map(img_rgb)
    h, _ = np.histogram(edges, bins=num_bins, range=(0.0, 1.0))
    h = h.astype(np.float64) / edges.size
    return h.reshape(1, -1)  # (1, K)


def compute_dataset_histogram(
    image_dir: str,
    num_bins: int = NUM_BINS,
    max_samples: int = None,
    mode: str = "rgb",
) -> dict:
    """
    Compute mean histogram over all images in a directory.
    mode: "rgb" or "edge"

    Returns a dict with:
        - mean_hist:  np.ndarray (C, K)  — mean normalized histogram
        - std_hist:   np.ndarray (C, K)  — std across images
        - num_images: int
        - num_bins:   int
        - color_space: str
    """
    image_dir = Path(image_dir)
    image_paths = sorted([
        p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
    ])

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if max_samples is not None and max_samples < len(image_paths):
        rng = np.random.default_rng(42)
        image_paths = list(rng.choice(image_paths, max_samples, replace=False))

    # Online Welford accumulator — O(1) memory, numerically stable
    C = 1 if mode == "edge" else 3
    mean_acc = np.zeros((C, num_bins), dtype=np.float64)
    M2_acc   = np.zeros((C, num_bins), dtype=np.float64)
    count = 0

    for p in tqdm(image_paths, desc=f"Computing {mode} histograms"):
        try:
            img = load_image_rgb(p)
            if mode == "edge":
                h = compute_edge_histogram(img, num_bins=num_bins)   # (1, K)
            else:
                h = compute_image_histogram(img, num_bins=num_bins)  # (3, K)
            count += 1
            delta     = h - mean_acc
            mean_acc += delta / count
            delta2    = h - mean_acc
            M2_acc   += delta * delta2
        except Exception as e:
            print(f"  Skipping {p.name}: {e}")

    std_hist = np.sqrt(M2_acc / max(count - 1, 1))

    return {
        "mean_hist":   mean_acc.tolist(),
        "std_hist":    std_hist.tolist(),
        "num_images":  count,
        "num_bins":    num_bins,
        "color_space": mode if mode == "edge" else COLOR_SPACE,
    }


# ─────────────────────────────────────────────
# Persistence: Save / Load
# ─────────────────────────────────────────────

def save_histogram(data: dict, dataset_name: str, log_dir: str = LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    path = Path(log_dir) / f"{dataset_name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


def load_histogram(dataset_name: str, log_dir: str = LOG_DIR) -> dict:
    path = Path(log_dir) / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved histogram for '{dataset_name}' at {path}")
    with open(path) as f:
        data = json.load(f)
    data["mean_hist"] = np.array(data["mean_hist"])
    data["std_hist"]  = np.array(data["std_hist"])
    return data


def list_saved_datasets(log_dir: str = LOG_DIR) -> list:
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []
    return [p.stem for p in sorted(log_dir.glob("*.json"))]


# ─────────────────────────────────────────────
# Distance metrics
# ─────────────────────────────────────────────

def histogram_distance(h1: np.ndarray, h2: np.ndarray, method: str = "jsd") -> np.ndarray:
    """
    Compute per-channel distance between two mean histograms.

    Supported methods:
        - "jsd":         Jensen-Shannon Divergence (symmetric, bounded [0,1])
        - "chi2":        Chi-squared distance
        - "bhattacharyya": Bhattacharyya distance
        - "l1":          L1 / Total Variation distance
    """
    assert h1.shape == h2.shape, "Histogram shapes must match"
    C = h1.shape[0]
    distances = []

    for c in range(C):
        p, q = h1[c], h2[c]
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)

        if method == "jsd":
            d = jensenshannon(p, q) ** 2
        elif method == "chi2":
            d = 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-12))
        elif method == "bhattacharyya":
            bc = np.sum(np.sqrt(p * q + 1e-12))
            d  = -np.log(bc + 1e-12)
        elif method == "l1":
            d = np.sum(np.abs(p - q))
        else:
            raise ValueError(f"Unknown method: {method}")

        distances.append(d)

    return np.array(distances)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

DATASET_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


def plot_single(dataset_name: str, log_dir: str = LOG_DIR, plot_dir: str = PLOT_DIR):
    """Plot per-channel histogram with std shading for one dataset."""
    data      = load_histogram(dataset_name, log_dir)
    mean_h    = data["mean_hist"]
    std_h     = data["std_hist"]
    K         = mean_h.shape[1]
    bins      = np.arange(K)
    ch_names  = CHANNEL_NAMES.get(data["color_space"], ["C1","C2","C3"])
    ch_colors = CHANNEL_COLORS.get(data["color_space"], ["#2c3e50"])
    C         = mean_h.shape[0]

    fig, axes = plt.subplots(1, C, figsize=(5 * C, 4), sharey=False)
    if C == 1:
        axes = [axes]
    fig.suptitle(f"Histogram — {dataset_name}  (N={data['num_images']})",
                 fontsize=13, fontweight="bold")

    for c, ax in enumerate(axes):
        ax.plot(bins, mean_h[c], color=ch_colors[c % len(ch_colors)], linewidth=1.8, label="mean")
        ax.fill_between(
            bins,
            mean_h[c] - std_h[c],
            mean_h[c] + std_h[c],
            alpha=0.25, color=ch_colors[c % len(ch_colors)], label="±1 std"
        )
        ax.set_title(ch_names[c] if c < len(ch_names) else f"C{c}", fontsize=11)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Normalized frequency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = Path(plot_dir) / f"{dataset_name}_histogram.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.show()


def plot_comparison(dataset_names: list, log_dir: str = LOG_DIR, plot_dir: str = PLOT_DIR,
                    distance_method: str = "jsd"):
    """
    Overlay mean histograms of multiple datasets per channel.
    Also prints pairwise distances.
    """
    datasets = {name: load_histogram(name, log_dir) for name in dataset_names}
    K = list(datasets.values())[0]["mean_hist"].shape[1]
    bins = np.arange(K)
    color_space = list(datasets.values())[0]["color_space"]
    ch_names = CHANNEL_NAMES.get(color_space, ["C1","C2","C3"])

    C = list(datasets.values())[0]["mean_hist"].shape[0]
    fig, axes = plt.subplots(1, C, figsize=(6 * C, 4.5), sharey=False)
    if C == 1:
        axes = [axes]
    fig.suptitle(f"Histogram Comparison  ({color_space.upper()})",
                 fontsize=13, fontweight="bold")

    for c, ax in enumerate(axes):
        for idx, (name, data) in enumerate(datasets.items()):
            color = DATASET_COLORS[idx % len(DATASET_COLORS)]
            mean_h = data["mean_hist"]
            ax.plot(bins, mean_h[c], color=color, linewidth=1.8, label=name)

        ax.set_title(ch_names[c] if c < len(ch_names) else f"C{c}", fontsize=11)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Normalized frequency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    tag = "_vs_".join(dataset_names)
    for ext in ["pdf", "png"]:
        out = Path(plot_dir) / f"comparison_{tag}.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.show()

    # ── Pairwise distances ──
    ch_names = CHANNEL_NAMES.get(color_space, ["C1", "C2", "C3"])
    C = list(datasets.values())[0]["mean_hist"].shape[0]
    header = "  " + f"{'Pair':<30}" + "".join(f"{ch_names[c] if c < len(ch_names) else f'C{c}':>8}" for c in range(C)) + f"{'mean':>8}"
    print(f"\nPairwise {distance_method.upper()} distances (per channel):")
    print(header)
    print("  " + "-" * (30 + 8 * C + 8))
    names = list(datasets.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            h1 = datasets[names[i]]["mean_hist"]
            h2 = datasets[names[j]]["mean_hist"]
            d  = histogram_distance(h1, h2, method=distance_method)
            pair = f"{names[i]} vs {names[j]}"
            row = f"  {pair:<30}" + "".join(f"{d[c]:>8.4f}" for c in range(C)) + f"{d.mean():>8.4f}"
            print(row)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Color histogram analysis for domain gap visualization.")
    parser.add_argument("--compute",     action="store_true", help="Compute and save histogram for a dataset")
    parser.add_argument("--plot",        action="store_true", help="Plot histogram for a single saved dataset")
    parser.add_argument("--compare",     action="store_true", help="Compare histograms of multiple datasets")
    parser.add_argument("--compare_all", action="store_true", help="Compare all saved datasets")
    parser.add_argument("--dataset",     type=str,            help="Dataset name (used for saving/loading)")
    parser.add_argument("--datasets",    nargs="+",           help="List of dataset names for comparison")
    parser.add_argument("--path",        type=str,            help="Path to image directory (for --compute)")
    parser.add_argument("--bins",        type=int, default=NUM_BINS, help=f"Number of histogram bins (default: {NUM_BINS})")
    parser.add_argument("--max_samples", type=int, default=None,     help="Max images to sample (default: all)")
    parser.add_argument("--distance",    type=str, default="jsd",
                        choices=["jsd", "chi2", "bhattacharyya", "l1"],
                        help="Distance metric for comparison (default: jsd)")
    parser.add_argument("--mode",         type=str, default="rgb", choices=["rgb", "edge"],
                        help="Histogram mode: 'rgb' or 'edge' (default: rgb)")
    parser.add_argument("--log_dir",     type=str, default=LOG_DIR,  help=f"Log directory (default: {LOG_DIR})")
    parser.add_argument("--plot_dir",    type=str, default=PLOT_DIR, help=f"Plot directory (default: {PLOT_DIR})")

    args = parser.parse_args()

    if args.compute:
        assert args.dataset and args.path, "--compute requires --dataset and --path"
        dataset_key = f"{args.dataset}_{args.mode}"
        print(f"\nComputing {args.mode} histogram for '{args.dataset}' from {args.path} ...")
        data = compute_dataset_histogram(args.path, num_bins=args.bins,
                                         max_samples=args.max_samples, mode=args.mode)
        save_histogram(data, dataset_key, log_dir=args.log_dir)
        print(f"  Done. {data['num_images']} images processed → saved as '{dataset_key}'")

    if args.plot:
        assert args.dataset, "--plot requires --dataset"
        plot_single(args.dataset, log_dir=args.log_dir, plot_dir=args.plot_dir)

    if args.compare:
        assert args.datasets and len(args.datasets) >= 2, "--compare requires at least 2 --datasets"
        plot_comparison(args.datasets, log_dir=args.log_dir, plot_dir=args.plot_dir,
                        distance_method=args.distance)

    if args.compare_all:
        saved = list_saved_datasets(args.log_dir)
        if len(saved) < 2:
            print(f"Not enough saved datasets for comparison. Found: {saved}")
        else:
            print(f"Comparing all saved datasets: {saved}")
            plot_comparison(saved, log_dir=args.log_dir, plot_dir=args.plot_dir,
                            distance_method=args.distance)

    if not any([args.compute, args.plot, args.compare, args.compare_all]):
        parser.print_help()


if __name__ == "__main__":
    main()