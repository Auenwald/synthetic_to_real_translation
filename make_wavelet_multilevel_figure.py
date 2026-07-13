#!/usr/bin/env python3
"""
Create a multi-level wavelet decomposition figure for a GTA5 image.

Default input:
    ./gta5/images/00010.png

Default outputs:
    ./figures/wavelet_decomposition_multilevel_gta5.pdf
    ./figures/wavelet_decomposition_multilevel_gta5.png

Example:
    python make_wavelet_multilevel_figure.py
    python make_wavelet_multilevel_figure.py --levels 2 --wavelet haar
    python make_wavelet_multilevel_figure.py --levels 3 --detail-percentile 98
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt


def load_image_as_gray(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = plt.imread(path)

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    if img.ndim == 2:
        return img, img

    if img.ndim == 3 and img.shape[-1] == 3:
        gray = (
            0.2126 * img[..., 0]
            + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2]
        )
        return img, gray

    raise ValueError(f"Unsupported image shape: {img.shape}")


def normalize_image_for_display(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, (low, high))

    if hi - lo < 1e-8:
        return np.zeros_like(x)

    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def normalize_detail_for_display(x: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    limit = np.percentile(np.abs(x), percentile)
    limit = max(limit, 1e-8)

    x = np.clip(x, -limit, limit)
    return (x + limit) / (2.0 * limit)


def plot_panel(ax, arr, title: str, kind: str, detail_percentile: float) -> None:
    if kind == "rgb":
        if arr.ndim == 2:
            ax.imshow(arr, interpolation="nearest", cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(arr, 0.0, 1.0), interpolation="nearest")

    elif kind == "approx":
        ax.imshow(
            normalize_image_for_display(arr),
            interpolation="nearest",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

    elif kind == "detail":
        ax.imshow(
            normalize_detail_for_display(arr, percentile=detail_percentile),
            interpolation="nearest",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def make_multilevel_wavelet_figure(
    input_path: Path,
    output_pdf: Path,
    output_png: Path,
    wavelet: str = "haar",
    levels: int = 2,
    dpi: int = 300,
    detail_percentile: float = 99.0,
) -> None:
    vis_img, gray = load_image_as_gray(input_path)

    # Multi-level 2D DWT.
    # PyWavelets returns:
    # [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=levels)
    final_approx = coeffs[0]
    detail_coeffs = coeffs[1:]  # coarsest to finest

    # Convert to dictionary: level 1 = finest details, level n = coarsest details
    details_by_level = {}
    for level_idx, details in enumerate(reversed(detail_coeffs), start=1):
        details_by_level[level_idx] = details  # (cH, cV, cD)

    # Layout:
    # row 0: input + final approximation
    # row 1: level 1 details
    # row 2: level 2 details
    # ...
    fig = plt.figure(figsize=(10.5, 3.0 + 2.1 * levels))
    gs = fig.add_gridspec(nrows=levels + 1, ncols=6)

    ax_input = fig.add_subplot(gs[0, 0:3])
    ax_approx = fig.add_subplot(gs[0, 3:6])

    plot_panel(ax_input, vis_img, "Input image", "rgb", detail_percentile)
    plot_panel(ax_approx, final_approx, f"Approximation LL{levels}", "approx", detail_percentile)

    for level in range(1, levels + 1):
        cH, cV, cD = details_by_level[level]

        ax_h = fig.add_subplot(gs[level, 0:2])
        ax_v = fig.add_subplot(gs[level, 2:4])
        ax_d = fig.add_subplot(gs[level, 4:6])

        # Level 1 = finest detail level.
        level_label = f"Level {level}"
        plot_panel(ax_h, cH, f"{level_label}: Horizontal detail", "detail", detail_percentile)
        plot_panel(ax_v, cV, f"{level_label}: Vertical detail", "detail", detail_percentile)
        plot_panel(ax_d, cD, f"{level_label}: Diagonal detail", "detail", detail_percentile)

    fig.tight_layout(pad=0.9)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PDF: {output_pdf}")
    print(f"Saved PNG: {output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./gta5/images/00010.png"),
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("./figures/wavelet_decomposition_multilevel_gta5.pdf"),
        help="Path to the output PDF.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("./figures/wavelet_decomposition_multilevel_gta5.png"),
        help="Path to the output PNG.",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="haar",
        help="Wavelet family, e.g. haar, db2, bior1.3.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=2,
        help="Number of DWT decomposition levels.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output.",
    )
    parser.add_argument(
        "--detail-percentile",
        type=float,
        default=99.0,
        help="Symmetric percentile clipping for detail subbands. Lower values increase contrast.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.levels < 1:
        raise ValueError("--levels must be >= 1")

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input image not found: {args.input}\n"
            "Run the script from the project root or pass --input explicitly."
        )

    make_multilevel_wavelet_figure(
        input_path=args.input,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        wavelet=args.wavelet,
        levels=args.levels,
        dpi=args.dpi,
        detail_percentile=args.detail_percentile,
    )
