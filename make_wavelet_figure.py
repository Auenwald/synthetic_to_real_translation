#!/usr/bin/env python3
"""
Create a single-level wavelet decomposition figure for a GTA5 image.

Default input:
    ./gta5/images/00010.png

Default outputs:
    ./figures/wavelet_decomposition_gta5.pdf
    ./figures/wavelet_decomposition_gta5.png

Example:
    python make_wavelet_figure_2x3.py
    python make_wavelet_figure_2x3.py --input ./gta5/images/00010.png --wavelet haar
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import matplotlib.gridspec as gridspec


def load_image_as_gray(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image and return:
        rgb_img: RGB image for visualization
        gray:    2D grayscale image used for the DWT
    """
    img = plt.imread(path)

    # Remove alpha channel if present
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    img = img.astype(np.float32)

    # Convert images stored as 0..255 to 0..1
    if img.max() > 1.0:
        img = img / 255.0

    if img.ndim == 2:
        gray = img
        rgb_img = np.stack([gray, gray, gray], axis=-1)

    elif img.ndim == 3 and img.shape[-1] == 3:
        rgb_img = img

        # Rec. 709 luminance conversion
        gray = (
            0.2126 * img[..., 0]
            + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2]
        )

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return rgb_img, gray


def normalize_image_for_display(x: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.0) -> np.ndarray:
    """
    Normalize a non-signed image/subband to [0, 1] for visualization.
    Used for the approximation subband.
    """
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, (percentile_low, percentile_high))

    if hi - lo < 1e-8:
        return np.zeros_like(x)

    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def normalize_detail_for_display(x: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Normalize signed wavelet detail coefficients for visualization.

    Detail coefficients can be positive or negative and are centered around zero.
    This function clips them symmetrically around zero and maps them to [0, 1],
    so that values close to zero appear gray.
    """
    x = np.asarray(x, dtype=np.float32)

    limit = np.percentile(np.abs(x), percentile)
    limit = max(limit, 1e-8)

    x = np.clip(x, -limit, limit)
    return (x + limit) / (2.0 * limit)


def make_wavelet_figure(
    input_path: Path,
    output_pdf: Path,
    output_png: Path,
    wavelet: str = "haar",
    dpi: int = 300,
    detail_percentile: float = 99.0,
) -> None:
    rgb_img, gray = load_image_as_gray(input_path)

    # Single-level 2D DWT on the grayscale image.
    # LL is the low-frequency approximation subband.
    # LH, HL, and HH are high-frequency detail subbands.
    LL, (LH, HL, HH) = pywt.dwt2(gray, wavelet)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    axes = axes.ravel()

    panels = [
        ("RGB input", rgb_img, "rgb"),
        ("Grayscale input", gray, "gray"),
        ("Approximation (LL)", LL, "approx"),
        ("Horizontal detail", LH, "detail"),
        ("Vertical detail", HL, "detail"),
        ("Diagonal detail", HH, "detail"),
    ]

    for ax, (title, arr, kind) in zip(axes, panels):
        if kind == "rgb":
            ax.imshow(np.clip(arr, 0.0, 1.0), interpolation="nearest")

        elif kind == "gray":
            ax.imshow(
                arr,
                interpolation="nearest",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )

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

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.tight_layout(pad=0.8)

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
        default=Path("./figures/wavelet_decomposition_gta5.pdf"),
        help="Path to the output PDF.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("./figures/wavelet_decomposition_gta5.png"),
        help="Path to the output PNG.",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="haar",
        help="Wavelet family, e.g. haar, db2, bior1.3.",
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

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input image not found: {args.input}\n"
            "Run the script from the project root or pass --input explicitly."
        )

    make_wavelet_figure(
        input_path=args.input,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        wavelet=args.wavelet,
        dpi=args.dpi,
        detail_percentile=args.detail_percentile,
    )
