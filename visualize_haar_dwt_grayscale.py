#!/usr/bin/env python3
"""Visualize a single-level 2D Haar-DWT of a grayscale image.

The input RGB image is converted to luminance-based grayscale and decomposed
into one low-frequency approximation and three signed detail subbands. The
publication-oriented output uses a 2 x 3 layout:

    Row 1: RGB input | grayscale input | low-frequency approximation
    Row 2: horizontal details | vertical details | diagonal details

For improved visibility, the detail panels display absolute coefficient
responses using one shared robust grayscale range. The saved ``.npz`` file
still contains the unmodified signed coefficients.

Dependencies:

    pip install pillow numpy matplotlib torch pytorch-wavelets

Example:

    python visualize_grayscale_haar_dwt.py image.png \
        --output-dir grayscale_haar_dwt --resize 760 1280
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grayscale-haar-dwt")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from pytorch_wavelets import DWTForward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2x3 visualization of a grayscale single-level Haar-DWT."
    )
    parser.add_argument("input_image", type=Path, help="Path to an RGB input image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grayscale_haar_dwt"),
        help="Output directory (default: grayscale_haar_dwt).",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional resize applied before grayscale conversion.",
    )
    parser.add_argument(
        "--swap-horizontal-vertical",
        action="store_true",
        help=(
            "Swap the first two detail panels if the H/V naming convention "
            "used in the thesis follows the reverse LH/HL assignment."
        ),
    )
    parser.add_argument(
        "--detail-percentile",
        type=float,
        default=99.5,
        help=(
            "Percentile of absolute detail coefficients used as the shared "
            "upper display limit (default: 99.5)."
        ),
    )
    parser.add_argument(
        "--approximation-percentiles",
        nargs=2,
        type=float,
        default=(0.5, 99.5),
        metavar=("LOW", "HIGH"),
        help="Display percentiles for the approximation subband.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Computation device (default: auto).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def load_rgb(
    path: Path,
    resize: tuple[int, int] | None,
    device: torch.device,
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    image = Image.open(path).convert("RGB")
    if resize is not None:
        height, width = resize
        image = image.resize((width, height), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def rgb_to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0,1] to luminance using standard fixed weights."""
    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        dtype=rgb.dtype,
        device=rgb.device,
    ).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


@torch.no_grad()
def haar_dwt(grayscale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    transform = DWTForward(J=1, wave="haar").to(grayscale.device)
    approximation, details_per_level = transform(grayscale)
    details = details_per_level[0]  # [B, 1, 3, H/2, W/2]: LH, HL, HH
    return approximation, details


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def finite_values(array: np.ndarray) -> np.ndarray:
    values = array[np.isfinite(array)]
    if values.size == 0:
        raise ValueError("An output array contains no finite values.")
    return values


def approximation_limits(
    approximation: np.ndarray,
    percentiles: tuple[float, float],
) -> tuple[float, float]:
    low_percentile, high_percentile = percentiles
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("Approximation percentiles must satisfy 0 <= low < high <= 100.")
    low, high = np.percentile(
        finite_values(approximation),
        [low_percentile, high_percentile],
    )
    if high <= low:
        high = low + 1e-8
    return float(low), float(high)


def detail_limit(details: list[np.ndarray], percentile: float) -> float:
    if not 0.0 < percentile <= 100.0:
        raise ValueError("The detail percentile must be in (0, 100].")
    all_absolute = np.concatenate(
        [np.abs(finite_values(detail)).ravel() for detail in details]
    )
    return max(float(np.percentile(all_absolute, percentile)), 1e-8)


def create_figure(
    rgb: np.ndarray,
    grayscale: np.ndarray,
    approximation: np.ndarray,
    details: list[np.ndarray],
    output_dir: Path,
    dpi: int,
    approximation_percentiles: tuple[float, float],
    detail_percentile: float,
    swap_horizontal_vertical: bool,
) -> None:
    detail_labels = ["Horizontal details", "Vertical details", "Diagonal details"]
    if swap_horizontal_vertical:
        details[0], details[1] = details[1], details[0]

    approximation_vmin, approximation_vmax = approximation_limits(
        approximation,
        approximation_percentiles,
    )
    shared_detail_limit = detail_limit(details, detail_percentile)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.2, 8.4),
        layout="constrained",
    )

    axes[0, 0].imshow(np.clip(rgb, 0.0, 1.0))
    axes[0, 0].set_title("(a) RGB input")

    axes[0, 1].imshow(grayscale, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("(b) Grayscale input")

    axes[0, 2].imshow(
        approximation,
        cmap="gray",
        vmin=approximation_vmin,
        vmax=approximation_vmax,
    )
    axes[0, 2].set_title("(c) Low-frequency approximation")

    for axis, detail, label, panel_letter in zip(
        axes[1],
        details,
        detail_labels,
        ["d", "e", "f"],
    ):
        axis.imshow(
            np.abs(detail),
            cmap="gray",
            vmin=0.0,
            vmax=shared_detail_limit,
        )
        axis.set_title(f"({panel_letter}) {label}")

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    figure.savefig(
        output_dir / "grayscale_haar_dwt.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_dir / "grayscale_haar_dwt.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resize = tuple(args.resize) if args.resize is not None else None
    rgb_tensor = load_rgb(args.input_image, resize, device)
    grayscale_tensor = rgb_to_grayscale(rgb_tensor)
    approximation_tensor, details_tensor = haar_dwt(grayscale_tensor)

    rgb = np.moveaxis(to_numpy(rgb_tensor)[0], 0, -1)
    grayscale = to_numpy(grayscale_tensor)[0, 0]
    approximation = to_numpy(approximation_tensor)[0, 0]
    raw_details = to_numpy(details_tensor)[0, 0]
    details = [raw_details[index] for index in range(3)]

    create_figure(
        rgb,
        grayscale,
        approximation,
        details,
        args.output_dir,
        args.dpi,
        tuple(args.approximation_percentiles),
        args.detail_percentile,
        args.swap_horizontal_vertical,
    )

    np.savez_compressed(
        args.output_dir / "grayscale_haar_dwt_arrays.npz",
        rgb=rgb,
        grayscale=grayscale,
        approximation=approximation,
        lh=raw_details[0],
        hl=raw_details[1],
        hh=raw_details[2],
    )

    metadata = {
        "input_image": str(args.input_image.resolve()),
        "device": str(device),
        "resize_height_width": resize,
        "grayscale_weights_rgb": [0.299, 0.587, 0.114],
        "wavelet": "haar",
        "decomposition_levels": 1,
        "raw_orientation_order": ["LH", "HL", "HH"],
        "display_orientation_labels": (
            ["Horizontal (HL)", "Vertical (LH)", "Diagonal (HH)"]
            if args.swap_horizontal_vertical
            else ["Horizontal (LH)", "Vertical (HL)", "Diagonal (HH)"]
        ),
        "detail_display": (
            "absolute coefficient responses with a shared grayscale range "
            f"from zero to the {args.detail_percentile}th absolute percentile; "
            "the NPZ file retains the signed coefficients"
        ),
        "tensor_shapes": {
            "rgb_hwc": list(rgb.shape),
            "grayscale_hw": list(grayscale.shape),
            "approximation_hw": list(approximation.shape),
            "each_detail_hw": list(raw_details[0].shape),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved figure and coefficient arrays to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()