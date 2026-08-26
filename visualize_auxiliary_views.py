#!/usr/bin/env python3
"""Create one publication-ready comparison of all auxiliary views.

The figure uses four rows:

    1. the geometrically augmented RGB image, centered;
    2. multi-scale Scharr edges, centered logarithmic Fourier magnitude,
       and DCT coefficient magnitude;
    3. horizontal, vertical, and diagonal Haar-wavelet details; and
    4. hue, saturation, and value channels.

The exact model-input tensors are saved as ``.npy`` files. Display-only
operations are deliberately separate:

* DCT is shown as log(1 + abs(raw coefficients)), although the model receives
  signed, standardized coefficients.
* Each DWT panel shows the mean absolute response across the three RGB
  channels, although the model receives all nine signed, standardized detail
  channels.
* Hue is normalized from Kornia's [0, 2*pi) range to [0, 1] for display.

Dependencies:

    pip install pillow numpy matplotlib torch kornia torch-dct pytorch-wavelets

Example:

    python visualize_auxiliary_views.py image.png \
        --output-dir auxiliary_views --resize 760 1280
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-auxiliary-views")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import kornia
from pytorch_wavelets import DWTForward
from torch_dct import dct_2d


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_SIGMAS = (0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all image-derived auxiliary views in one figure."
    )
    parser.add_argument("input_image", type=Path, help="Path to an RGB image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("auxiliary_views"),
        help="Output directory (default: auxiliary_views).",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional size used before generating the views.",
    )
    parser.add_argument(
        "--mean",
        nargs=3,
        type=float,
        default=DEFAULT_MEAN,
        metavar=("R", "G", "B"),
        help="Auxiliary-branch normalization mean (default: ImageNet).",
    )
    parser.add_argument(
        "--std",
        nargs=3,
        type=float,
        default=DEFAULT_STD,
        metavar=("R", "G", "B"),
        help="Auxiliary-branch normalization standard deviation (default: ImageNet).",
    )
    parser.add_argument(
        "--no-normalize-auxiliary",
        action="store_true",
        help="Generate edge/frequency views directly from RGB values in [0,1].",
    )
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=DEFAULT_SIGMAS,
        help="Gaussian scales for Scharr filtering (default: 0.5 1.0 2.0).",
    )
    parser.add_argument(
        "--swap-wavelet-hv",
        action="store_true",
        help=(
            "Swap the first two DWT display panels if the horizontal/vertical "
            "naming convention in the thesis uses the reverse LH/HL mapping."
        ),
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


def normalize_rgb(
    image: torch.Tensor,
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(tuple(mean), device=image.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(tuple(std), device=image.device).view(1, 3, 1, 1)
    if torch.any(std_tensor <= 0):
        raise ValueError("All normalization standard deviations must be positive.")
    return (image - mean_tensor) / std_tensor


def standardize_spatially(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.mean(dim=(2, 3), keepdim=True)
    std = tensor.std(dim=(2, 3), keepdim=True)
    return (tensor - mean) / (std + 1e-6)


@torch.no_grad()
def multiscale_scharr(
    image: torch.Tensor,
    sigmas: Iterable[float],
) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3)

    responses = []
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("Every Gaussian sigma must be positive.")
        blurred = kornia.filters.gaussian_blur2d(
            gray, (5, 5), (float(sigma), float(sigma))
        )
        gradient_x = F.conv2d(blurred, kernel_x, padding=1)
        gradient_y = F.conv2d(blurred, kernel_y, padding=1)
        responses.append(torch.sqrt(gradient_x.square() + gradient_y.square()))

    maximum = torch.stack(responses, dim=0).amax(dim=0)
    return maximum / (maximum.amax(dim=(2, 3), keepdim=True) + 1e-6)


@torch.no_grad()
def fourier_magnitude(image: torch.Tensor) -> torch.Tensor:
    coefficients = torch.fft.fft2(image)
    centered = torch.fft.fftshift(coefficients, dim=(-2, -1))
    magnitude = centered.abs().mean(dim=1, keepdim=True)
    return standardize_spatially(torch.log1p(magnitude))


@torch.no_grad()
def dct_representation(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = dct_2d(image).mean(dim=1, keepdim=True)
    return standardize_spatially(raw), raw


@torch.no_grad()
def wavelet_representation(image: torch.Tensor) -> torch.Tensor:
    transform = DWTForward(J=1, wave="haar").to(image.device)
    _, high_frequency = transform(image)
    details = high_frequency[0]  # [B, C, 3, H/2, W/2]: LH, HL, HH
    batch, channels, orientations, height, width = details.shape
    details = details.reshape(batch, channels * orientations, height, width)
    details = F.interpolate(
        details,
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return standardize_spatially(details)


@torch.no_grad()
def hsv_representation(raw_rgb: torch.Tensor) -> torch.Tensor:
    return kornia.color.rgb_to_hsv(raw_rgb)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze(0).numpy().astype(np.float32)


def robust_limits(array: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [100.0 - percentile, percentile])
    if high <= low:
        return float(low), float(low + 1.0)
    return float(low), float(high)


def show_scalar(
    axis: plt.Axes,
    array: np.ndarray,
    *,
    cmap: str = "gray",
    fixed_range: tuple[float, float] | None = None,
) -> None:
    vmin, vmax = fixed_range or robust_limits(array)
    axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks([])
    axis.set_yticks([])


def add_single_panel(
    subfigure: plt.SubFigure,
    array: np.ndarray,
    title: str,
    *,
    cmap: str | None = None,
    fixed_range: tuple[float, float] | None = None,
) -> None:
    axis = subfigure.subplots()
    if array.ndim == 3:
        axis.imshow(np.clip(array, 0.0, 1.0))
        axis.set_xticks([])
        axis.set_yticks([])
    else:
        show_scalar(axis, array, cmap=cmap or "gray", fixed_range=fixed_range)
    axis.set_title(title, fontsize=11, pad=7)


def create_overview(
    arrays: dict[str, np.ndarray],
    raw_dct: np.ndarray,
    output_dir: Path,
    dpi: int,
    swap_wavelet_hv: bool,
) -> None:
    rgb = np.moveaxis(arrays["rgb"], 0, -1)
    edge = arrays["edge"][0]
    fft = arrays["fft"][0]
    dct_display = np.log1p(np.abs(raw_dct[0]))

    # Reshape the nine channels as RGB x (LH, HL, HH), then average absolute
    # responses across RGB solely for visualization.
    wavelet = arrays["dwt"].reshape(3, 3, *arrays["dwt"].shape[-2:])
    wavelet_panels = [np.abs(wavelet[:, index]).mean(axis=0) for index in range(3)]
    wavelet_labels = ["H (LH)", "V (HL)", "D (HH)"]
    if swap_wavelet_hv:
        wavelet_panels[0], wavelet_panels[1] = wavelet_panels[1], wavelet_panels[0]
        wavelet_labels = ["H (HL)", "V (LH)", "D (HH)"]

    hsv = arrays["hsv"]
    hsv_panels = [
        np.mod(hsv[0], 2.0 * math.pi) / (2.0 * math.pi),
        hsv[1],
        hsv[2],
    ]

    # RGB is shown as the common source in a centered first row. The derived
    # single-channel views, DWT orientations, and HSV channels receive their
    # own rows of three equally sized panels.
    figure = plt.figure(figsize=(15.5, 14.2), layout="constrained")
    rows = figure.subfigures(
        4,
        1,
        height_ratios=(1.0, 1.0, 1.0, 1.0),
        hspace=0.065,
    )
    rgb_row = rows[0].subfigures(1, 3, wspace=0.025)
    derived_panels = rows[1].subfigures(1, 3, wspace=0.025)
    wavelet_panels_fig = rows[2].subfigures(1, 3, wspace=0.025)
    hsv_panels_fig = rows[3].subfigures(1, 3, wspace=0.025)

    # Empty side cells center RGB while keeping its width equal to all other
    # panels in the figure.
    for side_panel in (rgb_row[0], rgb_row[2]):
        side_axis = side_panel.subplots()
        side_axis.axis("off")
    add_single_panel(rgb_row[1], rgb, "(a) Geometrically augmented RGB")

    add_single_panel(
        derived_panels[0],
        edge,
        "(b) Multi-scale Scharr edges",
        fixed_range=(0.0, 1.0),
    )
    add_single_panel(
        derived_panels[1],
        fft,
        "(c) Centered log Fourier magnitude",
    )
    add_single_panel(
        derived_panels[2],
        dct_display,
        "(d) Log DCT coefficient magnitude",
        cmap="gray",
    )

    for subfigure, array, label, panel_letter in zip(
        wavelet_panels_fig,
        wavelet_panels,
        wavelet_labels,
        ["e", "f", "g"],
    ):
        add_single_panel(
            subfigure,
            array,
            f"({panel_letter}) Haar detail: {label}",
            cmap="gray",
        )

    for subfigure, array, label, cmap, panel_letter in zip(
        hsv_panels_fig,
        hsv_panels,
        ["Hue", "Saturation", "Value"],
        ["hsv", "gray", "gray"],
        ["h", "i", "j"],
    ):
        add_single_panel(
            subfigure,
            array,
            f"({panel_letter}) HSV: {label}",
            cmap=cmap,
        )

    figure.savefig(
        output_dir / "auxiliary_views_overview.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_dir / "auxiliary_views_overview.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resize = tuple(args.resize) if args.resize is not None else None
    raw_rgb = load_rgb(args.input_image, resize, device)
    auxiliary_rgb = (
        raw_rgb
        if args.no_normalize_auxiliary
        else normalize_rgb(raw_rgb, args.mean, args.std)
    )

    dct, raw_dct = dct_representation(auxiliary_rgb)
    tensors = {
        "rgb": raw_rgb,
        "edge": multiscale_scharr(auxiliary_rgb, args.sigmas),
        "fft": fourier_magnitude(auxiliary_rgb),
        "dct": dct,
        "dwt": wavelet_representation(auxiliary_rgb),
        "hsv": hsv_representation(raw_rgb),
    }
    arrays = {name: to_numpy(tensor) for name, tensor in tensors.items()}
    raw_dct_array = to_numpy(raw_dct)

    for name, array in arrays.items():
        np.save(args.output_dir / f"{name}.npy", array)
    np.save(args.output_dir / "dct_raw.npy", raw_dct_array)

    create_overview(
        arrays,
        raw_dct_array,
        args.output_dir,
        args.dpi,
        args.swap_wavelet_hv,
    )

    metadata = {
        "input_image": str(args.input_image.resolve()),
        "device": str(device),
        "resize_height_width": resize,
        "auxiliary_normalization": (
            None
            if args.no_normalize_auxiliary
            else {"mean": list(args.mean), "std": list(args.std)}
        ),
        "scharr_sigmas": list(args.sigmas),
        "tensor_shapes_chw": {name: list(array.shape) for name, array in arrays.items()},
        "display_only_operations": {
            "dct": "log(1 + abs(raw channel-averaged DCT coefficients))",
            "dwt": "mean absolute response across RGB, shown per orientation",
            "hue": "Kornia radians in [0, 2*pi) scaled to [0, 1]",
        },
        "wavelet_display_labels": (
            ["H (HL)", "V (LH)", "D (HH)"]
            if args.swap_wavelet_hv
            else ["H (LH)", "V (HL)", "D (HH)"]
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved figure and tensors to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()