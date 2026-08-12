#!/usr/bin/env python3
"""Generate the five image-derived auxiliary views used in the thesis.

The script creates

1. normalized multi-scale Scharr edges (1 channel),
2. centered log Fourier magnitude (1 channel),
3. global two-dimensional DCT coefficients (1 channel),
4. single-level Haar wavelet detail coefficients (9 channels), and
5. HSV channels (3 channels).

Exact tensors are written as ``.npy`` files. Publication-oriented PNG and PDF
figures are visualizations of these tensors. The wavelet overview shows the
mean absolute response of the nine signed detail channels. The DCT overview
shows the logarithmically compressed coefficient magnitude, whereas the model
receives the signed, standardized coefficients stored in ``dct.npy``.

Dependencies (in addition to PyTorch):

    pip install pillow numpy matplotlib kornia torch-dct pytorch-wavelets

Example:

    python visualize_auxiliary_views.py input.png --output-dir auxiliary_views

To use raw RGB values for the edge/frequency views instead of ImageNet-
normalized values:

    python visualize_auxiliary_views.py input.png \
        --output-dir auxiliary_views --no-normalize-auxiliary
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

# Prevent matplotlib from trying to write into a read-only user directory.
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
        description="Generate Scharr, FFT, DCT, Haar-wavelet, and HSV views."
    )
    parser.add_argument("input_image", type=Path, help="Path to an RGB input image.")
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
        help="Optional resize before view generation.",
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
        help="Use RGB values in [0,1] for Scharr/FFT/DCT/wavelet generation.",
    )
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=DEFAULT_SIGMAS,
        help="Gaussian scales for multi-scale Scharr (default: 0.5 1.0 2.0).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Computation device (default: auto).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300).")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def load_rgb(path: Path, resize: tuple[int, int] | None, device: torch.device) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    image = Image.open(path).convert("RGB")
    if resize is not None:
        height, width = resize
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def normalize_rgb(images: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(tuple(mean), device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(tuple(std), device=images.device).view(1, 3, 1, 1)
    if torch.any(std_tensor <= 0):
        raise ValueError("All normalization standard deviations must be positive.")
    return (images - mean_tensor) / std_tensor


@torch.no_grad()
def multiscale_scharr_edges(
    images: torch.Tensor,
    sigmas: Iterable[float] = DEFAULT_SIGMAS,
) -> torch.Tensor:
    """Reproduce the one-channel multi-scale Scharr implementation."""
    gray = images.mean(dim=1, keepdim=True)
    device = images.device

    kernel_x = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3)

    edges_per_scale = []
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("Every Scharr smoothing sigma must be positive.")
        blurred = kornia.filters.gaussian_blur2d(gray, (5, 5), (sigma, sigma))
        grad_x = F.conv2d(blurred, kernel_x, padding=1)
        grad_y = F.conv2d(blurred, kernel_y, padding=1)
        edges_per_scale.append(torch.sqrt(grad_x.square() + grad_y.square()))

    edges = torch.stack(edges_per_scale, dim=0).amax(dim=0)
    return edges / (edges.amax(dim=(2, 3), keepdim=True) + 1e-6)


@torch.no_grad()
def fft_magnitude_1ch(images: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft2(images)
    centered = torch.fft.fftshift(spectrum, dim=(-2, -1))
    magnitude = torch.abs(centered).mean(dim=1, keepdim=True)
    log_magnitude = torch.log1p(magnitude)
    mean = log_magnitude.mean(dim=(2, 3), keepdim=True)
    std = log_magnitude.std(dim=(2, 3), keepdim=True)
    return (log_magnitude - mean) / (std + 1e-6)


@torch.no_grad()
def dct_coefficients_1ch(images: torch.Tensor) -> torch.Tensor:
    """Compute the signed global DCT coefficients before standardization."""
    return dct_2d(images).mean(dim=1, keepdim=True)


@torch.no_grad()
def standardize_dct(coefficients: torch.Tensor) -> torch.Tensor:
    """Apply the same spatial standardization as the model implementation."""
    mean = coefficients.mean(dim=(2, 3), keepdim=True)
    std = coefficients.std(dim=(2, 3), keepdim=True)
    return (coefficients - mean) / (std + 1e-6)


@torch.no_grad()
def wavelet_details_9ch(images: torch.Tensor) -> torch.Tensor:
    transform = DWTForward(J=1, wave="haar").to(images.device)
    _, high_frequency = transform(images)
    details = high_frequency[0]
    batch, channels, orientations, height, width = details.shape
    details = details.reshape(batch, channels * orientations, height, width)
    details = F.interpolate(
        details,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    mean = details.mean(dim=(2, 3), keepdim=True)
    std = details.std(dim=(2, 3), keepdim=True)
    return (details - mean) / (std + 1e-6)


@torch.no_grad()
def hsv_3ch(raw_rgb: torch.Tensor) -> torch.Tensor:
    # Kornia represents hue in [0, 2*pi), while saturation/value remain in [0,1].
    return kornia.color.rgb_to_hsv(raw_rgb)


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().squeeze(0).numpy().astype(np.float32)


def robust_unit_interval(array: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(finite, [lower, upper])
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def save_scalar_image(
    array: np.ndarray,
    path: Path,
    dpi: int,
    cmap: str = "gray",
    symmetric: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.axis("off")
    if symmetric:
        bound = float(np.percentile(np.abs(array[np.isfinite(array)]), 99.0))
        bound = max(bound, 1e-8)
        axis.imshow(array, cmap=cmap, vmin=-bound, vmax=bound)
    else:
        axis.imshow(robust_unit_interval(array), cmap=cmap, vmin=0.0, vmax=1.0)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def create_hsv_preview(hsv: np.ndarray) -> np.ndarray:
    """Encode H, S, V as display RGB channels; this is not HSV-to-RGB conversion."""
    hue = np.mod(hsv[0], 2.0 * math.pi) / (2.0 * math.pi)
    saturation = np.clip(hsv[1], 0.0, 1.0)
    value = np.clip(hsv[2], 0.0, 1.0)
    return np.stack([hue, saturation, value], axis=-1).astype(np.float32)


def create_wavelet_orientation_previews(wavelet: np.ndarray) -> list[np.ndarray]:
    # DWTForward orders the orientation dimension as LH, HL, HH. After reshape,
    # channels are grouped as RGB x (LH, HL, HH).
    reshaped = wavelet.reshape(3, 3, wavelet.shape[-2], wavelet.shape[-1])
    return [np.mean(np.abs(reshaped[:, index]), axis=0) for index in range(3)]


def save_channel_figure(
    channels: list[np.ndarray],
    titles: list[str],
    path: Path,
    dpi: int,
    cmaps: list[str] | None = None,
) -> None:
    figure, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4.5))
    axes = np.atleast_1d(axes)
    cmaps = cmaps or ["gray"] * len(channels)
    for axis, channel, title, cmap in zip(axes, channels, titles, cmaps):
        axis.imshow(robust_unit_interval(channel), cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_overview(
    rgb: np.ndarray,
    edges: np.ndarray,
    fft: np.ndarray,
    dct_display: np.ndarray,
    wavelet_preview: np.ndarray,
    hsv_preview: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    panels = [
        (np.moveaxis(rgb, 0, -1), "Geometrically augmented RGB", None),
        (edges, "Multi-scale Scharr", "gray"),
        (fft, "Centered log FFT magnitude", "gray"),
        (dct_display, "Global DCT coefficient magnitude (log display)", "magma"),
        (wavelet_preview, "Mean absolute Haar detail response (display)", "gray"),
        (hsv_preview, "HSV channels encoded as H/S/V", None),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (array, title, cmap) in zip(axes.flat, panels):
        if array.ndim == 2:
            axis.imshow(robust_unit_interval(array), cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            axis.imshow(np.clip(array, 0.0, 1.0))
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_dir / "auxiliary_views_overview.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "auxiliary_views_overview.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    resize = tuple(args.resize) if args.resize is not None else None
    raw_rgb = load_rgb(args.input_image, resize=resize, device=device)
    auxiliary_rgb = (
        raw_rgb
        if args.no_normalize_auxiliary
        else normalize_rgb(raw_rgb, mean=args.mean, std=args.std)
    )

    with torch.no_grad():
        raw_dct = dct_coefficients_1ch(auxiliary_rgb)
        tensors = {
            "rgb": raw_rgb,
            "multiscale_scharr": multiscale_scharr_edges(auxiliary_rgb, args.sigmas),
            "fft_magnitude": fft_magnitude_1ch(auxiliary_rgb),
            "dct": standardize_dct(raw_dct),
            "wavelet_details": wavelet_details_9ch(auxiliary_rgb),
            "hsv": hsv_3ch(raw_rgb),
        }

    arrays = {name: as_numpy(tensor) for name, tensor in tensors.items()}
    for name, array in arrays.items():
        np.save(args.output_dir / f"{name}.npy", array)

    edges = arrays["multiscale_scharr"][0]
    fft = arrays["fft_magnitude"][0]
    dct = arrays["dct"][0]
    raw_dct_array = as_numpy(raw_dct)
    np.save(args.output_dir / "dct_coefficients_raw.npy", raw_dct_array)
    dct_display = np.log1p(np.abs(raw_dct_array[0]))
    wavelet_orientations = create_wavelet_orientation_previews(arrays["wavelet_details"])
    wavelet_preview = np.mean(np.abs(arrays["wavelet_details"]), axis=0)
    hsv = arrays["hsv"]
    hsv_preview = create_hsv_preview(hsv)

    save_scalar_image(edges, args.output_dir / "multiscale_scharr.png", args.dpi)
    save_scalar_image(fft, args.output_dir / "fft_magnitude.png", args.dpi)
    save_scalar_image(
        dct_display,
        args.output_dir / "dct_log_magnitude.png",
        args.dpi,
        cmap="magma",
    )
    save_scalar_image(
        wavelet_preview,
        args.output_dir / "wavelet_mean_absolute_response.png",
        args.dpi,
    )
    save_channel_figure(
        wavelet_orientations,
        ["LH details", "HL details", "HH details"],
        args.output_dir / "wavelet_orientations.png",
        args.dpi,
    )
    save_channel_figure(
        [np.mod(hsv[0], 2.0 * math.pi) / (2.0 * math.pi), hsv[1], hsv[2]],
        ["Hue", "Saturation", "Value"],
        args.output_dir / "hsv_channels.png",
        args.dpi,
        cmaps=["hsv", "gray", "gray"],
    )

    save_overview(
        rgb=arrays["rgb"],
        edges=edges,
        fft=fft,
        dct_display=dct_display,
        wavelet_preview=wavelet_preview,
        hsv_preview=hsv_preview,
        output_dir=args.output_dir,
        dpi=args.dpi,
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
        "notes": {
            "wavelet": "Exact model input has 9 channels: RGB x (LH, HL, HH).",
            "wavelet_display": "The overview uses the mean absolute response across the 9 signed detail channels.",
            "dct": "dct.npy contains the signed standardized model input; dct_coefficients_raw.npy contains signed coefficients before standardization.",
            "dct_display": "dct_log_magnitude.png and the overview show log(1 + abs(raw DCT coefficient)).",
            "hsv": "Kornia hue is stored in radians in [0, 2*pi).",
            "visualizations": "PNG/PDF files use display mappings; NPY files contain exact tensors.",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Generated auxiliary views in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
