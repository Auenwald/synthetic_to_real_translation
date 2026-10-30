#!/usr/bin/env python3
"""Plot RGB, a global DFT log-magnitude spectrum, and signed global DCT-II.

Install:
    python -m pip install numpy scipy matplotlib pillow
Run from the directory containing gta5/:
    python frequency_representations.py
    python frequency_representations.py --images 00001.png 00002.png
    python frequency_representations.py --representation model --save-arrays

Defaults: ./gta5/images/00001.png -> ./figures/frequency_representations_00001_theory.*
Each image produces PDF, PNG, a LaTeX figure snippet, and JSON settings.
No resizing, cropping, windowing, or image augmentation is applied.
RGB is converted from 8-bit intensities to float64 in [0, 1].

The transforms are applied to each RGB channel BEFORE channel averaging:
  FFT: fft2 -> fftshift -> abs -> channel mean -> log1p -> spatial z-score.
  DCT: dctn(type=2) -> signed channel mean -> spatial z-score.
The z-score uses ddof=1 and epsilon=1e-6, matching the supplied torch.std
default. NumPy/SciPy is used so that no GPU or PyTorch installation is needed.

--representation theory (default) plots the maps BEFORE z-scoring; model
plots the standardized maps. DCT signs are retained in both cases.
For visibility, DCT COLORS use SymLogNorm and both color ranges use percentile
limits. These display operations NEVER modify the transform/model arrays.
The FFT and DCT color bars are independent, not comparable energy scales.

The supplied dct_2d import/definition was not provided. This script explicitly
assumes DCT-II with norm='backward' (unnormalized SciPy convention).
Use --dct-norm ortho if your implementation uses orthonormal DCT-II.
This is not a claim of bitwise equivalence to an unspecified dct_2d function.

References:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html
  https://docs.pytorch.org/docs/stable/generated/torch.fft.fftshift.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn, fft2, fftfreq, fftshift


def standardize(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Spatial sample standard deviation, as in the supplied torch snippet."""
    return (values - values.mean()) / (values.std(ddof=1) + eps)


def compute_maps(rgb: np.ndarray, dct_norm: str = "backward") -> dict:
    """Input H x W x 3, float RGB in [0, 1]; return unmodified 2-D maps."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError("Expected an RGB image of shape H x W x 3, H,W >= 2.")
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Expected finite RGB values in [0, 1].")
    spectrum = fftshift(fft2(rgb, axes=(0, 1), norm="backward"), axes=(0, 1))
    fft_log_magnitude = np.log1p(np.abs(spectrum).mean(axis=2))
    del spectrum
    dct_mean = dctn(rgb, type=2, axes=(0, 1), norm=dct_norm).mean(axis=2)
    return {
        "fft_log_magnitude": fft_log_magnitude,
        "dct_mean_coefficients": dct_mean,
        "fft_standardized": standardize(fft_log_magnitude),
        "dct_standardized": standardize(dct_mean),
    }


def display_normalizations(fft_map, dct_map, percentile):
    low, high = np.percentile(fft_map, [100 - percentile, percentile])
    if high <= low:
        low, high = float(fft_map.min()), float(fft_map.max())
    if high <= low:
        low, high = low - 0.5, high + 0.5
    limit = float(np.percentile(np.abs(dct_map), percentile))
    if limit <= np.finfo(float).eps:
        limit = max(float(np.abs(dct_map).max()), 1.0)
    return Normalize(low, high), SymLogNorm(
        linthresh=limit * 0.01, linscale=1, vmin=-limit, vmax=limit, base=10
    )


def make_figure(rgb, maps, representation, percentile):
    height, width = rgb.shape[:2]
    model = representation == "model"
    fft_map = maps["fft_standardized" if model else "fft_log_magnitude"]
    dct_map = maps["dct_standardized" if model else "dct_mean_coefficients"]
    fft_norm, dct_norm = display_normalizations(fft_map, dct_map, percentile)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(9, max(2.9, 3 * height / width + 1.2)),
                     layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.045])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    axes[0].imshow(rgb, interpolation="nearest", aspect="auto")
    axes[0].set_title("(a) RGB image")
    axes[0].set_axis_off()

    fx, fy = fftshift(fftfreq(width)), fftshift(fftfreq(height))
    fft_extent = (fx[0] - 0.5 / width, fx[-1] + 0.5 / width,
                  fy[0] - 0.5 / height, fy[-1] + 0.5 / height)
    fft_image = axes[1].imshow(
        fft_map, cmap="magma", norm=fft_norm, origin="lower",
        extent=fft_extent, interpolation="nearest", aspect="auto", rasterized=True
    )
    axes[1].set_title("(b) DFT log-magnitude")
    axes[1].set_xlabel(r"Horizontal frequency $f_x$ (cycles/pixel)")
    axes[1].set_ylabel(r"Vertical frequency $f_y$")
    axes[1].set_xticks([-0.5, 0, 0.5])
    axes[1].set_yticks([-0.5, 0, 0.5])
    axes[1].set_xlim(fft_extent[:2])
    axes[1].set_ylim(fft_extent[2:])

    dct_image = axes[2].imshow(
        dct_map, cmap="RdBu_r", norm=dct_norm, origin="upper",
        interpolation="nearest", aspect="auto", rasterized=True
    )
    axes[2].set_title("(c) Signed DCT coefficients")
    axes[2].set_xlabel(r"Horizontal frequency index $l$")
    axes[2].set_ylabel(r"Vertical frequency index $k$")
    axes[2].set_xticks([0, (width - 1) // 2, width - 1])
    axes[2].set_yticks([0, (height - 1) // 2, height - 1])
    cb_fft = fig.colorbar(fft_image, cax=fig.add_subplot(grid[1, 1]),
                         orientation="horizontal", extend="both")
    cb_fft.locator = MaxNLocator(nbins=4)
    cb_fft.update_ticks()
    cb_fft.set_label("Standardized log-magnitude" if model else "Log-magnitude")
    cb_dct = fig.colorbar(dct_image, cax=fig.add_subplot(grid[1, 2]),
                         orientation="horizontal", extend="both")
    # Fixed symmetric ticks keep the signed, nonlinear color scale readable.
    vmax, threshold = dct_norm.vmax, dct_norm.linthresh
    ticks = [-vmax, -threshold, 0.0, threshold, vmax]
    cb_dct.set_ticks(ticks, labels=[f"{value:.2g}" for value in ticks])
    cb_dct.set_label(("Standardized coefficients" if model else "Coefficients")
                     + " (symlog color scale)")
    settings = {
        "fft_color_limits": [float(fft_norm.vmin), float(fft_norm.vmax)],
        "dct_color_limits": [float(dct_norm.vmin), float(dct_norm.vmax)],
        "dct_symlog_linear_threshold": float(dct_norm.linthresh),
    }
    return fig, settings


def latex_figure(stem, representation, percentile, dct_norm):
    standardization = (
        "Both resulting maps are standardized separately over their spatial dimensions. "
        if representation == "model" else
        "The maps are shown before the spatial standardization used for model input. "
    )
    lower = 100 - percentile
    normalization = ("unnormalized" if dct_norm == "backward" else "orthonormal")
    caption = (
        r"\textbf{Global frequency-domain representations of a GTA5 image.} "
        r"(a) RGB input; (b) channel-averaged DFT magnitudes, computed via FFT "
        r"and displayed after centering and $\log(1+x)$ scaling; "
        f"(c) signed, channel-averaged {normalization} DCT-II coefficients. "
        r"Both transforms are applied globally and independently to each RGB channel. "
        r"Zero frequency is located at the center of (b) and the upper-left corner of (c). "
        + standardization +
        r"Panel (c) uses a symmetric logarithmic color scale for visualization only. "
        f"For visibility, the color limits in (b) are the {lower:g}th and "
        f"{percentile:g}th percentiles; in (c), they are symmetric around zero, "
        f"using the {percentile:g}th percentile of absolute values. "
        r"The two color scales are independent. Source: own illustration based on "
        r"an image from GTA5~\cite{gta5}."
    )
    return (
        "\\begin{figure}[htbp]\n    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{figures/{stem}.pdf}}\n"
        "    \\caption[Global DFT and DCT representations]{%\n"
        f"    {caption}}}\n"
        f"    \\label{{fig:{stem}}}\n\\end{{figure}}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path, default=Path("gta5/images"))
    parser.add_argument("--images", nargs="+", default=["00001.png"])
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--representation", choices=["theory", "model"], default="theory")
    parser.add_argument("--dct-norm", choices=["backward", "ortho"], default="backward")
    parser.add_argument("--percentile", type=float, default=99.5,
                        help="Upper display percentile, 50 < p <= 100 (default: 99.5).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save-arrays", action="store_true",
                        help="Also save raw and standardized maps in a compressed NPZ.")
    args = parser.parse_args()
    if not 50 < args.percentile <= 100 or args.dpi <= 0:
        parser.error("Require 50 < --percentile <= 100 and --dpi > 0.")
    paths = [Path(name) if Path(name).is_absolute() else args.image_dir / name
             for name in args.images]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Image file(s) not found: " + ", ".join(missing))
    stems = ["frequency_representations_"
             + re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)
             + "_" + args.representation for path in paths]
    if len(stems) != len(set(stems)):
        parser.error("Image filenames would create duplicate output names.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source, stem in zip(paths, stems):
        with Image.open(source) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"),
                             dtype=np.float64) / 255.0
        maps = compute_maps(rgb, args.dct_norm)
        fig, display_settings = make_figure(rgb, maps, args.representation,
                                           args.percentile)
        base = args.output_dir / stem
        for extension in ("pdf", "png"):
            destination = base.with_suffix("." + extension)
            fig.savefig(destination, dpi=args.dpi)
            print(destination)
        plt.close(fig)
        base.with_suffix(".tex").write_text(
            latex_figure(stem, args.representation, args.percentile, args.dct_norm),
            encoding="utf-8")
        metadata = {
            "source_image": str(source), "image_shape_HWC": list(rgb.shape),
            "input_scaling": "8-bit RGB / 255; no resize, crop, or window",
            "transforms": "global, channel-wise, then channel-averaged",
            "dct_type": 2, "dct_norm": args.dct_norm,
            "representation": args.representation,
            "standardization_ddof": 1, "standardization_epsilon": 1e-6,
            "display_percentile": args.percentile,
            "display_changes_arrays": False, **display_settings,
        }
        base.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n",
                                             encoding="utf-8")
        if args.save_arrays:
            np.savez_compressed(base.with_suffix(".npz"), **maps)


if __name__ == "__main__":
    main()
