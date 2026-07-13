import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
image_path = Path("./gta5/images/00020.png")
output_path = Path("./augmentation_overview_gta5_00020_compact.pdf")

np.random.seed(42)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_for_panel(img: np.ndarray, size=(320, 180)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def rotate_image(img: np.ndarray, angle: float = 10.0) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def crop_and_resize(img: np.ndarray, crop_ratio: float = 0.70) -> np.ndarray:
    h, w = img.shape[:2]

    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2

    crop = img[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def horizontal_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def gaussian_blur(img: np.ndarray, kernel_size: int = 41) -> np.ndarray:
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)


def brightness_contrast(img: np.ndarray, alpha: float = 1.30, beta: int = 20) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_gaussian_noise(img: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Load image and create augmentations
# ------------------------------------------------------------
img = load_rgb(image_path)

geometric = [
    ("Original", img),
    ("Rotation", rotate_image(img, angle=10)),
    ("Crop + Resize", crop_and_resize(img, crop_ratio=0.70)),
    ("Horizontal Flip", horizontal_flip(img)),
]

photometric = [
    ("Original", img),
    ("Gaussian Blur", gaussian_blur(img, kernel_size=41)),
    ("Brightness / Contrast", brightness_contrast(img, alpha=1.30, beta=20)),
    ("Gaussian Noise", add_gaussian_noise(img, sigma=18.0)),
]

panel_size = (320, 180)
geometric = [(title, resize_for_panel(im, panel_size)) for title, im in geometric]
photometric = [(title, resize_for_panel(im, panel_size)) for title, im in photometric]


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(10.2, 3.75),
    dpi=300,
)

for ax in axes.ravel():
    ax.axis("off")

# Top row: geometric augmentations
for col, (title, aug_img) in enumerate(geometric):
    axes[0, col].imshow(aug_img)
    axes[0, col].set_title(title, fontsize=8.5, pad=1.0)

# Bottom row: photometric augmentations
for col, (title, aug_img) in enumerate(photometric):
    axes[1, col].imshow(aug_img)
    axes[1, col].set_title(title, fontsize=8.5, pad=1.0)

# Row labels
fig.text(
    0.012,
    0.735,
    "Geometric",
    rotation=90,
    va="center",
    ha="center",
    fontsize=9,
    fontweight="bold",
)

fig.text(
    0.012,
    0.285,
    "Photometric",
    rotation=90,
    va="center",
    ha="center",
    fontsize=9,
    fontweight="bold",
)

# Vertically compact, modest horizontal spacing
plt.subplots_adjust(
    left=0.045,
    right=0.995,
    top=0.965,
    bottom=0.02,
    wspace=0.035,
    hspace=-0.1,
)

plt.savefig(
    output_path,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.003,
)

plt.close()

print(f"Saved figure to: {output_path.resolve()}")