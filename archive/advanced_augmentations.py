from scipy.ndimage import gaussian_filter
import numpy as np
import albumentations as A

# Hilfsfunktion, um die Maske weichzuzeichnen (weiche Übergänge)
def soften_mask(mask, kernel_size=31):
    # Gaussian Blur auf die Maske anwenden für weiche Übergänge
    return gaussian_filter(mask.astype(np.float32), sigma=kernel_size/6)


class LocalBrightnessContrast(A.ImageOnlyTransform):
    def __init__(self, mask_func, brightness_limit=0.4, contrast_limit=0.4, always_apply=True, p=0.5):
        super().__init__(always_apply, p)
        if isinstance(brightness_limit, (float, int)):
            brightness_limit = (-brightness_limit, brightness_limit)
        self.mask_func = mask_func
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def apply(self, img, **params):
        mask = self.mask_func(img.shape[:2])
        transform = A.RandomBrightnessContrast(
            brightness_limit=self.brightness_limit,
            contrast_limit=self.contrast_limit,
            p=1.0
        )
        transformed = transform(image=img.copy())['image']
        img[mask] = transformed[mask]
        return img


class LocalLightSpot(A.ImageOnlyTransform):
    def __init__(self, mask_func, brightness_limit=(0.2, 0.6), always_apply=True, p=0.5):
        super().__init__(always_apply, p)
        self.mask_func = mask_func
        self.brightness_limit = brightness_limit

    def apply(self, img, **params):
        mask = self.mask_func(img.shape[:2])
        # Erzeuge zufällige positive Helligkeitssteigerung
        brightness_increase = np.random.uniform(*self.brightness_limit)
        # Weiche Maske (optional, siehe vorherige Beispiele)
        softened_mask = soften_mask(mask, kernel_size=31)
        
        # Helligkeit auf Bild addieren im maskierten Bereich (Skalierung auf 0-255 beachten)
        img_float = img.astype(np.float32) / 255.0
        img_float += softened_mask[..., None] * brightness_increase
        img_float = np.clip(img_float, 0, 1)
        return (img_float * 255).astype(img.dtype)


def random_shape_mask(image_shape):
    h, w = image_shape
    mask = np.zeros((h, w), dtype=bool)
    
    shape_type = np.random.choice(['circle', 'rectangle', 'ellipse'])
    
    if shape_type == 'circle':
        center_x = np.random.randint(0, w)
        center_y = np.random.randint(0, h)
        radius = np.random.randint(min(h, w) // 10, min(h, w) // 5)
        
        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        mask = dist_sq <= radius**2
    
    elif shape_type == 'rectangle':
        x1 = np.random.randint(0, w - 1)
        y1 = np.random.randint(0, h - 1)
        x2 = np.random.randint(x1 + 1, min(w, x1 + w // 4))
        y2 = np.random.randint(y1 + 1, min(h, y1 + h // 4))
        mask[y1:y2, x1:x2] = True
    
    else:  # ellipse
        center_x = np.random.randint(0, w)
        center_y = np.random.randint(0, h)
        axis_x = np.random.randint(min(w, 20), min(w, 50))
        axis_y = np.random.randint(min(h, 20), min(h, 50))
        
        Y, X = np.ogrid[:h, :w]
        norm_x = ((X - center_x) / axis_x) ** 2
        norm_y = ((Y - center_y) / axis_y) ** 2
        mask = (norm_x + norm_y) <= 1
    
    return mask

def random_shape_mask(image_shape):
    shape_type = np.random.choice(['circle', 'rectangle', 'ellipse', 'triangle'])
    h, w = image_shape
    mask = np.zeros((h, w), dtype=bool)
    
    if shape_type == 'circle':
        radius = np.random.randint(int(min(h, w) * 0.1), int(min(h, w) * 0.2))
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - cx)**2 + (Y - cy)**2
        mask = dist_sq <= radius**2
    
    elif shape_type == 'rectangle':
        max_w = int(w * 0.25)
        max_h = int(h * 0.25)
        x1 = np.random.randint(0, w - max_w)
        y1 = np.random.randint(0, h - max_h)
        x2 = x1 + np.random.randint(max_w // 2, max_w)
        y2 = y1 + np.random.randint(max_h // 2, max_h)
        mask[y1:y2, x1:x2] = True
    
    elif shape_type == 'ellipse':
        axis_x = np.random.randint(int(w * 0.1), int(w * 0.2))
        axis_y = np.random.randint(int(h * 0.1), int(h * 0.2))
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        Y, X = np.ogrid[:h, :w]
        norm_x = ((X - cx) / axis_x) ** 2
        norm_y = ((Y - cy) / axis_y) ** 2
        mask = (norm_x + norm_y) <= 1
    
    else:  # triangle
        from skimage.draw import polygon
        
        size = np.random.randint(int(min(h, w) * 0.1), int(min(h, w) * 0.2))
        cx = np.random.randint(size, w - size)
        cy = np.random.randint(size, h - size)
        
        pts_x = np.array([cx, cx + size, cx])
        pts_y = np.array([cy, cy - size//2, cy + size//2])
        rr, cc = polygon(pts_y, pts_x, shape=mask.shape)
        mask[rr, cc] = True
    
    return mask


def multiple_random_shapes_mask(image_shape, min_shapes=3, max_shapes=5):
    num_shapes = np.random.randint(min_shapes, max_shapes + 1)
    combined_mask = np.zeros(image_shape, dtype=bool)
    for _ in range(num_shapes):
        mask = random_shape_mask(image_shape)
        combined_mask |= mask  # Logisches ODER über alle Masken
    
    return combined_mask

global_brightness = A.RandomBrightnessContrast(
    brightness_limit=(-0.2, 0.5),
    contrast_limit=0.5,
    p=1.0
)

local_brightness = LocalBrightnessContrast(
    mask_func=lambda shape: multiple_random_shapes_mask(shape, 3, 5),
    brightness_limit=(-0.2, 0.5),
    contrast_limit=0.5,
    p=1.0
)

# Instanziere das LocalLightSpot-Objekt
local_light_spot = LocalLightSpot(
    mask_func=lambda shape: multiple_random_shapes_mask(shape, 3, 5),
    brightness_limit=(0.2, 0.6),
    p=1.0  # Immer anwenden, wenn transform ausgewählt wird
)

