import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, default_collate
from datasets.dataset_cityscapes import *
from datasets.dataset_synthia import *
from datasets.dataset_synthia_branched import SynthiaBranched
from datasets.dataset_bdd import *
from datasets.dataset_gta5 import *
import kornia
import torch.nn.functional as F
import cv2


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_fast_hist(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, device="cpu"):
    """
    Baut eine Confusion-Matrix direkt in PyTorch.
    preds:   [N, H, W] (argmax über Klassen)
    targets: [N, H, W] (ground truth)
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    indices = targets * num_classes + preds
    hist = torch.bincount(
        indices,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).to(device)

    return hist

def compute_mIoU_and_per_class_from_hist(conf_matrix: torch.Tensor):
    num_classes = conf_matrix.shape[0]

    TP = conf_matrix.diag()
    FP = conf_matrix.sum(dim=0) - TP
    FN = conf_matrix.sum(dim=1) - TP

    per_class_IoU = TP / (TP + FP + FN + 1e-6)
    per_class_dict = {int(c): float(per_class_IoU[c].item())
                      for c in range(num_classes) if conf_matrix.sum(dim=1)[c] > 0}

    mean_iou = sum(per_class_dict.values()) / len(per_class_dict) if len(per_class_dict) > 0 else 0.0
    return mean_iou, per_class_dict


def sobel_edges(images):
    # images: [B, 3, H, W] -> Graustufen + Sobel
    gray = images.mean(dim=1, keepdim=True)  # [B,1,H,W]
    kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=images.device).view(1,1,3,3)
    kernel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=images.device).view(1,1,3,3)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    edges = torch.sqrt(grad_x**2 + grad_y**2)
    edges = edges / (edges.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)  # normalize per image
    return edges

def multiscale_scharr_edges(images, sigmas=(0.5, 1.0, 2.0)):
    # images: [B, 3, H, W]
    device = images.device
    gray = images.mean(dim=1, keepdim=True)  # [B,1,H,W]
    
    # Scharr-Kernel
    kernel_x = torch.tensor([[3, 0, -3],
                             [10, 0, -10],
                             [3, 0, -3]], dtype=torch.float32, device=device).view(1,1,3,3)
    kernel_y = torch.tensor([[3, 10, 3],
                             [0, 0, 0],
                             [-3, -10, -3]], dtype=torch.float32, device=device).view(1,1,3,3)
    
    edges_multi = []
    for s in sigmas:
        blurred = kornia.filters.gaussian_blur2d(gray, (5, 5), (s, s))
        grad_x = F.conv2d(blurred, kernel_x, padding=1)
        grad_y = F.conv2d(blurred, kernel_y, padding=1)
        edges_multi.append(torch.sqrt(grad_x**2 + grad_y**2))
    
    # Max-Pooling über Skalen
    edges = torch.max(torch.stack(edges_multi, dim=0), dim=0)[0]
    
    # Normierung pro Bild
    edges = edges / (edges.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return edges


def get_augmentation(dataset_name, split, seed=None):
    name = dataset_name.lower()
    if split == "train":
        if "synthia" in name:
            return aug_train_dg(seed=seed)
        if "gta5" in name:
            return aug_train_dg(seed=seed)
        # fallback train
        return aug_train_dg(seed=seed)
    else:
        return aug_eval()
    


BASE_H, BASE_W = 512, 1024   # oder 384, 768

def aug_train_synthia(seed=None):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(BASE_H, BASE_W),
            scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0
        ),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], mask_interpolation=cv2.INTER_NEAREST, seed=seed
    )


def aug_train_gta5(seed=None):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(BASE_H, BASE_W),
            scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3,5), p=1.0),
            A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
        ], p=0.15),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], mask_interpolation=cv2.INTER_NEAREST, seed=seed)



def aug_eval():
    return A.Compose([
        A.Resize(BASE_H, BASE_W),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], mask_interpolation=cv2.INTER_NEAREST)



def aug_train_dg(seed=None):
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.RandomResizedCrop(
            size=(BASE_H, BASE_W),
            scale=(0.75, 1.0),
            ratio=(0.85, 1.2),
            p=1.0
        ),

        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=1.0),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.08,
                p=1.0),
        ], p=0.8),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.01, 0.03), p=1.0)
        ], p=0.2),

        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], mask_interpolation=cv2.INTER_NEAREST, seed=seed)



def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    if len(batch[0]) == 3:
        imgs_rgb    = default_collate([b[0] for b in batch])
        imgs_struct = default_collate([b[1] for b in batch])
        masks       = default_collate([b[2] for b in batch])
        return imgs_rgb, imgs_struct, masks
    else:
        imgs   = default_collate([b[0] for b in batch])
        masks  = default_collate([b[1] for b in batch])
        return imgs, masks


def get_dataloader_from_dataset(path, dataset_name, split, batch_size, shuffle, use_synthia_shapes=False, seed=0, num_workers=1, num_classes=19):
    if "cityscapes" in dataset_name:
        print("Use cityscapes as the target dataset")
        dataset = CityScapes(path, split='val', transform=get_augmentation('cityscapes', 'val'), num_classes=num_classes)
    elif "bdd" in dataset_name:
        print("Use bdd as the target dataset")
        dataset = BDD(path, split='val', transform=get_augmentation('bdd', 'val'), num_classes=num_classes)

    elif "synthiastyle" in dataset_name:
        print("Use synthia-style as the source dataset")
        if split == "train":
            dataset = SynthiaStyle(root_dir=path, split='train', transform=get_augmentation('synthia', 'train', seed=seed), use_synthia_shapes=use_synthia_shapes)
        else:
            dataset = SynthiaStyle(root_dir=path, split='val', transform=get_augmentation('synthia', 'val', seed=seed))

    elif "synthiamixed" in dataset_name:
        print("Use synthia-mixed as the source dataset")
        if split == "train":
            dataset = SynthiaMixed(root_dir='./synthia', split='train', transform=get_augmentation('synthia', 'train'))
        else:
            dataset = SynthiaMixed(root_dir='./synthia', split='val', transform=get_augmentation('synthia', 'val'))


    elif "synthiabranched" in dataset_name:
        print("Use SynthiaBranched as source dataset")
        if split == "train":
            dataset = SynthiaBranched(
                root_dir=path,
                split='train',
                transform_geo=aug_geo(seed=seed),
                transform_color=aug_color_normalize(),
                transform_normalize=aug_normalize_only(),
                use_synthia_shapes=use_synthia_shapes
            )
        else:
            dataset = SynthiaBranched(
                root_dir=path,
                split='val',
                transform_geo=aug_eval_geo(),
                transform_color=aug_normalize_only(),
                transform_normalize=aug_normalize_only(),
        )

    elif "synthia" in dataset_name:
        print("Use synthia as the source dataset")
        if split == "train":
            dataset = Synthia(root_dir=path, split='train', transform=get_augmentation('synthia', 'train', seed=seed), use_synthia_shapes=use_synthia_shapes)
        else:
            dataset = Synthia(root_dir=path, split='val', transform=get_augmentation('synthia', 'val', seed=seed))


    elif "gta5" in dataset_name:
        print("Use Gta 5 as the source dataset")
        if split == "train":
            dataset = GTA5(root_dir=path, split='train', transform=get_augmentation('gta5', 'train', seed=seed))
        else:
            dataset = GTA5(root_dir=path, split='val', transform=get_augmentation('gta5', 'val', seed=seed))

    g = torch.Generator()
    g.manual_seed(seed)
    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2, prefetch_factor=1, worker_init_fn=seed_worker, generator=g, collate_fn=collate_skip_none)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2, prefetch_factor=2, worker_init_fn=seed_worker, generator=g, collate_fn=collate_skip_none)





def aug_eval_geo():
    """Resize für Val – kein Color, kein Tensor (Geometrie-Schritt)"""
    return A.Compose([
        A.Resize(BASE_H, BASE_W),
    ], mask_interpolation=cv2.INTER_NEAREST)

def aug_geo(seed=None):
    """Nur Geometrie – wird auf beide Branches angewendet"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(BASE_H, BASE_W),
            scale=(0.75, 1.0),
            ratio=(0.85, 1.2),
            p=1.0
        ),
    ], mask_interpolation=cv2.INTER_NEAREST, seed=seed)


def aug_color_normalize():
    """Photometrische Augmentierung + Normalize nur für RGB"""
    return A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.01, 0.03), p=1.0)
        ], p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def aug_normalize_only():
    """Nur Normalize für Structural-Branch"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_image_size(dataset_name):
    return (BASE_H, BASE_W)