import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from datasets.dataset_cityscapes import *
from datasets.dataset_synthia import *
from datasets.dataset_synthia_style import *
from datasets.dataset_synthia_mixed import *
from datasets.dataset_bdd import *
from datasets.dataset_gta5 import *
from archive.advanced_augmentations import *
import kornia
import torch.nn.functional as F




# def compute_mIoU_and_per_class(preds, labels, num_classes):
#     preds = preds.view(-1)
#     labels = labels.view(-1)
    
#     mask = (labels >= 0) & (labels < num_classes)
#     preds = preds[mask]
#     labels = labels[mask]

#     conf_matrix = torch.bincount(
#         num_classes * labels + preds,
#         minlength=num_classes**2
#     ).reshape(num_classes, num_classes).float()

#     TP = conf_matrix.diag()
#     FP = conf_matrix.sum(dim=0) - TP
#     FN = conf_matrix.sum(dim=1) - TP

#     per_class_IoU = TP / (TP + FP + FN + 1e-6)
#     per_class_dict = {int(c): float(per_class_IoU[c].item())
#                       for c in range(num_classes) if conf_matrix.sum(dim=1)[c] > 0}

#     mean_iou = sum(per_class_dict.values()) / len(per_class_dict)
#     return mean_iou, per_class_dict


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


# def get_augmentation(dataset_name, split):
#     dataset_name = dataset_name.lower()

#     if 'synthia' in dataset_name:
#         if split == 'train':
#             # return A.Compose([
#             #     # A.HorizontalFlip(p=0.5),
#             #     # A.Blur(blur_limit=(3, 7), p=0.5),
#             #      A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.5), contrast_limit=0.5, p=0.5),

#             #     A.OneOf([local_brightness, global_brightness, local_light_spot], p=1.0),

#             #     # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
#             #     # A.RandomRotate90(p=0.5),
#             #     A.Resize(380, 640),
#                 #   A.RandomCrop(width=640, height=380),
#             #     # A.RandomCrop(width=WIDTH, height=HEIGHT),
#             #     # A.Resize(512, 1024),
#                 #   A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#                 #   ToTensorV2(),
#             # ]) 

#             # return A.Compose([
#             #     A.RandomResizedCrop(size=(380, 640), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
#             #     A.HorizontalFlip(p=0.5),
            
#             #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.25),
#             #     A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.5), contrast_limit=0.5, p=0.25),
            
#             #     # Optional: leichte Unschärfe oder Rauschen für Sensorvariationen
#             #     A.OneOf([
#             #         A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#             #         A.GaussNoise(p=0.5)
#             #     ], p=0.3),
            
#             #     # Normalisierung für ResNet/VGG-Backbones
#             #     A.Normalize(mean=(0.485, 0.456, 0.406),
#             #                 std=(0.229, 0.224, 0.225)),
#             #     ToTensorV2(),
#             #  ])

#             return A.Compose([
#                 # 1. Geometrische Augs
#                 # A.RandomResizedCrop(size=(380, 640), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
#                 A.RandomCrop(width=640, height=380),
#                 A.HorizontalFlip(p=0.5),

#                 # 2. Domain-Randomization (stärkere Farb/Stil Änderungen)
#                 A.OneOf([
#                     A.ColorJitter(
#                         brightness=0.3,   # Helligkeit ±30 % (stabiler als 50 %)
#                         contrast=0.2,     # Kontrast ±20 %
#                         saturation=0.2,   # Sättigung ±20 %
#                         hue=0.05           # Farbton ±5 % für subtile Farbverschiebung
#                     ),
#                     A.RandomBrightnessContrast(
#                         brightness_limit=(-0.15, 0.3),  # Helligkeit leicht asymmetrisch: -15 % bis +30 %
#                         contrast_limit=0.3               # Kontrast ±30 % für stabileres Training
#                     )
#                 ], p=0.7),
#                 A.GaussianBlur(blur_limit=(3, 5), p=0.2),

#                 A.Normalize(mean=(0.485, 0.456, 0.406),
#                             std=(0.229, 0.224, 0.225)),
#                 ToTensorV2(),
#             ])
        
#         else:
#             return A.Compose(
#             [
#                 A.Resize(380, 640),
#                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#                 ToTensorV2(),
#             ])
    
#     elif dataset_name == 'gta5':
#         if split == 'train':
#             return A.Compose([
#                 # A.HorizontalFlip(p=0.5),
#                 # A.Blur(blur_limit=(3, 7), p=0.5),
#                 # A.RandomBrightnessContrast(p=0.2),
#                 A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.5), contrast_limit=0.5, p=0.5),
#                 # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
#                 # A.RandomRotate90(p=0.5),
#                 # A.Resize(380, 640),
#                 A.RandomCrop(width=512, height=1024),
#                 # A.RandomCrop(width=256, height=256),
#                 # A.RandomCrop(width=WIDTH, height=HEIGHT),
#                 # A.Resize(512, 1024),
#                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#                 ToTensorV2(),
#             ]) 
#         else:
#             return A.Compose(
#             [
#                 A.Resize(512, 1024),
#                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#                 ToTensorV2(),
#             ])

#     elif dataset_name == 'cityscapes':
#         return A.Compose([
#             # A.SmallestMaxSize(max_size=160),
#             # A.CenterCrop(height=128, width=128),
#             # A.Resize(256, 512),
#             A.Resize(512, 1024),
#             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#             ToTensorV2(),
#         ])
    
#     elif dataset_name == 'bdd':
#         ''' bdd case '''
#         return A.Compose([
#         # A.SmallestMaxSize(max_size=160),
#         # A.CenterCrop(height=128, width=128),
#         # A.Resize(256, 512),
#         A.Resize(360, 640),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])


def get_augmentation(dataset_name, split):
    dataset_name = dataset_name.lower()

    if 'synthia' in dataset_name:
        if split == 'train':
            return A.Compose([
                # A.RandomCrop(width=640, height=380),
                A.HorizontalFlip(p=0.5),
                # A.RandomResizedCrop(size=(380, 640), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                A.RandomResizedCrop(size=(760, 1280), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.05, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.2, 0.2), 
                        contrast_limit=(-0.2, 0.2),
                        p=0.5),
                ], p=0.7),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                # A.Resize(380, 640),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    elif 'gta5' in dataset_name:
        if split == 'train':
            return A.Compose([
                A.RandomCrop(width=512, height=1024),
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.5), contrast_limit=0.5, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(512, 1024),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    elif 'cityscapes' in dataset_name:
        return A.Compose([
            # A.Resize(512, 1024),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    elif 'bdd' in dataset_name:
        return A.Compose([
            # A.Resize(360, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    else:
        # Fallback: Resize + Normalize
        return A.Compose([
            A.Resize(512, 1024),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def get_dataloader_from_dataset(path, dataset_name, split, batch_size, shuffle, use_synthia_shapes=False):
    if "cityscapes" in dataset_name:
        print("Use cityscapes as the target dataset")
        dataset = CityScapes(path, split='val', transform=get_augmentation('cityscapes', ''))
    elif "bdd" in dataset_name:
        print("Use bdd as the target dataset")
        dataset = BDD(path, split='val', transform=get_augmentation('bdd', ''))

    elif "synthiastyle" in dataset_name:
        print("Use synthia-style as the source dataset")
        if split == "train":
            dataset = SynthiaStyle(root_dir=path, split='train', transform=get_augmentation('synthia', 'train'), use_synthia_shapes=use_synthia_shapes)
        else:
            dataset = SynthiaStyle(root_dir=path, split='val', transform=get_augmentation('synthia', 'val'))

    elif "synthiamixed" in dataset_name:
        print("Use synthia-mixed as the source dataset")
        if split == "train":
            dataset = SynthiaMixed(root_dir='./synthia', split='train', transform=get_augmentation('synthia', 'train'))
        else:
            dataset = SynthiaMixed(root_dir='./synthia', split='val', transform=get_augmentation('synthia', 'val'))

    elif "synthia" in dataset_name:
        print("Use synthia as the source dataset")
        if split == "train":
            dataset = Synthia(root_dir=path, split='train', transform=get_augmentation('synthia', 'train'), use_synthia_shapes=use_synthia_shapes)
        else:
            dataset = Synthia(root_dir=path, split='val', transform=get_augmentation('synthia', 'val'))

    elif "gta5" in dataset_name:
        print("Use Gta 5 as the source dataset")
        if split == "train":
            dataset = GTA5(root_dir=path, split='train', transform=get_augmentation('gta5', 'train'))
        else:
            dataset = GTA5(root_dir=path, split='val', transform=get_augmentation('gta5', 'val'))

 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)



def get_image_size(dataset_name):
    if 'synthia' in dataset_name:
        return (760, 1280) # return (380, 640)
    elif 'gta5' in dataset_name:
        return (512, 1024)
    elif 'bdd' in dataset_name:
        return (720, 1280) #return (360, 640)
    elif 'cityscapes' in dataset_name:
        return (1024, 2048) # return (512, 1024)
    else:
        return (512, 1024)