import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from dataset_cityscapes import *
from dataset_synthia import *
from dataset_bdd import *
from dataset_gta5 import *

def get_augmentation(dataset_name, split):
    if dataset_name == 'synthia':
        if split == 'train':
            return A.Compose([
                # A.HorizontalFlip(p=0.5),
                # A.Blur(blur_limit=(3, 7), p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5),
                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.Resize(380, 640),
                A.RandomCrop(width=640, height=380),
                # A.RandomCrop(width=256, height=256),
                # A.RandomCrop(width=WIDTH, height=HEIGHT),
                # A.Resize(512, 1024),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]) 
        else:
            return A.Compose(
            [
                A.Resize(380, 640),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
    
    elif dataset_name == 'gta5':
        if split == 'train':
            return A.Compose([
                # A.HorizontalFlip(p=0.5),
                # A.Blur(blur_limit=(3, 7), p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5),
                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                # A.RandomRotate90(p=0.5),
                # A.Resize(380, 640),
                A.RandomCrop(width=512, height=1024),
                # A.RandomCrop(width=256, height=256),
                # A.RandomCrop(width=WIDTH, height=HEIGHT),
                # A.Resize(512, 1024),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]) 
        else:
            return A.Compose(
            [
                A.Resize(512, 1024),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    elif dataset_name == 'cityscapes':
        return A.Compose([
            # A.SmallestMaxSize(max_size=160),
            # A.CenterCrop(height=128, width=128),
            # A.Resize(256, 512),
            A.Resize(512, 1024),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    elif dataset_name == 'bdd':
        ''' bdd case '''
        return A.Compose([
        # A.SmallestMaxSize(max_size=160),
        # A.CenterCrop(height=128, width=128),
        # A.Resize(256, 512),
        A.Resize(360, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_dataloader_from_dataset(path, dataset_name, split, batch_size, shuffle):
    if "cityscapes" in dataset_name:
        print("Use cityscapes as the target dataset")
        dataset = CityScapes(path, split='val', transform=get_augmentation('cityscapes', ''))
    elif "bdd" in dataset_name:
        print("Use bdd as the target dataset")
        dataset = BDD(path, split='val', transform=get_augmentation('bdd', ''))

    elif "synthia" in dataset_name:
        print("Use synthia as the source dataset")
        if split == "train":
            dataset = Synthia(root_dir=path, split='train', transform=get_augmentation('synthia', 'train'))
        else:
            dataset = Synthia(root_dir=path, split='val', transform=get_augmentation('synthia', 'val'))
    elif "gta5" in dataset_name:
        print("Use Gta 5 as the source dataset")
        if split == "train":
            dataset = GTA5(root_dir=path, split='train', transform=get_augmentation('gta5', 'train'))
        else:
            dataset = GTA5(root_dir=path, split='val', transform=get_augmentation('gta5', 'val'))

 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def get_image_size(dataset_name):
    if 'synthia' in dataset_name:
        return (380, 640)
    elif 'gta5' in dataset_name:
        return (512, 1024)
    elif 'bdd' in dataset_name:
        return (360, 640)
    elif 'cityscapes' in dataset_name:
        return (512, 1024)
    else:
        return (512, 1024)