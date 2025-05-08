from PIL import Image
from torch.utils.data import DataLoader, Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os

ignore_index = 255

label_to_trainid = {
    0:ignore_index,
    1:ignore_index,
    2:ignore_index,
    3:ignore_index,
    4:ignore_index,
    5:ignore_index,
    6:ignore_index,
    7:0,
    8:1, 
    9:ignore_index,
    10:ignore_index,
    11:2,
    12:3,
    13:4,
    14:ignore_index,
    15:ignore_index,
    16:ignore_index,
    17:5,
    18:ignore_index,
    19:6, 
    20:7,
    21:8,
    22:9,
    23:10,
    24:11,
    25:12,
    26:13,
    27:14,
    28:15,
    29:ignore_index,
    30:ignore_index,
    31:16,
    32:17,
    33:18,
    34:ignore_index
    #34:-1
}

class GTA5(Dataset):
    def __init__(self, root_dir, split='train', num_classes=19, transform=None):
        self.label_to_trainid = label_to_trainid
        self.root_dir = root_dir
        self.split = split
        self.images = sorted(glob.glob(f'{root_dir}/images/*.png'))
        self.masks = sorted(glob.glob(f'{root_dir}/labels/*.png'))

        train_split_index = int(0.8*len(self.images))
        val_test_size = int(0.1*len(self.images))

        if self.split == 'train':
            self.images = self.images[0:train_split_index]
            self.masks = self.masks[0:train_split_index]
        elif self.split == 'val':
            self.images = self.images[train_split_index:train_split_index+val_test_size]
            self.masks = self.masks[train_split_index:train_split_index+val_test_size]
        else:
            self.images = self.images[train_split_index+val_test_size:]
            self.masks = self.masks[train_split_index+val_test_size:]


        self.transform = transform
        self.num_classes = num_classes
    
    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))

        mask_copy = mask.copy()
        for k, v in self.label_to_trainid.items():
            mask_copy[mask == k] = v
        mask = mask_copy

        if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
            print(f"[WARN] Auflösung mismatch bei idx {index}: image={img.shape}, label={mask.shape}")
            return None, None


        # albumentations
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            if transformed['image'].shape[1] != transformed['mask'].shape[0] or transformed['image'].shape[2] != transformed['mask'].shape[1]:
                print(f"[WARN] Auflösung mismatch bei idx {index}: image={transformed['image'].shape}, label={transformed['mask'].shape}")

            return transformed['image'], transformed['mask']
        else:
            return img, mask


    def __len__(self):
        return len(self.images)
