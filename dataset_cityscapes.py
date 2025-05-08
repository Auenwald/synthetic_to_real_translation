from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import matplotlib.pyplot as plt
# from utils import utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cityscapes_shared_labels as cityscapes_labels


trainid_to_name = cityscapes_labels.trainId2name
label_to_trainid = cityscapes_labels.label2trainid


ignore_index = 255

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        
zero_pad = 256 * 3 - len(palette) 
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
        """
        Colorize a segmentation mask.
        """
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask


class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', num_classes=16, transform=None):
        self.label_to_trainid = cityscapes_labels.label2trainid if num_classes == 16 else cityscapes_labels.label2trainid_19_classes

        self.root_dir = root_dir
        self.images = sorted(glob.glob(f'{self.root_dir}/leftImg8bit/{split}/*/*.png'))
        self.masks = sorted(glob.glob(f'{self.root_dir}/gtFine/{split}/*/*labelIds.png'))
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))


        # label transformation
        mask_copy = mask.copy()
        for k, v in self.label_to_trainid.items():
            mask_copy[mask == k] = v
        mask = mask_copy
            

        # albumentations
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            return transformed['image'], transformed['mask']
        else:
            return img, mask

    def __len__(self):
        return len(self.images)
