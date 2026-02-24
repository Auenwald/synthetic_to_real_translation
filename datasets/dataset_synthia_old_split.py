from PIL import Image
from torch.utils.data import DataLoader, Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import imageio
import random

# num_classes = 19
num_classes = 16
ignore_label = 255

# trainid_to_trainid = {
#         0: ignore_label,  # void
#         1: 10,            # sky
#         2: 2,             # building
#         3: 0,             # road
#         4: 1,             # sidewalk
#         5: 4,             # fence
#         6: 8,             # vegetation
#         7: 5,             # pole
#         8: 13,            # car
#         9: 7,             # traffic sign
#         10: 11,           # pedestrian - person
#         11: 18,           # bicycle
#         12: 17,           # motorcycle
#         13: ignore_label, # parking-slot
#         14: ignore_label, # road-work
#         15: 6,            # traffic light
#         16: 9,            # terrain - not present!
#         17: 12,           # rider
#         18: 14,           # truck - not present!
#         19: 15,           # bus
#         20: 16,           # train - - not present!
#         21: 3,            # wall
#         22: ignore_label  # Lanemarking
#         }


trainid_to_trainid = {
        0: ignore_label,  # void
        1: 9,            # sky
        2: 2,             # building
        3: 0,             # road
        4: 1,             # sidewalk
        5: 4,             # fence
        6: 8,             # vegetation
        7: 5,             # pole
        8: 12,            # car
        9: 7,             # traffic sign
        10: 10,           # pedestrian - person
        11: 15,           # bicycle
        12: 14,           # motorcycle
        13: ignore_label, # parking-slot
        14: ignore_label, # road-work
        15: 6,            # traffic light
        16: ignore_label, # terrain - not present!
        17: 11,           # rider
        18: ignore_label, # truck - not present!
        19: 13,           # bus
        20: ignore_label, # train - - not present!
        21: 3,            # wall
        22: ignore_label  # Lanemarking
        }



palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
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


class Synthia(Dataset):
    def __init__(self, root_dir, split='train', transform=None, use_synthia_shapes=False):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(f'{root_dir}/RGB/*.png'))
        self.shapes = sorted(glob.glob(f'{root_dir}/GT/COLOR/*.png'))
        self.masks = sorted(glob.glob(f'{root_dir}/GT/LABELS/*.png'))
        self.use_synthia_shapes = use_synthia_shapes

        self.num_classes = 16

        self.split = split
        self.transform = transform

        SCENE_LEN = 700
        TRAIN_SCENES = 12
        VAL_SCENE = 12  # 0-indexed

        TRAIN_END = SCENE_LEN * TRAIN_SCENES
        VAL_START = TRAIN_END
        VAL_END = VAL_START + SCENE_LEN

        if self.split == "train":
            self.images = (
                self.images[:VAL_START] +
                self.images[VAL_END:]
            )
            self.shapes = (
                self.shapes[:VAL_START] +
                self.shapes[VAL_END:]
            )
            self.masks = (
                self.masks[:VAL_START] +
                self.masks[VAL_END:]
            )

        elif self.split == "val":
            self.images = self.images[VAL_START:VAL_END]
            self.shapes = self.shapes[VAL_START:VAL_END]
            self.masks = self.masks[VAL_START:VAL_END]

        else:
            raise ValueError("No source test split for DG")



    def __getitem__(self, index):

        if self.use_synthia_shapes and random.random() < 0.5:
            img = Image.open(self.shapes[index]).convert('RGB')
        else:
            img = Image.open(self.images[index]).convert('RGB')

        # maybe necessary to install imageio plugins via: imageio.plugins.freeimage.download()
        mask = np.asarray(imageio.imread(self.masks[index], format='PNG-FI'))[:, :, 0]
        img = np.array(img)

        # label transformation
        mask = np.array(mask, dtype=np.uint8)
        mask_copy = mask.copy()
        for k, v in trainid_to_trainid.items():
            mask_copy[mask == k] = v

        mask = mask_copy
        # mask = np.array(Image.fromarray(mask_copy.astype(np.uint8)))

        # albumentations
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            return transformed['image'], transformed['mask']
        else:
            return img, mask


    def __len__(self):
        return len(self.images)