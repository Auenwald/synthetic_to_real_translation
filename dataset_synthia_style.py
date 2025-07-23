from PIL import Image
from torch.utils.data import DataLoader, Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import imageio

# num_classes = 19
num_classes = 16
ignore_label = 255



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


class SynthiaStyle(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.images = glob.glob(f'{root_dir}/Style/*.png')
        self.masks = glob.glob(f'{root_dir}/GT/LABELS/*.png')

        self.num_classes = 16

        self.split = split
        self.transform = transform
        
        if self.split == 'train':
            self.images = self.images[0:7700]
            self.masks = self.masks[0:7700]
        elif self.split == 'val':
            self.images = self.images[7700:7700+700]
            self.masks = self.masks[7700:7700+700]
        else:
            self.images = self.images[7700+700:7700+700+700]
            self.masks = self.masks[7700+700:7700+700+700]

    def __getitem__(self, index):

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

        # albumentations
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            return transformed['image'], transformed['mask']
        else:
            return img, mask


    def __len__(self):
        return len(self.images)