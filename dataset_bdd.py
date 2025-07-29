from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import glob
import albumentations as A

ignore_index = 255

label_to_trainid_16_classes = {
  # sourceid : targetid

  0: 0,             # road
  1: 1,             # sidewalk
  2: 2,             # building
  3: 3,             # wall
  4: 4,             # fence
  5: 5,             # pole
  6: 6,             # traffic light
  7: 7,             # traffic sign
  8: 8,             # vegetation  
  9: ignore_index,  # terrain - not present!
  10: 9,            # sky
  11: 10,           # person
  12: 11,           # rider 
  13: 12,           # car
  14: ignore_index, # truck - not present!
  15: 13,           # bus
  16: ignore_index, # train - not present!
  17: 14,           # motorcycle 
  18: 15,           # bicycle
}

label_to_trainid_19_classes = {
  # sourceid : targetid

  0: 0,             # road
  1: 1,             # sidewalk
  2: 2,             # building
  3: 3,             # wall
  4: 4,             # fence
  5: 5,             # pole
  6: 6,             # traffic light
  7: 7,             # traffic sign
  8: 8,             # vegetation  
  9: 9,             # terrain
  10: 10,           # sky
  11: 11,           # person
  12: 12,           # rider 
  13: 13,           # car
  14: 14,           # truck 
  15: 15,           # bus
  16: 16,           # train
  17: 17,           # motorcycle 
  18: 18,           # bicycle
}

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


class BDD(Dataset):
    def __init__(self, root_dir, split='val', num_classes=16, transform=None):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.label_to_trainid = label_to_trainid_16_classes if num_classes == 16 else label_to_trainid_19_classes
        self.images = sorted(glob.glob(f'{root_dir}/images/{split}/*.jpg'))
        self.masks = sorted(glob.glob(f'{root_dir}/masks/{split}/*.png'))

        # self.masks = glob.glob(...)
        self.transform = transform
        pass


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