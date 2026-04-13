from datasets.dataset_synthia import Synthia
import numpy as np
import imageio
from PIL import Image

ignore_label = 255

trainid_to_trainid = {
        0: ignore_label,
        1: 9,
        2: 2,
        3: 0,
        4: 1,
        5: 4,
        6: 8,
        7: 5,
        8: 12,
        9: 7,
        10: 10,
        11: 15,
        12: 14,
        13: ignore_label,
        14: ignore_label,
        15: 6,
        16: ignore_label,
        17: 11,
        18: ignore_label,
        19: 13,
        20: ignore_label,
        21: 3,
        22: ignore_label
}


class SynthiaBranched(Synthia):
    def __init__(self, root_dir, split,
                 transform_geo,
                 transform_color,
                 transform_normalize,
                 use_synthia_shapes=False,
                 list_dir=None,
                 split_tag="seed1337_85-15"):
        super().__init__(
            root_dir=root_dir,
            split=split,
            transform=None,
            use_synthia_shapes=use_synthia_shapes,
            list_dir=list_dir,
            split_tag=split_tag
        )
        self.transform_geo       = transform_geo
        self.transform_color     = transform_color
        self.transform_normalize = transform_normalize

    def __getitem__(self, index):
        img  = Image.open(self.images[index]).convert("RGB")
        mask = np.asarray(imageio.imread(self.masks[index], format="PNG-FI"))[:, :, 0]
        img  = np.array(img)
        mask = np.array(mask, dtype=np.uint8)

        # Label-Mapping
        mask_copy = mask.copy()
        for k, v in trainid_to_trainid.items():
            mask_copy[mask == k] = v
        mask = mask_copy

        # 1) Geometrie einmal auf beide anwenden → identischer Crop/Flip
        geo      = self.transform_geo(image=img, mask=mask)
        img_geo  = geo["image"]   # numpy [H, W, 3]
        mask_out = geo["mask"]

        # 2) RGB-Branch: Color + Blur + Noise + Normalize → Tensor
        rgb_aug = self.transform_color(image=img_geo)["image"]

        # 3) Struct-Branch: nur Normalize → Tensor
        struct  = self.transform_normalize(image=img_geo)["image"]

        return rgb_aug, struct, mask_out