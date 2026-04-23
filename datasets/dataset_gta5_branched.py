from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

ignore_index = 255

label_to_trainid = {
    0: ignore_index,
    1: ignore_index,
    2: ignore_index,
    3: ignore_index,
    4: ignore_index,
    5: ignore_index,
    6: ignore_index,
    7: 0,
    8: 1,
    9: ignore_index,
    10: ignore_index,
    11: 2,
    12: 3,
    13: 4,
    14: ignore_index,
    15: ignore_index,
    16: ignore_index,
    17: 5,
    18: ignore_index,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: ignore_index,
    30: ignore_index,
    31: 16,
    32: 17,
    33: 18,
    34: ignore_index,
}


class GTA5(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        list_dir=None,
        split_tag="seed1337_85-15",
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.num_classes = 19

        list_base = Path(list_dir) if list_dir else self.root_dir
        train_list = list_base / f"gta5_train_{split_tag}.txt"
        val_list   = list_base / f"gta5_val_{split_tag}.txt"

        if split == "train":
            list_path = train_list
        elif split == "val":
            list_path = val_list
        else:
            raise ValueError("For GTA5 source training, only 'train' and 'val' are supported")

        if not list_path.exists():
            raise FileNotFoundError(f"Split file not found: {list_path}")

        names = [
            ln.strip()
            for ln in list_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

        image_dir = self.root_dir / "images"
        label_dir = self.root_dir / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        image_map = {p.name: p for p in image_dir.glob("*.png")}
        label_map = {p.name: p for p in label_dir.glob("*.png")}

        missing = []
        self.images, self.masks = [], []

        for nm in names:
            if nm not in image_map or nm not in label_map:
                missing.append(nm)
                continue
            self.images.append(str(image_map[nm]))
            self.masks.append(str(label_map[nm]))

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files from split list missing. Example: {missing[:5]}"
            )

    def encode_mask(self, mask):
        mask_copy = np.full(mask.shape, ignore_index, dtype=np.uint8)
        for k, v in label_to_trainid.items():
            mask_copy[mask == k] = v
        return mask_copy

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = np.array(img, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        mask = self.encode_mask(mask)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            return transformed["image"], transformed["mask"]

        return img, mask

    def __len__(self):
        return len(self.images)


class GTA5Branched(GTA5):
    def __init__(
        self,
        root_dir,
        split,
        transform_geo,
        transform_color,
        transform_normalize,
        list_dir=None,
        split_tag="seed1337_85-15",
    ):
        super().__init__(
            root_dir=root_dir,
            split=split,
            transform=None,
            list_dir=list_dir,
            split_tag=split_tag,
        )

        self.transform_geo = transform_geo
        self.transform_color = transform_color
        self.transform_normalize = transform_normalize

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)

            img = np.array(img, dtype=np.uint8)
            mask = np.array(mask, dtype=np.uint8)

        except Exception as e:
            print(f"[SKIP] {img_path}")
            return None

        mask = self.encode_mask(mask)

        geo = self.transform_geo(image=img, mask=mask)
        img_geo  = geo["image"]
        mask_out = geo["mask"]

        rgb_aug = self.transform_color(image=img_geo)["image"]
        struct  = self.transform_normalize(image=img_geo)["image"]

        return rgb_aug, struct, mask_out