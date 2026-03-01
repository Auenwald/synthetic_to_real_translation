from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import imageio
import random


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


def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

class Synthia(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        use_synthia_shapes=False,
        list_dir=None,
        split_tag="seed1337_85-15", 
        base_seed = 0
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_synthia_shapes = use_synthia_shapes
        self.num_classes = 16

        self.base_seed = int(base_seed)  # <-- NEU
        self.epoch = 0                   # <-- NEU (optional)

        list_base = Path(list_dir) if list_dir else self.root_dir
        train_list = list_base / f"synthia_train_{split_tag}.txt"
        val_list   = list_base / f"synthia_val_{split_tag}.txt"

        if split == "train":
            list_path = train_list
        elif split == "val":
            list_path = val_list
        else:
            raise ValueError("No source test split for DG")

        if not list_path.exists():
            raise FileNotFoundError(f"Split file not found: {list_path}")

        names = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

        rgb_dir = self.root_dir / "RGB"
        color_dir = self.root_dir / "GT" / "COLOR"
        labels_dir_candidates = [
            self.root_dir / "GT" / "labels",
            self.root_dir / "GT" / "LABELS",
        ]
        labels_dir = next((p for p in labels_dir_candidates if p.exists()), None)
        if labels_dir is None:
            raise FileNotFoundError(f"Could not find labels dir in: {labels_dir_candidates}")

        # Basename -> Path (robust)
        rgb_map = {p.name: p for p in rgb_dir.glob("*.png")}
        color_map = {p.name: p for p in color_dir.glob("*.png")}
        label_map = {p.name: p for p in labels_dir.glob("*.png")}

        missing = []
        self.images, self.shapes, self.masks = [], [], []
        for nm in names:
            if nm not in rgb_map or nm not in color_map or nm not in label_map:
                missing.append(nm)
                continue
            self.images.append(str(rgb_map[nm]))
            self.shapes.append(str(color_map[nm]))
            self.masks.append(str(label_map[nm]))

        if missing:
            raise FileNotFoundError(f"{len(missing)} files from split list missing. Example: {missing[:5]}")

    def __getitem__(self, index):
        s = self.base_seed + index + self.epoch * 1_000_000
        random.seed(s)
        np.random.seed(s)

        img = Image.open(self.shapes[index]).convert("RGB")

        mask = np.asarray(imageio.imread(self.masks[index], format="PNG-FI"))[:, :, 0]
        img = np.array(img)

        mask = np.array(mask, dtype=np.uint8)
        mask_copy = mask.copy()
        for k, v in trainid_to_trainid.items():
            mask_copy[mask == k] = v
        mask = mask_copy

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            return transformed["image"], transformed["mask"]
        return img, mask

    def __len__(self):
        return len(self.images)