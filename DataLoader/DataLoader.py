"""
Oxford-IIIT Pet Dataset — DataLoader
=====================================
Produces separate train, val, and test DataLoaders with two label sets:
    y1 — species label  (0 = Cat, 1 = Dog)
    y2 — breed label    (0–36, 37 classes total)

Label format is controlled by `one_hot=True/False`:
    False (default) — integer class indices, compatible with nn.CrossEntropyLoss
    True            — one-hot float vectors,  compatible with soft-label losses

"""

import os
import random
from pathlib import Path
from typing import Literal, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

NUM_BREEDS = 37
NUM_SPECIES = 2  # 0 = Cat, 1 = Dog

# ImageNet stats — standard for pretrained ViT
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

class BatchAugmenter:
    """
    Applies CutMix and/or MixUp at the batch level.
    Pass to the training loop after DataLoader yields a batch.

    Args:
        augs : same AUGMENTATIONS dict used for get_train_transform
        num_classes : number of output classes (for one-hot encoding)
    """
    def __init__(self, augs: dict, num_classes: int):
        self.augs = augs
        self.num_classes = num_classes
        self.mixup_alpha  = augs.get("mixup_alpha",  0.4)
        self.cutmix_alpha = augs.get("cutmix_alpha", 1.0)

    def _to_onehot(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(labels, self.num_classes).float()

    def _mixup(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        idx = torch.randperm(x.size(0))
        x_mix = lam * x + (1 - lam) * x[idx]
        y_mix = lam * self._to_onehot(y) + (1 - lam) * self._to_onehot(y[idx])
        return x_mix, y_mix

    def _cutmix(self, x, y):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        idx = torch.randperm(x.size(0))
        _, _, H, W = x.shape

        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

        lam_actual = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        y_mix = lam_actual * self._to_onehot(y) + (1 - lam_actual) * self._to_onehot(y[idx])
        return x_mix, y_mix

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        apply_p = self.augs.get("batch_aug_prob", 0.2)
    
        if np.random.rand() > apply_p:
            return x, y
    
        do_mixup  = self.augs.get("mixup")
        do_cutmix = self.augs.get("cutmix")
    
        if do_mixup and do_cutmix:
            if np.random.rand() < 0.5:
                return self._mixup(x, y)
            else:
                return self._cutmix(x, y)
        elif do_mixup:
            return self._mixup(x, y)
        elif do_cutmix:
            return self._cutmix(x, y)
        else:
            return x, y
            

def get_train_transform(augs: dict, image_size: int = 224) -> transforms.Compose:
    pipeline = [transforms.Resize((256, 256))]

    if augs.get("horizontal_flip"):
        pipeline.append(transforms.RandomHorizontalFlip(p=0.5))

    if augs.get("rotation"):
        min_deg = augs.get("rotation_min_degrees", 0)
        max_deg = augs.get("rotation_max_degrees", 15)
        pipeline.append(transforms.RandomRotation(degrees=(min_deg, max_deg)))

    if augs.get("random_crop"):
        pipeline.append(transforms.RandomResizedCrop(
            image_size,
            scale=augs.get("crop_scale", (0.85, 1.0)),
            ratio=augs.get("crop_ratio", (0.9, 1.1)),
        ))
    else:
        pipeline.append(transforms.Resize((image_size, image_size)))

    if augs.get("color_jitter"):
        pipeline.append(transforms.ColorJitter(
            brightness=augs.get("jitter_brightness", 0.3),
            contrast=augs.get("jitter_contrast",   0.3),
            saturation=augs.get("jitter_saturation", 0.2),
            hue=augs.get("jitter_hue", 0.05),
        ))

    if augs.get("grayscale"):
        pipeline.append(transforms.RandomGrayscale(p=augs.get("grayscale_p", 0.05)))

    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD))

    if augs.get("random_erasing"):
        pipeline.append(transforms.RandomErasing(
            p=augs.get("erasing_p", 0.2),
            scale=augs.get("erasing_scale", (0.02, 0.10)),
        ))

    return transforms.Compose(pipeline)
    

def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _parse_annotation_file(filepath: str) -> list[dict]:
    """
    Parses trainval.txt or test.txt.
    Returns a list of dicts with zero-indexed labels ready for PyTorch.
    """
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue

            image_name = parts[0]
            class_id = int(parts[1])
            species_id = int(parts[2])

            records.append(
                {
                    "image_name": image_name,
                    "breed": class_id - 1,
                    "species": species_id - 1,
                }
            )
    return records


class CreateDataset(Dataset):
    """
    Args:
        records      : list of dicts from _parse_annotation_file
        images_dir   : path to the images/ folder
        transform    : torchvision transform applied to each image
        one_hot      : if True, returns one-hot float tensors for y1 and y2
                       if False, returns integer class index tensors
    """

    def __init__(
        self,
        records: list[dict],
        images_dir: str,
        transform: Optional[transforms.Compose] = None,
        one_hot: bool = False,
    ) -> None:
        self.records = records
        self.images_dir = Path(images_dir)
        self.transform = transform or get_eval_transform()
        self.one_hot = one_hot

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        record = self.records[idx]

        # load image
        img_path = self.images_dir / f"{record['image_name']}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # build labels
        species_idx = torch.tensor(record["species"], dtype=torch.long)
        breed_idx = torch.tensor(record["breed"], dtype=torch.long)

        if self.one_hot:
            y1 = F.one_hot(species_idx, num_classes=NUM_SPECIES).float()  # (2,)
            y2 = F.one_hot(breed_idx, num_classes=NUM_BREEDS).float()  # (37,)
        else:
            y1 = species_idx
            y2 = breed_idx 

        return image, (y1, y2)


def _pet_collate(batch):
    """
    Collates (image, (y1, y2)) tuples into
    (images_batch, (y1_batch, y2_batch)).
    """
    images = torch.stack([item[0] for item in batch])
    y1 = torch.stack([item[1][0] for item in batch])
    y2 = torch.stack([item[1][1] for item in batch])
    return images, (y1, y2)


def build_dataloaders(
    dataset_root: str,
    val_split: float = 0.2,
    batch_size: int = 32,
    one_hot: bool = False,
    image_size: int = 224,
    augs: dict = {},
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train, val, and test DataLoaders for the Oxford-IIIT Pet dataset.

    Args:
        dataset_root : root directory containing images/ and annotations/
        val_split    : fraction of trainval.txt to use for validation
        batch_size   : number of samples per batch
        one_hot      : False → integer labels
                       True  → one-hot float labels
        image_size   : spatial resolution fed to the model (default 224)
        num_workers  : DataLoader worker processes
        seed         : random seed for reproducible train/val split

    Returns:
        (train_loader, val_loader, test_loader)

    Label shapes per batch:
        one_hot=False : y1 → (B,)     y2 → (B,)
        one_hot=True  : y1 → (B, 2)   y2 → (B, 37)
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    root = Path(dataset_root)
    images_dir = root / "images"
    trainval_txt = root / "annotations" / "trainval.txt"
    test_txt = root / "annotations" / "test.txt"

    for p in [images_dir, trainval_txt, test_txt]:
        if not p.exists():
            raise FileNotFoundError(f"Expected path not found: {p}")

    # parse annotation files
    trainval_records = _parse_annotation_file(str(trainval_txt))
    test_records = _parse_annotation_file(str(test_txt))

    # reproducible train / val split
    rng = random.Random(seed)
    rng.shuffle(trainval_records)

    n_val = int(len(trainval_records) * val_split)
    n_train = len(trainval_records) - n_val

    train_records = trainval_records[:n_train]
    val_records = trainval_records[n_train:]

    print(
        f"Dataset split — "
        f"train: {len(train_records)} | "
        f"val: {len(val_records)} | "
        f"test: {len(test_records)}"
    )
    print(
        f"Label format  — "
        f"{'one-hot float' if one_hot else 'integer index'} | "
        f"y1: species (Cat/Dog) | y2: breed (0–36)"
    )

    # datasets
    train_dataset = CreateDataset(
        train_records,
        images_dir,
        transform=get_train_transform(augs,image_size),
        one_hot=one_hot,
    )
    val_dataset = CreateDataset(
        val_records,
        images_dir,
        transform=get_eval_transform(image_size),
        one_hot=one_hot,
    )
    test_dataset = CreateDataset(
        test_records,
        images_dir,
        transform=get_eval_transform(image_size),
        one_hot=one_hot,
    )

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_pet_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_pet_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_pet_collate,
    )

    return train_loader, val_loader, test_loader
