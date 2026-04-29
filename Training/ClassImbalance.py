import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Sampler, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

base_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

cat_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
    ]
)


class AugmentedMinorityDataset(torch.utils.data.Dataset):
    def __init__(self, subset, minority_classes, base_transform, minority_transform):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.minority_classes = set(minority_classes)
        self.base_transform = base_transform
        self.minority_transform = minority_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        if label in self.minority_classes:
            img = self.minority_transform(img)
        else:
            img = self.base_transform(img)

        return img, label


def make_weighted_sampler(dataset):
    labels = []
    for i in range(len(dataset)):
        label = dataset[i][1]
        if isinstance(label, tuple):
            labels.append(label[0].item() if hasattr(label[0], "item") else label[0])
        else:
            labels.append(label.item() if hasattr(label, "item") else label)

    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    return sampler


def split_cat_dog_indices(subset):
    labels = np.array(subset.dataset._labels)[subset.indices]
    cat_mask = labels < 12
    dog_mask = label >= 12
    cat_indices = np.array(subset.indices)[cat_mask]
    dog_indices = np.array(subset.indices)[dog_mask]

    return cat_indices, dog_indices


class BalancedBatchSampler(Sampler):
    def __init__(self, cat_indices, dog_indices, batch_size):
        assert beatch_size % 2 == 0
        self.cat_indices = list(cat_indices)
        self.dog_indices = list(dog_indices)
        self.batch_size = batch_size
        self.half = batch_size // 2

    def __iter__(self):
        cats = self.cat_indices.copy()
        dogs = self.dog_indices.copy()
        random.shuffle(cats)
        random.shuffle(dogs)
