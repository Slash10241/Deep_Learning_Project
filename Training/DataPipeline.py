import os
import random
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

import ClassImbalance


def get_stratified_subsets(dataset, subset_ratios=(1.0, 0.1, 0.01), random_state=42):
    """
    Returns stratified subsets of a dataset.

    Args:
        dataset: torchvision dataset (must have targets or labels)
        subset_ratios: tuple of ratios (e.g., (1.0, 0.1, 0.01))
        random_state: seed for reproducibility

    Returns:
        dict of {ratio: Subset}
    """
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    indices = np.arange(len(dataset))
    subsets = {}
    for ratio in subset_ratios:
        if ratio == 1.0:
            subsets[ratio] = Subset(dataset, indices)
            continue

        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=ratio, random_state=random_state
        )

        subset_idx, _ = next(splitter.split(indices, labels))
        subsets[ratio] = Subset(dataset, subset_idx)

    return subsets


def create_cat_imbalanced_subset(train_dataset, cat_fraction=0.2, seed=42):
    """
    Reduces cat samples to a fixed fraction while keeping all dog samples.

    Args:
        train_dataset: OxfordIIITPet training dataset
        cat_fraction: fraction of cat samples to keep (e.g., 0.2)
        seed: reproducibility

    Returns:
        Subset with imbalanced class distribution
    """
    random.seed(seed)
    np.random.seed(seed)
    labels = np.array([train_dataset.dataset[i][1] for i in train_dataset.indices])
    cat_indices = np.where(labels < 12)[0]
    dog_indices = np.where(labels >= 12)[0]
    n_cats_keep = int(len(cat_indices) * cat_fraction)
    selected_cat_indices = np.random.choice(cat_indices, n_cats_keep, replace=False)
    final_indices = np.concatenate([selected_cat_indices, dog_indices])
    np.random.shuffle(final_indices)

    return Subset(train_dataset, final_indices)


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def main():
    data = datasets.OxfordIIITPet(
        root="./", split="trainval", transform=None, download=False
    )
    test_data = datasets.OxfordIIITPet(
        root="./", split="test", transform=val_transform, download=False
    )
    classes = data.classes
    num_classes = len(classes)
    batch_size = 32
    train_size = int(0.9 * len(data))
    val_size = len(data) - train_size
    train_indices, val_indices = random_split(range(len(data)), [train_size, val_size])
    train_data = datasets.OxfordIIITPet(
        root="./", split="trainval", transform=train_transform, download=False
    )
    val_data = datasets.OxfordIIITPet(
        root="./", split="trainval", transform=val_transform, download=False
    )
    train_data = Subset(train_data, train_indices.indices)
    val_data = Subset(val_data, val_indices.indices)
    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, num_workers=os.cpu_count()
    )

    imbalanced_train = create_cat_imbalanced_subset(train_data)
    sampler = class_imbalance.make_weighted_sampler(imbalanced_train)
    loader = DataLoader(imbalanced_train, batch_size=batch_size, sampler=sampler)

    labels = [label for _, label in imbalanced_train]
    print("Original distribution:")
    print(Counter(labels))
    sampled_labels = []
    for i, (_, labels) in enumerate(loader):
        sampled_labels.extend(labels.tolist())
        if i >= 20:
            break

    print("Sampled distribution (first batches):")
    print(Counter(sampled_labels))


if __name__ == "__main__":
    main()
