import os
import random
import sys
from collections import Counter
from pathlib import Path

import ClassImbalance
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from DataLoader import CreateDataset, build_dataloaders

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
        train_dataset: CreateDataset instance with (image, (species, breed)) format
        cat_fraction: fraction of cat samples to keep (e.g., 0.2)
        seed: reproducibility

    Returns:
        Subset with imbalanced class distribution
    """
    random.seed(seed)
    np.random.seed(seed)
    indices = list(range(len(train_dataset)))
    labels = np.array([train_dataset[i][1][0].item() for i in indices])
    cat_indices = np.where(labels == 0)[0]
    dog_indices = np.where(labels == 1)[0]

    n_cats_keep = int(len(cat_indices) * cat_fraction)
    selected_cat_indices = np.random.choice(cat_indices, n_cats_keep, replace=False)
    final_indices = np.concatenate([selected_cat_indices, dog_indices])
    np.random.shuffle(final_indices)

    return Subset(train_dataset, final_indices)


def main():
    dataset_root = "../oxford-iiit-pet"
    batch_size = 32
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=dataset_root,
        val_split=0.2,
        batch_size=batch_size,
        one_hot=False,
        image_size=224,
        num_workers=os.cpu_count(),
        seed=42,
    )
    imbalanced_train = create_cat_imbalanced_subset(train_loader.dataset)
    sampler = ClassImbalance.make_weighted_sampler(imbalanced_train)
    loader = DataLoader(imbalanced_train, batch_size=batch_size, sampler=sampler)
    labels = [
        label[0].item() if hasattr(label[0], "item") else label[0]
        for _, label in imbalanced_train
    ]
    print("Original distribution:")
    print(Counter(labels))
    sampled_labels = []
    for i, (_, label_tuple) in enumerate(loader):
        sampled_labels.extend(
            label_tuple[0].tolist()
            if hasattr(label_tuple[0], "tolist")
            else label_tuple[0]
        )

        if i >= 20:
            break

    print("Sampled distribution (first batches):")
    print(Counter(sampled_labels))


if __name__ == "__main__":
    main()
