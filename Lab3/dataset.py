import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from config import Config


class PairedMNISTSVHN(Dataset):
    def __init__(self, mnist_dataset, svhn_dataset):
        self.mnist_dataset = mnist_dataset
        self.svhn_dataset = svhn_dataset

        self.mnist_by_label = self._group_by_label(mnist_dataset)
        self.svhn_by_label = self._group_by_label(svhn_dataset)

        self.pairs = self._create_pairs()

        self._validate_pairing()

    def _group_by_label(self, dataset):
        label_to_indices = {i: [] for i in range(10)}

        is_svhn = dataset.__class__.__name__ == "SVHN"

        for idx in range(len(dataset)):
            _, label = dataset[idx]

            if not (0 <= label <= 9):
                print(f"Warning: Invalid label {label} at index {idx}")
                continue

            label_to_indices[label].append(idx)
        return label_to_indices

    def _validate_pairing(self):
        labels_with_pairs = set()
        for _, _, label in self.pairs:
            labels_with_pairs.add(label)

        missing_labels = set(range(10)) - labels_with_pairs
        if missing_labels:
            raise ValueError(
                f"Missing digit classes in paired dataset: {sorted(missing_labels)}. "
                f"This indicates a label mapping issue between MNIST and SVHN."
            )

        label_counts = {i: 0 for i in range(10)}
        for _, _, label in self.pairs:
            label_counts[label] += 1

        print(f"Dataset pairing complete: {len(self.pairs)} total pairs")
        print(f"Samples per digit class: {dict(sorted(label_counts.items()))}")

    def _create_pairs(self):
        pairs = []
        for label in range(10):
            mnist_indices = self.mnist_by_label[label]
            svhn_indices = self.svhn_by_label[label]

            min_count = min(len(mnist_indices), len(svhn_indices))

            for i in range(min_count):
                pairs.append((mnist_indices[i], svhn_indices[i], label))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mnist_idx, svhn_idx, label = self.pairs[idx]

        mnist_img, _ = self.mnist_dataset[mnist_idx]
        svhn_img, _ = self.svhn_dataset[svhn_idx]

        return mnist_img, svhn_img, label


def get_mnist_transform(train=True):
    transform_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
    ]
    return transforms.Compose(transform_list)


def get_svhn_transform(train=True):
    transform_list = [
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0, 1] -> [-1, 1]
    ]
    return transforms.Compose(transform_list)


def load_datasets(config):
    """
    Load MNIST and SVHN datasets and create paired datasets.

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
    """
    mnist_train = datasets.MNIST(
        root=config.data_dir,
        train=True,
        transform=get_mnist_transform(train=True),
        download=True,
    )

    mnist_test = datasets.MNIST(
        root=config.data_dir,
        train=False,
        transform=get_mnist_transform(train=False),
        download=True,
    )

    svhn_train = datasets.SVHN(
        root=config.data_dir,
        split="train",
        transform=get_svhn_transform(train=True),
        download=True,
    )

    svhn_test = datasets.SVHN(
        root=config.data_dir,
        split="test",
        transform=get_svhn_transform(train=False),
        download=True,
    )

    train_dataset = PairedMNISTSVHN(mnist_train, svhn_train)
    test_dataset = PairedMNISTSVHN(mnist_test, svhn_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device.type == "cuda" else False,
    )

    return train_loader, test_loader
