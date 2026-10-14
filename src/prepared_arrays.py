"""Validated, pickle-free image inputs shared by preparation and training."""
from pathlib import Path
import numpy as np

CLASSES = {"cifar10": 10, "cifar100": 100, "fashionmnist": 10, "pathmnist": 9}
SHAPES = {"cifar10": (32,32,3), "cifar100": (32,32,3), "fashionmnist": (28,28), "pathmnist": (28,28,3)}


def read_arrays(path, dataset):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key]) for key in ("train_images", "train_labels", "test_images", "test_labels")}
    for split in ("train", "test"):
        x, y = arrays[split + "_images"], arrays[split + "_labels"]
        if y.ndim not in (1,2) or (y.ndim == 2 and y.shape[1] != 1):
            raise ValueError("Expected one label per image")
        y = y.reshape(-1)
        if x.dtype != np.uint8 or x.shape[1:] != SHAPES[dataset] or not len(x) or len(x) != len(y):
            raise ValueError(f"Invalid {dataset} {split} images/labels")
        if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= CLASSES[dataset])):
            raise ValueError("Labels outside the dataset class space")
        arrays[split + "_labels"] = y[:,None]
    return arrays


def image_dataset(path, dataset, train, transform):
    """A torchvision-compatible dataset with a fixed scientific class space."""
    import torch
    from PIL import Image
    arrays = read_arrays(path, dataset)
    split = "train" if train else "test"
    class Images(torch.utils.data.Dataset):
        def __init__(self):
            self.data = arrays[split + "_images"]
            self.targets = arrays[split + "_labels"].reshape(-1).tolist()
            self.classes = list(range(CLASSES[dataset]))
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, index):
            image = Image.fromarray(self.data[index])
            return (transform(image) if transform else image), self.targets[index]
    return Images()
