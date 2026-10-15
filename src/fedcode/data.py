"""Canonical FedCode data, deterministic loaders and exhaustive client partitions."""
from copy import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import wave

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, get_worker_info

DATASETS = ("cifar10", "cifar100", "speech_commands")
CLASSES = {"cifar10": 10, "cifar100": 100, "speech_commands": 12}
SPEECH_LABELS = ("unknown", "silence", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
CANONICAL_ROWS = {"cifar10": {"train": 50000, "test": 10000},
                  "cifar100": {"train": 50000, "test": 10000},
                  "speech_commands": {"train": 85511, "validation": 10102, "test": 4890}}


def checked_path(root, name, parent_cache=None):
    """Reject traversal, links and missing regular files in persisted manifests."""
    relative = PurePosixPath(str(name))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in str(name):
        raise ValueError(f"Unsafe data path: {name}")
    cache = {} if parent_cache is None else parent_cache
    root_key = ("root", Path(root))
    if root_key not in cache:
        cache[root_key] = Path(root).resolve(strict=True)
    root = cache[root_key]
    path = root / Path(*relative.parts)
    if path.parent not in cache:
        resolved = path.parent.resolve(strict=True)
        if resolved != path.parent or not resolved.is_relative_to(root):
            raise ValueError(f"Data path traverses a symbolic link: {name}")
        cache[path.parent] = resolved
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Data file is missing or a symbolic link: {name}")
    return path


def data_metadata(dataset, root="data"):
    """Read the stable data fingerprint; full hashes are checked by preparation."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    directory = Path(root).expanduser().resolve() / "prepared" / dataset
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prepare data first: python scripts/prepare_data.py --dataset {dataset} --root {root}")
    if path.is_symlink() or directory.is_symlink():
        raise ValueError("Prepared metadata cannot be a symbolic link")
    manifest = json.loads(path.read_text())
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError(f"Preparation manifest was modified: {path}")
    if manifest.get("schema") != 1 or manifest.get("dataset") != dataset or manifest.get("classes") != CLASSES[dataset]:
        raise ValueError(f"Unexpected preparation manifest: {path}")
    if manifest.get("rows") != CANONICAL_ROWS[dataset]:
        raise ValueError(f"Expected canonical {dataset} split sizes: {CANONICAL_ROWS[dataset]}")
    if dataset == "speech_commands" and manifest.get("class_names") != list(SPEECH_LABELS):
        raise ValueError("Speech Commands class order does not match FedCode")
    return manifest


def _labels(directory, dataset, split):
    labels = np.load(checked_path(directory, f"{split}_labels.npy"), mmap_mode="r", allow_pickle=False)
    expected = CANONICAL_ROWS[dataset][split]
    if labels.shape != (expected,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected {expected} integer labels for {dataset}/{split}")
    if not np.array_equal(np.unique(labels), np.arange(CLASSES[dataset])):
        raise ValueError(f"{dataset}/{split} must contain every class and no out-of-range labels")
    return labels


def read_waveform(path, start=0, length=None):
    """Decode original signed 16-bit, mono 16 kHz PCM without a codec backend."""
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError(f"Expected mono 16 kHz, 16-bit PCM: {path}")
        count = stream.getnframes() - start if length is None else length
        if start < 0 or count < 1 or start + count > stream.getnframes():
            raise ValueError(f"Invalid waveform segment: {path}")
        stream.setpos(start)
        samples = stream.readframes(count)
    if len(samples) != 2 * count:
        raise ValueError(f"Truncated waveform PCM payload: {path}")
    return torch.from_numpy(np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0).unsqueeze(0)


class VisionDataset(Dataset):
    def __init__(self, directory, dataset, split, metadata, training):
        self.labels = _labels(directory, dataset, split)
        self.targets, self.classes = self.labels, list(range(CLASSES[dataset]))
        self.images = np.load(checked_path(directory, f"{split}_images.npy"), mmap_mode="r", allow_pickle=False)
        if self.images.dtype != np.uint8 or self.images.shape != (len(self.labels), 32, 32, 3):
            raise ValueError("Expected canonical CIFAR uint8 RGB images")
        self.training, self.metadata = training, metadata
        self.fingerprint = metadata["fingerprint"]
        self.generator = torch.Generator().manual_seed(0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.from_numpy(np.array(self.images[index], copy=True)).permute(2, 0, 1).float().div_(255)
        if self.training:
            image = torch.nn.functional.pad(image, (4, 4, 4, 4))
            top, left = torch.randint(9, (2,), generator=self.generator).tolist()
            image = image[:, top:top + 32, left:left + 32]
            if bool(torch.rand((), generator=self.generator) < 0.5):
                image = image.flip(-1)
        mean = image.new_tensor((0.4914, 0.4822, 0.4465))[:, None, None]
        std = image.new_tensor((0.2023, 0.1994, 0.2010))[:, None, None]
        return (image - mean) / std, int(self.labels[index])


class AudioDataset(Dataset):
    def __init__(self, directory, root, split, metadata, training=False):
        from .audio import AudioTransform
        self.root, self.training, self.metadata = Path(root).resolve(), training, metadata
        self.labels = _labels(directory, "speech_commands", split)
        self.targets, self.classes = self.labels, list(SPEECH_LABELS)
        self.fingerprint = metadata["fingerprint"]
        self.paths = np.load(checked_path(directory, f"{split}_paths.npy"), allow_pickle=False)
        self.paths.setflags(write=False)
        self.offsets = np.load(checked_path(directory, f"{split}_offsets.npy"), mmap_mode="r", allow_pickle=False)
        if (self.paths.shape != self.labels.shape or self.paths.dtype.kind != "U"
                or self.offsets.shape != (len(self.labels), 2) or not np.issubdtype(self.offsets.dtype, np.integer)
                or np.any(self.offsets[:, 0] < 0) or np.any(self.offsets[:, 1] < 1)):
            raise ValueError("Invalid or misaligned waveform paths, offsets and labels")
        cache = {}
        for name in np.unique(self.paths):
            checked_path(self.root, str(name), cache)
        self.transform = AudioTransform(length=16000)

    def __len__(self):
        return len(self.labels)

    def raw(self, index):
        start, count = map(int, self.offsets[index])
        return read_waveform(self.root / str(self.paths[index]), start, count)

    def __getitem__(self, index):
        return self.transform(self.raw(index)), int(self.labels[index])


@lru_cache(maxsize=8)
def _cached_dataset(dataset, root, split, training, fingerprint):
    """Bounded per-process cache of read-only arrays and validated audio paths."""
    metadata = data_metadata(dataset, root)
    if metadata["fingerprint"] != fingerprint:
        raise ValueError("Preparation metadata changed while loading data; retry the load")
    directory = Path(root) / "prepared" / dataset
    if dataset == "speech_commands":
        return AudioDataset(directory, root, split, metadata, training)
    return VisionDataset(directory, dataset, split, metadata, training)


def load_data(dataset, root="data", train=True, split=None):
    """Load prepared data without downloading or repeating per-row path checks.

    Fresh manifest reads invalidate the bounded process-local cache if the
    fingerprint changes. Full content verification remains a launch-time check.
    """
    split = split or ("train" if train else "test")
    if dataset not in DATASETS or split not in CANONICAL_ROWS[dataset]:
        raise ValueError(f"Unknown dataset/split: {dataset}/{split}")
    root = str(Path(root).expanduser().resolve())
    metadata = data_metadata(dataset, root)
    data = copy(_cached_dataset(dataset, root, split, train and split == "train", metadata["fingerprint"]))
    # Caller-side augmentation state is not shared between cached dataset views.
    if hasattr(data, "generator"):
        data.generator = torch.Generator().manual_seed(0)
    return data


def load_datasets(dataset, root="data"):
    return load_data(dataset, root, train=True), load_data(dataset, root, train=False)


def partition_indices(labels, num_clients, mode="dirichlet", alpha=10.0, seed=42, min_size=2, max_attempts=100):
    """Partition every training row once; classwise Dirichlet follows source caps.

    The published near-IID setting is Dirichlet(alpha=10), while alpha=0.1 is
    non-IID. ``iid`` is an additional random, equal-size partition. No statement
    about a guaranteed fraction of classes per client is implied by alpha.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Expected a nonempty vector of nonnegative integer labels")
    if isinstance(num_clients, bool) or not isinstance(num_clients, (int, np.integer)) or num_clients < 1:
        raise ValueError("num_clients must be a positive integer")
    if (not isinstance(min_size, (int, np.integer)) or min_size < 1 or not isinstance(max_attempts, (int, np.integer))
            or max_attempts < 1 or len(labels) < num_clients * min_size):
        raise ValueError("Need enough rows for min_size per client and positive bounded max_attempts")
    if mode not in ("iid", "dirichlet") or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Choose iid or dirichlet and a finite positive alpha")
    rng = np.random.default_rng(seed)
    if mode == "iid":
        return [part.astype(np.int64) for part in np.array_split(rng.permutation(len(labels)), num_clients)]
    for _ in range(max_attempts):
        parts = [[] for _ in range(num_clients)]
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            proportions *= np.asarray([len(part) < len(labels) / num_clients for part in parts])
            total = proportions.sum()
            if not np.isfinite(total) or total <= 0:
                continue
            boundaries = (np.cumsum(proportions / total) * len(indices)).astype(np.int64)[:-1]
            for part, chunk in zip(parts, np.split(indices, boundaries)):
                part.extend(chunk.tolist())
        if sum(map(len, parts)) == len(labels) and min(map(len, parts)) >= min_size:
            return [rng.permutation(np.asarray(part, dtype=np.int64)) for part in parts]
    raise ValueError(f"Could not allocate min_size={min_size} in {max_attempts} attempts; increase alpha or reduce clients/min_size")


def label_counts(labels, partitions, num_classes):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("Labels outside the requested class space")
    histograms = []
    for part in partitions:
        part = np.asarray(part)
        if part.ndim != 1 or not np.issubdtype(part.dtype, np.integer) or np.any((part < 0) | (part >= len(labels))):
            raise ValueError("Partition indices outside the label vector")
        histograms.append(np.bincount(labels[part], minlength=num_classes).tolist())
    return histograms


def _seed_worker(worker_id):
    seed = torch.initial_seed()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    dataset = get_worker_info().dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if hasattr(dataset, "generator"):
        dataset.generator.manual_seed(seed)


def make_loader(dataset, batch_size=128, indices=None, shuffle=False, seed=42, num_workers=0):
    """Seed both sampler and augmentation RNGs; retain every evaluation example."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    dataset = copy(dataset)
    if hasattr(dataset, "generator"):
        dataset.generator = torch.Generator().manual_seed(seed)
    if indices is not None:
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or np.any((indices < 0) | (indices >= len(dataset))) or len(np.unique(indices)) != len(indices)):
            raise ValueError("Loader indices must be distinct valid rows")
        dataset = Subset(dataset, indices.tolist())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=torch.Generator().manual_seed(seed), worker_init_fn=_seed_worker,
                      drop_last=False)
