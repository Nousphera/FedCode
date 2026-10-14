"""Prepare or validate the exact dataset files consumed by this repository.

Imports use NPZ or already-extracted/prepared directories, never arbitrary pickle
or archives. Download mode uses the public dataset library's official source.
Existing outputs are never silently replaced. See docs/DATA_SETUP.md for sources.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tarfile

# Protect extraction by the pinned dataset libraries (including torchvision 0.21).
tarfile.TarFile.extraction_filter = staticmethod(tarfile.data_filter)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPOSITORY = "FedCode"
if REPOSITORY == "FiBA":
    from fiba.prepared_arrays import read_arrays
else:
    from prepared_arrays import read_arrays

DATASETS = ['cifar10', 'cifar100', 'spcm']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_files(root):
    paths = sorted(Path(root).rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Symlinks are not permitted in imported data")
    return [p for p in paths if p.is_file() and p.name != "preparation-manifest.json"]


def manifest(root, paths, dataset, metadata):
    return {"schema": 1, "dataset": dataset, "metadata": metadata,
            "files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(paths)]}


def record(root, name, payload, validate):
    path = root / (name + ".manifest.json")
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != payload:
            raise ValueError("Data differs from the saved preparation manifest")
    elif validate:
        raise FileNotFoundError(f"Preparation manifest missing: {path}; first import/prepare the data")
    else:
        with path.open("x") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dataset": payload["dataset"], "files": len(payload["files"]),
                      "metadata": payload["metadata"], "manifest": str(path)}, sort_keys=True))
    return payload


def image_path(root, dataset):
    if REPOSITORY == "FedCompress" and dataset == "pathmnist":
        return root / "raw/pathmnist/pathmnist.npz"
    return root / (dataset + ".npz")


def prepare_images(args):
    base = {"cifar10n": "cifar10", "cifar100n": "cifar100"}.get(args.dataset, args.dataset)
    path = image_path(args.root, base)
    if args.source:
        if args.validate:
            raise ValueError("Use --validate on --root without --source")
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
        read_arrays(args.source, base)
        path.parent.mkdir(parents=True, exist_ok=True)
        with args.source.open("rb") as src, path.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    elif args.download:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}; use --validate")
        if base == "pathmnist":
            import urllib.request
            path.parent.mkdir(parents=True, exist_ok=True)
            # Pin the official 28x28 artifact before accepting the data.
            with tempfile.TemporaryDirectory(dir=path.parent) as directory:
                temporary = Path(directory) / "pathmnist.npz"
                with urllib.request.urlopen("https://zenodo.org/records/10519652/files/pathmnist.npz?download=1") as src, temporary.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                md5 = hashlib.md5(temporary.read_bytes()).hexdigest()
                if md5 != "a8b06965200029087d5bd730944a56c1":
                    raise ValueError("Official PathMNIST checksum mismatch")
                read_arrays(temporary, base)
                with temporary.open("rb") as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Official library download URLs and hashes; cache is private to this preparation.
            with tempfile.TemporaryDirectory(dir=path.parent) as directory:
                if REPOSITORY in ("FedLN", "FedCompress"):
                    os.environ["KERAS_HOME"] = directory
                    import tensorflow as tf
                    loader = {"cifar10": tf.keras.datasets.cifar10, "cifar100": tf.keras.datasets.cifar100,
                              "fashionmnist": tf.keras.datasets.fashion_mnist}[base]
                    train, test = loader.load_data()
                    values = dict(train_images=train[0], train_labels=train[1], test_images=test[0], test_labels=test[1])
                else:
                    import torchvision
                    loader = torchvision.datasets.CIFAR10 if base == "cifar10" else torchvision.datasets.CIFAR100
                    train, test = [loader(directory, train=split, download=True) for split in (True, False)]
                    values = dict(train_images=train.data, train_labels=np.asarray(train.targets), test_images=test.data, test_labels=np.asarray(test.targets))
                temporary = Path(directory) / "prepared.npz"
                np.savez_compressed(temporary, **values)
                read_arrays(temporary, base)
                with temporary.open("rb") as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
    arrays = read_arrays(path, base)
    metadata = {"classes": 100 if base == "cifar100" else 9 if base == "pathmnist" else 10,
                "train_rows": len(arrays["train_images"]), "test_rows": len(arrays["test_images"]),
                "row_order": "source NPZ / official library order; no shuffle"}
    paths = [path]
    if args.dataset in ("cifar10n", "cifar100n"):
        if not args.human_labels or not args.human_noise_key:
            raise ValueError("CIFAR-N requires --human-labels safe NPZ and explicit --human-noise-key; see prepare_human_labels.py")
        from data.vision import human_annotations
        human_annotations(args.human_labels, args.human_noise_key, arrays["train_labels"].reshape(-1), args.dataset)
        metadata.update(human_labels_sha256=digest(args.human_labels), human_noise_key=args.human_noise_key)
    return record(args.root, args.dataset, manifest(args.root, paths, args.dataset, metadata), args.validate)


def prepare_directory(args):
    if args.source:
        if args.download or args.validate:
            raise ValueError("--source cannot be combined with --download or --validate")
        if args.root.exists():
            raise FileExistsError("Import destination already exists; validate it in place or choose a new root")
        source = args.source.resolve(strict=True)
        if source == args.root or source in args.root.parents:
            raise ValueError("Destination must not be inside the source")
        tree_files(source)  # reject links before copying
        shutil.copytree(source, args.root)
    if args.dataset == "eurosat":
        path = args.root / "eurosat/eurosat.npz"
        with np.load(path, allow_pickle=False) as data:
            x,y = data["images"], data["labels"]
            if x.shape != (27000,64,64,3) or x.dtype != np.uint8 or y.shape not in ((27000,), (27000,1)):
                raise ValueError("EuroSAT requires 27000 RGB 64x64 uint8 images in the chosen source order")
            if not np.issubdtype(y.dtype,np.integer) or np.any((y < 0) | (y >= 10)):
                raise ValueError("EuroSAT labels must be integers in [0,9]")
        metadata = {"train_rows":21600,"test_rows":5400,"classes":10,"historical_order_verified":False}
        paths = [path]
    elif args.dataset in ("ood-images", "ood-audio"):
        paths = tree_files(args.root)
        suffixes = {".wav"} if args.dataset == "ood-audio" else {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        paths = [p for p in paths if p.suffix.lower() in suffixes]
        if not paths:
            raise ValueError("No supported auxiliary files found")
        for path in paths:
            if args.dataset == "ood-audio":
                import wave
                with wave.open(str(path), "rb") as w:
                    if w.getframerate() != 16000 or w.getnchannels() != 1 or not w.getnframes():
                        raise ValueError("Auxiliary WAVs must be nonempty mono 16 kHz PCM")
            else:
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
        metadata = {"files":len(paths),"historical_subset_recovered":False}
    elif REPOSITORY == "FedLN":
        if args.download:
            raise ValueError("Use prepare_audio.py --download --tfds-cache CACHE --output ROOT, then this validator")
        os.environ["TFDS_DATA_DIR"] = str(args.root)
        from utils.features import source_digest
        from data.spcm import load_raw_data
        for split in ('train', 'test'):
            (ids, labels, _), _ = load_raw_data(split)
            if not len(ids):
                raise ValueError('Prepared audio split is empty')
        metadata = {"source_sha256": source_digest("spcm", None), "format":"FedLN waveform records and identity sidecars"}
        paths = tree_files(args.root / "speech_commands")
    elif REPOSITORY == "FedCode":
        from datasets.spcm import SpeechCommands
        if args.download: args.root.mkdir(parents=True, exist_ok=True)
        paths = []
        counts = {}
        seen = set()
        for split in ("training", "validation", "testing"):
            ds = SpeechCommands(root=str(args.root), url="speech_commands_v0.02", folder_in_archive="spcm", subset=split, download=args.download)
            counts[split] = len(ds)
            if not len(ds):
                raise ValueError("Speech Commands splits must be nonempty")
            for i in range(len(ds)):
                metadata_item = ds.get_metadata(i)
                import wave
                waveform = (Path(ds._archive) / metadata_item[0]).resolve(strict=True)
                if not waveform.is_relative_to(args.root) or waveform in seen:
                    raise ValueError("Audio split paths must be distinct files inside the prepared root")
                seen.add(waveform)
                with wave.open(str(waveform), "rb") as audio:
                    if audio.getframerate() != 16000 or audio.getnchannels() != 1 or not audio.getnframes():
                        raise ValueError("Speech Commands requires nonempty mono 16 kHz WAVs")
        paths = tree_files(args.root / "spcm")
        metadata = {"splits":counts,"labels":12,"silence_recipe_recovered":False}
    else:
        import tensorflow_datasets as tfds
        name = "speech_commands" if args.dataset == "spcm" else "voxforge"
        if args.download and name == "voxforge":
            raise ValueError("VoxForge requires manual archives/Beam preparation; see docs/DATA_SETUP.md")
        builder = tfds.builder(name, data_dir=str(args.root))
        if args.download:
            builder.download_and_prepare()
        counts = {}
        for split in ("train", "test"):
            ds = builder.as_dataset(split=split, shuffle_files=False)
            count = 0
            for example in tfds.as_numpy(ds):
                audio, label = np.asarray(example["audio"]), int(example["label"])
                if audio.ndim != 1 or not len(audio) or not 0 <= label < builder.info.features["label"].num_classes:
                    raise ValueError("Invalid TFDS audio or label")
                count += 1
            if count != builder.info.splits[split].num_examples or not count:
                raise ValueError("TFDS split count mismatch")
            counts[split] = count
        paths = tree_files(Path(str(builder.data_dir)))
        metadata = {"full_name":builder.info.full_name,"splits":counts,"tfds":tfds.__version__}
    return record(args.root, args.dataset, manifest(args.root, paths, args.dataset, metadata), args.validate)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--root", type=Path, required=True, help="Exact training data root (or auxiliary directory)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--source", type=Path, help="Import safe NPZ or prepared directory into a new destination")
    mode.add_argument("--download", action="store_true", help="Opt in to an official public download; may be large")
    mode.add_argument("--validate", action="store_true", help="Check schema and SHA-256 against the saved manifest")
    p.add_argument("--human-labels", type=Path)
    p.add_argument("--human-noise-key")
    args = p.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    if args.source: args.source = args.source.expanduser().resolve(strict=True)
    if args.dataset in ("cifar10", "cifar100", "cifar10n", "cifar100n", "fashionmnist", "pathmnist"):
        return prepare_images(args)
    if args.download and args.dataset not in ("spcm", "voxforge"):
        p.error("This dataset requires an explicit prepared --source; see docs/DATA_SETUP.md")
    return prepare_directory(args)


if __name__ == "__main__":
    main()
