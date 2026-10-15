"""Download, verify and prepare the paper datasets without TensorFlow."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import wave

import numpy as np

from .data import CANONICAL_ROWS, CLASSES, DATASETS, SPEECH_LABELS, checked_path, data_metadata, _labels

SPEECH = {
    "speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
    "speech_commands_test_set_v0.02.tar.gz": "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df",
}
SPEECH_BASE = "https://storage.googleapis.com/download.tensorflow.org/data/"
CIFAR10_MIRRORS = ("https://air-example-data.s3.us-west-2.amazonaws.com/cifar-10-python.tar.gz",)


def digest(path, algorithm="sha256"):
    checksum = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download(url, destination, checksum, algorithm="sha256", *, mirrors=()):
    """Resume partial downloads; publish only after the official checksum matches."""
    for mirror in mirrors:
        try:
            return download(mirror, destination, checksum, algorithm)
        except (urllib.error.URLError, TimeoutError):
            print("Download mirror unavailable; trying the next source", flush=True)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"Download destination is a symbolic link: {destination}")
    if destination.exists():
        if destination.is_symlink() or digest(destination, algorithm) != checksum:
            raise ValueError(f"Checksum mismatch: {destination}; remove this damaged download and retry")
        return destination
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink():
        raise ValueError(f"Download destination is a symbolic link: {partial}")
    if partial.exists() and digest(partial, algorithm) == checksum:
        partial.replace(destination)
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "FedCode-data/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"Downloading {destination.name}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        resume = offset > 0 and response.status == 206
        if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Server returned an incorrect download range")
        with partial.open("ab" if resume else "wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    if digest(partial, algorithm) != checksum:
        raise ValueError(f"Checksum mismatch: {partial}; remove this partial file and retry")
    partial.replace(destination)
    return destination


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or (not path.parts and name not in (".", "./")):
        raise ValueError(f"Unsafe archive member: {name}")
    return Path(*path.parts)


def extract_archive(archive, destination):
    """Extract verified archives atomically, rejecting links and special files."""
    archive, destination = Path(archive), Path(destination)
    checksum = digest(archive)
    marker = destination / ".archive.json"
    if destination.is_symlink() or marker.is_symlink():
        raise ValueError("Archive extraction cannot traverse symbolic links")
    if marker.is_file():
        if json.loads(marker.read_text()).get("sha256") != checksum:
            raise ValueError(f"Extraction belongs to another archive: {destination}")
        # The marker identifies the source but does not prove extracted integrity.
        with tarfile.open(archive) as source:
            cache = {}
            for member in source:
                relative = safe_member(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Archive links and special files are not supported: {member.name}")
                path = checked_path(destination, relative.as_posix(), cache)
                with source.extractfile(member) as reader:
                    expected = hashlib.file_digest(reader, "sha256").hexdigest()
                if path.stat().st_size != member.size or digest(path) != expected:
                    raise ValueError(f"Extracted file changed: {path}")
        return destination
    if destination.exists():
        raise FileExistsError(f"Unverified extraction exists: {destination}; choose an empty data root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        with tarfile.open(archive) as source:
            for member in source:
                target = stage / safe_member(member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
                else:
                    raise ValueError(f"Archive links and special files are not supported: {member.name}")
        (stage / ".archive.json").write_text(json.dumps({"sha256": checksum}) + "\n")
        stage.rename(destination)
    return destination


def save_array(directory, name, values):
    with (Path(directory) / f"{name}.npy").open("xb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)


def _file_entry(path, root):
    path = checked_path(root, Path(path).relative_to(root).as_posix())
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}


def finish_manifest(directory, dataset, metadata, raw_files=(), root=None):
    files = [_file_entry(path, directory) for path in sorted(Path(directory).glob("*.npy"))]
    external = [_file_entry(path, root) for path in sorted(set(raw_files))]
    manifest = {"schema": 1, "dataset": dataset, "classes": CLASSES[dataset],
                "files": files, "raw_files": external, **metadata}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (Path(directory) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest



@contextmanager
def staging(root, dataset):
    destination = Path(root) / "prepared" / dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Partial preparation exists: {destination}; inspect it before removal or choose a new root")
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=destination.parent) as temporary:
        stage = Path(temporary) / dataset
        stage.mkdir()
        yield stage
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)


def verify_dataset(dataset, root="data"):
    """Hash-check all persisted content once before a training launch."""
    root = Path(root).expanduser().resolve()
    manifest = data_metadata(dataset, root)
    directory = root / "prepared" / dataset
    expected = {f"{split}_{kind}.npy" for split in CANONICAL_ROWS[dataset]
                for kind in (("labels", "paths", "offsets") if dataset == "speech_commands" else ("labels", "images"))}
    if {entry["path"] for entry in manifest["files"]} != expected or len(manifest["files"]) != len(expected):
        raise ValueError("Manifest must cover exactly the required arrays")
    for base, entries in ((directory, manifest["files"]), (root, manifest["raw_files"])):
        cache = {}
        for entry in entries:
            path = checked_path(base, entry["path"], cache)
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError(f"Prepared data are missing or changed: {path}")
    raw_paths = {entry["path"] for entry in manifest["raw_files"]}
    for split in CANONICAL_ROWS[dataset]:
        labels = _labels(directory, dataset, split)
        if dataset == "speech_commands":
            paths = np.load(directory / f"{split}_paths.npy", allow_pickle=False)
            offsets = np.load(directory / f"{split}_offsets.npy", allow_pickle=False)
            if (paths.dtype.kind != "U" or paths.shape != labels.shape or offsets.shape != (len(labels), 2)
                    or not np.issubdtype(offsets.dtype, np.integer) or np.any(offsets[:, 0] < 0)
                    or np.any(offsets[:, 1] < 1) or not set(paths).issubset(raw_paths)):
                raise ValueError("Audio manifest must cover every waveform, with valid offsets")
        else:
            images = np.load(directory / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
            if images.dtype != np.uint8 or images.shape != (len(labels), 32, 32, 3):
                raise ValueError("Invalid canonical CIFAR image array")
    if dataset == "speech_commands" and manifest.get("class_names") != list(SPEECH_LABELS):
        raise ValueError("Speech Commands class order does not match FedCode")
    return manifest


def require_space(root, required_bytes):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < required_bytes:
        raise OSError(f"Preparing data requires about {required_bytes / 2**30:.1f} GiB additional free space; "
                      f"only {free / 2**30:.1f} GiB available at {root}. Choose --root on a larger volume.")


def _prepare_cifar(root, dataset, directory):
    from torchvision.datasets import CIFAR10, CIFAR100
    cls = CIFAR10 if dataset == "cifar10" else CIFAR100
    require_space(root, int(0.8 * 2**30))
    archive = download(cls.url, root / "downloads" / cls.filename, cls.tgz_md5, "md5",
                       mirrors=CIFAR10_MIRRORS if dataset == "cifar10" else ())
    raw = extract_archive(archive, root / "raw" / dataset)
    histograms = {}
    for split, expected in CANONICAL_ROWS[dataset].items():
        # Official batch checksums are verified by torchvision before unpickling.
        data = cls(raw, train=split == "train", download=False)
        labels = np.asarray(data.targets, dtype=np.int64)
        if (data.data.dtype != np.uint8 or data.data.shape != (expected, 32, 32, 3)
                or labels.shape != (expected,) or not np.array_equal(np.unique(labels), np.arange(CLASSES[dataset]))):
            raise ValueError(f"Unexpected canonical {dataset}/{split} data")
        save_array(directory, f"{split}_images", data.data)
        save_array(directory, f"{split}_labels", labels)
        histograms[split] = np.bincount(labels, minlength=CLASSES[dataset]).tolist()
    return finish_manifest(directory, dataset, {"rows": CANONICAL_ROWS[dataset], "source": cls.url,
                           "source_sha256": digest(archive), "class_counts": histograms,
                           "class_names": data.classes, "row_order": "Official Python batch order",
                           "preprocessing": "Train random crop 32 with padding 4 and horizontal flip; mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)"})


def _wave_frames(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError(f"Expected mono 16 kHz, 16-bit PCM: {path}")
        count = stream.getnframes()
        if not count or len(stream.readframes(count)) != 2 * count:
            raise ValueError(f"Empty or truncated waveform PCM payload: {path}")
    return count


def _split_list(main, filename):
    entries = (main / filename).read_text().splitlines()
    if not entries or len(entries) != len(set(entries)):
        raise ValueError(f"Empty or duplicate official split list: {filename}")
    cache = {}
    for entry in entries:
        checked_path(main, entry, cache)
    return set(entries)


def speech_rows(main, test):
    """TFDS 0.0.3 complete 12-class split, with FedCode class IDs.

    Every non-target word is retained. Background windows overlap only within
    one split; running_tap.wav is used exclusively for validation.
    """
    main, test = Path(main).resolve(strict=True), Path(test).resolve(strict=True)
    validation = _split_list(main, "validation_list.txt")
    excluded = _split_list(main, "testing_list.txt")
    if validation & excluded:
        raise ValueError("Official validation and testing lists overlap")
    validation.add("_background_noise_/running_tap.wav")
    checked_path(main, "_background_noise_/running_tap.wav")
    rows = {"train": [], "validation": [], "test": []}
    speakers = {split: set() for split in rows}
    for source, test_set in ((main, False), (test, True)):
        cache = {}
        for path in sorted(source.glob("*/*.wav")):
            relative = path.relative_to(source).as_posix()
            checked_path(source, relative, cache)
            word = path.parent.name
            if not test_set and relative in excluded:
                continue
            if test_set and word not in (*SPEECH_LABELS[2:], "_silence_", "_unknown_"):
                raise ValueError(f"Unexpected official test category: {word}")
            split = "test" if test_set else "validation" if relative in validation else "train"
            label = SPEECH_LABELS.index(word) if word in SPEECH_LABELS[2:] else 1 if word in ("_background_noise_", "_silence_") else 0
            count = _wave_frames(path)
            if word == "_background_noise_":
                for offset in range(0, count - 16000, 8000):
                    rows[split].append((path, label, offset, 16000))
            else:
                if count > 16000:
                    raise ValueError(f"Expected at most one second of speech: {path}")
                rows[split].append((path, label, 0, count))
                if "_nohash_" in path.stem:
                    speakers[split].add(path.stem.split("_nohash_")[0])
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if speakers[left] & speakers[right]:
            raise ValueError(f"Speech Commands speaker overlap between {left} and {right}")
        if {r[0] for r in rows[left]} & {r[0] for r in rows[right]}:
            raise ValueError(f"Waveforms shared between {left} and {right}")
    for split, values in rows.items():
        if {row[1] for row in values} != set(range(12)):
            raise ValueError(f"Speech Commands {split} must contain every class, including unknown and silence")
    return rows


def _prepare_speech(root, directory):
    required = 0
    for name in SPEECH:
        main = name == "speech_commands_v0.02.tar.gz"
        compressed, extracted = (2.3, 3.6) if main else (0.12, 0.17)
        if not (root / "downloads" / name).is_file():
            partial = root / "downloads" / (name + ".part")
            required += max(0, int(compressed * 2**30) - (partial.stat().st_size if partial.is_file() else 0))
        if not (root / "raw" / name.removesuffix(".tar.gz") / ".archive.json").is_file():
            required += int(extracted * 2**30)
    require_space(root, required + int(0.15 * 2**30))
    artifacts = {name: download(SPEECH_BASE + name, root / "downloads" / name, checksum)
                 for name, checksum in SPEECH.items()}
    main = extract_archive(artifacts["speech_commands_v0.02.tar.gz"], root / "raw" / "speech_commands_v0.02")
    test = extract_archive(artifacts["speech_commands_test_set_v0.02.tar.gz"], root / "raw" / "speech_commands_test_set_v0.02")
    rows = speech_rows(main, test)
    counts = {split: len(values) for split, values in rows.items()}
    if counts != CANONICAL_ROWS["speech_commands"]:
        raise ValueError(f"Unexpected canonical Speech Commands counts: {counts}")
    files, histograms = [main / "validation_list.txt", main / "testing_list.txt"], {}
    for split, values in rows.items():
        labels = np.asarray([row[1] for row in values], dtype=np.int64)
        save_array(directory, f"{split}_paths", [row[0].relative_to(root).as_posix() for row in values])
        save_array(directory, f"{split}_labels", labels)
        save_array(directory, f"{split}_offsets", np.asarray([row[2:] for row in values], dtype=np.int64))
        files.extend(row[0] for row in values)
        histograms[split] = np.bincount(labels, minlength=12).tolist()
    return finish_manifest(directory, "speech_commands", {"rows": counts, "source": SPEECH_BASE,
                           "source_sha256": SPEECH, "class_names": SPEECH_LABELS,
                           "class_counts": histograms, "sample_rate": 16000,
                           "protocol": "official-tfds-0.0.3-12-class",
                           "split": "Train excludes official validation/testing lists; separate canonical test archive; speaker-disjoint",
                           "unknown": "All non-target words; no balancing, class weights or subsampling",
                           "silence": "One-second background windows, half-second stride, exclusive endpoint; running_tap reserved for validation; official test silence",
                           "row_order": "Sorted source paths, then increasing waveform offset",
                           "preprocessing": "Center pad/trim 16000; power mel64x101 fft400/hop160 125-7500Hz plus1e-3; no logarithm"}, files, root)


def prepare_dataset(dataset, root="data", validate=False):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    root = Path(root).expanduser().resolve()
    if (root / "prepared" / dataset / "manifest.json").is_file():
        manifest = verify_dataset(dataset, root)
    else:
        if validate:
            raise FileNotFoundError(f"{dataset} is not prepared under {root}")
        with staging(root, dataset) as directory:
            manifest = _prepare_speech(root, directory) if dataset == "speech_commands" else _prepare_cifar(root, dataset, directory)
        manifest = verify_dataset(dataset, root)
    print(f"{dataset}: ready ({manifest['rows']})", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and prepare FedCode datasets; omit --dataset to prepare all.")
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--validate", action="store_true", help="Verify prepared content without downloads")
    args = parser.parse_args(argv)
    result = [prepare_dataset(dataset, args.root, args.validate)
              for dataset in DATASETS if args.dataset in (dataset, "all")]
    return result[0] if len(result) == 1 else result


if __name__ == "__main__":
    main()
