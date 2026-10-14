"""Preparation contracts tested with bounded files, never a corpus download."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts/prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def fixture(path, shape=(32,32,3), classes=10):
    rng = np.random.default_rng(7)
    np.savez(path, train_images=rng.integers(0,256,(40,*shape),dtype=np.uint8),
             train_labels=np.arange(40) % classes,
             test_images=rng.integers(0,256,(10,*shape),dtype=np.uint8), test_labels=np.arange(10)%classes)


def test_prepare_import_validate_no_overwrite_and_tamper(tmp_path):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    args = ["--dataset", "cifar10", "--root", str(target)]
    first = prep.main([*args, "--source", str(source)])
    assert first["metadata"]["train_rows"] == 40
    assert prep.main([*args, "--validate"]) == first
    with pytest.raises(FileExistsError):
        prep.main([*args, "--source", str(source)])
    with (target / "cifar10.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        prep.main([*args, "--validate"])


@pytest.mark.parametrize("shape,classes", [((28,28,3),10),((32,32,3),11)])
def test_reject_wrong_shape_or_out_of_range_labels(tmp_path, shape, classes):
    source = tmp_path / "invalid.npz"
    fixture(source, shape, classes)
    with pytest.raises(ValueError):
        prep.main(["--dataset","cifar10","--source",str(source),"--root",str(tmp_path/"target")])
    assert not (tmp_path/"target/cifar10.npz").exists()


def test_prepared_arrays_reach_training_loader(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    fixture(source)
    root = tmp_path / "data"
    prep.main(["--dataset","cifar10","--source",str(source),"--root",str(root)])
    monkeypatch.setenv("TORCH_DATA_DIR", str(root))
    from datasets.cifar import get_cifar10_dataset
    ds, classes, count = get_cifar10_dataset(return_eval_ds=True, batch_size=4)
    x, y = next(iter(ds))
    assert classes == 10 and count == 10 and tuple(x.shape[1:]) == (3,32,32)


def test_public_archive_extraction_blocks_traversal(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as out:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        out.addfile(info, io.BytesIO(b"x"))
    with tarfile.open(archive) as source:
        with pytest.raises(tarfile.FilterError):
            source.extractall(tmp_path / "safe")
    assert not (tmp_path / "escaped").exists()


def test_speechcommands_fixture_validates_real_wave_headers_and_splits(tmp_path):
    import wave
    root = tmp_path / "audio"
    directory = root / "spcm/speech_commands_v0.02/yes"
    directory.mkdir(parents=True)
    for i in range(3):
        with wave.open(str(directory/f"fixture_nohash_{i}.wav"), "wb") as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
            output.writeframes(np.zeros(16000, np.int16).tobytes())
    (directory.parent/"validation_list.txt").write_text("yes/fixture_nohash_1.wav\n")
    (directory.parent/"testing_list.txt").write_text("yes/fixture_nohash_2.wav\n")
    args = ["--dataset", "spcm", "--root", str(root)]
    record = prep.main(args)
    assert record["metadata"]["splits"] == {"training":1,"validation":1,"testing":1}
    prep.main([*args, "--validate"])
    with wave.open(str(directory/"fixture_nohash_2.wav"), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(8000)
        output.writeframes(np.zeros(8000, np.int16).tobytes())
    with pytest.raises(ValueError, match="16 kHz"):
        prep.main([*args, "--validate"])
