import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_codebook_and_calibration_rounds_train_over_stock_flower(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--dataset",
            "synthetic",
            "--clients",
            "2",
            "--rounds",
            "3",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--clusters",
            "4",
            "--start-round",
            "1",
            "--broadcast-rate",
            "3",
            "--request-rate",
            "2",
            "--output",
            str(tmp_path / "run"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "run/results.json").read_text())
    assert all(
        r["parameters_changed"] and r["examples"] == 24 for r in report["rounds"]
    )
    modes = {
        (t["downlink"], t["uplink"]) for r in report["rounds"] for t in r["transfers"]
    }
    assert modes == {("calibration", "codebook"), ("codebook", "calibration")}
    assert not list((tmp_path / "run").glob("clients-*"))
    assert (tmp_path / "run/model-state.pt").is_file()
