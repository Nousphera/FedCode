"""Explicit experiment parameters for the paper models and Flower apps."""

import math
from pathlib import Path

METHODS = ("fedcode", "fedavg", "clustered-fedavg", "centralized")
MODELS = ("resnet20", "mobilenetv2", "yamnet")
DATASETS = ("cifar10", "cifar100", "speech_commands")


def paper_defaults(model):
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    return {"num-rounds": 60 if model == "resnet20" else 100,
            "warmup-rounds": 2 if model == "resnet20" else 4,
            "downstream-period": 5 if model == "resnet20" else 3}


def defaults():
    return {
        "method": "fedcode", "dataset": "cifar10", "model": "resnet20", "device": "cpu",
        "num-clients": 10, "num-rounds": 60, "participation": 1., "dirichlet-alpha": 10.,
        "local-epochs": 4, "batch-size": 128, "learning-rate": .001,
        "num-clusters": 64, "warmup-rounds": 2, "downstream-period": 5, "upstream-period": 2,
        "kmeans-seed": 0, "seed": 42, "data-seed": 42,
        "data-root": "data", "output": "results/fedcode.json",
        "torch-threads": 1, "num-workers": 0, "registration-timeout": 120., "reply-timeout": 86400.,
    }


def validate(cfg):
    cfg = dict(cfg)
    if set(cfg) != set(defaults()):
        raise ValueError(f"Unexpected or missing configuration keys: {set(cfg) ^ set(defaults())}")
    for key, value in cfg.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    for key in ("participation", "dirichlet-alpha", "learning-rate", "registration-timeout", "reply-timeout"):
        if type(cfg[key]) not in (int, float):
            raise ValueError(f"{key} must be a real number")
    if cfg["method"] not in METHODS or cfg["dataset"] not in DATASETS or cfg["model"] not in MODELS:
        raise ValueError("Unknown method, dataset or model")
    if (cfg["dataset"] == "speech_commands") != (cfg["model"] == "yamnet"):
        raise ValueError("Use YAMNet for Speech Commands, ResNet20 or MobileNetV2 for CIFAR")
    for key in ("num-clients", "num-rounds", "local-epochs", "torch-threads", "num-clusters"):
        if type(cfg[key]) != int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["batch-size"]) != int or cfg["batch-size"] < 2:
        raise ValueError("batch-size must be at least two for BatchNorm training")
    for key in ("seed", "data-seed", "kmeans-seed", "num-workers", "warmup-rounds",
                "downstream-period", "upstream-period"):
        if type(cfg[key]) != int or cfg[key] < 0 or cfg[key] >= 2 ** 32:
            raise ValueError(f"{key} must be a nonnegative 32-bit integer")
    if not 0 < cfg["participation"] <= 1:
        raise ValueError("participation must be in (0, 1]")
    for key in ("dirichlet-alpha", "learning-rate", "registration-timeout", "reply-timeout"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["device"] not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be cpu, mps or cuda; Flower assigns each client's GPU")
    if Path(cfg["output"]).suffix != ".json":
        raise ValueError("output must end in .json; weights are saved alongside it as .pt")
    return cfg


def calibration_round(round_number, cfg, direction, completed=None):
    """Calibration before codebooks begin, periodically, or for a new receiver."""
    if direction not in {"upstream", "downstream"} or round_number < 1:
        raise ValueError("Invalid communication direction or round")
    period = cfg[f"{direction}-period"]
    return (cfg["method"] == "clustered-fedavg"
            or round_number <= cfg["warmup-rounds"]
            or (period > 0 and round_number % period == 0)
            or (direction == "downstream" and completed == 0))
