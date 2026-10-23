import json
import platform
import random
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def output_dirs(run_name: str) -> dict:
    # Each run name ("smoke", "full") gets its own folders so runs never overwrite each other.
    dirs = {
        "metrics": RESULTS_DIR / "metrics" / run_name,
        "plots": RESULTS_DIR / "plots" / run_name,
        "logs": RESULTS_DIR / "logs" / run_name,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def environment_info() -> dict:
    packages = {}
    for name in ["torch", "torchvision", "opacus", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "packages": packages,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def checkpoint_path(run_name: str, name: str) -> Path:
    return RESULTS_DIR / "checkpoints" / run_name / f"{name}.pt"


def save_checkpoint(model, run_name: str, name: str) -> Path:
    path = checkpoint_path(run_name, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(model, run_name: str, name: str):
    model.load_state_dict(torch.load(checkpoint_path(run_name, name), map_location="cpu"))
    return model
