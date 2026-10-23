"""CIFAR-10 loading and the fixed data splits used by the main experiment."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from src.utils import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def load_cifar10(download: bool = True):
    """Returns (train_images, train_labels, test_images, test_labels) as uint8 / int64 numpy arrays."""
    train = CIFAR10(root=DATA_DIR, train=True, download=download)
    test = CIFAR10(root=DATA_DIR, train=False, download=download)
    return (train.data, np.array(train.targets, dtype=np.int64),
            test.data, np.array(test.targets, dtype=np.int64))


def preprocess(images: np.ndarray) -> torch.Tensor:
    # Fixed scaling to [-1, 1]. The constants do not depend on any data, so no statistics
    # are learned from the training members (which would otherwise also need privacy accounting).
    x = torch.from_numpy(images).float().div(255.0)
    x = (x - 0.5) / 0.5
    return x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW


def stratified_take(labels: np.ndarray, indices: np.ndarray, per_class: int, rng) -> tuple:
    """Takes `per_class` examples of every class from `indices`; returns (taken, remaining)."""
    taken = []
    for c in range(10):
        class_idx = indices[labels[indices] == c]
        if len(class_idx) < per_class:
            raise ValueError(f"class {c} has only {len(class_idx)} examples, need {per_class}")
        taken.append(rng.choice(class_idx, size=per_class, replace=False))
    taken = np.sort(np.concatenate(taken))
    remaining = np.setdiff1d(indices, taken)
    return taken, remaining


def make_splits(train_labels: np.ndarray, test_labels: np.ndarray, n_members: int,
                n_val: int, n_test: int, seed: int) -> dict:
    """
    Splits the official CIFAR-10 training set into disjoint, class-balanced groups:
      members     - used to train the target model
      non_members - never used for training; the attacker's "out" examples
      validation  - only used to monitor training
    The official test set (optionally a balanced subset) is used only to measure model utility.
    Members and non-members have identical class counts, so the class label alone carries
    no information about membership.
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(len(train_labels))
    members, pool = stratified_take(train_labels, pool, n_members // 10, rng)
    non_members, pool = stratified_take(train_labels, pool, n_members // 10, rng)
    validation, pool = stratified_take(train_labels, pool, n_val // 10, rng)

    test_pool = np.arange(len(test_labels))
    if n_test >= len(test_labels):
        test = test_pool
    else:
        test, _ = stratified_take(test_labels, test_pool, n_test // 10, rng)
    return {"members": members, "non_members": non_members, "validation": validation, "test": test}


def make_loader(images: torch.Tensor, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(images, torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_experiment_data(cfg: dict, seed: int) -> dict:
    """Loads CIFAR-10 and returns {split_name: (image_tensor, labels)} plus the raw indices."""
    train_x, train_y, test_x, test_y = load_cifar10()
    splits = make_splits(train_y, test_y, cfg["n_members"], cfg["n_val"], cfg["n_test"], seed)
    data = {"indices": splits}
    for name in ["members", "non_members", "validation"]:
        idx = splits[name]
        data[name] = (preprocess(train_x[idx]), train_y[idx])
    data["test"] = (preprocess(test_x[splits["test"]]), test_y[splits["test"]])
    return data
