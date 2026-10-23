"""Records verifiable facts about the downloaded CIFAR-10 files (sizes, hashes, counts).

    python -m src.provenance
"""
import hashlib
import time

import numpy as np

from src.data_loader import DATA_DIR, load_cifar10
from src.utils import RESULTS_DIR, save_json

# MD5 published in torchvision's CIFAR10 class for the official archive.
TORCHVISION_TGZ_MD5 = "c58f30108f718f92721af3b95e74349a"


def file_hash(path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    archive = DATA_DIR / "cifar-10-python.tar.gz"
    batches = sorted((DATA_DIR / "cifar-10-batches-py").glob("*"))
    train_x, train_y, test_x, test_y = load_cifar10(download=False)
    info = {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archive": {
            "file": archive.name,
            "bytes": archive.stat().st_size if archive.exists() else None,
            "md5": file_hash(archive, "md5") if archive.exists() else None,
            "sha256": file_hash(archive, "sha256") if archive.exists() else None,
            "archive_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(archive.stat().st_mtime))
            if archive.exists() else None,
        },
        "md5_matches_torchvision": archive.exists() and file_hash(archive, "md5") == TORCHVISION_TGZ_MD5,
        "extracted_files": {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p, "sha256")}
                            for p in batches if p.is_file()},
        "train_shape": list(train_x.shape),
        "test_shape": list(test_x.shape),
        "train_class_counts": np.bincount(train_y).tolist(),
        "test_class_counts": np.bincount(test_y).tolist(),
        "dtype": str(train_x.dtype),
    }
    save_json(info, RESULTS_DIR / "metrics" / "cifar10_provenance.json")
    print(f"archive md5 matches torchvision: {info['md5_matches_torchvision']}  "
          f"train {info['train_shape']}  test {info['test_shape']}")


if __name__ == "__main__":
    main()
