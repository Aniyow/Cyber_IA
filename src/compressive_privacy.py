"""Module 5: compressive privacy via PCA feature reduction on CIFAR-10.

Idea (Kung, 2017): release a low-dimensional projection of the data instead of the raw input,
keeping what the task needs and discarding detail. We compare an MLP trained on the raw pixels
(3072 values) with the same MLP trained on d PCA components, and measure:
  - task utility: test accuracy / F1
  - how much of the raw image can be recovered from the d components (reconstruction PSNR)
  - the same loss-threshold membership inference attack as in Module 1
PCA is fitted on the training members only, never on test data.

    python -m src.compressive_privacy --config configs/smoke.json
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from src.data_loader import load_experiment_data, make_loader
from src.evaluation import classification_metrics, predict
from src.membership_inference import run_attack
from src.model import MLP
from src.utils import get_device, load_config, output_dirs, save_json, set_seed


def flatten(images: torch.Tensor) -> np.ndarray:
    return images.reshape(len(images), -1).numpy()


def fit_pca(train_features: np.ndarray, n_components: int, seed: int) -> PCA:
    return PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(train_features)


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    # Pixels are in [-1, 1], so the peak-to-peak range is 2.
    mse = np.mean((original - reconstructed) ** 2)
    return float(10 * np.log10(4.0 / mse))


def train_mlp(features: np.ndarray, labels: np.ndarray, cfg: dict, device) -> tuple:
    model = MLP(features.shape[1], hidden=cfg["hidden"]).to(device)
    loader = make_loader(torch.from_numpy(features).float(), labels, cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    start = time.perf_counter()
    for _ in range(cfg["epochs"]):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
    return model, time.perf_counter() - start


def run(cfg: dict) -> None:
    cp = cfg["compressive"]
    seed = cp["seed"]
    dirs = output_dirs(cfg["run_name"])
    device = get_device()
    data = load_experiment_data(cfg, seed)

    raw = {name: flatten(data[name][0]) for name in ["members", "non_members", "test"]}
    labels = {name: data[name][1] for name in ["members", "non_members", "test"]}

    rows, reconstructions = [], {}
    for dims in [None] + cp["pca_components"]:
        if dims is None:
            feats = raw
            name, recon_psnr, explained = "raw_pixels", None, 1.0
            n_features = raw["members"].shape[1]
        else:
            pca = fit_pca(raw["members"], dims, seed)
            feats = {k: pca.transform(v).astype(np.float32) for k, v in raw.items()}
            recon = pca.inverse_transform(feats["test"])
            recon_psnr = psnr(raw["test"], recon)
            explained = float(pca.explained_variance_ratio_.sum())
            reconstructions[dims] = recon[:6]
            name, n_features = f"pca_{dims}", dims

        set_seed(seed)
        model, train_time = train_mlp(feats["members"], labels["members"], cp, device)
        outputs = {k: predict(model, torch.from_numpy(v).float(), labels[k], device) for k, v in feats.items()}
        test = classification_metrics(labels["test"], outputs["test"]["pred"])
        train_acc = classification_metrics(labels["members"], outputs["members"]["pred"])["accuracy"]
        attack = run_attack(outputs["members"], outputs["non_members"], labels["members"],
                            labels["non_members"], seed)
        rows.append({
            "representation": name,
            "n_features": n_features,
            "explained_variance": explained,
            "test_psnr_db": recon_psnr if recon_psnr is not None else "",
            "test_accuracy": test["accuracy"],
            "test_f1": test["f1"],
            "train_accuracy": train_acc,
            "mia_loss_auc": attack["loss"]["auc"],
            "mia_loss_advantage": attack["loss"]["advantage"],
            "label_only_auc": attack["label_only"]["auc"],
            "train_time_sec": train_time,
        })
        print(f"{name:11s} features {n_features:5d}  test_acc {test['accuracy']:.4f}  "
              f"train_acc {train_acc:.4f}  MIA AUC {attack['loss']['auc']:.4f}")

    with open(dirs["metrics"] / "compressive_privacy.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json({"config": cp, "n_members": cfg["n_members"], "n_test": cfg["n_test"], "results": rows},
              dirs["metrics"] / "compressive_privacy.json")

    # Original test images next to their PCA reconstructions.
    dims_list = cp["pca_components"]
    fig, axes = plt.subplots(len(dims_list) + 1, 6, figsize=(9, 1.6 * (len(dims_list) + 1)))
    rows_images = [raw["test"][:6]] + [reconstructions[d] for d in dims_list]
    titles = ["original"] + [f"PCA {d}" for d in dims_list]
    for r, (imgs, title) in enumerate(zip(rows_images, titles)):
        for c in range(6):
            img = imgs[c].reshape(3, 32, 32).transpose(1, 2, 0)
            axes[r, c].imshow(np.clip((img + 1) / 2, 0, 1))
            axes[r, c].axis("off")
        axes[r, 0].set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "pca_reconstructions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))
