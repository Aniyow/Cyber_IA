"""Standard (non-private) training of the target model.

Run on its own:  python -m src.train_baseline --config configs/smoke.json --seed 0
"""
import argparse
import time

import torch
from torch import nn

from src.data_loader import load_experiment_data, make_loader
from src.evaluation import evaluate_model
from src.model import SmallCNN
from src.utils import get_device, load_config, save_checkpoint, set_seed


def train_standard(model, train_loader, val_data, cfg: dict, device, log=print) -> dict:
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                weight_decay=cfg.get("weight_decay", 0.0))
    loss_fn = nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total += len(y)
        val = evaluate_model(model, *val_data, device)
        history.append({"epoch": epoch, "train_loss": total_loss / total, "val_accuracy": val["accuracy"]})
        log(f"[baseline] epoch {epoch:3d}  train_loss {total_loss / total:.4f}  val_acc {val['accuracy']:.4f}")
    return {"history": history, "train_time_sec": time.perf_counter() - start}


def run_baseline(cfg: dict, seed: int, data: dict, device, log=print):
    set_seed(seed)
    model = SmallCNN().to(device)
    loader = make_loader(data["members"][0], data["members"][1], cfg["baseline"]["batch_size"], shuffle=True)
    result = train_standard(model, loader, data["validation"], cfg["baseline"], device, log)
    return model, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    data = load_experiment_data(cfg, args.seed)
    model, result = run_baseline(cfg, args.seed, data, device)
    test = evaluate_model(model, *data["test"], device)
    path = save_checkpoint(model, cfg["run_name"], f"baseline_seed{args.seed}")
    print(f"test accuracy {test['accuracy']:.4f}  train time {result['train_time_sec']:.1f}s  saved {path}")
