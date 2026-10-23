"""DP-SGD training of the target model with Opacus.

Run on its own:  python -m src.train_private --config configs/smoke.json --epsilon 8 --seed 0
"""
import argparse
import time

import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch import nn

from src.data_loader import load_experiment_data, make_loader
from src.evaluation import evaluate_model
from src.model import SmallCNN
from src.utils import get_device, load_config, save_checkpoint, set_seed


def train_dp(model, train_loader, val_data, cfg: dict, target_epsilon: float, device, log=print):
    """
    Opacus replaces the optimizer step with DP-SGD: each example's gradient is clipped to
    `max_grad_norm`, the clipped gradients are summed, Gaussian noise with standard deviation
    noise_multiplier * max_grad_norm is added, and the result is averaged. Batches are formed by
    Poisson sampling, which is the sampling assumption used by the RDP accountant.
    The noise multiplier is chosen by Opacus so that the accountant reaches `target_epsilon`
    after `epochs` epochs; the epsilon we report is read back from the accountant after training.
    """
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise ValueError(f"model is not compatible with DP-SGD: {errors}")

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=cfg["delta"],
        epochs=cfg["epochs"],
        max_grad_norm=cfg["max_grad_norm"],
        poisson_sampling=True,
    )
    loss_fn = nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total = 0.0, 0
        # The memory manager splits each logical (Poisson-sampled) batch into smaller physical
        # batches to fit in GPU memory; the optimizer still steps once per logical batch.
        with BatchMemoryManager(data_loader=train_loader,
                                max_physical_batch_size=cfg["physical_batch_size"],
                                optimizer=optimizer) as memory_safe_loader:
            for x, y in memory_safe_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
                total += len(y)
        epsilon = privacy_engine.get_epsilon(cfg["delta"])
        val = evaluate_model(model, *val_data, device)
        history.append({"epoch": epoch, "train_loss": total_loss / max(total, 1),
                        "val_accuracy": val["accuracy"], "epsilon": epsilon})
        log(f"[dp eps={target_epsilon}] epoch {epoch:3d}  train_loss {total_loss / max(total, 1):.4f}  "
            f"val_acc {val['accuracy']:.4f}  epsilon {epsilon:.3f}")

    privacy = {
        "target_epsilon": target_epsilon,
        "achieved_epsilon": privacy_engine.get_epsilon(cfg["delta"]),
        "delta": cfg["delta"],
        "noise_multiplier": optimizer.noise_multiplier,
        "max_grad_norm": cfg["max_grad_norm"],
        "sample_rate": train_loader.sample_rate,
        # Configured loader batch size. Opacus sets sample_rate = 1 / number of batches, so the
        # actual expected batch is sample_rate * n_members (1,000 for 10,000 members and 1,024 here).
        "expected_batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "steps": sum(num_steps for _, _, num_steps in privacy_engine.accountant.history),
        "accountant": "rdp",
        "sampling": "poisson",
    }
    # model is an Opacus GradSampleModule; _module is the plain SmallCNN with the trained weights.
    return model._module, {"history": history, "train_time_sec": time.perf_counter() - start, "privacy": privacy}


def run_private(cfg: dict, seed: int, data: dict, target_epsilon: float, device, log=print):
    set_seed(seed)
    model = SmallCNN().to(device)
    loader = make_loader(data["members"][0], data["members"][1], cfg["dp"]["batch_size"], shuffle=True)
    return train_dp(model, loader, data["validation"], cfg["dp"], target_epsilon, device, log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    data = load_experiment_data(cfg, args.seed)
    model, result = run_private(cfg, args.seed, data, args.epsilon, device)
    test = evaluate_model(model, *data["test"], device)
    path = save_checkpoint(model, cfg["run_name"], f"dp_eps{args.epsilon:g}_seed{args.seed}")
    p = result["privacy"]
    print(f"test accuracy {test['accuracy']:.4f}  epsilon {p['achieved_epsilon']:.3f} (delta {p['delta']})  "
          f"noise multiplier {p['noise_multiplier']:.3f}  saved {path}")
