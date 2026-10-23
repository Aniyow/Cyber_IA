"""Main study (Modules 1 and 2): standard vs DP-SGD target models and the membership inference attack.

    python -m src.run_experiment --config configs/smoke.json
    python -m src.run_experiment --config configs/full.json
"""
import argparse
import csv
import hashlib
import time

import numpy as np

from src.data_loader import load_experiment_data
from src.evaluation import evaluate_model
from src.membership_inference import attack_model
from src.train_baseline import run_baseline
from src.train_private import run_private
from src.utils import environment_info, get_device, load_config, output_dirs, save_checkpoint, save_json


def split_summary(data: dict) -> dict:
    idx = data["indices"]
    members, non_members, val = set(idx["members"]), set(idx["non_members"]), set(idx["validation"])
    summary = {
        "sizes": {name: int(len(v)) for name, v in idx.items()},
        "members_non_members_disjoint": members.isdisjoint(non_members),
        "validation_disjoint": val.isdisjoint(members | non_members),
        "test_source": "official CIFAR-10 test batch (never used for training or attack fitting)",
        "class_counts": {name: np.bincount(data[name][1], minlength=10).tolist()
                         for name in ["members", "non_members", "validation", "test"]},
        # Hashes of the index lists let anyone check they reproduce the same split.
        "index_sha256": {name: hashlib.sha256(np.asarray(v, dtype=np.int64).tobytes()).hexdigest()
                         for name, v in idx.items()},
    }
    return summary


def result_row(model_name, seed, test, train_metrics, attack, train_time, privacy=None) -> dict:
    row = {
        "model": model_name,
        "seed": seed,
        "test_accuracy": test["accuracy"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "train_accuracy": train_metrics["accuracy"],
        "generalization_gap": train_metrics["accuracy"] - test["accuracy"],
        "mia_loss_auc": attack["loss"]["auc"],
        "mia_loss_advantage": attack["loss"]["advantage"],
        "mia_loss_precision": attack["loss"]["precision"],
        "mia_loss_recall": attack["loss"]["recall"],
        "mia_loss_f1": attack["loss"]["f1"],
        "mia_loss_tpr": attack["loss"]["tpr"],
        "mia_loss_fpr": attack["loss"]["fpr"],
        "mia_loss_tpr_at_1pct_fpr": attack["loss"]["tpr_at_1pct_fpr"],
        "mia_conf_auc": attack["confidence"]["auc"],
        "mia_conf_advantage": attack["confidence"]["advantage"],
        "random_guess_auc": attack["random_guess"]["auc"],
        "random_guess_advantage": attack["random_guess"]["advantage"],
        "label_only_auc": attack["label_only"]["auc"],
        "train_time_sec": train_time,
        "target_epsilon": "",
        "achieved_epsilon": "",
        "delta": "",
        "noise_multiplier": "",
        "max_grad_norm": "",
    }
    if privacy:
        for key in ["target_epsilon", "achieved_epsilon", "delta", "noise_multiplier", "max_grad_norm"]:
            row[key] = privacy[key]
    return row


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    run = cfg["run_name"]
    dirs = output_dirs(run)
    device = get_device()
    log_file = open(dirs["logs"] / "main_experiment.log", "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    save_json({"config": cfg, "config_file": config_path, "device": str(device),
               "environment": environment_info(),
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}, dirs["metrics"] / "run_info.json")

    rows, details = [], {}
    for seed in cfg["seeds"]:
        data = load_experiment_data(cfg, seed)
        save_json(split_summary(data), dirs["metrics"] / f"splits_seed{seed}.json")

        model, result = run_baseline(cfg, seed, data, device, log)
        save_checkpoint(model, run, f"baseline_seed{seed}")
        test = evaluate_model(model, *data["test"], device)
        train_metrics = evaluate_model(model, *data["members"], device)
        attack = attack_model(model, data, seed, device)
        rows.append(result_row("standard", seed, test, train_metrics, attack, result["train_time_sec"]))
        details[f"standard_seed{seed}"] = {"history": result["history"], "attack": attack}
        log(f"seed {seed} standard: test_acc {test['accuracy']:.4f}  MIA AUC {attack['loss']['auc']:.4f}")

        for eps in cfg["dp"]["target_epsilons"]:
            model, result = run_private(cfg, seed, data, eps, device, log)
            save_checkpoint(model, run, f"dp_eps{eps:g}_seed{seed}")
            test = evaluate_model(model, *data["test"], device)
            train_metrics = evaluate_model(model, *data["members"], device)
            attack = attack_model(model, data, seed, device)
            privacy = result["privacy"]
            rows.append(result_row(f"dp_eps{eps:g}", seed, test, train_metrics, attack,
                                   result["train_time_sec"], privacy))
            details[f"dp_eps{eps:g}_seed{seed}"] = {"history": result["history"], "attack": attack,
                                                     "privacy": privacy}
            log(f"seed {seed} dp eps={eps}: test_acc {test['accuracy']:.4f}  MIA AUC {attack['loss']['auc']:.4f}  "
                f"achieved eps {privacy['achieved_epsilon']:.3f}  sigma {privacy['noise_multiplier']:.3f}")

        # Save after every seed so a crash does not lose finished results.
        with open(dirs["metrics"] / "main_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        save_json(details, dirs["metrics"] / "main_details.json")

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
