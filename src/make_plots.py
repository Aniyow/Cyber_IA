"""Builds the summary table and plots for the main study from the saved metrics files only.

    python -m src.make_plots --config configs/full.json
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import load_config, output_dirs

SUMMARY_COLUMNS = ["test_accuracy", "test_precision", "test_recall", "test_f1", "train_accuracy",
                   "generalization_gap", "mia_loss_auc", "mia_loss_advantage", "mia_loss_precision",
                   "mia_loss_recall", "mia_loss_f1", "mia_loss_tpr", "mia_loss_fpr", "mia_loss_tpr_at_1pct_fpr",
                   "mia_conf_auc", "random_guess_auc", "label_only_auc", "achieved_epsilon",
                   "noise_multiplier", "train_time_sec"]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    order = list(dict.fromkeys(results["model"]))  # keep the order the models were run in
    grouped = results.groupby("model")[SUMMARY_COLUMNS]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=0).add_suffix("_std"))
    summary["n_seeds"] = results.groupby("model").size()
    return summary.loc[order].reset_index()


def bar_with_error(ax, summary, column, ylabel, title, color):
    ax.bar(summary["model"], summary[f"{column}_mean"], yerr=summary[f"{column}_std"], capsize=4, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dirs = output_dirs(cfg["run_name"])
    results = pd.read_csv(dirs["metrics"] / "main_results.csv")
    results["achieved_epsilon"] = pd.to_numeric(results["achieved_epsilon"], errors="coerce")
    results["noise_multiplier"] = pd.to_numeric(results["noise_multiplier"], errors="coerce")
    summary = summarize(results)
    summary.to_csv(dirs["metrics"] / "main_summary.csv", index=False)
    plots = dirs["plots"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_with_error(ax, summary, "test_accuracy", "test accuracy", "Standard vs DP-SGD test accuracy", "tab:blue")
    fig.tight_layout()
    fig.savefig(plots / "accuracy_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_with_error(ax, summary, "mia_loss_auc", "attack ROC-AUC", "Loss-threshold membership inference", "tab:red")
    ax.axhline(0.5, color="black", linestyle="--", label="random guessing (0.5)")
    ax.set_ylim(0.4, max(0.75, summary["mia_loss_auc_mean"].max() + 0.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "mia_auc_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_with_error(ax, summary, "mia_loss_advantage", "TPR - FPR", "Membership advantage (attack-eval set)",
                   "tab:orange")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(plots / "mia_advantage_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_with_error(ax, summary, "train_time_sec", "seconds", "Training time per model", "tab:gray")
    fig.tight_layout()
    fig.savefig(plots / "training_time.png", dpi=150)
    plt.close(fig)

    # Trade-off plots: one point per DP model and seed; the standard model has no finite epsilon,
    # so it is drawn as a horizontal reference line.
    dp = results[results["model"] != "standard"]
    std_rows = results[results["model"] == "standard"]
    if len(dp):
        for column, ylabel, name in [("test_accuracy", "test accuracy", "accuracy_vs_epsilon.png"),
                                     ("mia_loss_auc", "attack ROC-AUC", "mia_auc_vs_epsilon.png")]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(dp["achieved_epsilon"], dp[column], alpha=0.5, label="DP-SGD (each seed)")
            means = dp.groupby("model")[["achieved_epsilon", column]].mean().sort_values("achieved_epsilon")
            ax.plot(means["achieved_epsilon"], means[column], "o-", label="DP-SGD (mean)")
            ax.axhline(std_rows[column].mean(), color="tab:green", linestyle="--", label="standard (no DP)")
            if column == "mia_loss_auc":
                ax.axhline(0.5, color="black", linestyle=":", label="random guessing")
            ax.set_xscale("log")
            ax.set_xlabel("achieved epsilon (delta = %g)" % cfg["dp"]["delta"])
            ax.set_ylabel(ylabel)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots / name, dpi=150)
            plt.close(fig)

    # ROC curves for the first seed.
    with open(dirs["metrics"] / "main_details.json", encoding="utf-8") as f:
        details = json.load(f)
    seed = cfg["seeds"][0]
    fig, ax = plt.subplots(figsize=(5, 5))
    for key, value in details.items():
        if key.endswith(f"_seed{seed}"):
            roc = value["attack"]["roc_loss"]
            ax.plot(roc["fpr"], roc["tpr"], label=f"{key.replace(f'_seed{seed}', '')} "
                                                  f"(AUC {value['attack']['loss']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="random guessing")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"Loss attack ROC, attack-eval set (seed {seed})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "mia_roc_curves.png", dpi=150)
    plt.close(fig)

    # Training curves for the first seed.
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, value in details.items():
        if key.endswith(f"_seed{seed}"):
            hist = value["history"]
            ax.plot([h["epoch"] for h in hist], [h["val_accuracy"] for h in hist],
                    label=key.replace(f"_seed{seed}", ""))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "validation_curves.png", dpi=150)
    plt.close(fig)

    cols = ["model", "n_seeds", "test_accuracy_mean", "test_f1_mean", "mia_loss_auc_mean",
            "mia_loss_advantage_mean", "achieved_epsilon_mean", "train_time_sec_mean"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
