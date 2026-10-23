"""Loss- and confidence-threshold membership inference attack (Yeom et al., 2018).

Threat model: the attacker can query the trained model for its output probabilities, knows the
true label of the record it is testing, and has a small labelled set of known members and known
non-members (attack-train) to choose a decision threshold. The final attack is then measured on
a separate attack-eval set it has never seen.

Run on a saved model:
    python -m src.membership_inference --config configs/smoke.json --checkpoint baseline_seed0
"""
import argparse

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

from src.data_loader import load_experiment_data
from src.evaluation import predict
from src.model import SmallCNN
from src.utils import get_device, load_checkpoint, load_config


def split_attack_sets(member_labels: np.ndarray, non_member_labels: np.ndarray, seed: int) -> dict:
    """
    Builds the attack data: label 1 = member, 0 = non-member. Half of each class within the members
    and within the non-members goes to attack-train, the other half to attack-eval, so both
    sets are balanced in membership and in image class.
    Returns boolean masks over the concatenation [members, non_members].
    """
    rng = np.random.default_rng(seed + 1000)
    class_labels = np.concatenate([member_labels, non_member_labels])
    is_member = np.concatenate([np.ones(len(member_labels), int), np.zeros(len(non_member_labels), int)])
    in_attack_train = np.zeros(len(is_member), dtype=bool)
    for membership in (0, 1):
        for c in np.unique(class_labels):
            idx = np.where((is_member == membership) & (class_labels == c))[0]
            chosen = rng.choice(idx, size=len(idx) // 2, replace=False)
            in_attack_train[chosen] = True
    return {"is_member": is_member, "class_labels": class_labels,
            "train_mask": in_attack_train, "eval_mask": ~in_attack_train}


def fit_threshold(scores: np.ndarray, is_member: np.ndarray) -> float:
    """Threshold with the best accuracy on attack-train. Predict 'member' when score >= threshold."""
    candidates = np.unique(scores)
    best_threshold, best_acc = candidates[0], -1.0
    for t in candidates:
        acc = np.mean((scores >= t).astype(int) == is_member)
        if acc > best_acc:
            best_threshold, best_acc = t, acc
    return float(best_threshold)


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    return float(tpr[fpr <= max_fpr].max()) if np.any(fpr <= max_fpr) else 0.0


def attack_metrics(scores: np.ndarray, is_member: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        is_member, predicted, average="binary", zero_division=0)
    tp = np.sum((predicted == 1) & (is_member == 1))
    fp = np.sum((predicted == 1) & (is_member == 0))
    tpr = tp / np.sum(is_member == 1)
    fpr = fp / np.sum(is_member == 0)
    roc_fpr, roc_tpr, _ = roc_curve(is_member, scores)
    return {
        "auc": float(roc_auc_score(is_member, scores)),
        "accuracy": float(np.mean(predicted == is_member)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "advantage": float(tpr - fpr),  # membership advantage, Yeom et al. (2018)
        "tpr_at_1pct_fpr": tpr_at_fpr(roc_fpr, roc_tpr, 0.01),
        "threshold": threshold,
    }


def run_attack(member_out: dict, non_member_out: dict, member_labels, non_member_labels, seed: int) -> dict:
    """member_out / non_member_out are outputs of evaluation.predict on the two groups."""
    sets = split_attack_sets(member_labels, non_member_labels, seed)
    is_member, tr, ev = sets["is_member"], sets["train_mask"], sets["eval_mask"]

    # Higher score = more likely a member. Members tend to have lower loss / higher confidence.
    scores = {
        "loss": -np.concatenate([member_out["loss"], non_member_out["loss"]]),
        "confidence": np.concatenate([member_out["confidence"], non_member_out["confidence"]]),
    }
    results = {}
    for name, s in scores.items():
        threshold = fit_threshold(s[tr], is_member[tr])
        results[name] = attack_metrics(s[ev], is_member[ev], threshold)

    # Baseline 1: random guessing (scores carry no information; expected AUC 0.5, advantage 0).
    rng = np.random.default_rng(seed + 2000)
    random_scores = rng.random(len(is_member))
    results["random_guess"] = attack_metrics(random_scores[ev], is_member[ev], 0.5)

    # Baseline 2: label only. Score each eval example by the member rate of its class in
    # attack-train. Because the splits are class-balanced this should stay at AUC ~0.5,
    # which checks that the attack is not just learning image classes.
    labels = sets["class_labels"]
    class_rate = {c: is_member[tr][labels[tr] == c].mean() for c in np.unique(labels)}
    label_scores = np.array([class_rate[c] for c in labels])
    results["label_only"] = {"auc": float(roc_auc_score(is_member[ev], label_scores[ev]))}

    roc_fpr, roc_tpr, _ = roc_curve(is_member[ev], scores["loss"][ev])
    results["roc_loss"] = {"fpr": roc_fpr.tolist(), "tpr": roc_tpr.tolist()}
    results["sizes"] = {"attack_train": int(tr.sum()), "attack_eval": int(ev.sum()),
                        "attack_eval_members": int(is_member[ev].sum())}
    return results


def attack_model(model, data: dict, seed: int, device) -> dict:
    member_out = predict(model, *data["members"], device)
    non_member_out = predict(model, *data["non_members"], device)
    result = run_attack(member_out, non_member_out, data["members"][1], data["non_members"][1], seed)
    result["mean_member_loss"] = float(member_out["loss"].mean())
    result["mean_non_member_loss"] = float(non_member_out["loss"].mean())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="e.g. baseline_seed0 or dp_eps8_seed0")
    parser.add_argument("--seed", type=int, default=0, help="must match the seed the model was trained with")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    data = load_experiment_data(cfg, args.seed)
    model = load_checkpoint(SmallCNN(), cfg["run_name"], args.checkpoint).to(device)
    result = attack_model(model, data, args.seed, device)
    for name in ["loss", "confidence", "random_guess"]:
        r = result[name]
        print(f"{name:12s} AUC {r['auc']:.4f}  advantage {r['advantage']:.4f}  "
              f"precision {r['precision']:.4f}  recall {r['recall']:.4f}  F1 {r['f1']:.4f}")
    print(f"label_only   AUC {result['label_only']['auc']:.4f}")
