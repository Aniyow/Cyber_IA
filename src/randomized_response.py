"""Module 3: local differential privacy with randomized response (Warner, 1965).

Each respondent answers a sensitive yes/no question. Before anything leaves their device they
report their true answer with probability p = e^eps / (1 + e^eps) and the opposite answer otherwise.
The ratio P(report | true=yes) / P(report | true=no) is at most e^eps, so every individual report
satisfies eps-local differential privacy. The collector never sees true answers but can still
estimate the population proportion because p is public.

    python -m src.randomized_response --config configs/smoke.json
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_config, output_dirs, save_json


def truth_probability(epsilon: float) -> float:
    return float(np.exp(epsilon) / (1 + np.exp(epsilon)))


def randomize(true_answers: np.ndarray, epsilon: float, rng) -> np.ndarray:
    keep = rng.random(len(true_answers)) < truth_probability(epsilon)
    return np.where(keep, true_answers, 1 - true_answers)


def estimate_proportion(reports: np.ndarray, epsilon: float) -> float:
    # E[observed yes rate] = p * pi + (1 - p) * (1 - pi)  =>  pi = (observed - (1 - p)) / (2p - 1)
    # This estimator is unbiased; it can fall outside [0, 1] for small samples, which we keep
    # so the reported error is honest (clipping would add bias).
    p = truth_probability(epsilon)
    return float((reports.mean() - (1 - p)) / (2 * p - 1))


def theoretical_std(epsilon: float, n: int) -> float:
    # The true answers are fixed, so the only randomness is each respondent's coin flip:
    # every report differs from the truth with probability 1 - p, so Var(observed rate) = p(1 - p) / n.
    p = truth_probability(epsilon)
    return float(np.sqrt(p * (1 - p) / n) / (2 * p - 1))


def run(cfg: dict) -> None:
    rr = cfg["randomized_response"]
    dirs = output_dirs(cfg["run_name"])
    rng = np.random.default_rng(rr["seed"])

    # Synthetic survey: 1 = "yes" to the sensitive question.
    n = rr["n_respondents"]
    true_answers = (rng.random(n) < rr["true_proportion"]).astype(int)
    actual = float(true_answers.mean())

    rows, examples = [], {"original": true_answers[:15].tolist()}
    for eps in rr["epsilons"]:
        estimates, flip_rates = [], []
        for _ in range(rr["trials"]):
            reports = randomize(true_answers, eps, rng)
            estimates.append(estimate_proportion(reports, eps))
            flip_rates.append(float(np.mean(reports != true_answers)))
        estimates = np.array(estimates)
        examples[f"eps_{eps:g}"] = randomize(true_answers[:15], eps, np.random.default_rng(0)).tolist()
        rows.append({
            "epsilon": eps,
            "truth_probability": truth_probability(eps),
            "n": n,
            "actual_proportion": actual,
            "mean_estimate": float(estimates.mean()),
            "mean_abs_error": float(np.mean(np.abs(estimates - actual))),
            "rmse": float(np.sqrt(np.mean((estimates - actual) ** 2))),
            "empirical_std": float(estimates.std()),
            "theoretical_std": theoretical_std(eps, n),
            "mean_flip_rate": float(np.mean(flip_rates)),
        })

    # Effect of the number of respondents at one fixed epsilon.
    size_rows = []
    for size in rr["sample_sizes"]:
        sample = (rng.random(size) < rr["true_proportion"]).astype(int)
        errors = [abs(estimate_proportion(randomize(sample, rr["size_epsilon"], rng), rr["size_epsilon"])
                      - sample.mean()) for _ in range(rr["trials"])]
        size_rows.append({"n": size, "epsilon": rr["size_epsilon"], "mean_abs_error": float(np.mean(errors))})

    with open(dirs["metrics"] / "randomized_response.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json({"config": rr, "results": rows, "sample_size_results": size_rows,
               "first_15_responses": examples}, dirs["metrics"] / "randomized_response.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    eps_values = [r["epsilon"] for r in rows]
    axes[0].plot(eps_values, [r["mean_abs_error"] for r in rows], "o-", label="empirical MAE")
    axes[0].plot(eps_values, [r["theoretical_std"] * np.sqrt(2 / np.pi) for r in rows], "--",
                 label="theoretical MAE")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("local privacy parameter epsilon")
    axes[0].set_ylabel("mean absolute error of estimated proportion")
    axes[0].set_title(f"Randomized response (n = {n})")
    axes[0].legend()
    axes[1].plot([r["n"] for r in size_rows], [r["mean_abs_error"] for r in size_rows], "o-")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("number of respondents")
    axes[1].set_ylabel("mean absolute error")
    axes[1].set_title(f"Effect of sample size (epsilon = {rr['size_epsilon']})")
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "randomized_response_error.png", dpi=150)
    plt.close(fig)

    for r in rows:
        print(f"eps {r['epsilon']:5.2f}  p_truth {r['truth_probability']:.3f}  actual {r['actual_proportion']:.4f}  "
              f"mean estimate {r['mean_estimate']:.4f}  MAE {r['mean_abs_error']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))
