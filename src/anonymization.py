"""Module 4: anonymization, k-anonymity and data perturbation on a small synthetic patient table.

Quasi-identifiers (QIs): age, zipcode, sex. Sensitive attribute: diagnosis.
Anonymization uses full-domain generalization (every record's QI is generalized to the same level)
plus suppression of the few records that still sit in classes smaller than k (Samarati, 2001;
Sweeney, 2002). k-anonymity is a syntactic property of the released table, not a differential
privacy guarantee.

    python -m src.anonymization --config configs/smoke.json
"""
import argparse
import csv
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.utils import load_config, output_dirs, save_json

QUASI_IDENTIFIERS = ["age", "zipcode", "sex"]
SENSITIVE = "diagnosis"
MAX_LEVEL = {"age": 4, "zipcode": 5, "sex": 1}
DIAGNOSES = ["Flu", "Asthma", "Diabetes", "Hypertension", "Heart disease"]


def make_synthetic_patients(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 86, size=n)
    region = rng.choice(["130", "131", "148", "149", "152"], size=n)
    zipcode = [r + f"{rng.integers(0, 100):02d}" for r in region]
    sex = rng.choice(["F", "M"], size=n)
    # Diagnosis depends on age so that the QIs carry some useful signal for the utility test.
    old = (age - 18) / 67
    weights = np.stack([1.2 - old, np.full(n, 0.6), 0.3 + 0.6 * old, 0.1 + 1.0 * old, 0.05 + 0.8 * old], axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    diagnosis = [DIAGNOSES[rng.choice(5, p=w)] for w in weights]
    income = np.round(20000 + 900 * np.minimum(age, 60) + rng.normal(0, 8000, n), -2)
    return pd.DataFrame({"age": age, "zipcode": zipcode, "sex": sex,
                         "diagnosis": diagnosis, "annual_income": income})


def generalize_age(age: pd.Series, level: int) -> pd.Series:
    if level == 0:
        return age.astype(str)
    if level == 4:
        return pd.Series("*", index=age.index)
    width = {1: 5, 2: 10, 3: 20}[level]
    low = (age // width) * width
    return low.astype(str) + "-" + (low + width - 1).astype(str)


def generalize_zip(zipcode: pd.Series, level: int) -> pd.Series:
    # Level L replaces the last L digits with '*'.
    return zipcode.str[:5 - level] + "*" * level


def generalize(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    out = df.copy()
    out["age"] = generalize_age(df["age"], levels["age"])
    out["zipcode"] = generalize_zip(df["zipcode"], levels["zipcode"])
    out["sex"] = df["sex"] if levels["sex"] == 0 else "*"
    return out


def equivalence_class_sizes(df: pd.DataFrame) -> pd.Series:
    """Size of the equivalence class (records sharing all QI values) that each record belongs to."""
    return df.groupby(QUASI_IDENTIFIERS)[QUASI_IDENTIFIERS[0]].transform("size")


def verify_k_anonymity(df: pd.DataFrame, k: int) -> dict:
    sizes = df.groupby(QUASI_IDENTIFIERS).size()
    return {"k": k, "satisfied": bool(len(sizes) == 0 or sizes.min() >= k),
            "num_classes": int(len(sizes)),
            "min_class_size": int(sizes.min()) if len(sizes) else 0,
            "mean_class_size": float(sizes.mean()) if len(sizes) else 0.0,
            "max_class_size": int(sizes.max()) if len(sizes) else 0}


def information_loss(levels: dict) -> float:
    # Average fraction of the generalization hierarchy climbed (0 = original, 1 = fully suppressed QIs).
    return float(np.mean([levels[q] / MAX_LEVEL[q] for q in QUASI_IDENTIFIERS]))


def anonymize(df: pd.DataFrame, k: int, max_suppression_rate: float) -> tuple:
    """
    Tries every combination of generalization levels and keeps the one with the lowest information
    loss for which suppressing the records in classes smaller than k removes at most
    max_suppression_rate of the table. Returns (anonymized table, chosen levels, suppressed count).
    """
    best = None
    for a, z, s in itertools.product(range(5), range(6), range(2)):
        levels = {"age": a, "zipcode": z, "sex": s}
        generalized = generalize(df, levels)
        too_small = equivalence_class_sizes(generalized) < k
        if too_small.sum() > max_suppression_rate * len(df):
            continue
        key = (information_loss(levels), int(too_small.sum()))
        if best is None or key < best[0]:
            best = (key, levels, generalized[~too_small], int(too_small.sum()))
    if best is None:
        raise ValueError(f"no generalization satisfies k={k}")
    return best[2], best[1], best[3]


def l_diversity(df: pd.DataFrame) -> dict:
    distinct = df.groupby(QUASI_IDENTIFIERS)[SENSITIVE].nunique()
    return {"min_distinct_diagnoses": int(distinct.min()),
            "share_of_homogeneous_classes": float((distinct == 1).mean())}


def age_to_number(value: str) -> float:
    # Generalized ages become the midpoint of their range; a fully suppressed age gets the middle of 18-85.
    if value == "*":
        return 51.5
    if "-" in value:
        low, high = value.split("-")
        return (int(low) + int(high)) / 2
    return float(value)


def utility_accuracy(df: pd.DataFrame, seed: int) -> float:
    """How well the released QIs still predict diagnosis. Age is numeric, zipcode and sex are categories."""
    X = pd.get_dummies(df[["zipcode", "sex"]].astype(str))
    X["age"] = df["age"].astype(str).map(age_to_number) / 85
    y = df[SENSITIVE]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def perturb(df: pd.DataFrame, scale: float, rng) -> pd.DataFrame:
    """Additive Laplace noise on age and income. `scale` is a fraction of each column's std."""
    out = df.copy()
    for col in ["age", "annual_income"]:
        out[col] = df[col] + rng.laplace(0, scale * df[col].std(), len(df))
    return out


def linkage_rate(original: pd.DataFrame, perturbed: pd.DataFrame) -> float:
    """Share of perturbed records whose nearest original record (on age, income) is the true one."""
    cols = ["age", "annual_income"]
    std = original[cols].std().values
    a = original[cols].values / std
    b = perturbed[cols].values / std
    dist = ((b[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)
    return float(np.mean(dist.argmin(axis=1) == np.arange(len(a))))


def run(cfg: dict) -> None:
    ac = cfg["anonymization"]
    dirs = output_dirs(cfg["run_name"])
    df = make_synthetic_patients(ac["n_records"], ac["seed"])
    df.to_csv(dirs["metrics"] / "synthetic_patients.csv", index=False)

    original_sizes = equivalence_class_sizes(df)
    baseline = {"unique_record_rate": float((original_sizes == 1).mean()),
                **verify_k_anonymity(df, 2), "utility_accuracy": utility_accuracy(df, ac["seed"])}

    k_rows = []
    for k in ac["k_values"]:
        anon, levels, suppressed = anonymize(df, k, ac["max_suppression_rate"])
        check = verify_k_anonymity(anon, k)
        k_rows.append({
            "k": k, "age_level": levels["age"], "zip_level": levels["zipcode"], "sex_level": levels["sex"],
            "suppressed_records": suppressed, "suppression_rate": suppressed / len(df),
            "information_loss": information_loss(levels), "verified": check["satisfied"],
            "num_classes": check["num_classes"], "min_class_size": check["min_class_size"],
            "mean_class_size": check["mean_class_size"], "max_class_size": check["max_class_size"],
            "unique_record_rate": float((equivalence_class_sizes(anon) == 1).mean()),
            **l_diversity(anon), "utility_accuracy": utility_accuracy(anon, ac["seed"]),
        })
        if k == ac["k_values"][0]:
            anon.head(10).to_csv(dirs["metrics"] / f"anonymized_sample_k{k}.csv", index=False)

    rng = np.random.default_rng(ac["seed"] + 1)
    p_rows = []
    for scale in ac["noise_scales"]:
        noisy = perturb(df, scale, rng) if scale > 0 else df.copy()
        p_rows.append({
            "noise_scale": scale,
            "age_mae": float(np.abs(noisy["age"] - df["age"]).mean()),
            "income_mae": float(np.abs(noisy["annual_income"] - df["annual_income"]).mean()),
            "mean_age_error": float(abs(noisy["age"].mean() - df["age"].mean())),
            "mean_income_error": float(abs(noisy["annual_income"].mean() - df["annual_income"].mean())),
            "age_income_corr": float(noisy["age"].corr(noisy["annual_income"])),
            "linkage_rate": linkage_rate(df, noisy),
        })

    for name, rows in [("k_anonymity", k_rows), ("perturbation", p_rows)]:
        with open(dirs["metrics"] / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    save_json({"config": ac, "original_table": baseline, "k_anonymity": k_rows, "perturbation": p_rows},
              dirs["metrics"] / "anonymization.json")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = [r["k"] for r in k_rows]
    axes[0].bar([str(k) for k in ks], [r["min_class_size"] for r in k_rows], label="smallest class")
    axes[0].plot([str(k) for k in ks], ks, "r_", markersize=30, label="required k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("records")
    axes[0].set_title("k-anonymity check")
    axes[0].legend()
    axes[1].plot(ks, [r["utility_accuracy"] for r in k_rows], "o-", label="anonymized")
    axes[1].axhline(baseline["utility_accuracy"], color="gray", linestyle="--", label="original")
    axes[1].plot(ks, [r["information_loss"] for r in k_rows], "s-", label="information loss")
    axes[1].set_xlabel("k")
    axes[1].set_title("Utility vs information loss")
    axes[1].legend()
    scales = [r["noise_scale"] for r in p_rows]
    axes[2].plot(scales, [r["linkage_rate"] for r in p_rows], "o-", label="linkage success rate")
    axes[2].plot(scales, [r["age_income_corr"] for r in p_rows], "s-", label="age-income correlation")
    axes[2].set_xlabel("Laplace noise scale (x column std)")
    axes[2].set_title("Data perturbation")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(dirs["plots"] / "anonymization_utility.png", dpi=150)
    plt.close(fig)

    print(f"original: unique QI combinations {baseline['unique_record_rate']:.3f}, "
          f"utility {baseline['utility_accuracy']:.3f}")
    for r in k_rows:
        print(f"k={r['k']:3d} levels(age,zip,sex)=({r['age_level']},{r['zip_level']},{r['sex_level']})  "
              f"suppressed {r['suppressed_records']}  min class {r['min_class_size']}  verified {r['verified']}  "
              f"utility {r['utility_accuracy']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(load_config(parser.parse_args().config))
