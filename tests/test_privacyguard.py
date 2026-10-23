import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import anonymization as anon
from src import randomized_response as rr
from src.compressive_privacy import fit_pca
from src.data_loader import DATA_DIR, make_loader, make_splits, preprocess
from src.membership_inference import attack_metrics, fit_threshold, run_attack, split_attack_sets
from src.model import SmallCNN
from src.train_private import train_dp

CIFAR_PRESENT = (DATA_DIR / "cifar-10-batches-py" / "data_batch_1").exists()


def fake_labels(n, seed=0):
    return np.random.default_rng(seed).integers(0, 10, size=n)


# ---- dataset and splits ----

def test_splits_are_disjoint_and_class_balanced():
    train_labels, test_labels = fake_labels(5000), fake_labels(1000, 1)
    s = make_splits(train_labels, test_labels, n_members=1000, n_val=500, n_test=500, seed=0)
    assert set(s["members"]).isdisjoint(s["non_members"])
    assert set(s["validation"]).isdisjoint(set(s["members"]) | set(s["non_members"]))
    assert len(s["members"]) == len(s["non_members"]) == 1000
    # Same class counts for members and non-members, so class alone says nothing about membership.
    assert np.array_equal(np.bincount(train_labels[s["members"]], minlength=10),
                          np.bincount(train_labels[s["non_members"]], minlength=10))


def test_splits_are_reproducible():
    labels = fake_labels(5000)
    a = make_splits(labels, labels[:1000], 1000, 500, 500, seed=3)
    b = make_splits(labels, labels[:1000], 1000, 500, 500, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_preprocess_shape_and_range():
    images = np.random.default_rng(0).integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
    x = preprocess(images)
    assert x.shape == (4, 3, 32, 32)
    assert x.min() >= -1 and x.max() <= 1


@pytest.mark.skipif(not CIFAR_PRESENT, reason="CIFAR-10 not downloaded yet")
def test_cifar10_loads_with_expected_sizes():
    from src.data_loader import load_cifar10
    train_x, train_y, test_x, test_y = load_cifar10(download=False)
    assert train_x.shape == (50000, 32, 32, 3) and test_x.shape == (10000, 32, 32, 3)
    assert np.bincount(train_y).tolist() == [5000] * 10


# ---- membership inference ----

def test_attack_sets_are_separate_and_balanced():
    member_labels, non_member_labels = np.repeat(np.arange(10), 20), np.repeat(np.arange(10), 20)
    sets = split_attack_sets(member_labels, non_member_labels, seed=0)
    assert not np.any(sets["train_mask"] & sets["eval_mask"])
    ev = sets["eval_mask"]
    assert sets["is_member"][ev].sum() == (1 - sets["is_member"][ev]).sum()


def test_attack_metrics_perfect_and_random():
    is_member = np.array([1, 1, 1, 0, 0, 0])
    perfect = attack_metrics(np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), is_member, threshold=0.5)
    assert perfect["auc"] == 1.0 and perfect["advantage"] == 1.0 and perfect["f1"] == 1.0
    assert fit_threshold(np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), is_member) == pytest.approx(0.7)


def test_attack_detects_lower_member_loss_and_baselines_stay_near_half():
    rng = np.random.default_rng(0)
    n = 2000
    labels = np.repeat(np.arange(10), n // 10)
    members = {"loss": rng.exponential(0.2, n), "confidence": rng.uniform(0.8, 1.0, n)}
    non_members = {"loss": rng.exponential(1.0, n), "confidence": rng.uniform(0.3, 1.0, n)}
    result = run_attack(members, non_members, labels, labels, seed=0)
    assert result["loss"]["auc"] > 0.7
    assert abs(result["random_guess"]["auc"] - 0.5) < 0.05
    assert abs(result["label_only"]["auc"] - 0.5) < 0.05


# ---- differential privacy ----

def test_dp_sgd_reaches_target_epsilon():
    torch.manual_seed(0)
    x = torch.randn(512, 3, 32, 32)
    y = np.random.default_rng(0).integers(0, 10, 512)
    loader = make_loader(x, y, batch_size=64, shuffle=True)
    cfg = {"lr": 0.1, "momentum": 0.0, "delta": 1e-5, "epochs": 1, "max_grad_norm": 1.0,
           "batch_size": 64, "physical_batch_size": 32}
    _, result = train_dp(SmallCNN(), loader, (x[:64], y[:64]), cfg, target_epsilon=5.0,
                         device=torch.device("cpu"), log=lambda *_: None)
    p = result["privacy"]
    assert p["noise_multiplier"] > 0
    assert 0 < p["achieved_epsilon"] <= 5.0 + 1e-6
    assert p["achieved_epsilon"] > 4.0  # Opacus picks the smallest noise that reaches the target
    assert p["steps"] > 0 and p["sample_rate"] == pytest.approx(64 / 512)


def test_batchnorm_model_is_rejected_for_dp():
    bad = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.Flatten(), nn.Linear(8 * 30 * 30, 10))
    x = torch.randn(64, 3, 32, 32)
    y = np.zeros(64, dtype=np.int64)
    with pytest.raises(ValueError):
        train_dp(bad, make_loader(x, y, 16, True), (x, y), {"lr": 0.1, "momentum": 0.0}, 1.0,
                 torch.device("cpu"))


# ---- randomized response ----

def test_randomized_response_flip_rate_and_unbiased_estimate():
    rng = np.random.default_rng(0)
    truth = (rng.random(200_000) < 0.3).astype(int)
    eps = 1.0
    reports = rr.randomize(truth, eps, rng)
    assert np.mean(reports != truth) == pytest.approx(1 - rr.truth_probability(eps), abs=0.005)
    assert rr.estimate_proportion(reports, eps) == pytest.approx(truth.mean(), abs=0.01)


def test_randomized_response_privacy_ratio():
    for eps in [0.1, 1.0, 3.0]:
        p = rr.truth_probability(eps)
        assert p / (1 - p) == pytest.approx(np.exp(eps))


# ---- k-anonymity ----

def test_k_anonymity_verification():
    df = anon.make_synthetic_patients(400, seed=0)
    assert not anon.verify_k_anonymity(df, 2)["satisfied"]  # raw table has unique records
    for k in [2, 5, 10]:
        released, levels, suppressed = anon.anonymize(df, k, max_suppression_rate=0.05)
        assert anon.verify_k_anonymity(released, k)["satisfied"]
        assert suppressed <= 0.05 * len(df)


def test_verify_detects_small_class():
    df = pd.DataFrame({"age": ["20-24"] * 3 + ["25-29"], "zipcode": ["130**"] * 4,
                       "sex": ["F"] * 4, "diagnosis": ["Flu"] * 4})
    check = anon.verify_k_anonymity(df, 2)
    assert not check["satisfied"] and check["min_class_size"] == 1


# ---- feature reduction ----

def test_pca_output_dimensions():
    features = np.random.default_rng(0).normal(size=(300, 3072)).astype(np.float32)
    for d in [8, 32]:
        pca = fit_pca(features, d, seed=0)
        assert pca.transform(features[:10]).shape == (10, d)
        assert pca.inverse_transform(pca.transform(features[:10])).shape == (10, 3072)
