# IA-1 Report: Problem Formulation and Preliminary Implementation

**Project title:** Privacy-Preserving Techniques in Machine Learning: Membership Inference Attacks, Differential Privacy, and Privacy–Utility Trade-offs

**Subject:** Security and Privacy for Machine Learning

**Student:** Sai Anirudh  **Roll number:** 16014223071  **Date:** 22 October 2026

---

## 1. Problem statement

> "Machine learning models may reveal whether an individual's data was used during training. This project investigates membership inference attacks and evaluates whether differential privacy can reduce this risk while maintaining model accuracy, using a public dataset and Python-based experiments."

A trained model is a function of its training data, and models often behave measurably differently on the examples they were trained on. For example, they usually have lower loss and higher confidence on those examples. An attacker who can query the model can use this difference to guess whether a given record was in the training set. This is called membership inference. If the training set is sensitive (patients of a clinic, customers of a bank), confirming that a person's record was used can itself be a privacy breach. Differential privacy (DP) gives a mathematical limit on how much any single record can affect the trained model. DP-SGD is the standard way to train neural networks with DP. The question this project studies experimentally is how much DP-SGD reduces the measured membership leakage, and at what cost in accuracy.

## 2. Objectives and scope

**Objectives**

1. Train a conventional image classifier on CIFAR-10.
2. Implement a membership inference attack and measure training-data leakage.
3. Implement DP-SGD.
4. Compare standard and DP models on predictive utility and attack effectiveness.
5. Demonstrate the techniques of all five course modules.
6. Document experiments, results, limitations and reproducibility instructions.

**Scope.** The main study (Modules 1 and 2) is a controlled experiment on CIFAR-10 with a small CNN, a threshold-based black-box attack and Opacus DP-SGD. Modules 3 and 4 are smaller demonstrations on synthetic data. Module 5 applies PCA to CIFAR-10 pixels. Out of scope: state-of-the-art attacks (e.g. LiRA with many shadow models), large models, federated learning, and any real personal data.

**Expected outcomes.** A working, tested codebase; measured attack performance against standard and DP models across several privacy budgets; a privacy–utility comparison; and module demonstrations with metrics suited to each technique. No particular result is assumed in advance.

## 3. Research questions

- **RQ1:** Can a membership inference attack distinguish training examples from non-training examples?
- **RQ2:** Does DP-SGD reduce the effectiveness of the evaluated membership inference attack?
- **RQ3:** What changes in model accuracy occur when differential privacy is applied?
- **RQ4:** How do the privacy settings influence the observed privacy–utility trade-off?
- **RQ5:** What are the benefits and limitations of the additional privacy techniques covered in the course?

## 4. Conceptual explanation and methodology

**Membership inference (Shokri et al., 2017; Yeom et al., 2018).** The attacker gets a record (x, y) and must decide whether it was in the training set. Yeom et al. showed that overfitting alone is enough for a simple attack: predict "member" when the model's loss on (x, y) is below a threshold. Their *membership advantage* (TPR − FPR) measures how much better than chance the attack is. Carlini et al. (2022) argue that attacks should also be judged at low false-positive rates, because an attacker who is confident about a few people can already cause harm.

**Differential privacy (Dwork & Roth, 2014).** A randomized algorithm M is (ε, δ)-DP if, for any two datasets D and D′ that differ in one record and any set of outputs S, Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D′) ∈ S] + δ. A smaller ε means any one record can change the output distribution less. Yeom et al. (2018) show that for an ε-DP training algorithm the membership advantage of *any* attack is at most e^ε − 1. This bound only says something useful when ε is small.

**DP-SGD (Abadi et al., 2016).** In each step, each example's gradient is clipped to norm C, Gaussian noise N(0, σ²C²) is added to the sum, and the result is averaged. Poisson subsampling and a privacy accountant turn σ, the sampling rate and the number of steps into an (ε, δ) guarantee. This project uses Opacus (Yousefpour et al., 2021) and its Rényi-DP accountant (Mironov, 2017).

**Methodology.** Full details are in `reports/methodology.md`. In short:

- Disjoint, class-balanced member, non-member, validation and test sets.
- The attack threshold is fitted on attack-train and all metrics are reported on a separate attack-eval set.
- Random-guess and label-only baselines check that the attack is learning membership, not class labels.
- The same CNN, splits, epochs, test set and attack are used for the standard and DP models.

## 5. Mapping to the five official modules

| # | Module | Technique | Implementation | Evaluation | IA-1 status |
|---|---|---|---|---|---|
| 1 | Privacy Considerations in Machine Learning | Loss / confidence threshold membership inference | `src/membership_inference.py` | ROC-AUC, precision, recall, F1, TPR/FPR, advantage, TPR@1%FPR vs random and label-only baselines | Implemented and evaluated on the standard model |
| 2 | Differential Privacy for Machine Learning | DP-SGD (Opacus, RDP accountant) | `src/train_private.py` | Accuracy, F1, attack metrics, achieved ε at δ = 1e-5, training time | Implemented; smoke run and tuning only |
| 3 | Local Differential Privacy for Machine Learning | Randomized response | `src/randomized_response.py` | Estimation error vs ε and n | Implemented; smoke run |
| 4 | Privacy-Preserving Data Mining Techniques | Generalization/suppression k-anonymity, Laplace perturbation | `src/anonymization.py` | Equivalence-class check, suppression, information loss, utility, linkage rate | Implemented; smoke run |
| 5 | Compressive Privacy for Machine Learning | PCA on CIFAR-10 pixels | `src/compressive_privacy.py` | Dimensions vs accuracy, reconstruction PSNR, attack AUC | Implemented; smoke run |

## 6. Workflow

```
CIFAR-10 (official train 50k, test 10k)
   │  class-balanced split, seed k
   ├── members (10k) ───────────────► train target model ──┬─ standard SGD
   │                                                         └─ DP-SGD (Opacus, ε ∈ {1,3,8,32}, δ=1e-5)
   ├── non-members (10k, never trained on)                         │
   ├── validation (2k, monitoring only)                            ▼
   └── test (official 10k) ──────────────────────────► utility: accuracy, P/R/F1
                                                                   │
members + non-members ─► per-example loss / confidence from the model
        ├─ attack-train (50%) ─► choose threshold
        └─ attack-eval  (50%) ─► AUC, P/R/F1, TPR/FPR, advantage  (+ random / label-only baselines)

Separate demos:  synthetic survey ─► randomized response (Module 3)
                 synthetic patients ─► k-anonymity + perturbation (Module 4)
                 CIFAR-10 members ─► PCA(d) ─► MLP ─► accuracy, PSNR, attack AUC (Module 5)
```

## 7. Dataset description and provenance

The primary dataset is CIFAR-10 (Krizhevsky, 2009). It has 60,000 32×32 colour images in 10 mutually exclusive classes. It was downloaded on 22 October 2026 through `torchvision.datasets.CIFAR10` from the official University of Toronto page. The archive's MD5 (`c58f30108f718f92721af3b95e74349a`) matches the value published on that page. The page states no license and only asks users to cite the technical report, so the license is recorded as *Unknown*. Hashes, sizes, preprocessing and split sizes are in `reports/dataset_provenance.md` and `results/metrics/cifar10_provenance.json`. Modules 3 and 4 use synthetic data generated from fixed seeds.

## 8. Preliminary implementation and baseline

**Completed at IA-1:**

- Project structure, configs (`configs/smoke.json`, `configs/full.json`) and the shared codebase in `src/`.
- Data pipeline with disjoint, class-balanced splits and a fixed, data-independent normalization.
- `SmallCNN` (GroupNorm, about 0.8 M parameters), standard training and DP-SGD training with Opacus.
- The membership attack with both baselines.
- First versions of Modules 3–5.
- 14 unit tests (`tests/test_privacyguard.py`).
- An end-to-end smoke run.
- A first full-size run of the standard model (3 seeds).

**Environment:** Windows 11, Python 3.12.5, PyTorch 2.16.0.dev (CUDA), Opacus 1.6.0, NVIDIA RTX 5050 Laptop GPU (8 GB).

## 9. Preliminary observations (actual runs)

**Smoke run** (1,000 members, 3 epochs, seed 0; snapshot in `results/archive/ia1_smoke_snapshot/`). This run only checks that the pipeline works; the models are far from trained.

| Model | Test acc. | Train acc. | Attack AUC (loss) | Advantage | Achieved ε |
|---|---|---|---|---|---|
| Standard | 0.239 | 0.275 | 0.533 | 0.046 | — |
| DP-SGD, target ε = 8 | 0.220 | 0.224 | 0.524 | 0.020 | 7.994 |

Modules 3–5 also ran on smoke settings. Randomized response estimated a true proportion of 0.302 with mean absolute error 0.0385 at ε = 0.5 and 0.0024 at ε = 4 (n = 2,000). k-anonymity was verified for k = 2 and 5 on a 500-record table. PCA to 64 components gave 33.7% test accuracy against 31.4% on raw pixels (3 epochs).

**First full-size standard model** (10,000 members, 30 epochs; `results/archive/full_v1_dp_lr0.25_metrics/main_results.csv`):

| Seed | Test acc. | Train acc. | Attack AUC (loss) | Advantage | Random-guess AUC | Label-only AUC |
|---|---|---|---|---|---|---|
| 0 | 0.639 | 0.991 | 0.729 | 0.422 | 0.505 | 0.500 |
| 1 | 0.627 | 0.969 | 0.714 | 0.381 | 0.505 | 0.500 |
| 2 | 0.681 | 1.000 | 0.752 | 0.515 | 0.512 | 0.500 |

The standard model nearly memorizes its 10,000 training images (train accuracy 97–100% against 63–68% test accuracy). The loss attack separates members from non-members well above chance (AUC 0.71–0.75), while both baselines stay at 0.50. This is a preliminary answer to RQ1 that the final runs must confirm.

## 10. Proposed evaluation plan (for IA-2)

1. Train DP-SGD models at target ε ∈ {1, 3, 8, 32} (δ = 1e-5) with 3 seeds each, and report the ε achieved according to the accountant.
2. Apply the identical attack to every model and compare AUC, advantage, P/R/F1 and TPR at 1% FPR.
3. Plot accuracy and attack AUC against achieved ε. Compare the measured advantage with the theoretical bound e^ε − 1.
4. Complete Modules 3–5 at full size and report the metrics listed in the methodology.
5. Re-run all tests and confirm that every table and figure is generated from saved outputs.

## 11. Known limitations and remaining work

- Only one simple attack family is used. Stronger attacks (shadow models, LiRA) could show more leakage, so low attack AUC on a DP model is not proof of safety.
- DP hyperparameters were tuned on the private member data without privacy accounting.
- The CNN is small and trained without augmentation, so absolute accuracy is far below the state of the art.
- **Remaining:** full DP-SGD runs, full Modules 3–5, final plots, and the IA-2 report.

## 12. Individual-contribution statement

This is an individual project.

| Student | Roll number | Contribution |
|---|---|---|
| Sai Anirudh | 16014223071 | All work: problem formulation and module mapping, dataset selection and preparation, implementation of the data pipeline, CNN, standard and DP-SGD training, membership inference attack and Modules 3–5, tests, smoke and preliminary runs, and this report |

**Declaration:** I confirm that the contribution described above is accurate. An AI coding assistant was used to help write parts of the code and draft the reports. I ran all experiments on my own machine, checked the results against the saved outputs, and take responsibility for the submitted work.

## References

See the reference list in `reports/IA2_report.md` (same sources).
