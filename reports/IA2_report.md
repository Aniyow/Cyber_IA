# IA-2 Report: Implementation, Evaluation and Technical Outcomes

**Project title:** Privacy-Preserving Techniques in Machine Learning: Membership Inference Attacks, Differential Privacy, and Privacy–Utility Trade-offs

**Subject:** Security and Privacy for Machine Learning

**Student:** Sai Anirudh  **Roll number:** 16014223071  **Date:** 22 October 2026

All numbers in this report come from `results/metrics/full/` (final run). Unless stated otherwise, values are the mean ± population standard deviation over seeds 0, 1 and 2. Figures are in `results/plots/full/`.

---

## 1. Final problem statement and objectives

> "Machine learning models may reveal whether an individual's data was used during training. This project investigates membership inference attacks and evaluates whether differential privacy can reduce this risk while maintaining model accuracy, using a public dataset and Python-based experiments."

**Objectives**

1. Train a conventional image classifier (CIFAR-10).
2. Implement a membership inference attack to measure training-data leakage.
3. Implement DP-SGD.
4. Compare standard and DP models on utility and attack effectiveness.
5. Demonstrate the techniques of all five course modules.
6. Document experiments, results, limitations and reproducibility.

**Research questions:**
- RQ1: Can a membership inference attack distinguish training from non-training examples?
- RQ2: Does DP-SGD reduce the effectiveness of the evaluated attack?
- RQ3: What changes in model accuracy occur when DP is applied?
- RQ4: How do the privacy settings influence the privacy–utility trade-off?
- RQ5: What are the benefits and limitations of the other course techniques?

## 2. Changes since IA-1, with justification

| Change | Reason | Evidence |
|---|---|---|
| DP-SGD learning rate changed from 0.25 to 0.1 for all ε | In full run v1 (lr 0.25) the ε = 1 models diverged: train loss rose from 2.29 to 4.13 and test accuracy averaged 0.173. At ε = 8 the two learning rates had almost the same validation accuracy (0.5385 vs 0.5355), and lr 0.1 trained stably at ε = 1. | `results/archive/dp_learning_rate_tuning.md`, `results/archive/full_v1_dp_lr0.25_metrics/` |
| Losses and confidences computed in float64 | In float32, many confidently classified training images had a loss of exactly 0. These ties collapsed the low-FPR part of the ROC curve, and v1 reported TPR = 0 at 1% FPR for all standard models. AUC was barely affected (seed 0: 0.7292 in both versions). | `src/evaluation.py::predict` |
| Corrected theoretical variance for randomized response | The first version included population-sampling variance, but the true answers are fixed in this experiment. The corrected formula `p(1−p)/(n(2p−1)²)` matches the simulation (§8.3). | `src/randomized_response.py::theoretical_std` |
| Full-size Modules 3–5, multi-seed DP study, plots, IA-2 documentation | Planned work from IA-1 | `results/metrics/full/` |

The v1 run is kept unchanged in `results/archive/` and is not mixed with the final results.

## 3. Mapping to the five official modules

| # | Module | Technique | Implementation | Evaluation | Evidence | Status | Limitations |
|---|---|---|---|---|---|---|---|
| 1 | Privacy Considerations in Machine Learning | Loss- and confidence-threshold membership inference (Yeom et al., 2018) | `src/membership_inference.py` | AUC, P/R/F1, TPR/FPR, advantage, TPR@1%FPR vs random and label-only baselines on a held-out attack-eval set | `main_results.csv`, `mia_roc_curves.png` | Implemented and experimentally evaluated | One simple attack; no shadow models or LiRA |
| 2 | Differential Privacy for Machine Learning | DP-SGD (Abadi et al., 2016) with Opacus, RDP accountant | `src/train_private.py`, `src/run_experiment.py` | Accuracy, F1, attack metrics, achieved (ε, δ), noise multiplier, time; 4 budgets × 3 seeds | `main_summary.csv`, `accuracy_vs_epsilon.png`, `mia_auc_vs_epsilon.png` | Implemented and experimentally evaluated | Hyperparameter tuning not privacy-accounted |
| 3 | Local Differential Privacy for Machine Learning | Randomized response (Warner, 1965) | `src/randomized_response.py` | MAE, RMSE, empirical vs theoretical std over 500 trials; 7 values of ε, 4 values of n | `randomized_response.csv/json`, `randomized_response_error.png` | Implemented and demonstrated on synthetic data | Binary attribute only; not applied to the CIFAR-10 model |
| 4 | Privacy-Preserving Data Mining Techniques | Generalization and suppression k-anonymity (Samarati, 2001; Sweeney, 2002); Laplace noise perturbation | `src/anonymization.py` | Equivalence-class verification, suppression, information loss, uniqueness, homogeneity, utility, linkage rate | `k_anonymity.csv`, `perturbation.csv`, `anonymization_utility.png` | Implemented and demonstrated on synthetic data | Not a formal guarantee; synthetic data |
| 5 | Compressive Privacy for Machine Learning | PCA feature reduction (Kung, 2017) | `src/compressive_privacy.py` | Retained dimensions vs accuracy/F1, reconstruction PSNR, attack AUC | `compressive_privacy.csv`, `pca_reconstructions.png` | Implemented and experimentally evaluated (1 seed) | Single seed; no formal guarantee |

## 4. Architecture and workflow

```
configs/full.json ─► src/run_all.py
   ├─ run_experiment.py ── data_loader.py (CIFAR-10, class-balanced disjoint splits)
   │      for seed in {0,1,2}:
   │        train_baseline.py  ─┐
   │        train_private.py ×4 ┼─► evaluation.py (test utility)
   │                            └─► membership_inference.py (attack-train → threshold, attack-eval → metrics)
   │      → main_results.csv, main_details.json, splits_seed*.json, run_info.json
   ├─ make_plots.py  (reads only the saved CSV/JSON → main_summary.csv + figures)
   ├─ provenance.py  (hashes, sizes → cifar10_provenance.json)
   ├─ randomized_response.py   (Module 3)
   ├─ anonymization.py         (Module 4)
   └─ compressive_privacy.py   (Module 5, reuses the CIFAR-10 split and the attack)
```

## 5. Dataset provenance and preprocessing

CIFAR-10 (Krizhevsky, 2009) was downloaded on 22 October 2026 with `torchvision.datasets.CIFAR10`. The archive's MD5 `c58f30108f718f92721af3b95e74349a` matches the value on the official page, and its SHA-256 is `6d958be0…c001ce`. It contains 50,000 training and 10,000 test images, with 5,000 and 1,000 per class respectively. No license is stated on the official page, which only asks users to cite the technical report, so the license is recorded as *Unknown*.

Preprocessing is the fixed scaling `(x/255 − 0.5)/0.5`, with no data-dependent statistics and no augmentation. For each seed the full run uses 10,000 members, 10,000 non-members and 2,000 validation images, all from the official training set, disjoint and class-balanced (1,000 / 1,000 / 200 per class). The test set is the full official test set. Disjointness was checked for all three seeds (`splits_seed*.json`: `members_non_members_disjoint = true`, `validation_disjoint = true`). Full details are in `reports/dataset_provenance.md`.

## 6. Implementation details

- **Target model:** `SmallCNN`, with 815,018 parameters: 3 × (conv-GN-ReLU-conv-GN-ReLU-maxpool), then 2048→256→10. GroupNorm is used because BatchNorm is incompatible with per-example gradients. Opacus' `ModuleValidator` is run before DP training.
- **Standard training:** SGD (lr 0.05, momentum 0.9), batch 128, 30 epochs.
- **DP-SGD:** Opacus 1.6.0 `make_private_with_epsilon`, SGD (lr 0.1, momentum 0.9), Poisson sampling with rate q = 0.1 (Opacus sets q = 1 / number of batches of the original 1,024-sized loader, i.e. an expected batch of 1,000), clipping norm C = 1.0, 30 epochs = 300 steps, δ = 1e-5, RDP accountant. Logical batches are split into physical batches of 512 by `BatchMemoryManager`.
- **Attack:**
  - Score = −loss (also max confidence).
  - Threshold chosen for best accuracy on attack-train (5,000 members + 5,000 non-members).
  - Metrics reported on attack-eval (another 5,000 + 5,000).
  - Baselines: random scores, and a label-only score.
- **Environment:** Windows 11, Python 3.12.5, torch 2.16.0.dev20261006+cu132, Opacus 1.6.0, scikit-learn 1.7.2, NVIDIA RTX 5050 Laptop GPU (`run_info.json`).
- **Tests:** 14 pytest tests covering splits and disjointness, CIFAR-10 sizes, attack metrics and baselines, attack-set separation, DP accounting (achieved ε ≤ target, noise > 0, sample rate), rejection of BatchNorm for DP, randomized response flip rate, ε ratio and unbiasedness, k-anonymity verification, and PCA dimensions. The last run gave **14 passed, 0 failed**.

## 7. Experimental methodology and evaluation metrics

| Experiment | Metrics |
|---|---|
| Target utility | Test accuracy, macro precision, recall, F1; train accuracy; generalization gap |
| Membership inference | ROC-AUC; accuracy, precision, recall, F1 at the attack-train threshold; TPR, FPR; advantage = TPR − FPR; TPR at FPR ≤ 1% |
| Privacy accounting | Achieved ε at δ = 1e-5 (RDP), noise multiplier σ, C, q, steps |
| Cost | Wall-clock training time |
| Module 3 | MAE, RMSE, empirical and theoretical std of the estimated proportion |
| Module 4 | Equivalence-class sizes, k verified, suppression rate, information loss, unique-record rate, homogeneity, logistic-regression utility, per-record MAE, mean error, correlation, linkage rate |
| Module 5 | Retained dimensions, explained variance, test accuracy/F1, PSNR, attack AUC |

## 8. Results

### 8.1 Main study: standard vs DP-SGD (3 seeds)

| Model | Achieved ε (δ=1e-5) | σ | Test acc. | Test F1 | Train acc. | Attack AUC (loss) | Advantage | TPR@1%FPR | Attack AUC (conf.) | Train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Standard | — (no DP) | — | 0.641 ± 0.015 | 0.640 ± 0.014 | 0.990 ± 0.006 | **0.736 ± 0.008** | 0.440 ± 0.024 | 0.012 ± 0.002 | 0.693 ± 0.013 | 34.2 ± 3.6 |
| DP-SGD | 0.993 | 7.19 | 0.314 ± 0.008 | 0.307 ± 0.006 | 0.326 ± 0.009 | 0.512 ± 0.007 | 0.020 ± 0.013 | 0.012 ± 0.001 | 0.500 ± 0.003 | 99.5 ± 19.5 |
| DP-SGD | 2.998 | 2.77 | 0.463 ± 0.004 | 0.453 ± 0.004 | 0.486 ± 0.006 | 0.519 ± 0.001 | 0.021 ± 0.008 | 0.013 ± 0.004 | 0.502 ± 0.009 | 91.5 ± 16.6 |
| DP-SGD | 7.999 | 1.36 | 0.522 ± 0.012 | 0.514 ± 0.014 | 0.551 ± 0.011 | 0.524 ± 0.003 | 0.032 ± 0.008 | 0.011 ± 0.004 | 0.505 ± 0.005 | 82.1 ± 1.4 |
| DP-SGD | 31.994 | 0.66 | 0.539 ± 0.012 | 0.532 ± 0.013 | 0.578 ± 0.015 | 0.528 ± 0.003 | 0.036 ± 0.008 | 0.010 ± 0.002 | 0.506 ± 0.006 | 92.2 ± 14.5 |

Baselines on the same attack-eval sets: random-guessing AUC 0.507 ± 0.004; label-only AUC 0.500 for every model and seed.

Attack precision, recall and F1 at the chosen threshold (from `main_summary.csv`):

| Model | Precision | Recall (= TPR) | FPR | F1 |
|---|---|---|---|---|
| Standard | 0.660 ± 0.009 | 0.909 ± 0.024 | 0.469 ± 0.020 | 0.764 ± 0.011 |
| DP ε≈1 | 0.512 ± 0.009 | 0.473 ± 0.045 | 0.453 ± 0.055 | 0.490 ± 0.021 |
| DP ε≈3 | 0.509 ± 0.003 | 0.568 ± 0.135 | 0.548 ± 0.130 | 0.529 ± 0.064 |
| DP ε≈8 | 0.517 ± 0.004 | 0.502 ± 0.135 | 0.470 ± 0.129 | 0.500 ± 0.065 |
| DP ε≈32 | 0.515 ± 0.006 | 0.666 ± 0.131 | 0.631 ± 0.138 | 0.575 ± 0.045 |

Figures:

- `accuracy_comparison.png`, `mia_auc_comparison.png`, `mia_advantage_comparison.png`, `training_time.png` (bars with seed std);
- `accuracy_vs_epsilon.png`, `mia_auc_vs_epsilon.png` (trade-off plots);
- `mia_roc_curves.png` (seed 0); `validation_curves.png`.

### 8.2 Privacy accounting

Opacus selected σ for each target and reported the achieved ε after training. The achieved values were 0.993, 2.998, 7.999 and 31.994. All are at or just below their targets and identical across seeds, since σ, q and the number of steps do not depend on the seed. The DP test (`test_dp_sgd_reaches_target_epsilon`) checks independently that the achieved ε is positive, does not exceed the target, and is close to it.

These guarantees cover the 10,000 member images used in DP training. They hold under the accountant's assumptions: Poisson sampling, a fixed C, and all intermediate models being released. They do **not** cover:

- the hyperparameter search;
- the choice of the class-balanced subset;
- any image outside the member set.

### 8.3 Module 3: randomized response (n = 10,000, true proportion 0.3008, 500 trials per ε)

| ε | P(truthful) | Mean estimate | MAE | Empirical std | Theoretical std |
|---|---|---|---|---|---|
| 0.1 | 0.525 | 0.3025 | 0.0784 | 0.0965 | 0.1000 |
| 0.25 | 0.562 | 0.2998 | 0.0314 | 0.0394 | 0.0399 |
| 0.5 | 0.622 | 0.3009 | 0.0159 | 0.0201 | 0.0198 |
| 1 | 0.731 | 0.3015 | 0.0077 | 0.0097 | 0.0096 |
| 2 | 0.881 | 0.3010 | 0.0034 | 0.0042 | 0.0043 |
| 4 | 0.982 | 0.3008 | 0.0011 | 0.0014 | 0.0014 |
| 8 | 0.9997 | 0.3008 | 0.00015 | 0.00019 | 0.00018 |

At ε = 1, the MAE is 0.0794 for n = 100, 0.0236 for n = 1,000, 0.0076 for n = 10,000 and 0.0024 for n = 100,000. Each tenfold increase in n reduces the error by about √10. Figure: `randomized_response_error.png`.

### 8.4 Module 4: k-anonymity and perturbation (2,000 synthetic records, max suppression 2%)

The original table is not even 2-anonymous: 97.7% of records have a unique (age, zipcode, sex) combination. The diagnosis classifier reaches 0.278 accuracy on it, against a majority-class rate of 0.248.

| k | Levels (age, zip, sex) | Suppressed | Info. loss | Classes | Min / mean class size | Verified | Min distinct diagnoses | Homogeneous classes | Utility acc. |
|---|---|---|---|---|---|---|---|---|---|
| 2 | (0, 3, 0) | 26 (1.3%) | 0.200 | 376 | 2 / 5.3 | yes | 1 | 4.3% | 0.297 |
| 5 | (1, 2, 0) | 31 (1.6%) | 0.217 | 140 | 5 / 14.1 | yes | 2 | 0% | 0.318 |
| 10 | (1, 4, 0) | 0 | 0.350 | 30 | 18 / 66.7 | yes | 3 | 0% | 0.315 |
| 25 | (2, 4, 0) | 22 (1.1%) | 0.433 | 15 | 38 / 131.9 | yes | 5 | 0% | 0.305 |
| 50 | (4, 2, 0) | 0 | 0.467 | 10 | 174 / 200.0 | yes | 5 | 0% | 0.243 |

Laplace perturbation of age and income (`perturbation.csv`):

| Noise scale (× std) | Age MAE (years) | Income MAE | Error of mean age | Age–income corr. | Linkage success |
|---|---|---|---|---|---|
| 0 | 0.00 | 0 | 0.000 | 0.794 | 0.950 |
| 0.05 | 0.96 | 760 | 0.015 | 0.789 | 0.209 |
| 0.1 | 1.93 | 1,497 | 0.066 | 0.781 | 0.076 |
| 0.25 | 4.93 | 3,644 | 0.043 | 0.697 | 0.018 |
| 0.5 | 9.95 | 7,278 | 0.546 | 0.535 | 0.011 |
| 1.0 | 19.94 | 14,990 | 0.048 | 0.229 | 0.003 |

Linkage success is below 1.0 even without noise because some synthetic records share the same (age, income) values. Figure: `anonymization_utility.png`.

### 8.5 Module 5: PCA feature reduction (seed 0; MLP, 30 epochs)

| Representation | Dimensions | Explained variance | Test PSNR (dB) | Test acc. | Test F1 | Train acc. | Attack AUC |
|---|---|---|---|---|---|---|---|
| Raw pixels | 3,072 | 1.000 | — | 0.439 | 0.438 | 0.965 | 0.822 |
| PCA | 8 | 0.623 | 16.3 | 0.325 | 0.324 | 0.532 | 0.670 |
| PCA | 32 | 0.802 | 19.0 | 0.457 | 0.456 | 1.000 | 0.845 |
| PCA | 128 | 0.921 | 22.9 | 0.478 | 0.477 | 1.000 | 0.857 |
| PCA | 512 | 0.985 | 29.6 | 0.448 | 0.448 | 1.000 | 0.898 |

Figure: `pca_reconstructions.png` shows six test images and their reconstructions from each PCA representation.

## 9. Interpretation of the privacy–utility trade-off

**RQ1: Yes.** Against the standard CNN, the loss attack reached AUC 0.736 ± 0.008 and membership advantage 0.440 ± 0.024. Both baselines stayed at chance, so the signal is membership, not class identity. The cause is visible in the gap between train accuracy (0.990) and test accuracy (0.641). The model is very confident on images it has seen, so a simple threshold on the loss separates most members from non-members. However, the attack identifies members mainly in bulk. At 1% FPR it found only 1.2% of members, about what chance would give. Following Carlini et al. (2022), this shows that a good AUC does not mean the attack can confidently identify specific individuals.

**RQ2: Yes, for this attack.** All DP-SGD models reduced the attack AUC to 0.512–0.528 and the advantage to 0.020–0.036, compared with 0.736 and 0.440 for the standard model. The confidence attack fell to between 0.500 and 0.506. The loss attack stayed slightly above the random baseline (0.507 ± 0.004) at ε ≈ 8 and ε ≈ 32, so some measurable signal remains. The theoretical bound of Yeom et al. (advantage ≤ e^ε − 1) is larger than 1 for every ε tested (it is 1.70 already at ε = 0.993). It therefore says nothing about these models; the observed reduction is an empirical result for this attack only. A stronger attack could find more leakage.

**RQ3.** DP-SGD cost accuracy at every budget. Test accuracy fell from 0.641 to 0.539 at ε ≈ 32, 0.522 at ε ≈ 8, 0.463 at ε ≈ 3 and 0.314 at ε ≈ 1. Training was also 2.4–2.9× slower, because of per-example gradients and memory-managed physical batches. The two kinds of model are not completely like-for-like: they use different batch sizes and learning rates, as is usual for DP-SGD.

**RQ4.** As ε grows, accuracy rises steeply from ε ≈ 1 to ε ≈ 8 and then levels off. Attack AUC rises only slightly (0.512 → 0.528). Most of the attack reduction is already present at ε ≈ 32, where accuracy is 0.539. This is likely because clipping and noise limit memorization: train accuracy is 0.578 vs test 0.539, a gap of 0.04 compared with 0.35 for the standard model. Even weak formal guarantees came with much less overfitting here. Between the DP models, the trade-off for this attack is mostly about accuracy, since attack AUC changes little. A practitioner would pick ε by the guarantee they need to claim, not by the measured attack AUC.

**RQ5:** see §10.

## 10. Cross-module comparison

| Technique | What is protected | Formal guarantee | Main cost observed | Main limitation observed |
|---|---|---|---|---|
| Membership inference (M1) | — (measures risk) | — | — | Low TPR at low FPR; one attack only |
| DP-SGD (M2) | Each training example's influence on the model | (ε, δ)-DP, central model | 10–33 points of test accuracy, ~2.5× training time | Trusted curator needed; tuning not accounted |
| Randomized response (M3) | Each respondent's answer, before collection | ε-local DP | Estimation error grows quickly as ε falls (MAE 0.078 at ε = 0.1 vs 0.0077 at ε = 1, n = 10k) | Needs large n; per-record answers are noisy |
| k-anonymity (M4) | Linking a record via QIs | None (syntactic) | Information loss 0.20–0.47, suppression up to 1.6% | At k = 2, 4.3% of classes had one diagnosis, so it was disclosed (homogeneity) |
| Perturbation (M4) | Exact numeric values | None as implemented | Correlation 0.79 → 0.23 at the largest noise | Means preserved, record-level values degraded; no ε |
| PCA (M5) | Raw input detail | None | 11 points of accuracy at 8 dims; none at 32–128 dims | Did **not** reduce the membership attack at ≥ 32 dims |

The main contrast is between **central DP** (DP-SGD) and **local DP** (randomized response). DP-SGD trusts the model trainer with raw data and protects what the released model reveals. Randomized response trusts nobody, because each person randomizes their own answer before sending it. The price is that much more noise is needed per person for a given accuracy, so local DP works best for simple statistics over large populations. Randomized response here protects only the synthetic survey answers; it does not protect the CIFAR-10 classifier.

**Module 4 interpretation:**
- Generalization removed all unique QI combinations for every k, and each released table passed the verification check.
- Information loss increased with k, and utility at k = 50 (0.243) fell below the original table (0.278) and to roughly the majority-class rate.
- At small k, utility was *higher* than on the original table (0.297–0.318). With 2,000 records, one-hot exact zipcodes give the classifier many sparse features. Coarser zipcodes acted as regularization. So this utility measure is noisy and should not be read as "anonymization improves data".
- At k = 2, 4.3% of equivalence classes contain only one diagnosis. An attacker who knows someone's age, zip and sex learns that person's diagnosis without re-identifying the record. This is the homogeneity attack that motivates l-diversity (Machanavajjhala et al., 2007).
- Perturbation shows a different trade-off. Small noise (0.1 × std) cut nearest-neighbour linkage from 95% to 7.6% while keeping the means and the correlation (0.781) almost unchanged.

**Module 5 interpretation:**
- 128 PCA dimensions (4% of the pixels, 92% of the variance) gave slightly better test accuracy than raw pixels (0.478 vs 0.439). Even 8 dimensions kept 0.325, compared with 0.1 for chance.
- Low-dimensional representations also limit how much image detail can be recovered: PSNR 16 dB at 8 dimensions vs 29.6 dB at 512. In the reconstruction figure, the 8-dimension images are colour blobs, while the 512-dimension images are recognisable.
- However, membership leakage did not fall with reduction. The MLP memorized the PCA features as easily as the pixels, and attack AUC was 0.85–0.90 at 32–512 dimensions, compared with 0.82 on raw pixels. Only the 8-dimension model, which could not fit its training data (train accuracy 0.53), had a lower AUC (0.67).
- So feature reduction may reduce exposure of the *input*, but by itself it does not protect *membership*, and it gives no formal guarantee. Its effect depends on how much the downstream model can still overfit.

## 11. Limitations and validity threats

1. **Single, simple attack.** A threshold attack is a lower bound on leakage. Shadow-model or LiRA attacks (Carlini et al., 2022) could show more leakage, especially at low FPR, and especially for the DP models at large ε.
2. **Strong attacker assumption.** The attacker is given labelled members to calibrate the threshold. This makes the attack stronger; it does not weaken the conclusions.
3. **Unaccounted hyperparameter tuning.** Four learning rates were tried on validation data before the final run, and the v1 result influenced the final choice. Both depend on the private member set. Papernot and Steinke (2022) show that tuning has its own privacy cost, so the reported ε slightly understates the total leakage of the whole process.
4. **Different optimisation settings.** The standard and DP models differ in batch size and learning rate, so part of the accuracy gap may be due to tuning rather than privacy.
5. **Small scale.** The study uses 10,000 training images, a small CNN, 30 epochs and no augmentation. Absolute accuracy is far below published DP CIFAR-10 results, and the size of the trade-off may differ with stronger models.
6. **Variance.** There are 3 seeds for the main study and 1 seed for Module 5. The threshold-based recall and FPR of the DP models vary a lot between seeds (std ≈ 0.13), because the member and non-member loss distributions almost completely overlap and the chosen threshold is unstable. AUC is the more reliable comparison.
7. **Synthetic data in Modules 3–4.** The generators are simple, and the Module 4 utility test (logistic regression on 2,000 records) is noisy.
8. **Data caveats.** CIFAR-10 contains no personal data and some test images are near-duplicates of training images (Barz & Denzler, 2020). Neither affects the attack, which only uses training-set images.
9. **Non-determinism.** Re-running gives the same numbers on this machine (the smoke run was reproduced exactly). Different GPUs or library versions may give small differences.

## 12. Conclusion

On CIFAR-10, a conventionally trained CNN leaked membership to a simple loss-threshold attack: AUC 0.736 and advantage 0.44, with both baselines at chance. However, the attack could not confidently identify individual members at low false-positive rates. DP-SGD with Opacus reduced the same attack to AUC 0.512–0.528 at all tested budgets (achieved ε from 0.99 to 32, δ = 1e-5). The cost was test accuracy: 0.641 fell to between 0.539 and 0.314, and training was about 2.5× slower. Between ε ≈ 8 and ε ≈ 32 the accuracy gain was small and the attack AUC barely changed. The formal bound on attack advantage is vacuous for every ε tested, so the reduction in leakage is an empirical result for this attack only. The other modules showed the expected trade-offs:
- Randomized response gave accurate population estimates only for moderate ε or large n.
- k-anonymity removed unique records but still disclosed attributes through homogeneous classes.
- PCA kept task accuracy at a fraction of the dimensions but did not reduce membership leakage.

**Possible improvements:**
- Stronger attacks (shadow models or LiRA).
- Privacy-accounted or public-data hyperparameter tuning.
- More seeds, and multiple seeds for Module 5.
- Data augmentation and DP-specific architectures to narrow the accuracy gap.
- Repeating the Module 4 and 5 analyses with DP-based alternatives (DP synthetic data, DP-PCA).

## 13. References

1. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep Learning with Differential Privacy. *Proc. ACM CCS*, 308–318. https://doi.org/10.1145/2976749.2978318
2. Agrawal, R., & Srikant, R. (2000). Privacy-Preserving Data Mining. *Proc. ACM SIGMOD*, 439–450. https://doi.org/10.1145/342009.335438
3. Barz, B., & Denzler, J. (2020). Do We Train on Test Data? Purging CIFAR of Near-Duplicates. *Journal of Imaging*, 6(6), 41. https://doi.org/10.3390/jimaging6060041
4. Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramèr, F. (2022). Membership Inference Attacks From First Principles. *IEEE S&P*. https://doi.org/10.1109/SP46214.2022.9833649
5. Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3–4), 211–407. https://doi.org/10.1561/0400000042
6. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Phil. Trans. R. Soc. A*, 374, 20150202. https://doi.org/10.1098/rsta.2015.0202
7. Kasiviswanathan, S. P., Lee, H. K., Nissim, K., Raskhodnikova, S., & Smith, A. (2011). What Can We Learn Privately? *SIAM Journal on Computing*, 40(3), 793–826. https://doi.org/10.1137/090756090
8. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto. https://www.cs.toronto.edu/~kriz/cifar.html
9. Kung, S. Y. (2017). Compressive Privacy: From Information/Estimation Theory to Machine Learning. *IEEE Signal Processing Magazine*, 34(1), 94–112. https://doi.org/10.1109/MSP.2016.2616720
10. Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M. (2007). ℓ-Diversity: Privacy Beyond k-Anonymity. *ACM TKDD*, 1(1). https://doi.org/10.1145/1217299.1217302
11. Mironov, I. (2017). Rényi Differential Privacy. *IEEE CSF*, 263–275. https://doi.org/10.1109/CSF.2017.11
12. Papernot, N., & Steinke, T. (2022). Hyperparameter Tuning with Renyi Differential Privacy. *ICLR 2022*. https://arxiv.org/abs/2110.03620
13. Samarati, P. (2001). Protecting Respondents' Identities in Microdata Release. *IEEE TKDE*, 13(6), 1010–1027. https://doi.org/10.1109/69.971193
14. Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership Inference Attacks Against Machine Learning Models. *IEEE S&P*, 3–18. https://doi.org/10.1109/SP.2017.41
15. Sweeney, L. (2002). k-Anonymity: A Model for Protecting Privacy. *Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems*, 10(5), 557–570. https://doi.org/10.1142/S0218488502001648
16. Warner, S. L. (1965). Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias. *JASA*, 60(309), 63–69. https://doi.org/10.1080/01621459.1965.10480775
17. Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. *IEEE CSF*, 268–282. https://doi.org/10.1109/CSF.2018.00027
18. Yousefpour, A., et al. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. arXiv:2109.12298. https://arxiv.org/abs/2109.12298

All DOIs (every entry except [8], [12] and [18], which are a technical report and arXiv/conference links) were checked against Crossref on 22 October 2026, including the page numbers given.

## 14. Reproducibility instructions

From the project folder (see `README.md` for environment setup):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q                                   # 14 tests
python -m src.run_all --config configs/smoke.json     # ~1–2 min pipeline check
python -m src.run_all --config configs/full.json      # ~25 min on an RTX 5050; produces everything in this report
```

`configs/full.json` fixes all seeds, sizes and hyperparameters. Every run writes `run_info.json` with package versions and GPU, and per-seed `splits_seed*.json` with SHA-256 hashes of the index lists. The summary table and all main-study figures are generated by `src/make_plots.py` from the saved CSV/JSON only.

## 15. Individual-contribution statement

This is an individual project.

| Student | Roll number | Contribution |
|---|---|---|
| Sai Anirudh | 16014223071 | All work: problem refinement, implementation of all five modules, DP-SGD tuning and privacy accounting, full experiments (15 CNN models over 3 seeds, Modules 3–5), diagnosis and correction of the v1 issues, evaluation and analysis, dataset provenance, documentation and this report |

**Declaration:** I confirm that the contribution described above is accurate. An AI coding assistant was used to help write parts of the code and draft the reports. I ran all experiments on my own machine, checked the results against the saved outputs, and take responsibility for the submitted work.
