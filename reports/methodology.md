# Methodology

This document describes how each experiment is set up, which data each step is allowed to see, and how the metrics are computed. The numbers produced by these procedures are in `results/metrics/<run>/` and are discussed in `reports/IA2_report.md`.

## 1. Main study: membership inference and DP-SGD (Modules 1 and 2)

### 1.1 Data roles

All data comes from CIFAR-10 (see `reports/dataset_provenance.md`). For every random seed, `src/data_loader.py::make_splits` draws the following disjoint groups from the official 50,000-image training set. Each group is sampled per class, so it contains the same number of images of every class.

| Role | Source | Used for | Never used for |
|---|---|---|---|
| Members | official train set | training the target model | — |
| Non-members | official train set, disjoint from members | the attacker's "out" examples | training anything |
| Validation | official train set, disjoint from both | monitoring validation accuracy per epoch | model selection, attack fitting |
| Test | official 10,000-image test set | reporting target-model utility | training, attack fitting |

The attack data is the union of members and non-members. `src/membership_inference.py::split_attack_sets` divides it into:

- **attack-train** (half of each class within the members and half within the non-members), used only to choose the attack threshold;
- **attack-eval** (the other half), used to report all attack metrics.

Disjointness is checked in code (`run_experiment.split_summary`, saved as `splits_seed<k>.json`) and in the tests.

### 1.2 Preprocessing

Images are scaled from `[0, 255]` to `[-1, 1]` with the fixed formula `(x / 255 - 0.5) / 0.5`. No mean or standard deviation is estimated from the data. This matters for DP-SGD: statistics computed from the private training images would themselves leak information and would need their own privacy budget. Labels are the integer class ids 0–9 used directly with cross-entropy loss. No data augmentation is used, so a "member" always means exactly the image the model was trained on.

### 1.3 Target models

Both target models use the same `SmallCNN` (`src/model.py`): three blocks of two 3×3 convolutions with GroupNorm and ReLU, each followed by 2×2 max pooling, then a 256-unit hidden layer and a 10-way output. GroupNorm replaces BatchNorm because BatchNorm mixes statistics across the examples in a batch, so per-example gradients (and therefore DP-SGD) are not defined for it. Opacus' `ModuleValidator` is run before private training and the training stops if the model is not compatible.

- **Standard model** (`src/train_baseline.py`): mini-batch SGD with momentum, shuffled batches.
- **DP-SGD model** (`src/train_private.py`): Opacus `PrivacyEngine.make_private_with_epsilon`. For each logical batch Opacus computes per-example gradients, clips each to L2 norm `max_grad_norm`, sums them, adds Gaussian noise with standard deviation `noise_multiplier × max_grad_norm`, and averages. Batches are drawn by Poisson sampling. Opacus sets the sampling rate to 1 / (number of batches of the original loader), which is 0.1 in the full config (expected batch 1,000; 300 steps over 30 epochs). Opacus searches for the smallest noise multiplier that keeps the RDP accountant at or below the target epsilon after the configured number of epochs. After training, the achieved epsilon is read back from the accountant (`privacy_engine.get_epsilon(delta)`) and that value, not the target, is reported. Large logical batches are processed in smaller physical batches with Opacus' `BatchMemoryManager`, which does not change the privacy analysis.

Delta is set to 1e-5, which is smaller than 1 / n_members for every configuration used.

The DP models use a larger batch and learning rate than the standard model. These are the usual adjustments for DP-SGD and are recorded in the config files. Architecture, data splits, number of epochs, test set and attack are the same.

### 1.4 Membership inference attack

The attack follows the loss-threshold attack of Yeom et al. (2018).

**Attacker assumptions.** The attacker (i) can query the trained model and obtain its output probabilities; (ii) knows the true label of the record they are testing; (iii) has a set of records whose membership is known (attack-train) to calibrate a threshold. The attacker does not see model weights or gradients and does not train shadow models. Assumption (iii) is stronger than in some real settings and makes the attack easier, so the measured leakage should be read as what a reasonably informed attacker could achieve with this simple method, not as an upper bound.

**Score.** For every member and non-member, the target model's per-example cross-entropy loss is computed; the membership score is the negative loss (lower loss = more likely a member). A second score, the maximum softmax confidence (which does not use the label), is also reported.

**Threshold.** The threshold that gives the best accuracy on attack-train is chosen (`fit_threshold`). It is then applied unchanged to attack-eval.

**Metrics on attack-eval.** ROC-AUC (threshold-free), accuracy, precision, recall and F1 for the "member" class, true- and false-positive rates at the chosen threshold, membership advantage `TPR − FPR` (Yeom et al., 2018), and TPR at FPR ≤ 1% (following Carlini et al., 2022, who argue that the low-FPR region matters most).

**Baselines and leakage checks.**

- *Random guessing*: uniformly random scores evaluated on the same attack-eval set (expected AUC 0.5, advantage 0).
- *Label only*: each attack-eval example is scored by the fraction of members among attack-train examples of the same class. Since members and non-members have identical class counts, this should give AUC ≈ 0.5. If it did not, the attack could be exploiting class differences rather than membership.

### 1.5 Repetition

Each configuration is run with several seeds. The seed controls the data split, the attack split and model initialisation. Results are reported as mean and population standard deviation across seeds.

## 2. Module 3: randomized response

`src/randomized_response.py` generates a synthetic binary survey (each respondent's answer is 1 with a fixed true probability). Each respondent reports the truth with probability `p = e^ε / (1 + e^ε)` and lies otherwise. Since `p / (1 − p) = e^ε`, each report satisfies ε-local differential privacy. The population proportion is estimated with the unbiased estimator `π̂ = (λ̂ − (1 − p)) / (2p − 1)`, where λ̂ is the observed "yes" rate. For each ε the experiment is repeated many times and the mean absolute error, RMSE, empirical standard deviation and the analytical standard deviation are reported. A second experiment fixes ε and varies the number of respondents.

## 3. Module 4: anonymization, k-anonymity and perturbation

`src/anonymization.py` creates a synthetic patient table with quasi-identifiers (age, 5-digit zipcode, sex), a sensitive attribute (diagnosis) and a numeric attribute (annual income).

- **Generalization hierarchies**: age → exact, 5-, 10-, 20-year ranges, `*`; zipcode → mask 0–5 trailing digits; sex → exact or `*`.
- **k-anonymization**: all 60 level combinations are tried (full-domain generalization). Records in equivalence classes smaller than k are suppressed if that removes at most a configured share of the table. The feasible combination with the lowest information loss (average normalized hierarchy height) is kept.
- **Verification**: `verify_k_anonymity` groups the released table by the QIs and checks that the smallest class has at least k records.
- **Re-identification proxy**: the share of records whose QI combination is unique.
- **Homogeneity check**: the smallest number of distinct diagnoses per class and the share of classes with only one diagnosis (an l-diversity style check, Machanavajjhala et al., 2007).
- **Utility**: accuracy of a logistic regression that predicts the diagnosis from the released QIs (70/30 split), compared with the original table.
- **Perturbation**: additive Laplace noise on age and income at several scales (fractions of each column's standard deviation). Utility is measured as per-record mean absolute error, error of the column means and the preserved age–income correlation. Risk is measured with a nearest-neighbour linkage attack: the share of perturbed records whose closest original record is the true one.

Neither k-anonymity nor this noise addition is a differential privacy mechanism; no ε is claimed for them.

## 4. Module 5: compressive privacy with PCA

`src/compressive_privacy.py` uses the same CIFAR-10 member / non-member / test split as the main study. Each image is flattened into 3,072 pixel values. PCA is fitted on the members only and used to project all splits to d components. The same MLP (two hidden layers, Adam) is trained on the raw pixels and on each PCA representation. Reported per representation: number of features, explained variance, test accuracy and macro-F1, reconstruction PSNR on test images (how much of the original image can be recovered from the released features), and the loss-threshold membership attack from Module 1 applied to that MLP.

The MLP on raw pixels is the baseline for this module. It is a different model from the CNN in Modules 1–2, so the two studies should not be compared on accuracy directly.
