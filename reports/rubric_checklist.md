# Rubric Compliance Checklist

Status values: **Complete**, **Partial** or **Not Complete**. This checklist records evidence; it does not predict marks.

## IA-1 (Problem Formulation and Preliminary Implementation)

| Requirement | Evidence | Status |
|---|---|---|
| Problem statement (fixed wording kept) | `IA1_report.md` §1, `IA2_report.md` §1, `README.md` | Complete |
| Objectives and scope | `IA1_report.md` §2 | Complete |
| Research questions RQ1–RQ5 | `IA1_report.md` §3 | Complete |
| Conceptual explanation (MIA, DP, DP-SGD) | `IA1_report.md` §4, `methodology.md` | Complete |
| Mapping to all five modules | `IA1_report.md` §5 | Complete |
| Workflow / architecture diagram | `IA1_report.md` §6 (text diagram) | Complete (text, not a drawn figure) |
| Dataset description and provenance | `dataset_provenance.md`, `cifar10_provenance.json` | Complete (license: Unknown, as on the official page) |
| Preliminary implementation and baseline | `src/`, smoke snapshot, v1 standard models | Complete |
| Actual preliminary observations | `IA1_report.md` §9 (smoke snapshot, v1 archive) | Complete |
| Proposed evaluation plan | `IA1_report.md` §10 | Complete |
| Limitations and remaining work | `IA1_report.md` §11 | Complete |
| Individual-contribution statement | `IA1_report.md` §12 (Sai Anirudh, 16014223071) | Complete |

## IA-2 (Implementation, Evaluation and Technical Outcomes)

| Requirement | Evidence | Status |
|---|---|---|
| Final problem statement and objectives | `IA2_report.md` §1 | Complete |
| Changes since IA-1 with justification | `IA2_report.md` §2, `results/archive/` | Complete |
| Module mapping with technique / implementation / evaluation / evidence / limitations | `IA2_report.md` §3 | Complete |
| Architecture and workflow | `IA2_report.md` §4 | Complete |
| Dataset provenance and preprocessing | `IA2_report.md` §5, `dataset_provenance.md` | Complete |
| Implementation details | `IA2_report.md` §6, `methodology.md` | Complete |
| Methodology and metrics | `IA2_report.md` §7 | Complete |
| Results tables and graphs from real outputs | `IA2_report.md` §8; `results/metrics/full/*.csv`; `results/plots/full/*.png` (11 figures) | Complete |
| DP accounting and achieved parameters | `IA2_report.md` §8.2; `main_details.json` (`privacy` entries) | Complete |
| Privacy–utility interpretation | `IA2_report.md` §9 | Complete |
| Cross-module comparison | `IA2_report.md` §10 | Complete |
| Limitations and validity threats | `IA2_report.md` §11 | Complete |
| Evidence-based conclusion | `IA2_report.md` §12 | Complete |
| References (verifiable) | `IA2_report.md` §13; DOIs checked on Crossref | Complete |
| Reproducibility instructions | `IA2_report.md` §14, `README.md`, `configs/` | Complete |
| Individual-contribution statement | `IA2_report.md` §15 (Sai Anirudh, 16014223071) | Complete |

## Technical requirements from the project brief

| Requirement | Evidence | Status |
|---|---|---|
| CIFAR-10 as primary dataset (M1, M2, M5) | `src/data_loader.py`, `cifar10_provenance.json` | Complete |
| Conventional image classifier | `src/model.py::SmallCNN`, `train_baseline.py` | Complete |
| Members / non-members disjoint | `splits_seed*.json` (`true` for all seeds), `test_splits_are_disjoint_and_class_balanced` | Complete |
| Attack-train separate from attack-eval | `split_attack_sets`, `test_attack_sets_are_separate_and_balanced` | Complete |
| Random-guessing baseline | `random_guess_*` columns in `main_results.csv` | Complete |
| Check that the attack does not learn classes | Class-balanced splits, label-only AUC = 0.500 | Complete |
| Attack AUC, P/R/F1, TPR/FPR, advantage | `main_results.csv`, `IA2_report.md` §8.1 | Complete |
| DP-SGD with per-example clipping and noise (Opacus) | `src/train_private.py`, `test_dp_sgd_reaches_target_epsilon` | Complete |
| Accountant, achieved ε, δ, C, σ, batch, epochs, sampling | `main_details.json` privacy entries | Complete |
| Multiple privacy configurations | ε ≈ 1, 3, 8, 32 × 3 seeds | Complete |
| Same test set and same attack for both models | `run_experiment.py` | Complete |
| Training time / overhead | `train_time_sec`, `training_time.png` | Complete |
| Module 3: RR, several ε, estimation error | `randomized_response.*` | Complete |
| Module 4: QIs, generalization/suppression, configurable k, verification, perturbation, utility | `anonymization.py`, `k_anonymity.csv`, `perturbation.csv` | Complete |
| Module 5: baseline vs reduced representation, dimension count, performance | `compressive_privacy.csv` | Complete (single seed) |
| Required plots (accuracy, AUC, advantage, acc vs ε, AUC vs ε, time, RR error, k-anon, PCA) | `results/plots/full/` | Complete |
| Tests for the listed areas | `tests/test_privacyguard.py`: 14 passed | Complete |
| Smoke and full configs | `configs/smoke.json`, `configs/full.json` | Complete |
| README with exact commands | `README.md` (commands executed during the build) | Complete |
| Shadow-model attack | Not implemented (optional in the brief) | Not Complete |
| Streamlit dashboard | Not implemented (optional) | Not Complete |
| Privacy-accounted hyperparameter tuning | Documented as a limitation | Not Complete |
| Multi-seed Module 5 | Only seed 0 | Partial |

## Before submission

- Re-read the reports so that every claim can be defended in the viva.
