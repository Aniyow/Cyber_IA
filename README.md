# PrivacyGuard ML

**Privacy-Preserving Techniques in Machine Learning: Membership Inference Attacks, Differential Privacy, and Privacy–Utility Trade-offs**

Course project for *Security and Privacy for Machine Learning* (IA-1 and IA-2).

> Machine learning models may reveal whether an individual's data was used during training. This project investigates membership inference attacks and evaluates whether differential privacy can reduce this risk while maintaining model accuracy, using a public dataset and Python-based experiments.

| Module | What is implemented | Code |
|---|---|---|
| 1. Privacy Considerations in Machine Learning | CIFAR-10 CNN, loss- and confidence-threshold membership inference, random and label-only baselines | `src/membership_inference.py`, `src/train_baseline.py` |
| 2. Differential Privacy for Machine Learning | DP-SGD with Opacus (per-example clipping, Gaussian noise, RDP accountant), 4 privacy levels × 3 seeds | `src/train_private.py`, `src/run_experiment.py` |
| 3. Local Differential Privacy for Machine Learning | Randomized response on a synthetic survey, unbiased estimation, error vs ε and n | `src/randomized_response.py` |
| 4. Privacy-Preserving Data Mining Techniques | Generalization + suppression k-anonymity with verification, Laplace perturbation, utility and linkage checks | `src/anonymization.py` |
| 5. Compressive Privacy for Machine Learning | PCA on CIFAR-10 pixels, accuracy / reconstruction / membership attack vs retained dimensions | `src/compressive_privacy.py` |

Reports are in `reports/`: `IA1_report.md`, `IA2_report.md`, `methodology.md`, `dataset_provenance.md` and `rubric_checklist.md`.

## Project layout

```
configs/         smoke.json (quick check) and full.json (reported results)
data/            CIFAR-10 is downloaded here (git-ignored)
src/             all code; each experiment script can be run with python -m src.<name>
tests/           pytest tests
results/metrics/<run>/   CSV / JSON outputs
results/plots/<run>/     PNG figures
results/logs/<run>/      training logs
results/checkpoints/     trained models (git-ignored)
results/archive/         first full run (v1), IA-1 smoke snapshot, DP tuning notes
reports/         IA-1, IA-2, methodology, provenance, rubric checklist
```

### What is in the repository

The code, configs, reports, metric files, plots and logs are committed. The repository does **not** contain:

- **CIFAR-10** (`data/`, about 340 MB). It downloads automatically from the official source on the first run (section 2).
- **Trained model checkpoints** (`results/checkpoints/`, about 54 MB). They are recreated by running the experiments (section 4).
- **The virtual environment** (`.venv/`) and Python caches.

## 1. Environment setup (Windows, PowerShell)

These steps match the machine the results were produced on: Windows 11, Python 3.12.5, an NVIDIA RTX 5050 Laptop GPU and a PyTorch build with CUDA already installed system-wide. Run them from the project folder (the folder that contains this README, e.g. after cloning the repository and running `cd privacyguard-ml`):

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps opacus==1.6.0 opt_einsum
python -m pip install pytest
```

`--system-site-packages` reuses the already installed `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `scipy` and `matplotlib`. `--no-deps` stops pip from trying to replace the installed PyTorch build.

On a fresh machine without PyTorch, install the right build for your hardware from https://pytorch.org/get-started/locally/ first, then:

```powershell
python -m pip install -r requirements.txt
```

Everything also runs on CPU, but the full DP-SGD run will be much slower.

## 2. Dataset preparation

CIFAR-10 (about 163 MB) downloads automatically the first time any CIFAR script runs. To download it and record hashes explicitly:

```powershell
python -c "from src.data_loader import load_cifar10; load_cifar10()"
python -m src.provenance
```

`src.provenance` writes `results/metrics/cifar10_provenance.json` and prints whether the archive MD5 matches the value torchvision expects.

## 3. Quick smoke test (about 1–2 minutes)

```powershell
python -m pytest -q
python -m src.run_all --config configs/smoke.json
```

The smoke config uses 1,000 members, 3 epochs and one DP setting. Its numbers only show that the pipeline works and are **not** results.

## 4. Full experiment (about 25 minutes on the RTX 5050)

```powershell
python -m src.run_all --config configs/full.json
```

This runs, in order: the main study (standard model plus DP-SGD at target ε = 1, 3, 8, 32; seeds 0, 1, 2), the plots and summary table, the provenance record, and Modules 3, 4 and 5.

## 5. Running parts individually

All commands take `--config configs/smoke.json` or `--config configs/full.json`.

```powershell
# Module 1 / 2: one standard model (saves results/checkpoints/<run>/baseline_seed0.pt)
python -m src.train_baseline --config configs/full.json --seed 0

# Module 2: one DP-SGD model at a target epsilon (saves dp_eps8_seed0.pt)
python -m src.train_private --config configs/full.json --epsilon 8 --seed 0

# Module 1: membership inference against a saved model (use the same seed it was trained with)
python -m src.membership_inference --config configs/full.json --checkpoint baseline_seed0 --seed 0
python -m src.membership_inference --config configs/full.json --checkpoint dp_eps8_seed0 --seed 0

# Whole main study (all seeds and privacy levels), then summary table and plots
python -m src.run_experiment --config configs/full.json
python -m src.make_plots --config configs/full.json

# Modules 3, 4, 5
python -m src.randomized_response --config configs/full.json
python -m src.anonymization --config configs/full.json
python -m src.compressive_privacy --config configs/full.json
```

## 6. Outputs

For run `<run>` (`smoke` or `full`):

| File | Contents |
|---|---|
| `results/metrics/<run>/run_info.json` | config, device, Python and package versions |
| `results/metrics/<run>/splits_seed<k>.json` | split sizes, class counts, disjointness checks, index hashes |
| `results/metrics/<run>/main_results.csv` | one row per model and seed: utility, attack and privacy metrics, training time |
| `results/metrics/<run>/main_summary.csv` | mean and std across seeds (input for the plots) |
| `results/metrics/<run>/main_details.json` | per-epoch histories, full attack results, ROC points, DP parameters |
| `results/metrics/<run>/randomized_response.{csv,json}` | Module 3 |
| `results/metrics/<run>/k_anonymity.csv`, `perturbation.csv`, `anonymization.json` | Module 4 |
| `results/metrics/<run>/compressive_privacy.{csv,json}` | Module 5 |
| `results/plots/<run>/*.png` | all figures |
| `results/logs/<run>/main_experiment.log` | training log |

## 7. Notes on validity

- Members, non-members and validation images are disjoint and class-balanced. The test set is the official CIFAR-10 test set and is used only to measure accuracy.
- The attack threshold is chosen on attack-train and all attack metrics are reported on the separate attack-eval set.
- The reported ε is read from Opacus' RDP accountant after training, with δ = 1e-5.
- The DP learning rate (0.25) was chosen from {0.1, 0.25, 0.5, 1.0} by validation accuracy at ε = 8. This tuning was not privacy-accounted (see Limitations in the IA-2 report).
