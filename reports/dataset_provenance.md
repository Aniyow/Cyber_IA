# Dataset Provenance

Machine-readable record: `results/metrics/cifar10_provenance.json` (produced by `python -m src.provenance`).

## 1. CIFAR-10 (primary dataset: Modules 1, 2 and 5)

| Item | Value |
|---|---|
| Name | CIFAR-10 (python version) |
| Official page | https://www.cs.toronto.edu/~kriz/cifar.html (on 2026-10-22 this URL redirected to https://cave.cs.toronto.edu/kriz/cifar.html) |
| Creators | Alex Krizhevsky, Vinod Nair, Geoffrey Hinton (University of Toronto) |
| Reference to cite | A. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, Technical Report, University of Toronto, 2009 |
| License / usage terms | No license is stated on the official page. The page only says: "If you're going to use this dataset, please cite the tech report". A formal license is therefore **Unknown**. |
| Version / release identifier | No version number is published. The archive is identified by its MD5 `c58f30108f718f92721af3b95e74349a`, which matches the value on the official page (checked 2026-10-22). |
| Access method | `torchvision.datasets.CIFAR10(root="data", download=True)` (torchvision 0.30.0.dev20261006). torchvision downloads `cifar-10-python.tar.gz` and checks the same MD5 before extracting. |
| Download date | 2026-10-22 (archive written 18:31:48 local time, IST) |
| Archive | `data/cifar-10-python.tar.gz`, 170,498,071 bytes, SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |
| Original size | 60,000 RGB images, 32×32, uint8; 50,000 training (5,000 per class) and 10,000 test (1,000 per class), checked after loading |
| Classes / labels | 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck. Used directly as integer targets. |
| Origin of images | According to the official page, a labelled subset of the "80 million tiny images" collection |

### Extracted file hashes (SHA-256)

| File | Bytes | SHA-256 |
|---|---|---|
| data_batch_1 | 31,035,704 | 54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd |
| data_batch_2 | 31,035,320 | 766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8 |
| data_batch_3 | 31,035,999 | 0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9 |
| data_batch_4 | 31,035,696 | 3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b |
| data_batch_5 | 31,035,623 | d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154 |
| test_batch | 31,035,526 | f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831 |
| batches.meta | 158 | f962466ef690d46b226450fb9aadc74ba4bc64a76aa526b5827fe4bc5c7125cb |

### Preprocessing

1. Pixels converted from uint8 `[0, 255]` to float and scaled to `[-1, 1]` with the fixed formula `(x/255 − 0.5)/0.5` (`src/data_loader.py::preprocess`). No statistic is estimated from any split.
2. Channel order changed from HWC to CHW for PyTorch.
3. For Module 5 only: images flattened to 3,072-value vectors, then projected with PCA fitted on the member split only.
4. No augmentation, no deduplication and no relabelling.

### Splitting (processed sizes)

Per seed, `make_splits` samples **class-balanced** subsets with `numpy.random.default_rng(seed)`:

| Split | Source | Smoke config | Full config |
|---|---|---|---|
| Members (target training set) | official train | 1,000 | 10,000 |
| Non-members | official train, disjoint | 1,000 | 10,000 |
| Validation | official train, disjoint | 500 | 2,000 |
| Unused | official train | 47,500 | 28,000 |
| Test | official test | 1,000 (balanced subset) | 10,000 (all) |

Members and non-members are each split 50/50 per class into attack-train and attack-eval. Split sizes, class counts, disjointness flags and SHA-256 hashes of the index lists are saved per seed in `results/metrics/<run>/splits_seed<k>.json`.

Using a 10,000-image subset rather than all 50,000 training images is a deliberate choice. It keeps enough unused images to form a same-sized, same-distribution non-member set, and it keeps DP-SGD training to a few minutes per model on the available laptop GPU (RTX 5050, 8 GB).

### Known limitations and privacy considerations

- CIFAR-10 contains photographs of objects and animals, not personal data. In this project "membership" is a stand-in for whether an individual's record was in the training set. Leakage measured here shows the mechanism; it does not measure harm to real people.
- The test set is known to contain some near-duplicates of training images (Barz & Denzler, 2020). This could slightly inflate test accuracy, but it does not affect the attack, which only uses training-set images.
- The license is unknown (see above); use is limited to academic coursework with citation.

## 2. Synthetic survey data (Module 3)

Generated in `src/randomized_response.py` with `numpy.random.default_rng(seed)`. The data is binary answers (1 = "yes" to a sensitive question) with a configured true proportion of 0.3. Smoke config: 2,000 respondents; full config: 10,000, plus extra populations of 100 to 100,000 for the sample-size study. No real person is represented. The data is regenerated from the seed and not stored, except for the first 15 original and randomized answers, which are saved in `randomized_response.json` as an illustration.

## 3. Synthetic patient table (Module 4)

Generated in `src/anonymization.py::make_synthetic_patients`. Columns: age (18–85), zipcode (5 digits from five invented 3-digit prefixes), sex, diagnosis (5 categories; probabilities depend on age) and annual income (depends on age plus Gaussian noise). Smoke config: 500 records; full config: 2,000. The generated table is saved as `results/metrics/<run>/synthetic_patients.csv` so the anonymization can be inspected. No real record is represented.
