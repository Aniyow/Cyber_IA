"""Runs every experiment for one config, in order.

    python -m src.run_all --config configs/smoke.json
    python -m src.run_all --config configs/full.json
"""
import argparse

from src import anonymization, compressive_privacy, make_plots, provenance, randomized_response, run_experiment
from src.utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    print("== Modules 1 and 2: standard vs DP-SGD + membership inference")
    run_experiment.main(args.config)
    make_plots.main(args.config)
    provenance.main()
    print("== Module 3: randomized response")
    randomized_response.run(cfg)
    print("== Module 4: anonymization, k-anonymity, perturbation")
    anonymization.run(cfg)
    print("== Module 5: PCA feature reduction")
    compressive_privacy.run(cfg)
