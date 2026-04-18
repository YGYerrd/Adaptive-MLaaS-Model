"""Configuration for the MLaaS data generator."""
from pathlib import Path

BASE_OUTPUT_DIR = Path("outputs")
RUNS_DIR = BASE_OUTPUT_DIR / "runs" 
MERGED_DIR = BASE_OUTPUT_DIR / "merged" 

import os
OVERRIDE = os.getenv("MLAAS_OUTDIR")
if OVERRIDE:
    BASE_OUTPUT_DIR = Path(OVERRIDE)
    RUNS_DIR = BASE_OUTPUT_DIR / "runs"
    MERGED_DIR = BASE_OUTPUT_DIR / "merged"

CONFIG = {
    "db_path": os.path.join("outputs", "federated.db"),
    "num_clients": 20,
    "num_rounds": 5,
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.01,
    "hidden_layers": [16, 32, 64, 128],
    "activation": "relu",
    "dropout": 0.0,
    "weight_decay": 0.0,
    "optimizer": "adam",
    "distribution_type": "iid",
    "distribution_param": None,
    "sample_size": 128,
    "sample_frac": None,
    "client_dropout_rate": 0.0,
    "task_type": "classification",
    "distribution_bins": 10,
    "dataset_args": None,
    "early_stopping_patience": None,
    "enable_perturbation_metrics": True,
    "perturbation_sample_count": 5,
    "perturbation_candidate_units": 12,
    "perturbation_trust_trials": 5,
    "perturbation_target_units": 2,
    "perturbation_random_strength": 0.02,
}
