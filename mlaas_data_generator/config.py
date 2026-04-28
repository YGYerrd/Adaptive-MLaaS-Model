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
    "seed": 42,
    "num_clients": 20,
    "num_rounds": 10,
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.01,
    "hidden_layers": [16, 32, 64, 128],
    "activation": "relu",
    "dropout": 0.0,
    "weight_decay": 0.0,
    "optimizer": "adam",
    "distribution_type": "iid",
    "distribution_param": None,
    "sample_size": 200,
    "sample_frac": None,
    "client_dropout_rate": 0.0,
    "task_type": "classification",
    "distribution_bins": 10,
    "dataset_args": None,
    "early_stopping_patience": None,
    "save_weights": False,
    "save_final_model_params": None,
    "final_model_params_dir": os.path.join("outputs", "final_model_params"),
    "update_signature_enabled": True,
    "update_signature_dim": 256,
    "update_signature_dir": None,
    "update_signature_max_source_elements": None,
    "enable_perturbation_metrics": True,
    "perturbation_final_round_only": True,
    "perturbation_sample_count": 10,
    "perturbation_candidate_units": 16,
    "perturbation_trust_trials": 5,
    "perturbation_target_units": 2,
    "perturbation_random_strength": 0.02,
    "perturbation_progress_logging": True,
    "perturbation_progress_sample_interval": 1,
    "explainability_random_trials": 8,
    "explainability_budget_fractions": [0.1, 0.2, 0.3],
    "explainability_meaningful_drop_threshold": 0.2,
    "explainability_selectivity_floor": 0.5,
}
