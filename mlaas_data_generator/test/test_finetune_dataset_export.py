import json

import numpy as np
import pandas as pd

from mlaas_data_generator.cli.cmd_export_finetune_dataset import (
    SERVICE_REQUIRED_COLUMNS,
    _normalise_service_rows,
    generate_compositions,
    generate_service_requests,
)


def _services() -> pd.DataFrame:
    rows = []
    for idx in range(8):
        task = "classification" if idx < 4 else "regression"
        rows.append(
            {
                "service_id": f"svc_{idx}",
                "run_id": f"run_{idx}",
                "task_family": task,
                "model_type": "hf_finetune" if idx % 2 == 0 else "mlp",
                "modality": "text" if idx % 2 == 0 else "tabular",
                "metric_score": 0.62 + idx * 0.03,
                "latency": 0.08 + idx * 0.01,
                "data_volume": 100 + idx * 10,
                "resource_cost_score": 0.2 + idx * 0.02,
                "data_distribution": "iid" if idx % 2 == 0 else "dirichlet",
                "model_update_signature": 1.0 + idx,
                "computation_time": 0.5 + idx * 0.1,
                "batch_size": 8 + (idx % 3),
                "reliability_score": 0.8,
                "trust_score": 0.75 + idx * 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_request_and_composition_generation_is_deterministic_and_valid():
    services = _services()
    requests = generate_service_requests(services, request_count=6, seed=123)
    second_requests = generate_service_requests(services, request_count=6, seed=123)
    compositions = generate_compositions(services, requests, candidates_per_request=4, seed=123)

    pd.testing.assert_frame_equal(requests, second_requests)
    assert len(requests) == 6
    assert len(compositions) == 24

    valid_service_ids = set(services["service_id"])
    for _, row in compositions.iterrows():
        service_ids = json.loads(row["service_ids"])
        assert set(service_ids).issubset(valid_service_ids)
        assert row["workflow_length"] == len(service_ids)

    for component in ["dhs", "mus", "shs", "ses", "hsq", "srs", "composability_score"]:
        assert np.isfinite(compositions[component]).all()
        assert ((compositions[component] >= 0.0) & (compositions[component] <= 1.0)).all()

    duplicate_sets = compositions.groupby("request_id")["service_ids"].apply(lambda values: values.duplicated().any())
    assert not duplicate_sets.any()

    for request_id, group in compositions.groupby("request_id"):
        selected = group[group["selected_flag"]]
        assert len(selected) == 1
        assert selected.iloc[0]["penalty_adjusted_score"] == group["penalty_adjusted_score"].max()


def test_normalised_service_rows_make_unsupported_fields_explicit():
    raw = pd.DataFrame(
        [
            {
                "run_id": "run_generic",
                "created_at": "2026-04-27 00:00:00",
                "dataset": "synthetic",
                "task_type": "regression",
                "model_type": "mlp",
                "Primary metric name": "rmse",
                "Primary metric": "0.5",
                "Auxiliary metric name": "mae",
                "Auxiliary metric": "0.3",
                "Latency": "Not Available",
                "Tail latency": "Not Available",
                "Participation rate": "1.0",
                "Reliability score": "1.0",
                "Mean compute time": "2.5",
                "Resource cost score": "0.4",
                "Cost efficiency": "1.2",
                "Model size": "512",
                "Downloads": "Not Available",
                "Likes": "Not Available",
                "Learning rate": "0.001",
                "Batch size": "16",
                "Data distribution": "iid",
                "Dataset distributions": "{}",
                "Dataset size": "120",
                "Explainability score": "Not Available",
            }
        ]
    )

    services = _normalise_service_rows(raw)
    row = services.iloc[0]

    assert set(SERVICE_REQUIRED_COLUMNS).issubset(set(services.columns))
    assert row["run_regime"] == "generic"
    assert bool(row["weights_exported"]) is False
    assert bool(row["latency_supported"]) is False
    assert row["latency"] == row["computation_time"] == 2.5
    assert bool(row["explainability_supported"]) is False
    assert bool(row["trust_supported"]) is False
    assert "latency_proxy" in row["missing_runtime_fields"]
