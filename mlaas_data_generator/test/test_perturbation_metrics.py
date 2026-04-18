import numpy as np

from mlaas_data_generator.federated.perturbation import run_perturbation_stage
from mlaas_data_generator.storage.writer import SQLiteWriter


_NEW_PERTURBATION_METRICS = [
    "trust_confidence_delta_std",
    "trust_confidence_delta_p95",
    "trust_confidence_delta_max",
    "trust_prediction_stability_min",
    "trust_score_p05",
    "trust_score_min",
    "explainability_confidence_drop_std",
    "explainability_confidence_drop_p50",
    "explainability_confidence_drop_p10",
    "explainability_confidence_drop_p90",
    "explainability_unit_fraction_p95",
    "explainability_score_p10",
]


class _FeatureModel:
    def predict(self, x, verbose=0):
        arr = np.asarray(x, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        p1 = 1.0 / (1.0 + np.exp(-arr[:, 0]))
        return np.stack([1.0 - p1, p1], axis=1)


def test_perturbation_stage_returns_explainability_and_trust_scores():
    x_eval = np.asarray(
        [
            [8.0, 0.1, -0.1],
            [7.0, 0.0, 0.2],
        ],
        dtype="float32",
    )
    y_eval = np.asarray([1, 1], dtype="int64")

    metrics = run_perturbation_stage(
        _FeatureModel(),
        x_eval,
        y_eval,
        task_family="classification",
        config={
            "seed": 7,
            "perturbation_sample_count": 2,
            "perturbation_candidate_units": 3,
            "perturbation_target_units": 1,
            "perturbation_trust_trials": 2,
            "perturbation_random_strength": 0.001,
        },
        client_id="client_1",
        round_idx=1,
    )

    assert metrics["perturbation_enabled_flag"] is True
    assert metrics["perturbation_supported_flag"] is True
    assert metrics["perturbation_sample_count"] == 2
    assert metrics["explainability_confidence_drop_mean"] > 0.0
    assert 0.0 <= metrics["explainability_score"] <= 1.0
    assert 0.0 <= metrics["explainability_score_p10"] <= metrics["explainability_score"] <= 1.0
    assert 0.0 <= metrics["trust_score"] <= 1.0
    assert 0.0 <= metrics["trust_score_min"] <= metrics["trust_score_p05"] <= 1.0
    assert metrics["trust_confidence_delta_max"] >= metrics["trust_confidence_delta_p95"] >= 0.0
    assert metrics["trust_confidence_delta_std"] >= 0.0
    assert metrics["explainability_confidence_drop_p10"] <= metrics["explainability_confidence_drop_p50"]
    assert metrics["explainability_confidence_drop_p50"] <= metrics["explainability_confidence_drop_p90"]
    assert metrics["explainability_confidence_drop_std"] >= 0.0
    assert 0.0 <= metrics["trust_prediction_stability_min"] <= 1.0
    assert 0.0 <= metrics["explainability_unit_fraction_p95"] <= 1.0
    for metric_name in _NEW_PERTURBATION_METRICS:
        assert metric_name in metrics
    assert isinstance(metrics["perturbation_samples"], list)
    assert "baseline_prediction" in metrics["perturbation_samples"][0]


def test_new_perturbation_metrics_are_seeded_with_domains(tmp_path):
    writer = SQLiteWriter(str(tmp_path / "metrics.db"))
    writer.start()
    try:
        writer.seed_metrics()
        rows = writer.conn.execute(
            f"""
            SELECT name, domain
            FROM metrics
            WHERE name IN ({",".join("?" for _ in _NEW_PERTURBATION_METRICS)})
            """,
            _NEW_PERTURBATION_METRICS,
        ).fetchall()
    finally:
        writer.finish()

    domains = dict(rows)
    assert set(domains) == set(_NEW_PERTURBATION_METRICS)
    assert domains["trust_score_min"] == "reliability"
    assert domains["explainability_score_p10"] == "quality"
    assert all(domain != "resource" for domain in domains.values())
