import numpy as np

from mlaas_data_generator.federated.orchestrator import FederatedDataGenerator
from mlaas_data_generator.federated.perturbation import run_perturbation_stage
from mlaas_data_generator.federated.strategies.base import TaskStrategy
from mlaas_data_generator.federated.strategies.keras_strategy import _generic_runtime_metrics
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
    "explainability_supported_flag",
    "explainability_task_family",
    "explainability_method",
    "explainability_quality_metric",
    "explainability_budget_fractions",
    "explainability_targeted_degradation_mean",
    "explainability_random_degradation_mean",
    "explainability_self_faithfulness_score",
    "explainability_self_faithfulness_score_p10",
    "explainability_semantic_supported_flag",
    "explainability_semantic_supported_rate",
    "explainability_semantic_target_source",
    "explainability_semantic_degradation_mean",
    "explainability_semantic_random_degradation_mean",
    "explainability_semantic_behavior_score",
    "explainability_semantic_behavior_score_p10",
    "explainability_semantic_sensitivity_score",
    "explainability_semantic_sensitivity_score_p10",
    "explainability_semantic_selectivity_score",
    "explainability_semantic_selectivity_score_p10",
    "explainability_semantic_alignment_score",
    "explainability_semantic_alignment_score_p10",
    "explainability_meaningful_drop_threshold",
    "explainability_selectivity_floor",
]


class _FeatureModel:
    def predict(self, x, verbose=0):
        arr = np.asarray(x, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        p1 = 1.0 / (1.0 + np.exp(-arr[:, 0]))
        return np.stack([1.0 - p1, p1], axis=1)


class _NoConfidenceModel:
    def predict(self, x, verbose=0):
        arr = np.asarray(x, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, 0]


class _KeywordTextModel:
    def predict(self, x, verbose=0):
        texts = x if isinstance(x, list) else [x]
        rows = []
        for text in texts:
            lowered = str(text).lower()
            p1 = 0.9 if "dog" in lowered and "legs" in lowered else 0.2
            rows.append([1.0 - p1, p1])
        return np.asarray(rows, dtype="float32")


class _DummyStrategy(TaskStrategy):
    def task_type(self):
        return "classification"


def test_strategy_perturbation_metrics_run_only_on_final_round_by_default(monkeypatch):
    calls = []

    def _fake_perturbation_stage(*args, **kwargs):
        calls.append(kwargs)
        return {"perturbation_supported_flag": True, "trust_score": 0.75}

    monkeypatch.setattr(
        "mlaas_data_generator.federated.perturbation.run_perturbation_stage",
        _fake_perturbation_stage,
    )

    strategy = _DummyStrategy(
        meta={},
        knobs={"num_rounds": 3},
        config={"enable_perturbation_metrics": True},
        x_test=np.asarray([[1.0]], dtype="float32"),
        y_test=np.asarray([1], dtype="int64"),
        metric_key="accuracy",
        save_weights=False,
    )

    assert strategy.perturbation_metrics(_FeatureModel(), client_id="client_1", round_idx=1) == {}
    assert calls == []

    metrics = strategy.perturbation_metrics(_FeatureModel(), client_id="client_1", round_idx=3)
    assert metrics["trust_score"] == 0.75
    assert len(calls) == 1
    assert calls[0]["client_id"] == "client_1"
    assert calls[0]["round_idx"] == 3


def test_generic_runtime_metrics_keep_trust_fields_final_round_only():
    metrics = _generic_runtime_metrics(
        _FeatureModel(),
        eval_latency_s=0.01,
        metric_score=0.9,
        loss=0.2,
        include_trust_metrics=False,
    )

    assert metrics["inference_latency_s"] == 0.01
    assert "trust_score" not in metrics
    assert "explainability_score" not in metrics
    assert "explainability_supported_flag" not in metrics


def test_orchestrator_drops_final_only_trust_metrics_from_non_final_rows():
    generator = object.__new__(FederatedDataGenerator)
    generator.config = {"perturbation_final_round_only": True}
    generator.knobs = {"num_rounds": 2}

    values = {
        "accuracy": 0.8,
        "loss": 0.4,
        "trust_score": 0.9,
        "explainability_score": 0.7,
        "perturbation_supported_flag": True,
        "robustness_score": 0.6,
    }

    non_final = generator._drop_non_final_trust_metrics(values, round_idx=1)
    assert non_final == {"accuracy": 0.8, "loss": 0.4}
    assert generator._drop_non_final_trust_metrics(values, round_idx=2) == values


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
    assert metrics["explainability_supported_flag"] is True
    assert metrics["explainability_task_family"] == "classification"
    assert metrics["explainability_method"] == "semantic_sensitivity_faithfulness_v1"
    assert metrics["explainability_quality_metric"] == "class_confidence"
    assert metrics["explainability_budget_fractions"] == [0.05, 0.1, 0.2]
    assert metrics["perturbation_sample_count"] == 2
    assert metrics["explainability_confidence_drop_mean"] > 0.0
    assert 0.0 <= metrics["explainability_score"] <= 1.0
    assert metrics["explainability_score"] > 0.0
    assert metrics["explainability_self_faithfulness_score"] > 0.0
    assert metrics["explainability_targeted_degradation_mean"] > metrics["explainability_random_degradation_mean"]
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
    assert "explainability_budget_scores" in metrics["perturbation_samples"][0]
    assert "explainability_self_faithfulness_budget_scores" in metrics["perturbation_samples"][0]


def test_task_faithfulness_score_uses_prediction_stability_when_confidence_is_unavailable():
    x_eval = np.asarray(
        [
            [9.0, 0.0, 0.0, 0.0],
            [8.0, 0.1, 0.1, 0.1],
        ],
        dtype="float32",
    )

    metrics = run_perturbation_stage(
        _NoConfidenceModel(),
        x_eval,
        task_family="generation",
        config={
            "seed": 11,
            "perturbation_sample_count": 2,
            "perturbation_candidate_units": 4,
            "perturbation_target_units": 1,
            "perturbation_trust_trials": 2,
            "explainability_random_trials": 4,
        },
        client_id="client_1",
        round_idx=1,
    )

    assert metrics["perturbation_supported_flag"] is True
    assert metrics["explainability_supported_flag"] is True
    assert metrics["explainability_quality_metric"] == "prediction_stability"
    assert 0.0 <= metrics["explainability_score"] <= 1.0
    assert metrics["explainability_score"] > 0.0


def test_semantic_text_units_contribute_to_hybrid_explainability_score():
    metrics = run_perturbation_stage(
        _KeywordTextModel(),
        ["How many legs does a dog have"],
        np.asarray([1], dtype="int64"),
        task_family="classification",
        config={
            "seed": 23,
            "perturbation_sample_count": 1,
            "perturbation_candidate_units": 8,
            "perturbation_target_units": 1,
            "perturbation_trust_trials": 1,
            "explainability_random_trials": 3,
            "explainability_meaningful_drop_threshold": 0.2,
        },
        client_id="client_1",
        round_idx=1,
    )

    assert metrics["perturbation_supported_flag"] is True
    assert metrics["explainability_semantic_supported_flag"] is True
    assert metrics["explainability_semantic_target_source"] == "content_word_heuristic"
    assert metrics["explainability_semantic_degradation_mean"] > 0.0
    assert metrics["explainability_semantic_sensitivity_score"] > 0.0
    assert 0.0 <= metrics["explainability_semantic_alignment_score"] <= 1.0
    assert 0.0 <= metrics["explainability_self_faithfulness_score"] <= 1.0
    assert 0.0 <= metrics["explainability_score"] <= 1.0
    sample = metrics["perturbation_samples"][0]
    assert sample["explainability_semantic_supported"] is True
    assert sample["explainability_semantic_sensitivity_budget_scores"]


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
