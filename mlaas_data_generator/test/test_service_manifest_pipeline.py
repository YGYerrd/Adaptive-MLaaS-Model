import json

import pandas as pd

from mlaas_data_generator.cli.manifest.hf_manifest_builder import MANIFEST_COLUMNS, build_hf_manifest
from mlaas_data_generator.cli.run_manifest import _resolve_row, _validate_row, run_manifest


def test_manifest_builder_emits_service_rows_without_federated_columns():
    df = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        training_regimes=["finetune_transfer", "inference_only"],
        knob_variants_per_pair=2,
        total_services=4,
        seed=123,
        manifest_profile="test",
    )

    assert not df.empty
    assert list(df.columns) == MANIFEST_COLUMNS
    assert df["service_id"].is_unique
    assert {"service_id", "service_config", "training_regime", "dataset_variant", "split_variant", "knob_variant"}.issubset(df.columns)
    assert not {"num_rounds", "client_participation_rate", "aggregation"}.intersection(df.columns)
    assert "num_clients" not in df.columns
    assert set(df["training_regime"]).issubset({"finetune_transfer", "inference_only"})
    assert set(df["resource_tier"]) == {"light"}
    for payload in df["service_config"]:
        assert isinstance(json.loads(payload), dict)


def test_manifest_resource_tier_caps_model_and_workload_size():
    df = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=3,
        datasets_per_model=2,
        training_regimes=["finetune_transfer"],
        resource_tier="light",
        knob_variants_per_pair=2,
        seed=123,
    )

    assert not df.empty
    assert set(df["resource_tier"]) == {"light"}
    assert set(df["model_resource_tier"]) == {"light"}
    assert df["max_samples"].max() <= 128
    assert df["max_length"].dropna().max() <= 96


def test_manifest_knob_variants_are_task_aware_and_distinct():
    df = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        training_regimes=["finetune_transfer"],
        resource_tier="medium",
        knob_variants_per_pair=4,
        seed=123,
    )

    assert list(df["knob_variant"]) == [0, 1, 2, 3]
    assert set(df["optimizer"]) == {"adamw"}
    assert len(set(df["batch_size"])) > 1
    assert set(df["learning_rate"]).issubset({2e-5, 3e-5, 5e-5, 1e-4})
    for payload in df["service_config"]:
        config = json.loads(payload)
        assert config["resource_tier"] == "medium"
        assert config["max_train_time_s"] == 120


def test_manifest_inference_uses_explicit_dataset_matches():
    df = build_hf_manifest(
        task_keys=["object_detection"],
        models_per_task=3,
        datasets_per_model=5,
        training_regimes=["inference_only"],
        resource_tier="medium",
        seed=123,
    )

    assert not df.empty
    assert set(df["training_regime"]) == {"inference_only"}
    assert set(df["dataset_name"]) == {"detection-datasets/coco"}
    assert set(df["learning_rate"].dropna()) == set()


def test_manifest_service_ids_are_deterministic():
    first = build_hf_manifest(task_keys=["text_classification"], models_per_task=1, datasets_per_model=1, total_services=2, seed=99)
    second = build_hf_manifest(task_keys=["text_classification"], models_per_task=1, datasets_per_model=1, total_services=2, seed=99)
    pd.testing.assert_series_equal(first["service_id"], second["service_id"])


def test_run_manifest_dry_run_validates_service_rows(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "service_id": "svc_manual",
                "enabled": True,
                "dataset": "hf",
                "model_type": "hf",
                "task_type": "classification",
                "hf_task": "sequence_classification",
                "hf_model_id": "distilbert-base-uncased",
                "dataset_name": "glue",
                "dataset_config": "sst2",
                "train_split": "train",
                "test_split": "validation",
                "text_column": "sentence",
                "label_column": "label",
                "training_regime": "inference_only",
                "batch_size": 4,
            }
        ]
    ).to_csv(manifest, index=False)

    results_path = run_manifest(str(manifest), dry_run=True, db_path=str(tmp_path / "services.db"))
    results = pd.read_csv(results_path)

    assert results.iloc[0]["service_id"] == "svc_manual"
    assert results.iloc[0]["status"] == "success"
    resolved = json.loads(results.iloc[0]["resolved_config_json"])
    assert resolved["benchmark_split"] == "validation"
    assert "num_rounds" not in resolved
    assert "num_clients" not in resolved


def test_resolved_manifest_row_gets_deterministic_service_id():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf",
            "task_type": "classification",
            "hf_task": "sequence_classification",
            "hf_model_id": "distilbert-base-uncased",
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "training_regime": "inference_only",
            "batch_size": 4,
        }
    )
    resolved = _resolve_row(row, {})
    assert resolved["service_id"].startswith("classification_")
    assert _validate_row(resolved).ok


def test_run_manifest_rejects_legacy_federated_columns():
    row = pd.Series(
        {
            "service_id": "svc_bad",
            "dataset": "hf",
            "model_type": "hf",
            "task_type": "classification",
            "hf_task": "sequence_classification",
            "hf_model_id": "distilbert-base-uncased",
            "dataset_name": "glue",
            "training_regime": "inference_only",
            "batch_size": 4,
            "num_rounds": 2,
        }
    )

    validation = _validate_row(_resolve_row(row, {}))

    assert not validation.ok
    assert "Federated columns" in validation.error
