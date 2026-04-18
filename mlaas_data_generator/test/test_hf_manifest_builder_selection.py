import pandas as pd

from mlaas_data_generator.cli.manifest import hf_manifest_builder as builder


def test_manifest_prefers_dataset_closer_to_requested_average(monkeypatch):
    models = {
        "demo_text_model": {
            "hf_model_id": "org/demo-text-model",
            "task_key": "text_classification",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["oversized_dataset", "target_dataset"],
            "explainability": {"supported": True},
        }
    }
    datasets = {
        "oversized_dataset": {
            "task_key": "text_classification",
            "dataset_name": "oversized-dataset",
            "text_column": "text",
            "label_column": "label",
            "loader_template": "hf_text_classification",
            "max_samples": 1600,
            "realism_score": 0.9,
        },
        "target_dataset": {
            "task_key": "text_classification",
            "dataset_name": "target-dataset",
            "text_column": "text",
            "label_column": "label",
            "loader_template": "hf_text_classification",
            "max_samples": 192,
            "realism_score": 1.4,
        },
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=11,
        manifest_profile="test",
        avg_sample_size=160,
    )

    assert len(df) == 1
    assert df.iloc[0]["dataset_name"] == "target-dataset"
    assert df.iloc[0]["fit_decision"] == "selected"
    assert "selected score=" in str(df.iloc[0]["fit_reason"])


def test_manifest_skips_unvalidated_high_risk_multimodal_pairs(monkeypatch):
    models = {
        "demo_clip": {
            "hf_model_id": "org/demo-clip",
            "task_key": "text_image_retrieval",
            "modality": "multimodal",
            "allowed_run_regimes": ["inference_only"],
            "dataset_keys": ["retrieval_dataset"],
        }
    }
    datasets = {
        "retrieval_dataset": {
            "task_key": "text_image_retrieval",
            "dataset_name": "retrieval-dataset",
            "image_column": "image",
            "text_column": "caption",
            "loader_template": "hf_image_text_retrieval",
            "max_samples": 256,
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_image_retrieval"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=7,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty

    datasets["retrieval_dataset"]["manifest_validated"] = True
    df = builder.build_hf_manifest(
        task_keys=["text_image_retrieval"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=7,
    )

    assert len(df) == 1
    assert df.iloc[0]["run_regime"] == "inference_only"


def test_manifest_shapes_average_sample_size_and_realistic_finetune_workload(monkeypatch):
    models = {
        "demo_text_model": {
            "hf_model_id": "org/demo-text-model",
            "task_key": "text_classification",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["dataset_a", "dataset_b"],
        }
    }
    datasets = {
        "dataset_a": {
            "task_key": "text_classification",
            "dataset_name": "dataset-a",
            "text_column": "text",
            "label_column": "label",
            "loader_template": "hf_text_classification",
            "max_samples": 1000,
        },
        "dataset_b": {
            "task_key": "text_classification",
            "dataset_name": "dataset-b",
            "text_column": "text",
            "label_column": "label",
            "loader_template": "hf_text_classification",
            "max_samples": 600,
        },
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=2,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=23,
        manifest_profile="benchmark",
        avg_sample_size=200,
    )

    assert len(df) == 2
    assert df["max_samples"].mean() >= 160
    assert df["max_samples"].mean() <= 240
    assert set(df["num_rounds"]).issubset({3, 4, 5})
    assert set(df["local_epochs"]).issubset({1, 2})
    assert (df["num_clients"] >= 1).all()
    assert (df["num_clients"] <= df["max_samples"]).all()


def test_manifest_total_runs_are_balanced_across_requested_tasks_and_cycle_variants(monkeypatch):
    models = {
        "demo_text_model": {
            "hf_model_id": "org/demo-text-model",
            "task_key": "text_classification",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["text_dataset"],
        },
        "demo_token_model": {
            "hf_model_id": "org/demo-token-model",
            "task_key": "token_classification",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["token_dataset"],
        },
    }
    datasets = {
        "text_dataset": {
            "task_key": "text_classification",
            "dataset_name": "text-dataset",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "text",
            "label_column": "label",
            "loader_template": "hf_text_classification",
            "max_samples": 256,
        },
        "token_dataset": {
            "task_key": "token_classification",
            "dataset_name": "token-dataset",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "tokens",
            "label_column": "ner_tags",
            "loader_template": "hf_token_classification",
            "max_samples": 256,
        },
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_classification", "token_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        total_runs=8,
        seed=31,
        manifest_profile="test",
        avg_sample_size=128,
    )

    assert len(df) == 8
    assert df.groupby("hf_task").size().to_dict() == {
        "sequence_classification": 4,
        "token_classification": 4,
    }
    assert {"train[:80%]", "train[:90%]"}.issubset(set(df["train_split"]))
    assert {"train[80%:]", "train[90%:]"}.issubset(set(df["test_split"]))
    assert any("__v3" in case_name for case_name in df["case_name"])
