import json

import pandas as pd

from mlaas_data_generator.cli.manifest import hf_manifest_builder as builder
from mlaas_data_generator.cli.run_manifest import _resolve_row, _validate_row


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
    assert bool(df.iloc[0]["save_weights"]) is False
    assert bool(df.iloc[0]["save_final_model_params"]) is True


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


def test_manifest_requires_finetune_validated_multimodal_model(monkeypatch):
    models = {
        "demo_clip": {
            "hf_model_id": "org/demo-clip",
            "task_key": "text_image_retrieval",
            "modality": "multimodal",
            "allowed_run_regimes": ["finetune_transfer"],
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
            "manifest_validated": True,
            "max_samples": 256,
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_image_retrieval"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=17,
    )

    assert df.empty

    models["demo_clip"]["finetune_validated"] = True
    models["demo_clip"]["retrieval_positive_policy"] = "diagonal_in_batch"
    df = builder.build_hf_manifest(
        task_keys=["text_image_retrieval"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=17,
    )

    assert len(df) == 1
    assert df.iloc[0]["run_regime"] == "finetune_transfer"
    assert df.iloc[0]["retrieval_positive_policy"] == "diagonal_in_batch"


def test_manifest_marks_inference_only_training_knobs_not_applicable(monkeypatch):
    models = {
        "demo_text_model": {
            "hf_model_id": "org/demo-text-model",
            "task_key": "text_classification",
            "allowed_run_regimes": ["inference_only"],
            "dataset_keys": ["text_dataset"],
        }
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
            "max_samples": 128,
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=29,
        manifest_profile="test",
    )

    assert len(df) == 1
    row = df.iloc[0]
    for column in builder.INFERENCE_ONLY_TRAINING_COLUMNS:
        assert row[column] == builder.NOT_APPLICABLE
    assert row["batch_size"] != builder.NOT_APPLICABLE
    assert bool(row["save_weights"]) is False

    resolved = _resolve_row(row, {})
    assert resolved["local_epochs"] == 1
    assert resolved["learning_rate"] == 0.001
    assert resolved["optimizer"] == "adam"
    assert resolved["save_weights"] is False
    assert _validate_row(resolved).ok


def test_manifest_strict_inference_requires_training_dataset_metadata(monkeypatch):
    models = {
        "base_text_model": {
            "hf_model_id": "org/base-text-model",
            "task_key": "text_classification",
            "allowed_run_regimes": ["inference_only"],
            "dataset_keys": ["text_dataset"],
        },
        "trained_text_model": {
            "hf_model_id": "org/trained-text-model",
            "task_key": "text_classification",
            "allowed_run_regimes": ["inference_only"],
            "dataset_keys": ["text_dataset"],
            "inference_dataset_keys": ["text_dataset"],
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
            "max_samples": 128,
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=2,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=31,
        manifest_profile="test",
        strict_inference_dataset_match=True,
    )

    assert len(df) == 1
    assert df.iloc[0]["hf_model_id"] == "org/trained-text-model"
    assert "strict_inference_dataset_match:text_dataset" in str(df.iloc[0]["fit_reason"])


def test_manifest_strict_inference_can_use_audit_dataset_tags(monkeypatch, tmp_path):
    models = {
        "audit_tagged_model": {
            "hf_model_id": "org/audit-tagged-model",
            "task_key": "text_classification",
            "allowed_run_regimes": ["inference_only"],
            "dataset_keys": ["text_dataset"],
        }
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
            "max_samples": 128,
        }
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "pipeline_tag": "text-classification",
                        "models": [
                            {
                                "hf_model_id": "org/audit-tagged-model",
                                "audit_dataset_tags": ["text-dataset"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        json_path=str(audit_path),
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=37,
        manifest_profile="test",
        strict_inference_dataset_match=True,
    )

    assert len(df) == 1
    assert df.iloc[0]["hf_model_id"] == "org/audit-tagged-model"


def test_manifest_filters_invalid_seq2seq_single_text_pairs(monkeypatch):
    models = {
        "demo_t5": {
            "hf_model_id": "org/demo-t5",
            "task_key": "text2text_generation",
            "allowed_run_regimes": ["inference_only"],
            "dataset_keys": ["bad_seq2seq"],
        }
    }
    datasets = {
        "bad_seq2seq": {
            "task_key": "text2text_generation",
            "dataset_name": "wikitext",
            "text_column": "text",
            "label_column": "text",
            "loader_template": "hf_seq2seq",
            "max_samples": 128,
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text2text_generation"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=17,
        manifest_profile="test",
    )

    assert df.empty

    datasets["bad_seq2seq"]["label_column"] = "summary"
    df = builder.build_hf_manifest(
        task_keys=["text2text_generation"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=17,
        manifest_profile="test",
    )

    assert len(df) == 1
    assert df.iloc[0]["text_column"] == "text"
    assert df.iloc[0]["label_column"] == "summary"


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
        dataset_split_variants_per_pair=3,
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
    assert any("__split2__svc0" in case_name for case_name in df["case_name"])


def test_manifest_service_variants_keep_dataset_split_fixed(monkeypatch):
    models = {
        "demo_text_model": {
            "hf_model_id": "org/demo-text-model",
            "task_key": "text_classification",
            "family": "bert",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["text_dataset"],
        }
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
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=3,
        dataset_split_variants_per_pair=1,
        seed=41,
        manifest_profile="test",
    )

    assert len(df) == 3
    assert set(df["train_split"]) == {"train"}
    assert set(df["test_split"]) == {"validation"}
    assert {json.loads(value)["service_variant_index"] for value in df["hf_service_meta_json"]} == {0, 1, 2}
    assert {json.loads(value)["split_variant_index"] for value in df["hf_service_meta_json"]} == {0}
    assert {name.rsplit("__", 1)[-1] for name in df["case_name"]} == {"svc0", "svc1", "svc2"}


def test_manifest_prefers_distinct_families_before_duplicate_families(monkeypatch):
    models = {
        "bert_large": {
            "hf_model_id": "org/bert-large",
            "task_key": "text_classification",
            "family": "bert",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["text_dataset"],
            "downloads": 100000,
        },
        "bert_base": {
            "hf_model_id": "org/bert-base",
            "task_key": "text_classification",
            "family": "bert",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["text_dataset"],
            "downloads": 90000,
        },
        "roberta_base": {
            "hf_model_id": "org/roberta-base",
            "task_key": "text_classification",
            "family": "roberta",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["text_dataset"],
            "downloads": 10,
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
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=2,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=17,
        manifest_profile="test",
    )

    assert len(df) == 2
    assert set(df["hf_model_id"]) == {"org/bert-large", "org/roberta-base"}


def test_manifest_respects_max_models_per_family_cap(monkeypatch):
    models = {
        "bert_large": {
            "hf_model_id": "org/bert-large",
            "task_key": "text_classification",
            "family": "bert",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["text_dataset"],
        },
        "bert_base": {
            "hf_model_id": "org/bert-base",
            "task_key": "text_classification",
            "family": "bert",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["text_dataset"],
        },
        "roberta_base": {
            "hf_model_id": "org/roberta-base",
            "task_key": "text_classification",
            "family": "roberta",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["text_dataset"],
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
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=3,
        max_models_per_family=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=19,
        manifest_profile="test",
    )

    selected_model_ids = set(df["hf_model_id"])
    assert len(df) == 2
    assert "org/roberta-base" in selected_model_ids
    assert len(selected_model_ids & {"org/bert-large", "org/bert-base"}) == 1
