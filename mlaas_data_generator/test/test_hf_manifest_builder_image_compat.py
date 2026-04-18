import pandas as pd

from mlaas_data_generator.cli.manifest import hf_manifest_builder as builder


def _registry_pair(*, model_labels: int, dataset_labels: int):
    models = {
        "demo_imgcls_model": {
            "hf_model_id": "org/demo-image-cls",
            "task_key": "image_classification",
            "allowed_run_regimes": ["finetune_transfer", "inference_only"],
            "dataset_keys": ["demo_imgcls_dataset"],
            "inference_num_labels": model_labels,
        }
    }
    datasets = {
        "demo_imgcls_dataset": {
            "task_key": "image_classification",
            "dataset_name": "demo-dataset",
            "train_split": "train",
            "test_split": "validation",
            "image_column": "image",
            "label_column": "label",
            "loader_template": "hf_image_classification",
            "num_classes": dataset_labels,
            "max_samples": 8,
        }
    }
    return models, datasets


def test_manifest_skips_incompatible_image_classification_inference_pair(monkeypatch):
    models, datasets = _registry_pair(model_labels=1000, dataset_labels=3)
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["image_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=1,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_manifest_keeps_finetune_pair_even_when_inference_is_incompatible(monkeypatch):
    models, datasets = _registry_pair(model_labels=1000, dataset_labels=3)
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["image_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=1,
    )

    assert len(df) == 1
    assert set(df["run_regime"]) == {"finetune_transfer"}
    assert df.iloc[0]["device"] != "cpu"


def test_manifest_keeps_image_finetune_gpu_eligible(monkeypatch):
    models, datasets = _registry_pair(model_labels=1000, dataset_labels=3)
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["image_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=1,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["device"] != "cpu"
    assert row["mixed_precision"] == True
    assert "forced device=cpu" not in row["fit_reason"]


def test_manifest_keeps_compatible_image_classification_inference_pair(monkeypatch):
    models, datasets = _registry_pair(model_labels=3, dataset_labels=3)
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["image_classification"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=1,
    )

    assert len(df) == 1
    assert set(df["run_regime"]) == {"inference_only"}


def test_manifest_uses_safe_detection_finetune_knobs(monkeypatch):
    models = {
        "demo_objdet_model": {
            "hf_model_id": "org/demo-objdet",
            "task_key": "object_detection",
            "allowed_run_regimes": ["finetune_transfer"],
            "dataset_keys": ["demo_objdet_dataset"],
        }
    }
    datasets = {
        "demo_objdet_dataset": {
            "task_key": "object_detection",
            "dataset_name": "demo-objdet-dataset",
            "train_split": "train",
            "test_split": "validation",
            "image_column": "image",
            "label_column": "objects",
            "loader_template": "hf_object_detection",
            "max_samples": 8,
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["object_detection"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["finetune_transfer"],
        variants_per_pair=1,
        seed=1,
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["optimizer"] == "adamw"
    assert float(row["learning_rate"]) in {2e-5, 5e-5, 1e-4}
    assert float(row["learning_rate"]) <= 1e-4


def test_manifest_emits_mask_column_for_image_segmentation(monkeypatch):
    models = {
        "demo_imgseg_model": {
            "hf_model_id": "org/demo-imgseg",
            "task_key": "image_segmentation",
            "allowed_run_regimes": ["inference_only"],
            "dataset_keys": ["demo_imgseg_dataset"],
        }
    }
    datasets = {
        "demo_imgseg_dataset": {
            "task_key": "image_segmentation",
            "dataset_name": "demo-imgseg-dataset",
            "train_split": "train",
            "test_split": "validation",
            "image_column": "image",
            "label_column": "annotation",
            "mask_column": "annotation",
            "loader_template": "hf_image_segmentation",
            "max_samples": 8,
        }
    }
    monkeypatch.setattr(builder, "MODEL_REGISTRY", models)
    monkeypatch.setattr(builder, "DATASET_REGISTRY", datasets)

    df = builder.build_hf_manifest(
        task_keys=["image_segmentation"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=1,
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["mask_column"] == "annotation"
