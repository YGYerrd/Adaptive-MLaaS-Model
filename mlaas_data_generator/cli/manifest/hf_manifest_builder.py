import argparse
import json
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mlaas_data_generator.registry import DATASET_REGISTRY, MODEL_REGISTRY


@dataclass(frozen=True)
class TaskSpec:
    pipeline_tag: str
    hf_task: str
    modality: str
    task_type: str
    task_label: str
    task_tag: str | None = None


@dataclass(frozen=True)
class ManifestProfile:
    name: str
    default_avg_sample_size: int
    finetune_rounds: tuple[int, ...]
    finetune_clients: tuple[int, ...]
    finetune_local_epochs: tuple[int, ...]
    finetune_participation: tuple[float, ...]
    timeout_s: int


@dataclass(frozen=True)
class ManifestCandidate:
    task_key: str
    task_spec: TaskSpec
    model: dict[str, Any]
    dataset_spec: dict[str, Any]
    run_regime: str
    variant_index: int
    pair_score: float
    fit_reason: str


@dataclass(frozen=True)
class GenericManifestCandidate:
    task_key: str
    case: dict[str, Any]
    variant_index: int


TASK_SPECS: dict[str, TaskSpec] = {
    "text_classification": TaskSpec("text-classification", "sequence_classification", "text", "classification", "textcls"),
    "token_classification": TaskSpec("token-classification", "token_classification", "text", "classification", "tokencls"),
    "sentence_similarity": TaskSpec("sentence-similarity", "sentence_similarity", "text", "classification", "pairscore"),
    "fill_mask": TaskSpec("fill-mask", "fill_mask", "text", "classification", "fillmask"),
    "text_generation": TaskSpec("text-generation", "causal_lm_generation", "text", "classification", "textgen", "language-modeling"),
    "text2text_generation": TaskSpec("text2text-generation", "seq2seq_generation", "text", "classification", "text2text", "summarization"),
    "image_classification": TaskSpec("image-classification", "image_classification", "image", "classification", "imgcls"),
    "object_detection": TaskSpec("object-detection", "image_detection", "image", "detection", "objdet"),
    "image_segmentation": TaskSpec("image-segmentation", "image_segmentation", "image", "segmentation", "imgseg"),
    "image_captioning": TaskSpec("image-to-text", "image_captioning", "multimodal", "generation", "imgcap", "captioning"),
    "text_image_retrieval": TaskSpec(
        "zero-shot-image-classification",
        "text_image_retrieval",
        "multimodal",
        "retrieval",
        "imgtxtret",
        "retrieval",
    ),
    "visual_question_answering": TaskSpec(
        "visual-question-answering",
        "visual_question_answering",
        "multimodal",
        "vqa",
        "vqa",
        "vqa",
    ),
}

MANIFEST_PROFILES: dict[str, ManifestProfile] = {
    "test": ManifestProfile(
        name="test",
        default_avg_sample_size=128,
        finetune_rounds=(1, 2),
        finetune_clients=(1, 2),
        finetune_local_epochs=(1,),
        finetune_participation=(1.0,),
        timeout_s=900,
    ),
    "balanced": ManifestProfile(
        name="balanced",
        default_avg_sample_size=768,
        finetune_rounds=(2, 3),
        finetune_clients=(2, 3, 4),
        finetune_local_epochs=(1, 2),
        finetune_participation=(0.75, 1.0),
        timeout_s=1800,
    ),
    "benchmark": ManifestProfile(
        name="benchmark",
        default_avg_sample_size=1600,
        finetune_rounds=(3, 4, 5),
        finetune_clients=(3, 4, 5, 6),
        finetune_local_epochs=(1, 2),
        finetune_participation=(0.5, 0.75, 1.0),
        timeout_s=3600,
    ),
}

STRICT_VALIDATION_TASK_KEYS = {"image_captioning", "text_image_retrieval"}

MANIFEST_COLUMNS = [
    "external_run_id",
    "dataset",
    "run_group_id",
    "case_name",
    "notes",
    "enabled",
    "measure_system_metrics",
    "mixed_precision",
    "num_rounds",
    "num_clients",
    "client_participation_rate",
    "local_epochs",
    "batch_size",
    "earning_rate",
    "learning_rate",
    "optimizer",
    "seed",
    "distribution",
    "sample_size",
    "max_samples",
    "max_length",
    "num_workers",
    "timeout_s",
    "weight_decay",
    "momentum",
    "dirichlet_alpha",
    "aggregation",
    "device",
    "save_weights",
    "model_type",
    "hf_task",
    "task_type",
    "task",
    "modality",
    "dataset_name",
    "dataset_config",
    "hf_model_id",
    "train_split",
    "test_split",
    "label_column",
    "mask_column",
    "text_column",
    "image_column",
    "task_tag",
    "run_regime",
    "explainability_enabled",
    "explainability_method",
    "explainability_target",
    "model_role",
    "input_schema",
    "fit_decision",
    "fit_reason",
    "realism_score",
    "domain_alignment",
    "dataset_hint",
    "hf_pipeline_tag",
    "hf_downloads",
    "hf_likes",
    "hf_author",
    "hf_url",
    "hf_service_meta_json",
]

GENERIC_MANIFEST_CASES: tuple[dict[str, Any], ...] = (
    {
        "task_key": "keras_image_classification",
        "task_label": "keras_imgcls",
        "dataset": "cifar10",
        "dataset_name": "cifar10",
        "task_type": "classification",
        "task": "classification",
        "modality": "image",
        "model_type": "cnn",
        "input_schema": "single_image",
        "source": "keras",
        "max_samples": 1200,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "notes": "Generated generic Keras image-classification run",
    },
    {
        "task_key": "sklearn_image_classification",
        "task_label": "sk_imgcls",
        "dataset": "cifar10",
        "dataset_name": "cifar10",
        "task_type": "classification",
        "task": "classification",
        "modality": "image",
        "model_type": "randomforest",
        "input_schema": "single_image_flattened",
        "source": "sklearn",
        "max_samples": 1000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "none",
        "notes": "Generated generic sklearn image-classification run",
    },
    {
        "task_key": "tabular_regression",
        "task_label": "tabreg",
        "dataset": "synthetic",
        "dataset_name": "synthetic",
        "task_type": "regression",
        "task": "regression",
        "modality": "tabular",
        "model_type": "mlp",
        "input_schema": "tabular_features",
        "source": "sklearn_synthetic",
        "max_samples": 1200,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "notes": "Generated synthetic tabular regression run",
    },
    {
        "task_key": "tabular_regression",
        "task_label": "tabreg",
        "dataset": "uci_wine_quality",
        "dataset_name": "uci_wine_quality",
        "task_type": "regression",
        "task": "regression",
        "modality": "tabular",
        "model_type": "randomforest",
        "input_schema": "tabular_features",
        "source": "uci",
        "max_samples": 1600,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "none",
        "notes": "Generated UCI tabular regression run",
    },
    {
        "task_key": "clustering",
        "task_label": "cluster",
        "dataset": "synthetic",
        "dataset_name": "synthetic",
        "task_type": "clustering",
        "task": "clustering",
        "modality": "tabular",
        "model_type": "kmeans",
        "input_schema": "tabular_features",
        "source": "sklearn_synthetic",
        "max_samples": 1200,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "none",
        "notes": "Generated synthetic sklearn clustering run",
    },
)

GENERIC_MANIFEST_TASK_KEYS = frozenset(case["task_key"] for case in GENERIC_MANIFEST_CASES)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _resolve_manifest_profile(profile_name: str | None) -> ManifestProfile:
    normalized = str(profile_name or "balanced").strip().lower()
    return MANIFEST_PROFILES.get(normalized, MANIFEST_PROFILES["balanced"])


def _normalise_positive_int(value: int | None, *, minimum: int = 1) -> int | None:
    parsed = _as_int(value)
    if parsed is None:
        return None
    return max(minimum, parsed)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _has_required_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        return all(_has_required_value(item) for item in value)
    return True


def _log_signal(value: Any, *, scale: float = 1.0) -> float:
    numeric = _as_float(value)
    if numeric is None or numeric <= 0:
        return 0.0
    return math.log1p(numeric) / max(scale, 1e-6)


def _minimum_samples_for_task(task_key: str, run_regime: str) -> int:
    if run_regime == "inference_only":
        return 16

    per_task = {
        "text_classification": 48,
        "token_classification": 64,
        "sentence_similarity": 48,
        "fill_mask": 64,
        "text_generation": 64,
        "text2text_generation": 64,
        "image_classification": 32,
        "object_detection": 24,
        "image_segmentation": 24,
        "image_captioning": 24,
        "text_image_retrieval": 24,
        "visual_question_answering": 24,
    }
    return per_task.get(task_key, 32)


def _max_clients_for_samples(task_key: str, max_samples: int) -> int:
    min_samples = max(1, _minimum_samples_for_task(task_key, "finetune_transfer") // 2)
    return max(1, int(max_samples) // min_samples)


def _task_dataset_validation_issues(
    *,
    task_key: str,
    task_spec: TaskSpec,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    run_regime: str,
) -> list[str]:
    issues: list[str] = []

    if not _has_required_value(model.get("hf_model_id") or model.get("model_id") or model.get("id")):
        issues.append("missing_model_id")
    if not _has_required_value(dataset_spec.get("dataset_name")):
        issues.append("missing_dataset_name")

    allowed_run_regimes = model.get("allowed_run_regimes") or []
    if allowed_run_regimes and run_regime not in allowed_run_regimes:
        issues.append(f"run_regime_not_allowed:{run_regime}")

    compatible_dataset_keys = model.get("dataset_keys")
    dataset_key = str(dataset_spec.get("registry_id") or "").strip()
    if compatible_dataset_keys and dataset_key and dataset_key not in set(compatible_dataset_keys):
        issues.append("dataset_not_in_model_dataset_keys")

    model_task = str(model.get("task_key") or "").strip()
    dataset_task = str(dataset_spec.get("task_key") or "").strip()
    if model_task and model_task != task_key:
        issues.append(f"model_task_mismatch:{model_task}")
    if dataset_task and dataset_task != task_key:
        issues.append(f"dataset_task_mismatch:{dataset_task}")

    model_modality = str(model.get("modality") or task_spec.modality).strip().lower()
    dataset_modality = str(dataset_spec.get("modality") or task_spec.modality).strip().lower()
    if model_modality != task_spec.modality:
        issues.append(f"model_modality_mismatch:{model_modality}")
    if dataset_modality != task_spec.modality:
        issues.append(f"dataset_modality_mismatch:{dataset_modality}")

    if task_key in STRICT_VALIDATION_TASK_KEYS and not dataset_spec.get("manifest_validated"):
        issues.append("task_requires_manifest_validated_dataset")

    if task_spec.modality == "text":
        if not _has_required_value(dataset_spec.get("text_column")):
            issues.append("missing_text_column")
        if task_key == "sentence_similarity":
            text_column = dataset_spec.get("text_column")
            if not isinstance(text_column, (list, tuple)) or len(text_column) != 2 or not _has_required_value(list(text_column)):
                issues.append("sentence_similarity_requires_text_pair")
        if not _has_required_value(dataset_spec.get("label_column")):
            issues.append("missing_label_column")

    elif task_spec.modality == "image":
        if not _has_required_value(dataset_spec.get("image_column")):
            issues.append("missing_image_column")
        if task_key == "image_classification" and not _has_required_value(dataset_spec.get("label_column")):
            issues.append("missing_label_column")
        if task_key == "object_detection":
            has_label_container = _has_required_value(dataset_spec.get("label_column"))
            has_box_columns = _has_required_value(dataset_spec.get("boxes_column")) and _has_required_value(dataset_spec.get("classes_column"))
            if not has_label_container and not has_box_columns:
                issues.append("missing_detection_annotations")
        if task_key == "image_segmentation" and not _has_required_value(dataset_spec.get("mask_column") or dataset_spec.get("label_column")):
            issues.append("missing_segmentation_mask_column")
        if task_key == "image_classification" and run_regime == "inference_only":
            is_compatible, reason = _is_image_classification_inference_pair_compatible(
                model=model,
                dataset_spec=dataset_spec,
            )
            if not is_compatible:
                issues.append(reason)

    elif task_spec.modality == "multimodal":
        if not _has_required_value(dataset_spec.get("image_column")):
            issues.append("missing_image_column")
        if not _has_required_value(dataset_spec.get("text_column")):
            issues.append("missing_text_column")
        if task_key == "visual_question_answering" and not _has_required_value(dataset_spec.get("label_column")):
            issues.append("missing_vqa_label_column")

    return issues


def _model_priority_score(model: dict[str, Any], *, audit_meta: dict[str, Any], profile: ManifestProfile) -> float:
    model_id = str(model.get("hf_model_id") or model.get("model_id") or model.get("id") or "").strip()
    audit_models = audit_meta.get("models", {}) if isinstance(audit_meta, dict) else {}
    model_audit = audit_models.get(model_id, {}) if isinstance(audit_models, dict) else {}

    allowed_run_regimes = model.get("allowed_run_regimes") or []
    explainability = model.get("explainability") if isinstance(model.get("explainability"), dict) else {}
    score = 1.0
    score += 0.35 * _log_signal(model.get("downloads", model_audit.get("downloads")), scale=4.0)
    score += 0.15 * _log_signal(model.get("likes", model_audit.get("likes")), scale=3.0)
    score += 0.10 * len(model.get("dataset_keys") or [])
    score += 0.20 if "finetune_transfer" in allowed_run_regimes else 0.0
    score += 0.10 if "inference_only" in allowed_run_regimes else 0.0
    score += 0.10 if explainability.get("supported", explainability.get("supports_gradients")) else 0.0
    if profile.name == "benchmark":
        score += 0.05 * _log_signal(model.get("downloads", model_audit.get("downloads")), scale=2.5)
    return score


def _dataset_priority_score(dataset_spec: dict[str, Any], *, target_avg_sample_size: int | None) -> float:
    realism = _as_float(dataset_spec.get("realism_score"))
    if realism is None:
        realism = 1.0

    max_samples = max(1, _as_int(dataset_spec.get("max_samples")) or 1)
    target_fit = 1.0
    if target_avg_sample_size and target_avg_sample_size > 0:
        target_fit = 1.0 / (1.0 + abs(math.log((max_samples + 1.0) / (target_avg_sample_size + 1.0))))

    domain_alignment = str(dataset_spec.get("domain_alignment") or "registry_curated").strip().lower()
    domain_bonus = 0.20 if domain_alignment in {"registry_curated", "curated", "benchmark"} else 0.0
    explainability = dataset_spec.get("explainability") if isinstance(dataset_spec.get("explainability"), dict) else {}
    explainability_bonus = 0.10 if explainability.get("supported", explainability.get("supports_feature_attribution")) else 0.0
    return realism + (0.75 * target_fit) + domain_bonus + explainability_bonus


def _pair_priority_score(
    *,
    task_key: str,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    run_regime: str,
    audit_meta: dict[str, Any],
    profile: ManifestProfile,
    target_avg_sample_size: int | None,
) -> tuple[float, str]:
    model_score = _model_priority_score(model, audit_meta=audit_meta, profile=profile)
    dataset_score = _dataset_priority_score(dataset_spec, target_avg_sample_size=target_avg_sample_size)
    regime_bonus = 0.25 if run_regime == "finetune_transfer" else 0.10
    score = model_score + dataset_score + regime_bonus
    reason = (
        f"selected score={score:.3f} "
        f"(model={model_score:.3f}, dataset={dataset_score:.3f}, regime={run_regime})"
    )
    if task_key in STRICT_VALIDATION_TASK_KEYS:
        reason += "; strict multimodal validation passed"
    return score, reason


def _is_image_classification_inference_pair_compatible(
    *,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
) -> tuple[bool, str]:
    dataset_key = str(dataset_spec.get("registry_id") or "").strip()
    explicit_allow = model.get("inference_dataset_keys")
    if isinstance(explicit_allow, (list, tuple, set)):
        if dataset_key and dataset_key in set(explicit_allow):
            return True, "explicit_inference_dataset_allowlist"
        return False, "dataset_not_in_model_inference_allowlist"

    model_labels = _as_int(model.get("inference_num_labels"))
    dataset_labels = _as_int(dataset_spec.get("num_classes") or dataset_spec.get("num_labels"))
    if model_labels is not None and dataset_labels is not None:
        if model_labels == dataset_labels:
            return True, "matching_label_space"
        return False, f"label_space_mismatch:model={model_labels},dataset={dataset_labels}"

    # Conservative fallback: if registry lacks explicit compatibility metadata,
    # do not emit inference-only image classification rows.
    return False, "missing_inference_compatibility_metadata"


def _dataset_split_variants(dataset_spec: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    configured = dataset_spec.get("split_variants")
    if isinstance(configured, (list, tuple)):
        for item in configured:
            if not isinstance(item, dict):
                continue
            train_split = item.get("train_split")
            test_split = item.get("test_split")
            if _has_required_value(train_split) and _has_required_value(test_split):
                variants.append({"train_split": train_split, "test_split": test_split})

    base_train = dataset_spec.get("train_split", "train")
    base_test = dataset_spec.get("test_split", "validation")
    if _has_required_value(base_train) and _has_required_value(base_test):
        variants.insert(0, {"train_split": base_train, "test_split": base_test})

    train_split = str(base_train or "").strip()
    if train_split and "[" not in train_split and "]" not in train_split:
        variants.extend(
            [
                {"train_split": f"{train_split}[:80%]", "test_split": f"{train_split}[80%:]"},
                {"train_split": f"{train_split}[:90%]", "test_split": f"{train_split}[90%:]"},
            ]
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for variant in variants:
        key = (str(variant["train_split"]), str(variant["test_split"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped or [{"train_split": "train", "test_split": "validation"}]


def _apply_dataset_variant(dataset_spec: dict[str, Any], variant_index: int) -> dict[str, Any]:
    dataset_variant = dict(dataset_spec)
    split_variants = _dataset_split_variants(dataset_spec)
    split_variant = split_variants[variant_index % len(split_variants)]
    dataset_variant.update(split_variant)
    dataset_variant["_variant_index"] = variant_index
    dataset_variant["_split_variant_index"] = variant_index % len(split_variants)
    return dataset_variant


def _sample_training_knobs(
    rng: random.Random,
    *,
    seed: int,
    task_key: str | None = None,
    run_regime: str | None = None,
    max_samples: int | None = None,
    num_clients: int = 1,
) -> dict[str, Any]:
    effective_max_samples = max(1, _as_int(max_samples) or 1)
    per_client_budget = effective_max_samples
    effective_total_samples = effective_max_samples * max(1, int(num_clients))
    allow_dirichlet = (
        run_regime != "inference_only"
        and per_client_budget >= 24
        and effective_total_samples >= 64
    )
    distribution = rng.choice(["iid", "dirichlet"]) if allow_dirichlet else "iid"
    dirichlet_alpha = rng.choice([0.1, 0.3, 0.5]) if distribution == "dirichlet" else None

    task_key = str(task_key or "").strip().lower()
    run_regime = str(run_regime or "").strip().lower()

    if task_key == "object_detection" and run_regime == "finetune_transfer":
        return {
            "batch_size": rng.choice([2, 4, 8]),
            "learning_rate": rng.choice([2e-5, 5e-5, 1e-4]),
            "optimizer": "adamw",
            "seed": seed,
            "distribution": distribution,
            "weight_decay": rng.choice([0.0, 0.01, 0.05]),
            "momentum": 0.0,
            "dirichlet_alpha": dirichlet_alpha,
            "save_weights": True,
        }

    batch_choices = [choice for choice in [4, 8, 16, 32] if choice <= max(4, per_client_budget)]
    if not batch_choices:
        batch_choices = [max(1, min(4, per_client_budget))]
    learning_rate = rng.choice([1e-5, 2e-5, 3e-5, 5e-5, 1e-4])

    optimizer = rng.choice(["adamw", "sgd", "rmsprop"])

    if optimizer == "adamw":
        learning_rate = rng.choice([1e-5, 2e-5, 3e-5, 5e-5, 1e-4])
        weight_decay = rng.choice([0.0, 0.01, 0.05])
        momentum = 0.0

    elif optimizer == "sgd":
        learning_rate = rng.choice([1e-3, 5e-3, 1e-2, 5e-2])
        weight_decay = rng.choice([0.0, 1e-4, 5e-4])
        momentum = rng.choice([0.0, 0.9])

    else:  # rmsprop
        learning_rate = rng.choice([1e-4, 5e-4, 1e-3, 5e-3])
        weight_decay = rng.choice([0.0, 1e-4, 1e-3])
        momentum = rng.choice([0.0, 0.9])


    return {
        "batch_size": rng.choice(batch_choices),
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "seed": seed,
        "distribution": distribution,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "dirichlet_alpha": dirichlet_alpha,
        "save_weights": True,
    }


def _sample_run_workload(
    rng: random.Random,
    *,
    task_key: str,
    run_regime: str,
    max_samples: int,
    profile: ManifestProfile,
) -> dict[str, Any]:
    if run_regime == "inference_only":
        return {
            "num_rounds": 1,
            "num_clients": 1,
            "client_participation_rate": 1.0,
            "local_epochs": 1,
            "num_workers": 2,
            "timeout_s": profile.timeout_s,
            "mixed_precision": task_key in {"image_classification", "object_detection", "image_segmentation"},
            "measure_system_metrics": True,
        }

    requested_clients = rng.choice(profile.finetune_clients)
    num_clients = max(1, requested_clients)
    return {
        "num_rounds": rng.choice(profile.finetune_rounds),
        "num_clients": num_clients,
        "client_participation_rate": rng.choice(profile.finetune_participation),
        "local_epochs": rng.choice(profile.finetune_local_epochs),
        "num_workers": 2,
        "timeout_s": profile.timeout_s,
        "mixed_precision": task_key in {"image_classification", "object_detection", "image_segmentation"},
        "measure_system_metrics": True,
    }


def _resolve_candidate_sample_sizes(
    candidates: list[ManifestCandidate],
    *,
    avg_sample_size: int | None,
    seed: int,
) -> list[int]:
    if not candidates:
        return []

    base_caps = [max(1, _as_int(candidate.dataset_spec.get("max_samples")) or 1) for candidate in candidates]
    if not avg_sample_size or avg_sample_size <= 0:
        return base_caps

    baseline_avg = sum(base_caps) / len(base_caps)
    scale = float(avg_sample_size) / baseline_avg if baseline_avg > 0 else 1.0

    resolved: list[int] = []
    for candidate, base_cap in zip(candidates, base_caps):
        sample_rng = random.Random(
            f"{seed}:samples:{candidate.task_key}:{candidate.model.get('hf_model_id')}:{candidate.dataset_spec.get('registry_id')}:{candidate.run_regime}:{candidate.variant_index}"
        )
        jitter = sample_rng.uniform(0.85, 1.15)
        minimum = min(base_cap, _minimum_samples_for_task(candidate.task_key, candidate.run_regime))
        proposed = int(round(base_cap * scale * jitter))
        resolved.append(max(minimum, min(base_cap, proposed)))

    return resolved


def _select_manifest_candidates(
    *,
    requested_task_keys: list[str],
    models_per_task: int,
    datasets_per_model: int,
    selected_run_regimes: list[str],
    variants_per_pair: int,
    target_avg_sample_size: int | None,
    audit_meta: dict[str, Any],
    profile: ManifestProfile,
) -> list[ManifestCandidate]:
    candidates: list[ManifestCandidate] = []

    for task_key in requested_task_keys:
        task_spec = TASK_SPECS.get(task_key)
        if task_spec is None:
            continue

        models = [
            {"registry_id": registry_id, **dict(model)}
            for registry_id, model in MODEL_REGISTRY.items()
            if model.get("task_key") == task_key
        ]
        datasets = [
            {"registry_id": registry_id, **dict(dataset)}
            for registry_id, dataset in DATASET_REGISTRY.items()
            if dataset.get("task_key") == task_key
        ]
        if not models or not datasets:
            continue

        ranked_models = sorted(
            models,
            key=lambda model: (
                -_model_priority_score(model, audit_meta=audit_meta, profile=profile),
                str(model.get("registry_id") or ""),
            ),
        )

        selected_models = 0
        for model in ranked_models:
            compatible_datasets: list[tuple[dict[str, Any], float, list[str], str]] = []
            for dataset in datasets:
                valid_run_regimes: list[str] = []
                best_score = float("-inf")
                best_reason = ""
                for run_regime in selected_run_regimes:
                    issues = _task_dataset_validation_issues(
                        task_key=task_key,
                        task_spec=task_spec,
                        model=model,
                        dataset_spec=dataset,
                        run_regime=run_regime,
                    )
                    if issues:
                        continue
                    pair_score, fit_reason = _pair_priority_score(
                        task_key=task_key,
                        model=model,
                        dataset_spec=dataset,
                        run_regime=run_regime,
                        audit_meta=audit_meta,
                        profile=profile,
                        target_avg_sample_size=target_avg_sample_size,
                    )
                    valid_run_regimes.append(run_regime)
                    if pair_score > best_score:
                        best_score = pair_score
                        best_reason = fit_reason

                if valid_run_regimes:
                    compatible_datasets.append((dataset, best_score, valid_run_regimes, best_reason))

            if not compatible_datasets:
                continue

            compatible_datasets.sort(
                key=lambda item: (-item[1], str(item[0].get("registry_id") or "")),
            )
            chosen_datasets = compatible_datasets[: max(0, datasets_per_model)]
            if not chosen_datasets:
                continue

            selected_models += 1
            for dataset, _, valid_run_regimes, _ in chosen_datasets:
                for run_regime in valid_run_regimes:
                    pair_score, fit_reason = _pair_priority_score(
                        task_key=task_key,
                        model=model,
                        dataset_spec=dataset,
                        run_regime=run_regime,
                        audit_meta=audit_meta,
                        profile=profile,
                        target_avg_sample_size=target_avg_sample_size,
                    )
                    for variant_index in range(max(1, variants_per_pair)):
                        candidates.append(
                            ManifestCandidate(
                                task_key=task_key,
                                task_spec=task_spec,
                                model=model,
                                dataset_spec=_apply_dataset_variant(dataset, variant_index),
                                run_regime=run_regime,
                                variant_index=variant_index,
                                pair_score=pair_score,
                                fit_reason=fit_reason,
                            )
                        )

            if selected_models >= max(0, models_per_task):
                break

    return candidates


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _load_audit_metadata(json_path: str | None) -> dict[str, Any]:
    if not json_path:
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        source = json.load(f)

    tasks = source.get("tasks") if isinstance(source, dict) else None
    if not isinstance(tasks, list):
        return {"raw": source}

    model_audit: dict[str, dict[str, Any]] = {}
    for task in tasks:
        for model in task.get("models", []) or []:
            if not isinstance(model, dict):
                continue
            model_id = str(model.get("hf_model_id") or model.get("model_id") or model.get("id") or "").strip()
            if model_id:
                model_audit[model_id] = {
                    "downloads": model.get("downloads"),
                    "likes": model.get("likes"),
                    "author": model.get("author"),
                    "url": model.get("url"),
                    "pipeline_tag": task.get("pipeline_tag"),
                    "audit_dataset_tags": model.get("audit_dataset_tags") or model.get("dataset_tags") or [],
                    "audit_raw_tags": model.get("audit_raw_tags") or [],
                }

    return {"models": model_audit}


def _regime_defaults(run_regime: str) -> dict[str, str]:
    if run_regime == "inference_only":
        return {"model_type": "hf", "model_role": "service"}
    return {"model_type": "hf_finetune", "model_role": "task_head"}


def _resolve_explainability_metadata(
    *,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    run_regime: str,
) -> dict[str, Any]:
    model_meta = model.get("explainability") if isinstance(model.get("explainability"), dict) else {}
    dataset_meta = dataset_spec.get("explainability") if isinstance(dataset_spec.get("explainability"), dict) else {}

    model_supported = bool(model_meta.get("supported", model_meta.get("supports_token_attribution") or model_meta.get("supports_gradients")))
    dataset_supported = bool(dataset_meta.get("supported", dataset_meta.get("supports_feature_attribution") or dataset_meta.get("supports_example_level_rationales")))
    enabled = model_supported and dataset_supported and run_regime in {"finetune_transfer", "inference_only"}

    dataset_methods = [str(method) for method in dataset_meta.get("preferred_methods", []) if str(method).strip()]
    model_methods = [str(method) for method in model_meta.get("preferred_methods", []) if str(method).strip()]
    preferred_method = next((method for method in dataset_methods if method in model_methods), None)
    if preferred_method is None:
        preferred_method = (model_methods or dataset_methods or [None])[0]

    return {
        "explainability_enabled": enabled,
        "explainability_method": preferred_method,
        "explainability_target": dataset_meta.get("target_type") or model_meta.get("target_type"),
    }


def _row_from_registry(
    *,
    run_group_id: str,
    run_index: int,
    task_spec: TaskSpec,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    knobs: dict[str, Any],
    workload: dict[str, Any],
    resolved_max_samples: int,
    run_regime: str,
    audit_meta: dict[str, Any],
    pair_score: float,
    fit_reason: str,
) -> dict[str, Any]:
    model_id = str(model.get("hf_model_id") or model.get("model_id") or model.get("id") or "").strip()
    audit_models = audit_meta.get("models", {}) if isinstance(audit_meta, dict) else {}
    model_audit = audit_models.get(model_id, {}) if isinstance(audit_models, dict) else {}
    author = model.get("author") or model_audit.get("author")
    if not author and "/" in model_id:
        author = model_id.split("/", 1)[0]

    model_defaults = _regime_defaults(run_regime)
    service_payload = {
        "source": "registry",
        "registry_task": next((task_key for task_key, spec in TASK_SPECS.items() if spec == task_spec), None),
        "run_regime": run_regime,
        "audit_json_used": bool(audit_meta),
        "pair_score": round(pair_score, 4),
        "variant_index": dataset_spec.get("_variant_index", 0),
        "split_variant_index": dataset_spec.get("_split_variant_index", 0),
    }
    explainability = _resolve_explainability_metadata(model=model, dataset_spec=dataset_spec, run_regime=run_regime)
    resolved_modality = str(dataset_spec.get("modality") or "").strip().lower()
    if resolved_modality not in {"text", "image", "multimodal"} or (
        task_spec.modality and resolved_modality != task_spec.modality
    ):
        resolved_modality = task_spec.modality or "text"

    row = {
        "external_run_id": f"hf_{task_spec.task_label}_{run_index:06d}",
        "dataset": "hf",
        "run_group_id": run_group_id,
        "case_name": (
            f"{model_id.replace('/', '_')}__{dataset_spec.get('registry_id', dataset_spec.get('dataset_name'))}"
            f"__{run_regime}__v{dataset_spec.get('_variant_index', 0)}"
        ),
        "notes": "Generated from scored registry-defined HF model and dataset compatibility",
        "enabled": True,
        "measure_system_metrics": workload["measure_system_metrics"],
        "mixed_precision": workload["mixed_precision"],
        "num_rounds": workload["num_rounds"],
        "num_clients": workload["num_clients"],
        "client_participation_rate": workload["client_participation_rate"],
        "local_epochs": workload["local_epochs"],
        "batch_size": knobs["batch_size"],
        "earning_rate": knobs["learning_rate"],
        "learning_rate": knobs["learning_rate"],
        "optimizer": knobs["optimizer"],
        "seed": knobs["seed"],
        "distribution": knobs["distribution"],
        "max_samples": resolved_max_samples,
        "max_length": dataset_spec.get("max_length", 128),
        "num_workers": workload["num_workers"],
        "timeout_s": workload["timeout_s"],
        "weight_decay": knobs["weight_decay"],
        "momentum": knobs["momentum"],
        "dirichlet_alpha": knobs["dirichlet_alpha"],
        "aggregation": "",
        "device": "",
        "save_weights": knobs["save_weights"],
        "model_type": model_defaults["model_type"] if run_regime == "inference_only" else (model.get("model_type") or model_defaults["model_type"]),
        "hf_task": task_spec.hf_task,
        "task_type": dataset_spec.get("task_type", task_spec.task_type),
        "modality": resolved_modality,
        "dataset_name": dataset_spec.get("dataset_name"),
        "dataset_config": dataset_spec.get("dataset_config"),
        "hf_model_id": model_id,
        "train_split": dataset_spec.get("train_split", "train"),
        "test_split": dataset_spec.get("test_split", "validation"),
        "label_column": dataset_spec.get("label_column", "label"),
        "mask_column": dataset_spec.get("mask_column"),
        "text_column": dataset_spec.get("text_column", "text"),
        "image_column": dataset_spec.get("image_column"),
        "task_tag": dataset_spec.get("task_tag", task_spec.task_tag),
        "run_regime": run_regime,
        "explainability_enabled": explainability["explainability_enabled"],
        "explainability_method": explainability["explainability_method"],
        "explainability_target": explainability["explainability_target"],
        "model_role": model_defaults["model_role"] if run_regime == "inference_only" else (model.get("model_role") or model_defaults["model_role"]),
        "input_schema": dataset_spec.get("input_schema", "single_text"),
        "fit_decision": "selected",
        "fit_reason": fit_reason,
        "realism_score": round(pair_score, 4),
        "domain_alignment": dataset_spec.get("domain_alignment", "registry_curated"),
        "dataset_hint": dataset_spec.get("dataset_hint") or dataset_spec.get("dataset_name"),
        "hf_pipeline_tag": task_spec.pipeline_tag,
        "hf_downloads": model.get("downloads", model_audit.get("downloads")),
        "hf_likes": model.get("likes", model_audit.get("likes")),
        "hf_author": author,
        "hf_url": model.get("url") or model_audit.get("url") or (f"https://huggingface.co/{model_id}" if model_id else None),
        "hf_service_meta_json": json.dumps(service_payload),
    }
    return row


def _selected_generic_cases(requested_task_keys: list[str]) -> list[dict[str, Any]]:
    requested = set(requested_task_keys)
    return [case for case in GENERIC_MANIFEST_CASES if case["task_key"] in requested]


def _allocate_even_quotas(total_runs: int, task_keys: list[str]) -> dict[str, int]:
    eligible_task_keys = _dedupe_preserve_order(task_keys)
    if not eligible_task_keys or total_runs <= 0:
        return {}

    base_quota = total_runs // len(eligible_task_keys)
    remainder = total_runs % len(eligible_task_keys)
    quotas = {task_key: base_quota for task_key in eligible_task_keys}
    for task_key in eligible_task_keys[:remainder]:
        quotas[task_key] += 1
    return quotas


def _balanced_hf_candidates(
    *,
    base_candidates: list[ManifestCandidate],
    requested_task_keys: list[str],
    total_runs: int | None,
) -> list[ManifestCandidate]:
    if total_runs is None:
        return base_candidates

    by_task: dict[str, list[ManifestCandidate]] = {}
    for candidate in base_candidates:
        by_task.setdefault(candidate.task_key, []).append(candidate)

    eligible_task_keys = [task_key for task_key in requested_task_keys if task_key in by_task]
    quotas = _allocate_even_quotas(total_runs, eligible_task_keys)

    selected: list[ManifestCandidate] = []
    for task_key in eligible_task_keys:
        pool = by_task[task_key]
        quota = quotas.get(task_key, 0)
        for run_offset in range(quota):
            base_candidate = pool[run_offset % len(pool)]
            variant_index = run_offset // len(pool)
            selected.append(
                ManifestCandidate(
                    task_key=base_candidate.task_key,
                    task_spec=base_candidate.task_spec,
                    model=base_candidate.model,
                    dataset_spec=_apply_dataset_variant(base_candidate.dataset_spec, variant_index),
                    run_regime=base_candidate.run_regime,
                    variant_index=variant_index,
                    pair_score=base_candidate.pair_score,
                    fit_reason=base_candidate.fit_reason,
                )
            )
    return selected


def _balanced_generic_candidates(
    *,
    cases: list[dict[str, Any]],
    requested_task_keys: list[str],
    total_runs: int | None,
    variants_per_pair: int,
) -> list[GenericManifestCandidate]:
    if not cases:
        return []

    by_task: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_task.setdefault(str(case["task_key"]), []).append(case)

    if total_runs is None:
        selected: list[GenericManifestCandidate] = []
        for task_key in requested_task_keys:
            for case in by_task.get(task_key, []):
                for variant_index in range(max(1, variants_per_pair)):
                    selected.append(GenericManifestCandidate(task_key=task_key, case=case, variant_index=variant_index))
        return selected

    eligible_task_keys = [task_key for task_key in requested_task_keys if task_key in by_task]
    quotas = _allocate_even_quotas(total_runs, eligible_task_keys)
    selected = []
    for task_key in eligible_task_keys:
        pool = by_task[task_key]
        quota = quotas.get(task_key, 0)
        for run_offset in range(quota):
            selected.append(
                GenericManifestCandidate(
                    task_key=task_key,
                    case=pool[run_offset % len(pool)],
                    variant_index=run_offset // len(pool),
                )
            )
    return selected


def _split_total_runs_by_family(
    *,
    total_runs: int | None,
    requested_task_keys: list[str],
    hf_candidates: list[ManifestCandidate],
    generic_cases: list[dict[str, Any]],
) -> tuple[int | None, int | None]:
    if total_runs is None:
        return None, None

    hf_task_keys = {candidate.task_key for candidate in hf_candidates}
    generic_task_keys = {str(case["task_key"]) for case in generic_cases}
    eligible_task_keys = [
        task_key
        for task_key in requested_task_keys
        if task_key in hf_task_keys or task_key in generic_task_keys
    ]
    quotas = _allocate_even_quotas(total_runs, eligible_task_keys)
    hf_total = sum(quota for task_key, quota in quotas.items() if task_key in hf_task_keys)
    generic_total = sum(quota for task_key, quota in quotas.items() if task_key in generic_task_keys and task_key not in hf_task_keys)
    return hf_total, generic_total


def _row_from_generic_case(
    *,
    run_group_id: str,
    run_index: int,
    case: dict[str, Any],
    profile: ManifestProfile,
    seed: int,
    variant_index: int,
    resolved_max_samples: int,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}:generic:{case['task_key']}:{case['dataset']}:{case['model_type']}:{variant_index}")
    is_non_federated = str(case["task_type"]).lower() == "clustering" or str(case["model_type"]).lower() == "randomforest"
    num_clients = 1 if is_non_federated else rng.choice(profile.finetune_clients)
    num_rounds = 1 if is_non_federated else rng.choice(profile.finetune_rounds)
    local_epochs = 1 if is_non_federated else rng.choice(profile.finetune_local_epochs)
    distribution = "iid" if case["task_type"] in {"regression", "clustering"} else rng.choice(["iid", "dirichlet"])
    dirichlet_alpha = rng.choice([0.1, 0.3, 0.5]) if distribution == "dirichlet" else None
    task_label = str(case["task_label"])
    source = str(case.get("source") or "generic")
    if str(case.get("optimizer") or "").lower() == "none":
        batch_size = case.get("batch_size", 32)
        learning_rate = case.get("learning_rate", 1e-3)
        optimizer = case.get("optimizer", "none")
        weight_decay = 0.0
        momentum = 0.0
    else:
        knobs = _sample_training_knobs(
            rng,
            seed=seed,
            task_key=str(case["task_key"]),
            run_regime="finetune_transfer",
            max_samples=resolved_max_samples,
            num_clients=num_clients,
        )
        batch_size = knobs["batch_size"]
        learning_rate = knobs["learning_rate"]
        optimizer = knobs["optimizer"]
        weight_decay = knobs["weight_decay"]
        momentum = knobs["momentum"]

    row = {
        "external_run_id": f"gen_{task_label}_{run_index:06d}",
        "dataset": case["dataset"],
        "run_group_id": run_group_id,
        "case_name": f"{source}_{case['model_type']}__{case['dataset']}__v{variant_index}",
        "notes": case.get("notes", "Generated generic registry run"),
        "enabled": True,
        "measure_system_metrics": True,
        "mixed_precision": False,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "client_participation_rate": 1.0,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "earning_rate": learning_rate,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "seed": seed,
        "distribution": distribution,
        "sample_size": resolved_max_samples,
        "max_samples": resolved_max_samples,
        "max_length": "",
        "num_workers": 2,
        "timeout_s": profile.timeout_s,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "dirichlet_alpha": dirichlet_alpha,
        "aggregation": "",
        "device": "",
        "save_weights": not is_non_federated,
        "model_type": case["model_type"],
        "hf_task": "",
        "task_type": case["task_type"],
        "task": case["task"],
        "modality": case["modality"],
        "dataset_name": case["dataset_name"],
        "dataset_config": "",
        "hf_model_id": "",
        "train_split": "",
        "test_split": "",
        "label_column": "",
        "mask_column": "",
        "text_column": "",
        "image_column": "",
        "task_tag": "",
        "run_regime": "generic",
        "explainability_enabled": False,
        "explainability_method": "",
        "explainability_target": "",
        "model_role": "task_head",
        "input_schema": case["input_schema"],
        "fit_decision": "selected",
        "fit_reason": f"selected generic {source} {case['task_type']} case",
        "realism_score": 1.0,
        "domain_alignment": "generic_curated",
        "dataset_hint": case["dataset_name"],
        "hf_pipeline_tag": "",
        "hf_downloads": "",
        "hf_likes": "",
        "hf_author": "",
        "hf_url": "",
        "hf_service_meta_json": json.dumps(
            {
                "source": source,
                "registry_task": case["task_key"],
                "run_regime": "generic",
                "variant_index": variant_index,
            }
        ),
    }
    if case["task_type"] == "clustering":
        row.update(
            {
                "fit_reason": f"selected generic {source} clustering case with kmeans",
            }
        )
    return row


def build_hf_manifest(
    *,
    json_path: str | None = None,
    task_keys: list[str] | None = None,
    models_per_task: int,
    datasets_per_model: int,
    run_regimes: list[str] | None = None,
    variants_per_pair: int = 1,
    total_runs: int | None = None,
    seed: int,
    manifest_profile: str = "balanced",
    avg_sample_size: int | None = None,
) -> pd.DataFrame:
    requested_task_keys = _dedupe_preserve_order(
        task_keys or [*list(TASK_SPECS.keys()), *sorted(GENERIC_MANIFEST_TASK_KEYS)]
    )
    audit_meta = _load_audit_metadata(json_path)
    run_group_id = str(uuid.uuid4())
    selected_run_regimes = run_regimes or ["finetune_transfer"]
    profile = _resolve_manifest_profile(manifest_profile)
    effective_avg_sample_size = avg_sample_size if avg_sample_size is not None else profile.default_avg_sample_size
    effective_total_runs = _normalise_positive_int(total_runs) if total_runs is not None else None

    hf_variant_budget = 1 if effective_total_runs is not None else variants_per_pair
    candidates = _select_manifest_candidates(
        requested_task_keys=requested_task_keys,
        models_per_task=models_per_task,
        datasets_per_model=datasets_per_model,
        selected_run_regimes=selected_run_regimes,
        variants_per_pair=hf_variant_budget,
        target_avg_sample_size=effective_avg_sample_size,
        audit_meta=audit_meta,
        profile=profile,
    )
    generic_cases = _selected_generic_cases(requested_task_keys)
    hf_total_runs, generic_total_runs = _split_total_runs_by_family(
        total_runs=effective_total_runs,
        requested_task_keys=requested_task_keys,
        hf_candidates=candidates,
        generic_cases=generic_cases,
    )
    candidates = _balanced_hf_candidates(
        base_candidates=candidates,
        requested_task_keys=requested_task_keys,
        total_runs=hf_total_runs,
    )

    resolved_sample_sizes = _resolve_candidate_sample_sizes(
        candidates,
        avg_sample_size=effective_avg_sample_size,
        seed=seed,
    )

    rows: list[dict[str, Any]] = []
    for candidate, resolved_max_samples in zip(candidates, resolved_sample_sizes):
        variant_rng = random.Random(
            f"{seed}:{candidate.task_key}:{candidate.model.get('hf_model_id')}:{candidate.dataset_spec.get('registry_id')}:{candidate.run_regime}:{candidate.variant_index}"
        )
        workload = _sample_run_workload(
            variant_rng,
            task_key=candidate.task_key,
            run_regime=candidate.run_regime,
            max_samples=resolved_max_samples,
            profile=profile,
        )
        knobs = _sample_training_knobs(
            variant_rng,
            seed=seed,
            task_key=candidate.task_key,
            run_regime=candidate.run_regime,
            max_samples=resolved_max_samples,
            num_clients=workload["num_clients"],
        )
        rows.append(
            _row_from_registry(
                run_group_id=run_group_id,
                run_index=len(rows) + 1,
                task_spec=candidate.task_spec,
                model=candidate.model,
                dataset_spec=candidate.dataset_spec,
                knobs=knobs,
                workload=workload,
                resolved_max_samples=resolved_max_samples,
                run_regime=candidate.run_regime,
                audit_meta=audit_meta,
                pair_score=candidate.pair_score,
                fit_reason=candidate.fit_reason,
            )
        )

    generic_candidates = _balanced_generic_candidates(
        cases=generic_cases,
        requested_task_keys=requested_task_keys,
        total_runs=generic_total_runs,
        variants_per_pair=variants_per_pair,
    )
    for generic_candidate in generic_candidates:
        case = generic_candidate.case
        base_cap = max(1, _as_int(case.get("max_samples")) or effective_avg_sample_size)
        resolved_max_samples = min(base_cap, max(_minimum_samples_for_task(str(case["task_key"]), "finetune_transfer"), effective_avg_sample_size))
        rows.append(
            _row_from_generic_case(
                run_group_id=run_group_id,
                run_index=len(rows) + 1,
                case=case,
                profile=profile,
                seed=seed,
                variant_index=generic_candidate.variant_index,
                resolved_max_samples=resolved_max_samples,
            )
        )

    df = pd.DataFrame(rows)
    return df.reindex(columns=MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame, output_path: Path, sheet_name: str = "runs") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return

    if suffix == ".xlsx":
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        return

    raise ValueError("Output path must end with .csv or .xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HF model+dataset run manifests from registries")
    parser.add_argument("--input-json", help="Optional HF audit JSON used only for metadata enrichment")
    parser.add_argument("--output", default="outputs/run_manifest.xlsx", help="Output .csv or .xlsx path")
    parser.add_argument("--sheet", default="runs", help="Sheet name for xlsx output")
    parser.add_argument("--task-keys", help="Comma-separated registry task keys")
    parser.add_argument("--models-per-task", type=int, default=10)
    parser.add_argument("--datasets-per-model", type=int, default=1)
    parser.add_argument("--run-regimes", help="Comma-separated run regimes")
    parser.add_argument("--variants-per-pair", type=int, default=1)
    parser.add_argument("--total-runs", type=int, help="Total rows to emit, split as evenly as possible across requested task keys")
    parser.add_argument("--manifest-profile", choices=sorted(MANIFEST_PROFILES), default="balanced")
    parser.add_argument("--avg-sample-size", type=int, help="Target average max_samples across emitted manifest rows")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_hf_manifest(
        json_path=args.input_json,
        task_keys=_parse_csv_arg(args.task_keys),
        models_per_task=args.models_per_task,
        datasets_per_model=args.datasets_per_model,
        run_regimes=_parse_csv_arg(args.run_regimes),
        variants_per_pair=args.variants_per_pair,
        total_runs=args.total_runs,
        seed=args.seed,
        manifest_profile=args.manifest_profile,
        avg_sample_size=args.avg_sample_size,
    )
    output_path = Path(args.output)
    save_manifest(df, output_path, sheet_name=args.sheet)
    avg_samples = float(df["max_samples"].mean()) if not df.empty else 0.0
    print(f"Wrote {len(df)} rows to {output_path} (avg max_samples={avg_samples:.1f})")


if __name__ == "__main__":
    main()
