from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mlaas_data_generator.config import DEFAULT_MANIFEST_PATH
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
    training_epochs: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    learning_rates: tuple[float, ...]
    timeout_s: int


TASK_SPECS: dict[str, TaskSpec] = {
    "text_classification": TaskSpec("text-classification", "sequence_classification", "text", "classification", "textcls"),
    "token_classification": TaskSpec("token-classification", "token_classification", "text", "classification", "tokencls"),
    "sentence_similarity": TaskSpec("sentence-similarity", "sentence_similarity", "text", "classification", "pairscore"),
    "fill_mask": TaskSpec("fill-mask", "fill_mask", "text", "classification", "fillmask"),
    "text_generation": TaskSpec("text-generation", "causal_lm_generation", "text", "generation", "textgen", "language-modeling"),
    "text2text_generation": TaskSpec("text2text-generation", "seq2seq_generation", "text", "generation", "text2text", "summarization"),
    "image_classification": TaskSpec("image-classification", "image_classification", "image", "classification", "imgcls"),
    "object_detection": TaskSpec("object-detection", "image_detection", "image", "detection", "objdet"),
    "image_segmentation": TaskSpec("image-segmentation", "image_segmentation", "image", "segmentation", "imgseg"),
    "image_captioning": TaskSpec("image-to-text", "image_captioning", "multimodal", "generation", "imgcap", "captioning"),
    "text_image_retrieval": TaskSpec("zero-shot-image-classification", "text_image_retrieval", "multimodal", "retrieval", "imgtxtret", "retrieval"),
    "visual_question_answering": TaskSpec("visual-question-answering", "visual_question_answering", "multimodal", "vqa", "vqa", "vqa"),
}


MANIFEST_PROFILES: dict[str, ManifestProfile] = {
    "test": ManifestProfile("test", default_avg_sample_size=128, training_epochs=(1,), batch_sizes=(4, 8), learning_rates=(5e-5, 1e-4), timeout_s=900),
    "balanced": ManifestProfile("balanced", default_avg_sample_size=768, training_epochs=(1, 2), batch_sizes=(8, 16), learning_rates=(2e-5, 5e-5, 1e-4), timeout_s=1800),
    "benchmark": ManifestProfile("benchmark", default_avg_sample_size=1600, training_epochs=(1, 2, 3), batch_sizes=(8, 16, 32), learning_rates=(2e-5, 5e-5, 1e-4), timeout_s=3600),
}


MANIFEST_COLUMNS = [
    "service_id",
    "enabled",
    "case_name",
    "notes",
    "dataset",
    "dataset_name",
    "dataset_config",
    "train_split",
    "test_split",
    "benchmark_split",
    "model_type",
    "hf_task",
    "hf_model_id",
    "task_type",
    "task",
    "task_tag",
    "modality",
    "input_schema",
    "output_schema",
    "training_regime",
    "dataset_variant",
    "split_variant",
    "knob_variant",
    "service_config",
    "split_strategy",
    "distribution_type",
    "distribution_param",
    "custom_distributions",
    "training_epochs",
    "batch_size",
    "learning_rate",
    "optimizer",
    "weight_decay",
    "momentum",
    "sample_size",
    "max_samples",
    "max_length",
    "timeout_s",
    "device",
    "mixed_precision",
    "save_weights",
    "num_workers",
    "text_column",
    "image_column",
    "label_column",
    "mask_column",
    "question_column",
    "answer_column",
    "ranking_label_column",
    "vqa_label_mode",
    "vqa_answer_vocab_size",
    "vqa_unseen_answer_policy",
    "retrieval_positive_policy",
    "missing_pair_handling",
    "on_decode_error",
    "report_decode_errors",
    "source_max_length",
    "target_max_length",
    "dynamic_padding",
    "column_mapping",
    "explainability_enabled",
    "enable_perturbation_metrics",
    "perturbation_stage_logging",
    "perturbation_progress_logging",
    "perturbation_sample_count",
    "perturbation_candidate_units",
    "perturbation_target_units",
    "perturbation_trust_trials",
    "perturbation_progress_sample_interval",
    "perturbation_random_strength",
    "explainability_method",
    "explainability_target",
    "service_source",
    "model_role",
    "fit_decision",
    "fit_reason",
    "realism_score",
    "domain_alignment",
    "dataset_hint",
    "hf_pipeline_tag",
    "hf_downloads",
    "hf_likes",
    "hf_dataset_id",
    "downloads",
    "likes",
    "model_size",
    "params_count",
    "pipeline_tag",
    "library_name",
    "license",
    "tags",
    "last_modified",
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
        "max_samples": 1200,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
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
        "max_samples": 1000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "none",
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
        "max_samples": 1200,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "adam",
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
        "max_samples": 1600,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "optimizer": "none",
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
        "max_samples": 1200,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "none",
        "clustering_k": 3,
    },
)
GENERIC_MANIFEST_TASK_KEYS = frozenset(case["task_key"] for case in GENERIC_MANIFEST_CASES)


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _resolve_manifest_profile(profile_name: str | None) -> ManifestProfile:
    return MANIFEST_PROFILES.get(str(profile_name or "balanced").strip().lower(), MANIFEST_PROFILES["balanced"])


def _normalise_positive_int(value: int | None, *, minimum: int = 1) -> int:
    if value is None:
        return minimum
    return max(minimum, int(value))


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _service_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    safe_prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix.lower()).strip("_") or "svc"
    return f"{safe_prefix}_{digest}"


def _training_regime_defaults(training_regime: str) -> dict[str, str]:
    if training_regime == "inference_only":
        return {"model_type": "hf", "model_role": "service"}
    return {"model_type": "hf_finetune", "model_role": "task_head"}


def _service_config(row: dict[str, Any]) -> str:
    payload = {
        "training_epochs": row.get("training_epochs"),
        "split_strategy": row.get("split_strategy"),
        "distribution_type": row.get("distribution_type"),
        "distribution_param": row.get("distribution_param"),
        "batch_size": row.get("batch_size"),
        "learning_rate": row.get("learning_rate"),
        "optimizer": row.get("optimizer"),
        "weight_decay": row.get("weight_decay"),
        "momentum": row.get("momentum"),
        "max_samples": row.get("max_samples"),
        "sample_size": row.get("sample_size"),
        "max_length": row.get("max_length"),
        "timeout_s": row.get("timeout_s"),
        "device": row.get("device"),
        "mixed_precision": row.get("mixed_precision"),
        "save_weights": row.get("save_weights"),
        "enable_perturbation_metrics": row.get("enable_perturbation_metrics"),
        "perturbation_sample_count": row.get("perturbation_sample_count"),
    }
    return json.dumps({k: v for k, v in payload.items() if v is not None}, sort_keys=True)


def _max_samples(dataset_spec: dict[str, Any], profile: ManifestProfile, avg_sample_size: int | None) -> int:
    requested = avg_sample_size if avg_sample_size is not None else profile.default_avg_sample_size
    dataset_cap = _as_int(dataset_spec.get("max_samples")) or requested
    return max(1, min(dataset_cap, int(requested)))


def _split_value(dataset_spec: dict[str, Any], key: str, variant: int) -> Any:
    value = dataset_spec.get(key)
    if isinstance(value, (list, tuple)) and value:
        return value[variant % len(value)]
    return value


def _row_from_registry(
    *,
    task_key: str,
    task_spec: TaskSpec,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    profile: ManifestProfile,
    training_regime: str,
    dataset_variant: int,
    split_variant: int,
    knob_variant: int,
    rng: random.Random,
    avg_sample_size: int | None,
) -> dict[str, Any]:
    defaults = _training_regime_defaults(training_regime)
    epochs = 0 if training_regime == "inference_only" else rng.choice(profile.training_epochs)
    row = {
        "enabled": True,
        "dataset": "hf",
        "dataset_name": dataset_spec.get("dataset_name"),
        "dataset_config": dataset_spec.get("dataset_config"),
        "train_split": _split_value(dataset_spec, "train_split", split_variant),
        "test_split": _split_value(dataset_spec, "test_split", split_variant),
        "benchmark_split": _split_value(dataset_spec, "test_split", split_variant),
        "model_type": defaults["model_type"],
        "hf_task": task_spec.hf_task,
        "hf_model_id": model.get("hf_model_id"),
        "task_type": task_spec.task_type,
        "task": task_key,
        "task_tag": task_spec.task_tag,
        "modality": task_spec.modality,
        "input_schema": dataset_spec.get("input_schema") or model.get("input_schema"),
        "output_schema": dataset_spec.get("label_format"),
        "training_regime": training_regime,
        "dataset_variant": dataset_variant,
        "split_variant": split_variant,
        "knob_variant": knob_variant,
        "split_strategy": "iid",
        "distribution_type": "iid",
        "distribution_param": None,
        "custom_distributions": None,
        "training_epochs": epochs,
        "batch_size": rng.choice(profile.batch_sizes),
        "learning_rate": (None if training_regime == "inference_only" else rng.choice(profile.learning_rates)),
        "optimizer": ("none" if training_regime == "inference_only" else "adam"),
        "weight_decay": 0.0,
        "momentum": 0.0,
        "sample_size": None,
        "max_samples": _max_samples(dataset_spec, profile, avg_sample_size),
        "max_length": dataset_spec.get("max_length") or model.get("max_length"),
        "timeout_s": profile.timeout_s,
        "device": "auto",
        "mixed_precision": False,
        "save_weights": False,
        "num_workers": 0,
        "text_column": dataset_spec.get("text_column"),
        "image_column": dataset_spec.get("image_column"),
        "label_column": dataset_spec.get("label_column"),
        "mask_column": dataset_spec.get("mask_column"),
        "question_column": dataset_spec.get("question_column"),
        "answer_column": dataset_spec.get("answer_column"),
        "ranking_label_column": dataset_spec.get("ranking_label_column"),
        "vqa_label_mode": dataset_spec.get("vqa_label_mode"),
        "vqa_answer_vocab_size": dataset_spec.get("vqa_answer_vocab_size"),
        "vqa_unseen_answer_policy": dataset_spec.get("vqa_unseen_answer_policy"),
        "retrieval_positive_policy": dataset_spec.get("retrieval_positive_policy"),
        "missing_pair_handling": dataset_spec.get("missing_pair_handling"),
        "on_decode_error": dataset_spec.get("on_decode_error"),
        "report_decode_errors": dataset_spec.get("report_decode_errors"),
        "source_max_length": dataset_spec.get("source_max_length"),
        "target_max_length": dataset_spec.get("target_max_length"),
        "dynamic_padding": dataset_spec.get("dynamic_padding"),
        "column_mapping": json.dumps(dataset_spec.get("column_mapping"), sort_keys=True) if dataset_spec.get("column_mapping") else None,
        "explainability_enabled": bool((model.get("explainability") or dataset_spec.get("explainability") or {}).get("supported", True)),
        "enable_perturbation_metrics": bool((model.get("explainability") or dataset_spec.get("explainability") or {}).get("supported", True)),
        "perturbation_stage_logging": True,
        "perturbation_progress_logging": False,
        "perturbation_sample_count": 1,
        "perturbation_candidate_units": 4,
        "perturbation_target_units": 1,
        "perturbation_trust_trials": 2,
        "perturbation_progress_sample_interval": 1,
        "perturbation_random_strength": 0.02,
        "explainability_method": _preferred_explainability(model, dataset_spec),
        "explainability_target": (model.get("explainability") or dataset_spec.get("explainability") or {}).get("target_type"),
        "service_source": "hf_registry",
        "model_role": defaults["model_role"],
        "fit_decision": "compatible",
        "fit_reason": f"registry-compatible task={task_key} training_regime={training_regime}",
        "realism_score": model.get("realism_score"),
        "domain_alignment": dataset_spec.get("domain_alignment"),
        "dataset_hint": dataset_spec.get("dataset_hint"),
        "hf_pipeline_tag": model.get("pipeline_tag") or task_spec.pipeline_tag,
        "hf_downloads": model.get("downloads"),
        "hf_likes": model.get("likes"),
        "hf_author": model.get("author"),
        "hf_url": model.get("url"),
        "hf_service_meta_json": json.dumps(
            {
                "model_family": model.get("family"),
                "dataset_registry_key": dataset_spec.get("registry_key"),
                "model_registry_key": model.get("registry_key"),
                "training_regime": training_regime,
                "dataset_variant": dataset_variant,
                "split_variant": split_variant,
                "knob_variant": knob_variant,
            },
            sort_keys=True,
        ),
    }
    row["case_name"] = (
        f"{task_spec.task_label}__{model.get('hf_model_id')}__{dataset_spec.get('dataset_name')}"
        f"__{training_regime}__d{dataset_variant}__s{split_variant}__k{knob_variant}"
    )
    row["notes"] = "Reviewed service row generated from model and dataset registries"
    row["service_id"] = _service_id(
        f"hf_{task_spec.task_label}",
        {
            "model": row["hf_model_id"],
            "dataset": row["dataset_name"],
            "dataset_config": row["dataset_config"],
            "training_regime": training_regime,
            "dataset_variant": dataset_variant,
            "split_variant": split_variant,
            "knob_variant": knob_variant,
        },
    )
    row["service_config"] = _service_config(row)
    return row


def _preferred_explainability(model: dict[str, Any], dataset_spec: dict[str, Any]) -> str | None:
    payload = model.get("explainability") or dataset_spec.get("explainability") or {}
    methods = payload.get("preferred_methods")
    if isinstance(methods, list) and methods:
        return str(methods[0])
    return None


def _model_candidates(task_key: str) -> list[dict[str, Any]]:
    candidates = []
    for registry_key, model in MODEL_REGISTRY.items():
        if model.get("task_key") == task_key:
            copied = dict(model)
            copied["registry_key"] = registry_key
            candidates.append(copied)
    return candidates


def _dataset_candidates(task_key: str, model: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = set(model.get("dataset_keys") or [])
    candidates = []
    for registry_key, dataset in DATASET_REGISTRY.items():
        if dataset.get("task_key") != task_key:
            continue
        if allowed and registry_key not in allowed:
            continue
        copied = dict(dataset)
        copied["registry_key"] = registry_key
        candidates.append(copied)
    return candidates


def _selected_generic_cases(requested_task_keys: list[str]) -> list[dict[str, Any]]:
    requested = set(requested_task_keys)
    return [case for case in GENERIC_MANIFEST_CASES if case["task_key"] in requested]


def _row_from_generic_case(
    *,
    case: dict[str, Any],
    profile: ManifestProfile,
    dataset_variant: int,
    split_variant: int,
    knob_variant: int,
    rng: random.Random,
    avg_sample_size: int | None,
) -> dict[str, Any]:
    row = {
        **case,
        "enabled": True,
        "case_name": f"{case['task_label']}__{case['dataset_name']}__generic__d{dataset_variant}__s{split_variant}__k{knob_variant}",
        "notes": "Reviewed generic service row",
        "training_regime": "generic",
        "dataset_variant": dataset_variant,
        "split_variant": split_variant,
        "knob_variant": knob_variant,
        "split_strategy": "iid",
        "distribution_type": "iid",
        "distribution_param": None,
        "custom_distributions": None,
        "training_epochs": rng.choice(profile.training_epochs),
        "batch_size": case.get("batch_size") or rng.choice(profile.batch_sizes),
        "learning_rate": case.get("learning_rate"),
        "optimizer": case.get("optimizer"),
        "weight_decay": 0.0,
        "momentum": 0.0,
        "sample_size": None,
        "max_samples": min(int(case.get("max_samples", profile.default_avg_sample_size)), int(avg_sample_size or profile.default_avg_sample_size)),
        "timeout_s": profile.timeout_s,
        "mixed_precision": False,
        "num_workers": 0,
        "service_source": "generic_registry",
        "model_role": "service",
        "fit_decision": "compatible",
        "fit_reason": "generic manifest-compatible task runner",
        "explainability_enabled": True,
        "enable_perturbation_metrics": True,
        "perturbation_stage_logging": True,
        "perturbation_progress_logging": False,
        "perturbation_sample_count": 1,
        "perturbation_candidate_units": 4,
        "perturbation_target_units": 1,
        "perturbation_trust_trials": 2,
        "perturbation_progress_sample_interval": 1,
        "perturbation_random_strength": 0.02,
        "device": "auto",
        "save_weights": False,
    }
    row["service_id"] = _service_id(
        f"gen_{case['task_label']}",
        {
            "model": row["model_type"],
            "dataset": row["dataset_name"],
            "training_regime": "generic",
            "dataset_variant": dataset_variant,
            "split_variant": split_variant,
            "knob_variant": knob_variant,
        },
    )
    row["service_config"] = _service_config(row)
    return row


def build_hf_manifest(
    json_path: str | None = None,
    task_keys: list[str] | None = None,
    models_per_task: int = 10,
    datasets_per_model: int = 1,
    training_regimes: list[str] | None = None,
    dataset_variants_per_pair: int = 1,
    split_variants_per_pair: int = 1,
    knob_variants_per_pair: int = 1,
    total_services: int | None = None,
    seed: int = 42,
    manifest_profile: str = "balanced",
    avg_sample_size: int | None = None,
    max_models_per_family: int | None = None,
    strict_inference_dataset_match: bool = False,
) -> pd.DataFrame:
    del json_path, strict_inference_dataset_match
    rng = random.Random(seed)
    profile = _resolve_manifest_profile(manifest_profile)
    requested_task_keys = task_keys or list(TASK_SPECS) + sorted(GENERIC_MANIFEST_TASK_KEYS)
    selected_training_regimes = training_regimes or ["finetune_transfer"]
    selected_training_regimes = [str(item).strip().lower() for item in selected_training_regimes if str(item).strip()]

    rows: list[dict[str, Any]] = []
    for task_key in requested_task_keys:
        task_spec = TASK_SPECS.get(task_key)
        if task_spec is None:
            continue
        models = _model_candidates(task_key)
        rng.shuffle(models)
        if max_models_per_family:
            models = _cap_models_per_family(models, int(max_models_per_family))
        for model in models[: _normalise_positive_int(models_per_task)]:
            datasets = _dataset_candidates(task_key, model)
            rng.shuffle(datasets)
            for dataset in datasets[: _normalise_positive_int(datasets_per_model)]:
                for training_regime in selected_training_regimes:
                    allowed = set(model.get("allowed_training_regimes") or [])
                    if allowed and training_regime not in allowed:
                        continue
                    for dataset_variant in range(_normalise_positive_int(dataset_variants_per_pair)):
                        for split_variant in range(_normalise_positive_int(split_variants_per_pair)):
                            for knob_variant in range(_normalise_positive_int(knob_variants_per_pair)):
                                rows.append(
                                    _row_from_registry(
                                        task_key=task_key,
                                        task_spec=task_spec,
                                        model=model,
                                        dataset_spec=dataset,
                                        profile=profile,
                                        training_regime=training_regime,
                                        dataset_variant=dataset_variant,
                                        split_variant=split_variant,
                                        knob_variant=knob_variant,
                                        rng=rng,
                                        avg_sample_size=avg_sample_size,
                                    )
                                )

    for case in _selected_generic_cases(requested_task_keys):
        for dataset_variant in range(_normalise_positive_int(dataset_variants_per_pair)):
            for split_variant in range(_normalise_positive_int(split_variants_per_pair)):
                for knob_variant in range(_normalise_positive_int(knob_variants_per_pair)):
                    rows.append(
                        _row_from_generic_case(
                            case=case,
                            profile=profile,
                            dataset_variant=dataset_variant,
                            split_variant=split_variant,
                            knob_variant=knob_variant,
                            rng=rng,
                            avg_sample_size=avg_sample_size,
                        )
                    )

    df = pd.DataFrame(rows)
    if total_services is not None and total_services >= 0:
        df = df.head(int(total_services))
    return df.reindex(columns=MANIFEST_COLUMNS)


def _cap_models_per_family(models: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    selected = []
    for model in models:
        family = str(model.get("family") or model.get("registry_key") or "unknown")
        if counts.get(family, 0) >= limit:
            continue
        counts[family] = counts.get(family, 0) + 1
        selected.append(model)
    return selected


def save_manifest(df: pd.DataFrame, output_path: Path, sheet_name: str = "services") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        with pd.ExcelWriter(output_path) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            pd.DataFrame([{"enabled": True}]).to_excel(writer, sheet_name="defaults", index=False)
        return
    raise ValueError("Output path must end with .csv or .xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviewed MLaaS service manifest rows")
    parser.add_argument("--input-json")
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--sheet", default="services")
    parser.add_argument("--task-keys")
    parser.add_argument("--models-per-task", type=int, default=10)
    parser.add_argument("--max-models-per-family", type=int)
    parser.add_argument("--datasets-per-model", type=int, default=1)
    parser.add_argument("--training-regimes")
    parser.add_argument("--dataset-variants-per-pair", type=int, default=1)
    parser.add_argument("--split-variants-per-pair", type=int, default=1)
    parser.add_argument("--knob-variants-per-pair", type=int, default=1)
    parser.add_argument("--total-services", type=int)
    parser.add_argument("--manifest-profile", choices=sorted(MANIFEST_PROFILES), default="balanced")
    parser.add_argument("--avg-sample-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_hf_manifest(
        json_path=args.input_json,
        task_keys=_parse_csv_arg(args.task_keys),
        models_per_task=args.models_per_task,
        datasets_per_model=args.datasets_per_model,
        training_regimes=_parse_csv_arg(args.training_regimes),
        dataset_variants_per_pair=args.dataset_variants_per_pair,
        split_variants_per_pair=args.split_variants_per_pair,
        knob_variants_per_pair=args.knob_variants_per_pair,
        total_services=args.total_services,
        seed=args.seed,
        manifest_profile=args.manifest_profile,
        avg_sample_size=args.avg_sample_size,
        max_models_per_family=args.max_models_per_family,
    )
    save_manifest(df, Path(args.output), sheet_name=args.sheet)
    print(f"Wrote {len(df)} service rows to {args.output}")


if __name__ == "__main__":
    main()
