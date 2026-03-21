import argparse
import json
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
    task_type: str
    task_label: str
    task_tag: str | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    "text_classification": TaskSpec("text-classification", "sequence_classification", "classification", "textcls"),
    "token_classification": TaskSpec("token-classification", "token_classification", "classification", "tokencls"),
    "sentence_similarity": TaskSpec("sentence-similarity", "sentence_similarity", "classification", "pairscore"),
    "fill_mask": TaskSpec("fill-mask", "fill_mask", "classification", "fillmask"),
    "text_generation": TaskSpec("text-generation", "causal_lm_generation", "classification", "textgen", "language-modeling"),
    "text2text_generation": TaskSpec("text2text-generation", "seq2seq_generation", "classification", "text2text", "summarization"),
    "image_classification": TaskSpec("image-classification", "image_classification", "classification", "imgcls"),
    "object_detection": TaskSpec("object-detection", "image_detection", "detection", "objdet"),
    "image_segmentation": TaskSpec("image-segmentation", "image_segmentation", "segmentation", "imgseg"),
    "image_captioning": TaskSpec("image-to-text", "image_captioning", "generation", "imgcap", "captioning"),
    "text_image_retrieval": TaskSpec(
        "zero-shot-image-classification",
        "text_image_retrieval",
        "retrieval",
        "imgtxtret",
        "retrieval",
    ),
    "visual_question_answering": TaskSpec(
        "visual-question-answering",
        "visual_question_answering",
        "vqa",
        "vqa",
        "vqa",
    ),
}

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
    "modality",
    "dataset_name",
    "dataset_config",
    "hf_model_id",
    "train_split",
    "test_split",
    "label_column",
    "text_column",
    "image_column",
    "task_tag",
    "run_regime",
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


def _sample_training_knobs(rng: random.Random, *, seed: int) -> dict[str, Any]:
    distribution = rng.choice(["iid", "dirichlet"])
    dirichlet_alpha = rng.choice([0.1, 0.3, 0.5]) if distribution == "dirichlet" else None
    batch_choices = [4, 8, 16, 32]
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


def _row_from_registry(
    *,
    run_group_id: str,
    run_index: int,
    task_spec: TaskSpec,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    knobs: dict[str, Any],
    run_regime: str,
    audit_meta: dict[str, Any],
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
        "dataset_pairing_source": "registry.dataset_keys",
    }
    if model_audit.get("audit_dataset_tags") or model_audit.get("audit_raw_tags"):
        service_payload["audit_only_metadata"] = {
            "dataset_tags": model_audit.get("audit_dataset_tags") or [],
            "raw_tags": model_audit.get("audit_raw_tags") or [],
            "used_for_pairing": False,
        }

    row = {
        "external_run_id": f"hf_{task_spec.task_label}_{run_index:06d}",
        "dataset": "hf",
        "run_group_id": run_group_id,
        "case_name": (
            f"{model_id.replace('/', '_')}__{dataset_spec.get('registry_id', dataset_spec.get('dataset_name'))}"
            f"__{run_regime}__v{dataset_spec.get('_variant_index', 0)}"
        ),
        "notes": "Generated from registry-defined HF model and dataset compatibility",
        "enabled": True,
        "measure_system_metrics": True,
        "mixed_precision": False,
        "num_rounds": 1,
        "num_clients": 1,
        "client_participation_rate": 1.0,
        "local_epochs": 1,
        "batch_size": knobs["batch_size"],
        "earning_rate": knobs["learning_rate"],
        "learning_rate": knobs["learning_rate"],
        "optimizer": knobs["optimizer"],
        "seed": knobs["seed"],
        "distribution": knobs["distribution"],
        "max_samples": dataset_spec.get("max_samples", 1000),
        "max_length": dataset_spec.get("max_length", 128),
        "num_workers": 2,
        "timeout_s": 1800,
        "weight_decay": knobs["weight_decay"],
        "momentum": knobs["momentum"],
        "dirichlet_alpha": knobs["dirichlet_alpha"],
        "aggregation": "",
        "device": "",
        "save_weights": knobs["save_weights"],
        "model_type": model_defaults["model_type"] if run_regime == "inference_only" else (model.get("model_type") or model_defaults["model_type"]),
        "hf_task": task_spec.hf_task,
        "task_type": dataset_spec.get("task_type", task_spec.task_type),
        "modality": dataset_spec.get("modality", "text"),
        "dataset_name": dataset_spec.get("dataset_name"),
        "dataset_config": dataset_spec.get("dataset_config"),
        "hf_model_id": model_id,
        "train_split": dataset_spec.get("train_split", "train"),
        "test_split": dataset_spec.get("test_split", "validation"),
        "label_column": dataset_spec.get("label_column", "label"),
        "text_column": dataset_spec.get("text_column", "text"),
        "image_column": dataset_spec.get("image_column"),
        "task_tag": dataset_spec.get("task_tag", task_spec.task_tag),
        "run_regime": run_regime,
        "model_role": model_defaults["model_role"] if run_regime == "inference_only" else (model.get("model_role") or model_defaults["model_role"]),
        "input_schema": dataset_spec.get("input_schema", "single_text"),
        "fit_decision": None,
        "fit_reason": None,
        "realism_score": dataset_spec.get("realism_score", 1.0),
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


def build_hf_manifest(
    *,
    json_path: str | None = None,
    task_keys: list[str] | None = None,
    models_per_task: int,
    datasets_per_model: int,
    run_regimes: list[str] | None = None,
    variants_per_pair: int = 1,
    seed: int,
) -> pd.DataFrame:
    requested_task_keys = task_keys or list(TASK_SPECS.keys())
    audit_meta = _load_audit_metadata(json_path)
    run_group_id = str(uuid.uuid4())
    rows: list[dict[str, Any]] = []
    selected_run_regimes = run_regimes or ["finetune_transfer"]

    for task_key in requested_task_keys:
        task_spec = TASK_SPECS.get(task_key)
        if task_spec is None:
            continue

        models = [
            {"registry_id": registry_id, **dict(model)}
            for registry_id, model in MODEL_REGISTRY.items()
            if model.get("task_key") == task_key
        ][: max(0, models_per_task)]
        datasets = [
            {"registry_id": registry_id, **dict(dataset)}
            for registry_id, dataset in DATASET_REGISTRY.items()
            if dataset.get("task_key") == task_key
        ]
        print(f"\nTASK: {task_key}")
        print(f"  models found: {len(models)}")
        print(f"  datasets found: {len(datasets)}")

        if models:
            for m in models:
                print(f"    model: {m.get('registry_id')} | hf_model_id={m.get('hf_model_id')} | dataset_keys={m.get('dataset_keys')} | allowed_run_regimes={m.get('allowed_run_regimes')}")
        if datasets:
                for d in datasets:
                    print(f"    dataset: {d.get('registry_id')} | dataset_name={d.get('dataset_name')}")

        if not models or not datasets:
            continue

        for model in models:
            compatible_dataset_keys = model.get("dataset_keys")
            compatible_datasets = [
                dataset for dataset in datasets
                if not compatible_dataset_keys or dataset.get("registry_id") in compatible_dataset_keys
            ]
            compatible_datasets = compatible_datasets[: max(0, datasets_per_model)]

            allowed_run_regimes = model.get("allowed_run_regimes") or selected_run_regimes

            print(f"  checking model {model.get('registry_id')}")
            print(f"    compatible datasets before cap: {[d.get('registry_id') for d in compatible_datasets]}")
            print(f"    selected run regimes: {selected_run_regimes}")
            print(f"    allowed run regimes: {allowed_run_regimes}")

            for dataset in compatible_datasets:
                for run_regime in selected_run_regimes:
                    if run_regime not in allowed_run_regimes:
                        continue
                    for variant_index in range(max(1, variants_per_pair)):
                        variant_rng = random.Random(f"{seed}:{task_key}:{model.get('hf_model_id')}:{dataset.get('registry_id')}:{run_regime}:{variant_index}")
                        dataset_variant = dict(dataset)
                        dataset_variant["_variant_index"] = variant_index
                        rows.append(
                            _row_from_registry(
                                run_group_id=run_group_id,
                                run_index=len(rows) + 1,
                                task_spec=task_spec,
                                model=model,
                                dataset_spec=dataset_variant,
                                knobs=_sample_training_knobs(variant_rng, seed=seed),
                                run_regime=run_regime,
                                audit_meta=audit_meta,
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
        seed=args.seed,
    )
    output_path = Path(args.output)
    save_manifest(df, output_path, sheet_name=args.sheet)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
