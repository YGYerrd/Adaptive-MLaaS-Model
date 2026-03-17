import argparse
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


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
}

# Kept for compatibility with tests/importers. The builder now prefers dataset specs from the input JSON.
SUPPORTED_DATASETS: dict[str, list[dict[str, Any]]] = {
    "text-classification": [],
    "token-classification": [],
    "sentence-similarity": [],
    "fill-mask": [],
    "text-generation": [],
    "text2text-generation": [],
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
    "num_shards",
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


def _sample_training_knobs(rng: random.Random) -> dict[str, Any]:
    distribution = rng.choice(["iid", "dirichlet", "shards"])
    num_shards = rng.choice([5, 10, 20]) if distribution == "shards" else None
    dirichlet_alpha = rng.choice([0.1, 0.3, 0.5]) if distribution == "dirichlet" else None

    batch_choices = [4, 8, 16, 32]
    learning_rate = rng.choice([1e-5, 2e-5, 3e-5, 5e-5, 1e-4])

    return {
        "batch_size": rng.choice(batch_choices),
        "learning_rate": learning_rate,
        "optimizer": "adamw",
        "seed": rng.randint(1, 1_000_000),
        "distribution": distribution,
        "num_shards": num_shards,
        "weight_decay": rng.choice([0.0, 0.01, 0.05]),
        "momentum": 0.0,
        "dirichlet_alpha": dirichlet_alpha,
        "save_weights": True,
    }


def _normalize_dataset_tag(tag: str) -> str:
    normalized = (tag or "").strip().lower().replace("dataset:", "")
    if normalized == "sst2":
        return "glue/sst2"
    return normalized


def _extract_model_datasets(model: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(model.get("paired_datasets"), list):
        return [d for d in model["paired_datasets"] if isinstance(d, dict)]
    if isinstance(model.get("datasets"), list):
        out: list[dict[str, Any]] = []
        for item in model["datasets"]:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                out.append({"dataset_name": _normalize_dataset_tag(item)})
        return out

    tags = model.get("dataset_tags") or []
    out = []
    for tag in tags:
        if isinstance(tag, str):
            ds_name = _normalize_dataset_tag(tag)
            if ds_name:
                out.append({"dataset_name": ds_name})
    return out


def _dataset_defaults(dataset_name: str, hf_task: str) -> dict[str, Any]:
    name = (dataset_name or "").strip()
    config = None
    if name.startswith("glue/"):
        _, config = name.split("/", 1)
        name = "glue"

    defaults = {
        "dataset_name": name,
        "dataset_config": config,
        "train_split": "train",
        "test_split": "validation",
        "text_column": "text",
        "label_column": "label",
        "max_samples": 1000,
        "max_length": 128,
        "input_schema": "single_text",
    }

    if hf_task == "token_classification":
        defaults.update({"text_column": "tokens", "label_column": "ner_tags", "input_schema": "token_sequence"})
    if hf_task in {"causal_lm_generation", "seq2seq_generation"}:
        defaults.update({"label_column": "text", "max_length": 256})

    return defaults


def _coalesce_dataset_spec(raw_dataset: dict[str, Any], hf_task: str) -> dict[str, Any]:
    dataset_name = str(raw_dataset.get("dataset_name") or "").strip()
    base = _dataset_defaults(dataset_name, hf_task)
    base.update({k: v for k, v in raw_dataset.items() if v is not None})

    if isinstance(base.get("dataset_name"), str) and "/" in base["dataset_name"] and not base.get("dataset_config"):
        if base["dataset_name"].startswith("glue/"):
            _, cfg = base["dataset_name"].split("/", 1)
            base["dataset_name"] = "glue"
            base["dataset_config"] = cfg

    return base


def _row_from_pair(
    run_group_id: str,
    run_index: int,
    task_spec: TaskSpec,
    model: dict[str, Any],
    dataset_spec: dict[str, Any],
    knobs: dict[str, Any],
) -> dict[str, Any]:
    model_id = str(model.get("model_id") or model.get("id") or "").strip()
    author = model.get("author")
    if not author and "/" in model_id:
        author = model_id.split("/", 1)[0]

    service_payload = {
        "source": "hf_dataset_manifest_json",
        "pipeline_tag": task_spec.pipeline_tag,
        "dataset_tags": model.get("dataset_tags", []),
    }

    row = {
        "external_run_id": f"hf_{task_spec.task_label}_{run_index:06d}",
        "dataset": "hf",
        "run_group_id": run_group_id,
        "case_name": f"{model_id.replace('/', '_')}__{dataset_spec.get('dataset_name')}",
        "notes": "Generated from pre-tagged HF model/dataset JSON",
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
        "num_shards": knobs["num_shards"],
        "max_samples": dataset_spec.get("max_samples", 1000),
        "max_length": dataset_spec.get("max_length", 128),
        "num_workers": 2,
        "timeout_s": 1800,
        "weight_decay": knobs["weight_decay"],
        "momentum": knobs["momentum"],
        "dirichlet_alpha": knobs["dirichlet_alpha"],
        "aggregation": "fedavg",
        "device": "auto",
        "save_weights": knobs["save_weights"],
        "model_type": "hf_finetune",
        "hf_task": task_spec.hf_task,
        "task_type": task_spec.task_type,
        "modality": "text",
        "dataset_name": dataset_spec.get("dataset_name"),
        "dataset_config": dataset_spec.get("dataset_config"),
        "hf_model_id": model_id,
        "train_split": dataset_spec.get("train_split", "train"),
        "test_split": dataset_spec.get("test_split", "validation"),
        "label_column": dataset_spec.get("label_column", "label"),
        "text_column": dataset_spec.get("text_column", "text"),
        "image_column": dataset_spec.get("image_column"),
        "task_tag": task_spec.task_tag,
        "run_regime": "finetune_transfer",
        "model_role": "task_head",
        "input_schema": dataset_spec.get("input_schema", "single_text"),
        "fit_decision": "json_paired",
        "fit_reason": "Paired in source JSON",
        "realism_score": 1.0,
        "domain_alignment": "json_declared",
        "dataset_hint": dataset_spec.get("dataset_name"),
        "hf_pipeline_tag": task_spec.pipeline_tag,
        "hf_downloads": model.get("downloads"),
        "hf_likes": model.get("likes"),
        "hf_author": author,
        "hf_url": model.get("url") or (f"https://huggingface.co/{model_id}" if model_id else None),
        "hf_service_meta_json": json.dumps(service_payload),
    }

    return row


def build_hf_manifest(
    *,
    json_path: str,
    models_per_task: int,
    datasets_per_model: int,
    seed: int,
) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        source = json.load(f)

    tasks = source.get("tasks") if isinstance(source, dict) else None
    if not isinstance(tasks, list):
        raise ValueError("Input JSON must contain a top-level 'tasks' list")

    rng = random.Random(seed)
    run_group_id = str(uuid.uuid4())
    rows: list[dict[str, Any]] = []

    for task in tasks:
        pipeline_tag = str(task.get("pipeline_tag") or "").strip()
        if not pipeline_tag:
            continue

        matched = next((spec for spec in TASK_SPECS.values() if spec.pipeline_tag == pipeline_tag), None)
        if not matched:
            continue

        models = [m for m in (task.get("models") or []) if isinstance(m, dict)]
        if not models:
            continue

        rng.shuffle(models)
        picked_models = models[: max(0, models_per_task)]

        for model in picked_models:
            raw_datasets = _extract_model_datasets(model)
            if not raw_datasets:
                continue

            deduped: list[dict[str, Any]] = []
            seen: set[tuple[Any, Any]] = set()
            for d in raw_datasets:
                spec = _coalesce_dataset_spec(d, matched.hf_task)
                key = (spec.get("dataset_name"), spec.get("dataset_config"))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(spec)

            rng.shuffle(deduped)
            for ds in deduped[: max(1, datasets_per_model)]:
                rows.append(
                    _row_from_pair(
                        run_group_id=run_group_id,
                        run_index=len(rows) + 1,
                        task_spec=matched,
                        model=model,
                        dataset_spec=ds,
                        knobs=_sample_training_knobs(rng),
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
    parser = argparse.ArgumentParser(description="Generate HF model+dataset run manifests from JSON")
    parser.add_argument("--input-json", required=True, help="Path to HF model/dataset pairing JSON")
    parser.add_argument("--output", default="outputs/run_manifest.xlsx", help="Output .csv or .xlsx path")
    parser.add_argument("--sheet", default="runs", help="Sheet name for xlsx output")
    parser.add_argument("--models-per-task", type=int, default=10)
    parser.add_argument("--datasets-per-model", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_hf_manifest(
        json_path=args.input_json,
        models_per_task=args.models_per_task,
        datasets_per_model=args.datasets_per_model,
        seed=args.seed,
    )
    output_path = Path(args.output)
    save_manifest(df, output_path, sheet_name=args.sheet)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
