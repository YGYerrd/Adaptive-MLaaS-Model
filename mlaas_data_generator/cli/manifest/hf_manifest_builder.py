import argparse
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
    "text_classification": TaskSpec(
        pipeline_tag="text-classification",
        hf_task="sequence_classification",
        task_type="classification",
        task_label="textcls",
    ),
    "token_classification": TaskSpec(
        pipeline_tag="token-classification",
        hf_task="token_classification",
        task_type="classification",
        task_label="tokencls",
    ),
    "sentence_similarity": TaskSpec(
        pipeline_tag="sentence-similarity",
        hf_task="sentence_similarity",
        task_type="classification",
        task_label="pairscore",
    ),
    "fill_mask": TaskSpec(
        pipeline_tag="fill-mask",
        hf_task="fill_mask",
        task_type="classification",
        task_label="fillmask",
    ),
    "text_generation": TaskSpec(
        pipeline_tag="text-generation",
        hf_task="causal_lm_generation",
        task_type="classification",
        task_label="textgen",
        task_tag="language-modeling",
    ),
    "text2text_generation": TaskSpec(
        pipeline_tag="text2text-generation",
        hf_task="seq2seq_generation",
        task_type="classification",
        task_label="text2text",
        task_tag="summarization",
    ),
    "image_classification": TaskSpec(
        pipeline_tag="image-classification",
        hf_task="image_classification",
        task_type="classification",
        task_label="imgcls",
    ),
    "object_detection": TaskSpec(
        pipeline_tag="object-detection",
        hf_task="image_detection",
        task_type="detection",
        task_label="objdet",
    ),
    "image_segmentation": TaskSpec(
        pipeline_tag="image-segmentation",
        hf_task="image_segmentation",
        task_type="segmentation",
        task_label="imgseg",
    ),
}

SUPPORTED_DATASETS: dict[str, list[dict[str, Any]]] = {
    "text-classification": [
        {
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "sentence",
            "label_column": "label",
            "max_samples": 1500,
            "max_length": 128,
        },
        {
            "dataset_name": "ag_news",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "label_column": "label",
            "max_samples": 2000,
            "max_length": 128,
        },
        {
            "dataset_name": "imdb",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "test",
            "text_column": "text",
            "label_column": "label",
            "max_samples": 2000,
            "max_length": 256,
        },
    ],
    "token-classification": [
        {
            "dataset_name": "conll2003",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "validation",
            "text_column": "tokens",
            "label_column": "ner_tags",
            "max_samples": 1000,
            "max_length": 128,
        },
        {
            "dataset_name": "wnut_17",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "validation",
            "text_column": "tokens",
            "label_column": "ner_tags",
            "max_samples": 900,
            "max_length": 128,
        },
    ],
    "sentence-similarity": [
        {
            "dataset_name": "glue",
            "dataset_config": "stsb",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "[sentence1, sentence2]",
            "label_column": "label",
            "max_samples": 1200,
            "max_length": 128,
        },
        {
            "dataset_name": "snli",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "validation",
            "text_column": "[premise, hypothesis]",
            "label_column": "label",
            "max_samples": 2000,
            "max_length": 128,
        },
        {
            "dataset_name": "glue",
            "dataset_config": "qqp",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "[question1, question2]",
            "label_column": "label",
            "max_samples": 2000,
            "max_length": 128,
        },
    ],
    "fill-mask": [
        {
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "text",
            "label_column": None,
            "max_samples": 2500,
            "max_length": 128,
        },
    ],
    "text-generation": [
        {
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "text",
            "label_column": "text",
            "max_samples": 1800,
            "max_length": 256,
            "task_tag": "language-modeling",
        },
    ],
    "text2text-generation": [
        {
            "dataset_name": "cnn_dailymail",
            "dataset_config": "3.0.0",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "article",
            "label_column": "highlights",
            "max_samples": 1200,
            "max_length": 256,
            "task_tag": "summarization",
        },
        {
            "dataset_name": "wmt14",
            "dataset_config": "de-en",
            "train_split": "train",
            "test_split": "test",
            "text_column": "translation.de",
            "label_column": "translation.en",
            "max_samples": 1000,
            "max_length": 192,
            "task_tag": "translation",
        },
    ],
    "image-classification": [
        {
            "dataset_name": "beans",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "validation",
            "text_column": "image",
            "label_column": "labels",
            "max_samples": 1000,
            "max_length": 224,
        },
        {
            "dataset_name": "food101",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "validation",
            "text_column": "image",
            "label_column": "label",
            "max_samples": 1200,
            "max_length": 224,
        },
    ],
    "object-detection": [
        {
            "dataset_name": "cppe-5",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "validation",
            "text_column": "image",
            "label_column": "objects",
            "max_samples": 800,
            "max_length": 512,
        },
        {
            "dataset_name": "coco",
            "dataset_config": "2017",
            "train_split": "train",
            "test_split": "validation",
            "text_column": "image",
            "label_column": "objects",
            "max_samples": 800,
            "max_length": 512,
        },
    ],
    "image-segmentation": [
        {
            "dataset_name": "scene_parse_150",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "validation",
            "text_column": "image",
            "label_column": "annotation",
            "max_samples": 700,
            "max_length": 512,
        },
        {
            "dataset_name": "segments/sidewalk-semantic",
            "dataset_config": None,
            "train_split": "train",
            "test_split": "validation",
            "text_column": "image",
            "label_column": "semantic_mask",
            "max_samples": 700,
            "max_length": 512,
        },
    ],
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
    "model_type",
    "hf_task",
    "task_type",
    "dataset_name",
    "dataset_config",
    "hf_model_id",
    "train_split",
    "test_split",
    "label_column",
    "text_column",
    "task_tag",
]


def _infer_family(model_id: str, arch: list[str] | None) -> str:
    if arch:
        return arch[0].lower()
    base = model_id.split("/")[-1].lower()
    for prefix in (
        "bert", "roberta", "distilbert", "albert", "deberta", "electra", "xlnet", "gpt2", "t5",
        "llama", "mistral", "xlm", "camembert", "longformer", "funnel", "mobilebert",
    ):
        if base.startswith(prefix):
            return prefix
    return base.split("-")[0]


def _model_downloads(model: Any) -> int:
    value = getattr(model, "downloads", 0)
    return int(value or 0)


def _select_diverse_models(
    models: list[Any],
    *,
    target_count: int,
    max_per_author: int,
    max_per_family: int,
    min_downloads: int,
) -> list[Any]:
    filtered = [m for m in models if _model_downloads(m) >= min_downloads]
    filtered.sort(key=lambda m: (_model_downloads(m), int(getattr(m, "likes", 0) or 0)), reverse=True)

    by_author: dict[str, int] = {}
    by_family: dict[str, int] = {}
    selected: list[Any] = []

    for model in filtered:
        model_id = getattr(model, "id", "") or ""
        if not model_id:
            continue
        author = model_id.split("/", 1)[0].lower() if "/" in model_id else "unknown"
        family = _infer_family(model_id, getattr(model, "architectures", None))

        if by_author.get(author, 0) >= max_per_author:
            continue
        if by_family.get(family, 0) >= max_per_family:
            continue

        selected.append(model)
        by_author[author] = by_author.get(author, 0) + 1
        by_family[family] = by_family.get(family, 0) + 1

        if len(selected) >= target_count:
            break

    return selected


def _fetch_models_for_tag(pipeline_tag: str, limit: int) -> list[Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    iterator = api.list_models(
        pipeline_tag=pipeline_tag,
        sort="downloads",
        limit=limit,
        full=True,
    )
    return list(iterator)


def _row_for(
    *,
    run_group_id: str,
    task_spec: TaskSpec,
    model_id: str,
    dataset_spec: dict[str, Any],
    run_index: int,
    training_knobs: dict[str, Any],
) -> dict[str, Any]:
    ds_slug = dataset_spec["dataset_name"].replace("/", "_")
    ext_id = f"hf_{task_spec.task_label}_{run_index:06d}"
    case_name = f"{task_spec.task_label}__{model_id.replace('/', '__')}__{ds_slug}"

    return {
        "external_run_id": ext_id,
        "dataset": "hf",
        "run_group_id": run_group_id,
        "case_name": case_name,
        "notes": f"Auto-generated HF manifest row for {task_spec.pipeline_tag}",
        "enabled": True,
        "measure_system_metrics": True,
        "mixed_precision": False,
        "num_rounds": 3,
        "num_clients": 5,
        "client_participation_rate": 1.0,
        "local_epochs": 1,
        "batch_size": training_knobs["batch_size"],
        "earning_rate": training_knobs["learning_rate"],
        "learning_rate": training_knobs["learning_rate"],
        "optimizer": training_knobs["optimizer"],
        "seed": training_knobs["seed"],
        "distribution": training_knobs["distribution"],
        "num_shards": training_knobs["num_shards"],
        "max_samples": dataset_spec["max_samples"],
        "max_length": dataset_spec["max_length"],
        "num_workers": 2,
        "timeout_s": 1800,
        "weight_decay": training_knobs["weight_decay"],
        "momentum": training_knobs["momentum"],
        "dirichlet_alpha": training_knobs["dirichlet_alpha"],
        "aggregation": "mean",
        "device": "cpu",
        "model_type": "hf",
        "hf_task": task_spec.hf_task,
        "task_type": task_spec.task_type,
        "dataset_name": dataset_spec["dataset_name"],
        "dataset_config": dataset_spec.get("dataset_config"),
        "hf_model_id": model_id,
        "train_split": dataset_spec["train_split"],
        "test_split": dataset_spec["test_split"],
        "label_column": dataset_spec.get("label_column"),
        "text_column": dataset_spec.get("text_column"),
        "task_tag": dataset_spec.get("task_tag") or task_spec.task_tag,
    }


def _sample_training_knobs(rng: random.Random) -> dict[str, Any]:
    optimizer = rng.choice(["adam", "adamw", "sgd", "adagrad"])
    learning_rate_options = {
        "adam": [1e-5, 2e-5, 3e-5, 5e-5],
        "adamw": [1e-5, 2e-5, 3e-5, 5e-5],
        "sgd": [1e-3, 3e-3, 1e-2],
        "adagrad": [5e-4, 1e-3, 2e-3],
    }

    distribution = rng.choices(["iid", "dirichlet", "shards"], weights=[0.4, 0.35, 0.25], k=1)[0]
    num_shards: int | None = None
    dirichlet_alpha: float | None = None

    if distribution == "shards":
        num_shards = rng.choice([2, 5, 10, 20])
    elif distribution == "dirichlet":
        dirichlet_alpha = rng.choice([0.1, 0.3, 0.5, 1.0])

    return {
        "batch_size": rng.choice([8, 16, 32, 64]),
        "learning_rate": rng.choice(learning_rate_options[optimizer]),
        "optimizer": optimizer,
        "seed": rng.randint(1, 1_000_000),
        "distribution": distribution,
        "num_shards": num_shards,
        "weight_decay": rng.choice([0.0, 1e-4, 5e-4, 1e-3]),
        "momentum": rng.choice([0.8, 0.9, 0.95]) if optimizer == "sgd" else 0.0,
        "dirichlet_alpha": dirichlet_alpha,
    }

def build_hf_manifest(
    *,
    models_per_task: int,
    datasets_per_model: int,
    fetch_limit_per_task: int,
    max_per_author: int,
    max_per_family: int,
    min_downloads: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    run_group_id = str(uuid.uuid4())
    rows: list[dict[str, Any]] = []

    for task_key, task_spec in TASK_SPECS.items():
        all_models = _fetch_models_for_tag(task_spec.pipeline_tag, fetch_limit_per_task)
        selected_models = _select_diverse_models(
            all_models,
            target_count=models_per_task,
            max_per_author=max_per_author,
            max_per_family=max_per_family,
            min_downloads=min_downloads,
        )

        dataset_pool = SUPPORTED_DATASETS[task_spec.pipeline_tag]
        for model in selected_models:
            model_id = getattr(model, "id")
            chosen_datasets = dataset_pool[:]
            rng.shuffle(chosen_datasets)
            chosen_datasets = chosen_datasets[: max(1, datasets_per_model)]

            for dataset_spec in chosen_datasets:
                row = _row_for(
                    run_group_id=run_group_id,
                    task_spec=task_spec,
                    model_id=model_id,
                    dataset_spec=dataset_spec,
                    run_index=len(rows) + 1,
                    training_knobs=_sample_training_knobs(rng),
                )
                rows.append(row)

        print(
            f"[{task_key}] fetched={len(all_models)} selected={len(selected_models)} rows={len(rows)}"
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
    parser = argparse.ArgumentParser(description="Generate HF model+dataset run manifests")
    parser.add_argument("--output", default="outputs/run_manifest.xlsx", help="Output .csv or .xlsx path")
    parser.add_argument("--sheet", default="runs", help="Sheet name for xlsx output")
    parser.add_argument("--models-per-task", type=int, default=100)
    parser.add_argument("--datasets-per-model", type=int, default=2)
    parser.add_argument("--fetch-limit-per-task", type=int, default=5000)
    parser.add_argument("--max-per-author", type=int, default=20)
    parser.add_argument("--max-per-family", type=int, default=35)
    parser.add_argument("--min-downloads", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_hf_manifest(
        models_per_task=args.models_per_task,
        datasets_per_model=args.datasets_per_model,
        fetch_limit_per_task=args.fetch_limit_per_task,
        max_per_author=args.max_per_author,
        max_per_family=args.max_per_family,
        min_downloads=args.min_downloads,
        seed=args.seed,
    )
    output_path = Path(args.output)
    save_manifest(df, output_path, sheet_name=args.sheet)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()