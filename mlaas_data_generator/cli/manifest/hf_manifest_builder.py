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


@dataclass(frozen=True)
class ModelFitAssessment:
    decision: str
    reason: str
    recommended_run_regime: str


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
    "image_captioning": TaskSpec(
        pipeline_tag="image-to-text",
        hf_task="image_captioning",
        task_type="classification",
        task_label="imgcap",
        task_tag="captioning",
    ),
    "text_image_retrieval": TaskSpec(
        pipeline_tag="zero-shot-image-classification",
        hf_task="text_image_retrieval",
        task_type="classification",
        task_label="imgret",
        task_tag="retrieval",
    ),
    "visual_question_answering": TaskSpec(
        pipeline_tag="visual-question-answering",
        hf_task="visual_question_answering",
        task_type="classification",
        task_label="vqa",
        task_tag="vqa",
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
            "input_schema": "single_text",
            "task_family": "text_classification",
            "label_space_type": "sentiment_binary",
            "num_labels": 2,
            "domain": "general_english",
            "preferred_run_regimes": ["finetune_exact", "inference"],
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
            "input_schema": "single_text",
            "task_family": "text_classification",
            "label_space_type": "topic_4way",
            "num_labels": 4,
            "domain": "news",
            "preferred_run_regimes": ["finetune_exact", "finetune_transfer"],
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
            "input_schema": "single_text",
            "task_family": "text_classification",
            "label_space_type": "sentiment_binary",
            "num_labels": 2,
            "domain": "movie_reviews",
            "preferred_run_regimes": ["finetune_exact", "finetune_transfer"],
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
            "input_schema": "token_sequence",
            "task_family": "token_classification",
            "label_space_type": "ner_conll2003",
            "num_labels": 9,
            "domain": "news",
            "preferred_run_regimes": ["finetune_exact", "finetune_transfer"],
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
            "input_schema": "token_sequence",
            "task_family": "token_classification",
            "label_space_type": "ner_wnut17",
            "num_labels": 13,
            "domain": "social_media",
            "preferred_run_regimes": ["finetune_exact", "finetune_transfer"],
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
            "input_schema": "text_pair",
            "task_family": "sentence_similarity",
            "label_space_type": "continuous_similarity",
            "num_labels": 1,
            "domain": "general_english",
            "preferred_run_regimes": ["finetune_exact", "finetune_transfer"],
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
            "input_schema": "text_pair",
            "task_family": "sentence_similarity",
            "label_space_type": "entailment_3way",
            "num_labels": 3,
            "domain": "general_english",
            "preferred_run_regimes": ["finetune_transfer"],
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
            "input_schema": "single_text_with_mask",
            "task_family": "masked_language_modeling",
            "label_space_type": "vocab",
            "num_labels": None,
            "domain": "general_english",
            "preferred_run_regimes": ["inference"],
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
            "input_schema": "single_text",
            "task_family": "causal_lm",
            "label_space_type": "next_token",
            "num_labels": None,
            "domain": "general_english",
            "preferred_run_regimes": ["inference", "finetune_transfer"],
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
            "input_schema": "single_text",
            "task_family": "seq2seq_generation",
            "label_space_type": "free_text",
            "num_labels": None,
            "domain": "news",
            "preferred_run_regimes": ["finetune_transfer", "inference"],
        },
    ],
}

PREFERRED_MODEL_OVERRIDES: dict[tuple[str, str | None], list[str]] = {
    ("glue", "sst2"): [
        "textattack/bert-base-uncased-SST-2",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ],
    ("ag_news", None): [
        "fabriceyhc/bert-base-uncased-ag_news",
        "cardiffnlp/twitter-roberta-base-topic-multi-all",
    ],
    ("conll2003", None): [
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "dslim/bert-base-NER",
    ],
    ("wnut_17", None): [
        "Jean-Baptiste/roberta-large-ner-english",
        "xlm-roberta-base-finetuned-conll03-english",
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
]


def _infer_family(model_id: str, arch: list[str] | None) -> str:
    if arch:
        return arch[0].lower()
    base = model_id.split("/")[-1].lower()
    for prefix in (
        "bert",
        "roberta",
        "distilbert",
        "albert",
        "deberta",
        "electra",
        "xlnet",
        "gpt2",
        "t5",
        "llama",
        "mistral",
        "xlm",
        "camembert",
        "longformer",
        "funnel",
        "mobilebert",
    ):
        if base.startswith(prefix):
            return prefix
    return base.split("-")[0]


def _model_downloads(model: Any) -> int:
    value = getattr(model, "downloads", 0)
    return int(value or 0)


def _normalize_model_profile(model: Any) -> dict[str, Any]:
    """Normalize Hugging Face model metadata into compatibility-friendly fields."""
    model_id = getattr(model, "id", "") or ""
    tags = [str(t).lower() for t in (getattr(model, "tags", None) or [])]
    family = _infer_family(model_id, getattr(model, "architectures", None))
    author = model_id.split("/", 1)[0].lower() if "/" in model_id else "unknown"
    suffix = model_id.split("/", 1)[-1].lower()

    task_family = "generic"
    if any(x in tags for x in ["text-classification", "sequence-classification"]):
        task_family = "text_classification"
    elif "token-classification" in tags:
        task_family = "token_classification"
    elif "sentence-similarity" in tags:
        task_family = "sentence_similarity"
    elif "fill-mask" in tags:
        task_family = "masked_language_modeling"
    elif "text-generation" in tags:
        task_family = "causal_lm"
    elif "text2text-generation" in tags:
        task_family = "seq2seq_generation"

    input_schemas = ["single_text"]
    if task_family in {"sentence_similarity"}:
        input_schemas = ["text_pair"]
    elif task_family == "token_classification":
        input_schemas = ["token_sequence"]
    elif task_family == "masked_language_modeling":
        input_schemas = ["single_text_with_mask", "single_text"]

    output_type = "labels" if "classification" in task_family else "text"
    if task_family == "sentence_similarity":
        output_type = "score"

    dataset_hint = None
    if "sst-2" in suffix or "sst2" in suffix:
        dataset_hint = "sst2"
    elif "ag_news" in suffix:
        dataset_hint = "ag_news"
    elif "conll" in suffix:
        dataset_hint = "conll2003"
    elif "wnut" in suffix:
        dataset_hint = "wnut_17"

    is_dataset_specific = dataset_hint is not None or any(
        tag in tags for tag in ["dataset:glue", "dataset:conll2003", "dataset:wnut_17", "dataset:ag_news"]
    )
    is_base_backbone = any(x in suffix for x in ["base", "large"]) and any(
        x in family for x in ["bert", "roberta", "deberta", "gpt", "t5"]
    ) and not is_dataset_specific
    is_zero_shot_capable = any(t in tags for t in ["zero-shot-classification", "sentence-transformers", "feature-extraction"])

    role = "base_backbone" if is_base_backbone else "task_head"
    if is_zero_shot_capable:
        role = "zero_shot_model"

    domain = "general"
    if any(x in suffix for x in ["twitter", "tweet", "wnut"]):
        domain = "social_media"
    elif "news" in suffix or dataset_hint in {"ag_news", "conll2003"}:
        domain = "news"

    return {
        "model_id": model_id,
        "author": author,
        "family": family,
        "role": role,
        "task_family": task_family,
        "input_schemas": input_schemas,
        "output_type": output_type,
        "domain": domain,
        "is_dataset_specific_checkpoint": is_dataset_specific,
        "dataset_hint": dataset_hint,
        "is_base_backbone": is_base_backbone,
        "is_zero_shot_capable": is_zero_shot_capable,
    }


def _assess_model_dataset_fit(model_profile: dict[str, Any], dataset_spec: dict[str, Any]) -> ModelFitAssessment:
    """Assess model/dataset compatibility with explicit semantic and label-space checks."""
    if dataset_spec["task_family"] not in {model_profile["task_family"], "sentence_similarity"} and not model_profile["is_base_backbone"]:
        return ModelFitAssessment("reject", "Task family mismatch for non-backbone checkpoint", "none")

    if dataset_spec["input_schema"] not in model_profile["input_schemas"] and not model_profile["is_base_backbone"]:
        return ModelFitAssessment("reject", "Input schema mismatch", "none")

    if model_profile["is_dataset_specific_checkpoint"]:
        hint = model_profile["dataset_hint"]
        dataset_name = dataset_spec["dataset_name"]
        dataset_config = dataset_spec.get("dataset_config")
        effective_name = dataset_config if dataset_name == "glue" and dataset_config else dataset_name

        if dataset_spec["task_family"] == "text_classification" and hint and hint != effective_name:
            return ModelFitAssessment(
                "reject",
                "Dataset-specific text-classification head has incompatible label space",
                "none",
            )

        if dataset_spec["task_family"] == "token_classification" and hint and hint != effective_name:
            return ModelFitAssessment(
                "reject",
                "Dataset-specific token-classification head has incompatible entity schema",
                "none",
            )

    preferred = dataset_spec.get("preferred_run_regimes", [])

    if dataset_spec["task_family"] == model_profile["task_family"]:
        if model_profile["is_dataset_specific_checkpoint"]:
            return ModelFitAssessment("exact_match", "Task and dataset-specific head align", "finetune_exact")
        return ModelFitAssessment("acceptable_transfer", "Task aligns and generic checkpoint is transferable", "finetune_transfer")

    if model_profile["is_zero_shot_capable"] and "inference" in preferred:
        return ModelFitAssessment("inference_only", "Zero-shot capable model used in inference regime", "inference")

    if model_profile["is_base_backbone"] and "finetune_transfer" in preferred:
        return ModelFitAssessment("acceptable_transfer", "Generic backbone can be adapted via transfer learning", "finetune_transfer")

    return ModelFitAssessment("reject", "No safe transfer path for this model-task pair", "none")


def _choose_model_type(run_regime: str) -> str:
    """Map run regime to manifest model_type values."""
    if run_regime == "inference":
        return "hf"
    if run_regime in {"finetune_exact", "finetune_transfer"}:
        return "hf_finetune"
    return "hf"


def _sample_training_knobs(rng: random.Random, run_regime: str, model_profile: dict[str, Any]) -> dict[str, Any]:
    """Generate regime-aware, realistic transformer training and inference knobs."""
    distribution = rng.choices(["iid", "dirichlet", "shards"], weights=[0.55, 0.3, 0.15], k=1)[0]
    num_shards: int | None = None
    dirichlet_alpha: float | None = None
    if distribution == "shards":
        num_shards = rng.choice([5, 10, 20])
    elif distribution == "dirichlet":
        dirichlet_alpha = rng.choice([0.1, 0.3, 0.5])

    if run_regime == "inference":
        return {
            "batch_size": rng.choice([8, 16, 32]),
            "learning_rate": 0.0,
            "optimizer": "adamw",
            "seed": rng.randint(1, 1_000_000),
            "distribution": distribution,
            "num_shards": num_shards,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "dirichlet_alpha": dirichlet_alpha,
            "save_weights": False,
        }

    family = model_profile["family"]
    if any(x in family for x in ["bert", "roberta", "deberta", "electra"]):
        learning_rate = rng.choice([1e-5, 2e-5, 3e-5, 5e-5])
    elif any(x in family for x in ["t5", "bart"]):
        learning_rate = rng.choice([3e-5, 5e-5, 1e-4])
    else:
        learning_rate = rng.choice([2e-5, 5e-5, 1e-4])

    return {
        "batch_size": rng.choice([8, 16, 32]),
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


def _compute_realism_score(
    *,
    model_profile: dict[str, Any],
    dataset_spec: dict[str, Any],
    fit: ModelFitAssessment,
    is_override: bool,
) -> tuple[float, bool]:
    """Score candidate realism based on task fit, domain alignment, and benchmark preferences."""
    if fit.decision == "reject":
        return (0.0, False)

    score = 0.0
    score += {"exact_match": 1.0, "acceptable_transfer": 0.8, "inference_only": 0.6}[fit.decision]

    domain_alignment = dataset_spec["domain"] == model_profile["domain"] or model_profile["domain"] == "general"
    if domain_alignment:
        score += 0.2

    if is_override:
        score += 0.25

    if model_profile["is_dataset_specific_checkpoint"] and fit.decision != "exact_match":
        score -= 0.3

    return (round(score, 4), domain_alignment)


def _row_for(
    *,
    run_group_id: str,
    task_spec: TaskSpec,
    model_profile: dict[str, Any],
    dataset_spec: dict[str, Any],
    run_index: int,
    training_knobs: dict[str, Any],
    fit: ModelFitAssessment,
    realism_score: float,
    domain_alignment: bool,
) -> dict[str, Any]:
    ds_slug = dataset_spec["dataset_name"].replace("/", "_")
    model_id = model_profile["model_id"]
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
        "save_weights": training_knobs["save_weights"],
        "model_type": _choose_model_type(fit.recommended_run_regime),
        "hf_task": task_spec.hf_task,
        "task_type": task_spec.task_type,
        "modality": dataset_spec.get("modality")
        or ("image" if task_spec.pipeline_tag.startswith("image") or task_spec.pipeline_tag == "object-detection" else "text"),
        "dataset_name": dataset_spec["dataset_name"],
        "dataset_config": dataset_spec.get("dataset_config"),
        "hf_model_id": model_id,
        "train_split": dataset_spec["train_split"],
        "test_split": dataset_spec["test_split"],
        "label_column": dataset_spec.get("label_column"),
        "text_column": dataset_spec.get("text_column"),
        "image_column": dataset_spec.get("image_column"),
        "task_tag": dataset_spec.get("task_tag") or task_spec.task_tag,
        "run_regime": fit.recommended_run_regime,
        "model_role": model_profile["role"],
        "input_schema": dataset_spec["input_schema"],
        "fit_decision": fit.decision,
        "fit_reason": fit.reason,
        "realism_score": realism_score,
        "domain_alignment": domain_alignment,
        "dataset_hint": model_profile["dataset_hint"],
    }


def _is_row_valid(row: dict[str, Any]) -> bool:
    """Validate final manifest rows and drop semantically invalid entries."""
    if row["fit_decision"] == "reject":
        return False
    if row["run_regime"] not in {"inference", "finetune_exact", "finetune_transfer"}:
        return False
    if row["model_type"] == "hf" and row["run_regime"] != "inference":
        return False
    if row["model_type"] == "hf_finetune" and row["run_regime"] == "inference":
        return False
    return True


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
        profile = _normalize_model_profile(model)
        if not profile["model_id"]:
            continue

        if by_author.get(profile["author"], 0) >= max_per_author:
            continue
        if by_family.get(profile["family"], 0) >= max_per_family:
            continue

        selected.append(model)
        by_author[profile["author"]] = by_author.get(profile["author"], 0) + 1
        by_family[profile["family"]] = by_family.get(profile["family"], 0) + 1

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

        dataset_pool = SUPPORTED_DATASETS.get(task_spec.pipeline_tag, [])
        for model in selected_models:
            model_profile = _normalize_model_profile(model)
            candidate_rows: list[tuple[float, dict[str, Any]]] = []

            for dataset_spec in dataset_pool:
                fit = _assess_model_dataset_fit(model_profile, dataset_spec)
                key = (dataset_spec["dataset_name"], dataset_spec.get("dataset_config"))
                overrides = PREFERRED_MODEL_OVERRIDES.get(key, [])
                is_override = model_profile["model_id"] in overrides
                realism_score, domain_alignment = _compute_realism_score(
                    model_profile=model_profile,
                    dataset_spec=dataset_spec,
                    fit=fit,
                    is_override=is_override,
                )
                if realism_score <= 0:
                    continue

                row = _row_for(
                    run_group_id=run_group_id,
                    task_spec=task_spec,
                    model_profile=model_profile,
                    dataset_spec=dataset_spec,
                    run_index=len(rows) + len(candidate_rows) + 1,
                    training_knobs=_sample_training_knobs(rng, fit.recommended_run_regime, model_profile),
                    fit=fit,
                    realism_score=realism_score,
                    domain_alignment=domain_alignment,
                )
                if _is_row_valid(row):
                    candidate_rows.append((realism_score, row))

            candidate_rows.sort(key=lambda x: x[0], reverse=True)
            for _, row in candidate_rows[: max(1, datasets_per_model)]:
                row["external_run_id"] = f"hf_{task_spec.task_label}_{len(rows) + 1:06d}"
                rows.append(row)

        print(f"[{task_key}] fetched={len(all_models)} selected={len(selected_models)} rows={len(rows)}")

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
