import argparse
import random
import uuid
from dataclasses import dataclass
import re
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
        "TweebankNLP/bertweet-wnut17-ner",
        "xlm-roberta-base",
    ],
}

ALLOWED_DATASET_TRANSFER_HINTS: dict[str, set[str]] = {
    "stsb": {"qqp", "mrpc"},
    "mrpc": {"qqp", "stsb"},
    "qqp": {"mrpc", "stsb"},
    "mnli": {"snli", "qnli", "rte"},
    "snli": {"mnli", "qnli", "rte"},
}

FROZEN_ONLY_MODEL_ROLES = {"embedding_model", "reranker_model", "zero_shot_classifier"}

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


def _dataset_schema(dataset_spec: dict[str, Any]) -> str:
    schema = dataset_spec.get("input_schema")
    if schema:
        return str(schema)
    text_column = str(dataset_spec.get("text_column") or "")
    if text_column.startswith("[") and text_column.endswith("]"):
        return "text_pair"
    if "tokens" in text_column:
        return "token_sequence"
    return "single_text"


def _effective_dataset_name(dataset_spec: dict[str, Any]) -> str:
    dataset_name = dataset_spec["dataset_name"]
    dataset_config = dataset_spec.get("dataset_config")
    if dataset_name == "glue" and dataset_config:
        return str(dataset_config)
    return str(dataset_name)


def _extract_num_labels(model: Any, tags: list[str]) -> int | None:
    for source in (getattr(model, "config", None), getattr(model, "cardData", None)):
        if isinstance(source, dict):
            value = source.get("num_labels")
            if isinstance(value, int):
                return value
    for tag in tags:
        m = re.search(r"num_labels:(\d+)", tag)
        if m:
            return int(m.group(1))
    return None


def _normalize_model_profile(model: Any) -> dict[str, Any]:
    """Normalize Hugging Face model metadata into compatibility-friendly fields."""
    model_id = getattr(model, "id", "") or ""
    tags = [str(t).lower() for t in (getattr(model, "tags", None) or [])]
    family = _infer_family(model_id, getattr(model, "architectures", None))
    author = model_id.split("/", 1)[0].lower() if "/" in model_id else "unknown"
    suffix = model_id.split("/", 1)[-1].lower()

    is_embedding_model = any(t in tags for t in ["sentence-transformers", "feature-extraction"])
    is_reranker_model = "reranker" in suffix or any("rerank" in t for t in tags)
    is_zero_shot_classifier = "zero-shot-classification" in tags

    task_family = "generic"
    if any(x in tags for x in ["text-classification", "sequence-classification"]):
        task_family = "text_classification"
    elif "token-classification" in tags:
        task_family = "token_classification"
    elif "sentence-similarity" in tags or is_embedding_model or is_reranker_model:
        task_family = "sentence_similarity"
    elif "fill-mask" in tags:
        task_family = "masked_language_modeling"
    elif "text-generation" in tags:
        task_family = "causal_lm"
    elif "text2text-generation" in tags:
        task_family = "seq2seq_generation"

    input_schemas = ["single_text"]
    if task_family == "sentence_similarity":
        input_schemas = ["text_pair", "single_text"]
    elif task_family == "token_classification":
        input_schemas = ["token_sequence"]
    elif task_family == "masked_language_modeling":
        input_schemas = ["single_text_with_mask", "single_text"]

    output_type = "labels" if "classification" in task_family else "text"
    if task_family == "sentence_similarity":
        output_type = "score"

    hint_keywords = {
        "sst2": ["sst-2", "sst2"],
        "ag_news": ["ag_news", "ag-news"],
        "conll2003": ["conll", "conll03"],
        "wnut_17": ["wnut", "wnut17"],
        "qqp": ["qqp", "quora"],
        "mrpc": ["mrpc"],
        "mnli": ["mnli", "multinli"],
        "qnli": ["qnli"],
        "rte": ["rte"],
        "stsb": ["stsb", "sts-b"],
        "imdb": ["imdb"],
        "snli": ["snli"],
        "cola": ["cola"],
        "squad": ["squad"],
    }
    dataset_hint = None
    for hint, keys in hint_keywords.items():
        if any(k in suffix for k in keys) or any(any(k in t for k in keys) for t in tags):
            dataset_hint = hint
            break

    known_dataset_tags = [
        "dataset:glue", "dataset:conll2003", "dataset:wnut_17", "dataset:ag_news", "dataset:imdb", "dataset:snli"
    ]
    is_dataset_specific = dataset_hint is not None or any(tag in tags for tag in known_dataset_tags)
    is_base_backbone = any(x in suffix for x in ["base", "large"]) and any(
        x in family for x in ["bert", "roberta", "deberta", "gpt", "t5", "llama", "mistral"]
    ) and not is_dataset_specific and task_family != "sentence_similarity"

    role = "task_head"
    if is_base_backbone:
        role = "base_backbone"
    elif is_embedding_model:
        role = "embedding_model"
    elif is_reranker_model:
        role = "reranker_model"
    elif is_zero_shot_classifier:
        role = "zero_shot_classifier"

    domain = "general"
    if any(x in suffix for x in ["twitter", "tweet", "wnut"]):
        domain = "social_media"
    elif "news" in suffix or dataset_hint in {"ag_news", "conll2003", "cnn_dailymail"}:
        domain = "news"
    elif "imdb" in suffix:
        domain = "movie_reviews"

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
        "is_zero_shot_capable": is_zero_shot_classifier,
        "is_embedding_model": is_embedding_model,
        "is_reranker_model": is_reranker_model,
        "num_labels": _extract_num_labels(model, tags),
    }


def _assess_model_dataset_fit(model_profile: dict[str, Any], dataset_spec: dict[str, Any]) -> ModelFitAssessment:
    """Assess model/dataset compatibility with explicit semantic and label-space checks."""
    dataset_task_family = dataset_spec["task_family"]
    dataset_schema = _dataset_schema(dataset_spec)
    if dataset_task_family != model_profile["task_family"] and not model_profile["is_base_backbone"]:
        if not (model_profile["is_zero_shot_capable"] and "inference" in dataset_spec.get("preferred_run_regimes", [])):
            return ModelFitAssessment("reject", "Task family mismatch for non-backbone checkpoint", "none")

    if dataset_schema not in model_profile["input_schemas"] and not model_profile["is_base_backbone"]:
        return ModelFitAssessment("reject", "Input schema mismatch", "none")

    effective_name = _effective_dataset_name(dataset_spec)
    hint = model_profile["dataset_hint"]
    if model_profile["is_dataset_specific_checkpoint"] and hint and hint != effective_name:
        allowed = ALLOWED_DATASET_TRANSFER_HINTS.get(hint, set())
        if effective_name not in allowed:
            return ModelFitAssessment("reject", "Dataset-specific head does not align with selected dataset", "none")

    expected_num_labels = dataset_spec.get("num_labels")
    observed_num_labels = model_profile.get("num_labels")
    if (
        expected_num_labels is not None
        and observed_num_labels is not None
        and dataset_task_family in {"text_classification", "token_classification", "sentence_similarity"}
        and model_profile["is_dataset_specific_checkpoint"]
        and expected_num_labels != observed_num_labels
    ):
        return ModelFitAssessment("reject", "Label-count mismatch between checkpoint head and dataset", "none")

    preferred = dataset_spec.get("preferred_run_regimes", [])
    if dataset_task_family == model_profile["task_family"]:
        if model_profile["is_dataset_specific_checkpoint"] and hint == effective_name:
            return ModelFitAssessment("exact_match", "Task and dataset-specific head align", "finetune_exact")
        if "finetune_transfer" in preferred:
            return ModelFitAssessment("acceptable_transfer", "Task aligns and generic checkpoint is transferable", "finetune_transfer")

    if model_profile["is_zero_shot_capable"] and "inference" in preferred:
        return ModelFitAssessment("inference_only", "Zero-shot classifier used in inference regime", "inference")

    if model_profile["is_base_backbone"] and "finetune_transfer" in preferred:
        return ModelFitAssessment("acceptable_transfer", "Generic backbone can be adapted via transfer learning", "finetune_transfer")

    return ModelFitAssessment("reject", "No safe transfer path for this model-task pair", "none")


def _choose_model_type(run_regime: str) -> str:
    """Map run regime to manifest model_type values."""
    if run_regime == "inference":
        return "hf"
    if run_regime in {"finetune_exact", "finetune_transfer"}:
        return "hf_finetune"
    raise ValueError(f"Unknown run regime: {run_regime}")


def _sample_training_knobs(
    rng: random.Random,
    run_regime: str,
    model_profile: dict[str, Any],
    dataset_spec: dict[str, Any],
) -> dict[str, Any]:
    """Generate task-aware transformer training and inference knobs."""
    distribution = rng.choices(["iid", "dirichlet", "shards"], weights=[0.55, 0.3, 0.15], k=1)[0]
    num_shards: int | None = None
    dirichlet_alpha: float | None = None
    if distribution == "shards":
        num_shards = rng.choice([5, 10, 20])
    elif distribution == "dirichlet":
        dirichlet_alpha = rng.choice([0.1, 0.3, 0.5])

    task_family = dataset_spec["task_family"]
    family = model_profile["family"]
    is_large_model = any(x in family for x in ["llama", "mistral", "t5"]) or "large" in family

    if run_regime == "inference":
        batch_choices = [4, 8, 16] if task_family == "token_classification" else [8, 16, 32]
        return {
            "batch_size": rng.choice(batch_choices),
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

    if task_family == "token_classification":
        batch_choices = [4, 8, 16]
    elif task_family == "seq2seq_generation":
        batch_choices = [2, 4, 8]
    else:
        batch_choices = [8, 16, 32]
    if is_large_model:
        batch_choices = [x for x in batch_choices if x <= 8] or [4]

    if task_family in {"seq2seq_generation", "causal_lm"}:
        learning_rate = rng.choice([1e-5, 2e-5, 3e-5])
    elif any(x in family for x in ["bert", "roberta", "deberta", "electra"]):
        learning_rate = rng.choice([1e-5, 2e-5, 3e-5, 5e-5])
    else:
        learning_rate = rng.choice([2e-5, 5e-5, 1e-4])

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


def _federated_defaults(run_regime: str, model_profile: dict[str, Any], dataset_spec: dict[str, Any]) -> dict[str, Any]:
    task_family = dataset_spec["task_family"]
    if run_regime == "inference":
        return {"num_rounds": 1, "num_clients": 1, "local_epochs": 1, "device": "cpu"}
    local_epochs = 2 if model_profile["is_base_backbone"] else 1
    if task_family == "token_classification":
        return {"num_rounds": 5, "num_clients": 5, "local_epochs": local_epochs, "device": "cpu"}
    if task_family in {"seq2seq_generation", "causal_lm"}:
        return {"num_rounds": 3, "num_clients": 3, "local_epochs": local_epochs, "device": "cpu"}
    return {"num_rounds": 4, "num_clients": 5, "local_epochs": local_epochs, "device": "cpu"}


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
    fed_defaults = _federated_defaults(fit.recommended_run_regime, model_profile, dataset_spec)

    return {
        "external_run_id": ext_id,
        "dataset": "hf",
        "run_group_id": run_group_id,
        "case_name": case_name,
        "notes": f"Auto-generated HF manifest row for {task_spec.pipeline_tag}",
        "enabled": True,
        "measure_system_metrics": True,
        "mixed_precision": False,
        "num_rounds": fed_defaults["num_rounds"],
        "num_clients": fed_defaults["num_clients"],
        "client_participation_rate": 1.0,
        "local_epochs": fed_defaults["local_epochs"],
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
        "device": fed_defaults["device"],
        "save_weights": training_knobs["save_weights"],
        "model_type": _choose_model_type(fit.recommended_run_regime),
        "hf_task": task_spec.hf_task,
        "task_type": task_spec.task_type,
        "modality": dataset_spec.get("modality") or "text",
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
        "input_schema": _dataset_schema(dataset_spec),
        "fit_decision": fit.decision,
        "fit_reason": fit.reason,
        "realism_score": realism_score,
        "domain_alignment": domain_alignment,
        "dataset_hint": model_profile["dataset_hint"],
    }


def _is_row_valid(row: dict[str, Any], model_profile: dict[str, Any], dataset_spec: dict[str, Any]) -> bool:
    """Validate final manifest rows and drop semantically invalid entries."""
    if row["fit_decision"] == "reject":
        return False
    if row["run_regime"] not in {"inference", "finetune_exact", "finetune_transfer"}:
        return False
    if row["model_type"] == "hf" and row["run_regime"] != "inference":
        return False
    if row["model_type"] == "hf_finetune" and row["run_regime"] == "inference":
        return False

    required_columns = {"dataset_name", "hf_model_id", "run_regime", "input_schema", "task_type", "model_type"}
    if any(row.get(c) in {None, ""} for c in required_columns):
        return False

    dataset_schema = _dataset_schema(dataset_spec)
    if dataset_schema == "text_pair" and "text_pair" not in model_profile["input_schemas"] and not model_profile["is_base_backbone"]:
        return False

    dataset_task_family = dataset_spec["task_family"]
    if dataset_task_family != model_profile["task_family"] and not model_profile["is_base_backbone"] and not model_profile["is_zero_shot_capable"]:
        return False

    effective_name = _effective_dataset_name(dataset_spec)
    hint = model_profile.get("dataset_hint")
    if model_profile["is_dataset_specific_checkpoint"] and hint and hint != effective_name:
        allowed = ALLOWED_DATASET_TRANSFER_HINTS.get(hint, set())
        if effective_name not in allowed:
            return False

    if (
        model_profile.get("num_labels") is not None
        and dataset_spec.get("num_labels") is not None
        and model_profile["is_dataset_specific_checkpoint"]
        and dataset_task_family in {"text_classification", "token_classification", "sentence_similarity"}
        and int(model_profile["num_labels"]) != int(dataset_spec["num_labels"])
    ):
        return False

    if row["run_regime"] != "inference" and model_profile["role"] in FROZEN_ONLY_MODEL_ROLES:
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
            candidates: list[dict[str, Any]] = []

            for dataset_spec in dataset_pool:
                fit = _assess_model_dataset_fit(model_profile, dataset_spec)
                key = (dataset_spec["dataset_name"], dataset_spec.get("dataset_config"))
                is_override = model_profile["model_id"] in PREFERRED_MODEL_OVERRIDES.get(key, [])
                realism_score, domain_alignment = _compute_realism_score(
                    model_profile=model_profile,
                    dataset_spec=dataset_spec,
                    fit=fit,
                    is_override=is_override,
                )
                if realism_score <= 0:
                    continue
                candidates.append(
                    {
                        "dataset_spec": dataset_spec,
                        "fit": fit,
                        "realism_score": realism_score,
                        "domain_alignment": domain_alignment,
                        "is_override": is_override,
                    }
                )

            candidates.sort(key=lambda c: (c["realism_score"], c["is_override"]), reverse=True)
            for candidate in candidates[: max(1, datasets_per_model)]:
                fit = candidate["fit"]
                row = _row_for(
                    run_group_id=run_group_id,
                    task_spec=task_spec,
                    model_profile=model_profile,
                    dataset_spec=candidate["dataset_spec"],
                    run_index=len(rows) + 1,
                    training_knobs=_sample_training_knobs(
                        rng,
                        fit.recommended_run_regime,
                        model_profile,
                        candidate["dataset_spec"],
                    ),
                    fit=fit,
                    realism_score=candidate["realism_score"],
                    domain_alignment=candidate["domain_alignment"],
                )
                if _is_row_valid(row, model_profile, candidate["dataset_spec"]):
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
