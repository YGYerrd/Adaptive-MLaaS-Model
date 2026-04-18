import numpy as np
import pandas as pd
import sqlite3

import mlaas_data_generator.cli.run_manifest as runner
from mlaas_data_generator.cli.manifest.hf_manifest_builder import MANIFEST_COLUMNS, build_hf_manifest
from mlaas_data_generator.cli.run_manifest import _build_dataset_args, _resolve_row
from mlaas_data_generator.federated.strategies.base import (
    canonical_generation_metrics,
    canonical_metric_names,
    canonical_task_family,
    metric_availability,
)
from mlaas_data_generator.federated.orchestrator import _resolve_database_run_id
from mlaas_data_generator.models.adapters.hf_task import CausalLMGenerationSpec
from mlaas_data_generator.registry import DATASET_REGISTRY, MODEL_REGISTRY
from mlaas_data_generator.runtime_compat import is_rocm_miopen_runtime_error
from mlaas_data_generator.storage.writer import make_writer


def test_generation_registries_cover_curated_tasks_and_metadata():
    generation_datasets = {
        dataset_id: spec for dataset_id, spec in DATASET_REGISTRY.items() if spec["task_key"] in {"text_generation", "text2text_generation"}
    }
    assert set(generation_datasets) == {"wikitext2_lm", "wikitext2_text2text", "cnn_dailymail"}
    assert generation_datasets["wikitext2_lm"]["pipeline_tag"] == "text-generation"
    assert generation_datasets["wikitext2_lm"]["loader_template"] == "hf_causal_lm"
    assert generation_datasets["wikitext2_lm"]["explainability"]["supports_feature_attribution"] is True
    assert generation_datasets["wikitext2_lm"]["explainability"]["preferred_methods"] == ["integrated_gradients", "token_saliency"]

    assert generation_datasets["cnn_dailymail"]["pipeline_tag"] == "text2text-generation"
    assert generation_datasets["cnn_dailymail"]["dataset_name"] == "cnn_dailymail"
    assert generation_datasets["cnn_dailymail"]["explainability"]["supports_example_level_rationales"] is True
    assert generation_datasets["cnn_dailymail"]["explainability"]["target_type"] == "summary_token"
    assert generation_datasets["wikitext2_text2text"]["dataset_name"] == "wikitext"
    assert generation_datasets["wikitext2_text2text"]["task_tag"] == "language-modeling"

    generation_models = {
        model_id: spec for model_id, spec in MODEL_REGISTRY.items() if spec["task_key"] in {"text_generation", "text2text_generation"}
    }
    assert {spec["hf_model_id"] for spec in generation_models.values()} == {
        "distilgpt2",
        "gpt2",
        "google/flan-t5-small",
        "t5-small",
    }
    assert generation_models["distilgpt2_textgen"]["pipeline_tag"] == "text-generation"
    assert generation_models["distilgpt2_textgen"]["loader_template"] == "hf_causal_lm"
    assert generation_models["distilgpt2_textgen"]["allowed_run_regimes"] == ["finetune_transfer", "inference_only"]
    assert generation_models["distilgpt2_textgen"]["explainability"]["requires_attention"] is False
    assert generation_models["flan-t5-small_text2text"]["pipeline_tag"] == "text2text-generation"
    assert generation_models["flan-t5-small_text2text"]["family"] == "t5"
    assert generation_models["flan-t5-small_text2text"]["explainability"]["supports_attention_rollout"] is True
    assert generation_models["flan-t5-small_text2text"]["explainability"]["preferred_methods"] == ["integrated_gradients", "attention_rollout"]

    assert "task_tag" in MANIFEST_COLUMNS
    assert "explainability_enabled" in MANIFEST_COLUMNS
    assert "explainability_method" in MANIFEST_COLUMNS
    assert "explainability_target" in MANIFEST_COLUMNS
    assert any(spec["task_key"] == "image_classification" for spec in DATASET_REGISTRY.values())
    assert any(spec["task_key"] == "visual_question_answering" for spec in MODEL_REGISTRY.values())


def test_manifest_builder_uses_flat_registries_for_generation_rows():
    df = build_hf_manifest(
        task_keys=["text_generation", "text2text_generation"],
        models_per_task=1,
        datasets_per_model=1,
        run_regimes=["inference_only"],
        variants_per_pair=1,
        seed=7,
    )

    assert set(df["hf_pipeline_tag"]) == {"text-generation", "text2text-generation"}
    assert set(df["dataset_name"]) == {"wikitext", "cnn_dailymail"}
    assert set(df["hf_model_id"]) == {"distilgpt2", "google/flan-t5-small"}
    assert set(df["run_regime"]) == {"inference_only"}
    assert set(df["model_role"]) == {"service"}
    assert set(df["explainability_enabled"]) == {True}
    assert set(df["explainability_method"]) == {"integrated_gradients"}
    assert set(df["explainability_target"]) == {"generated_token", "summary_token"}


def test_generation_metric_availability_by_subtype():
    assert canonical_generation_metrics(None, has_labels=True, hf_task="causal_lm_generation") == ("loss", "perplexity")
    assert canonical_generation_metrics("summarization", has_labels=True) == ("rouge1", "rouge2", "rougeL", "perplexity")
    assert canonical_generation_metrics("translation", has_labels=True) == ("sacrebleu", "perplexity")
    assert canonical_generation_metrics("translation", has_labels=False) == ("sacrebleu",)
    assert canonical_generation_metrics("captioning", has_labels=True) == ("cider", "bleu", "perplexity")

    assert canonical_metric_names("generation", "loss", hf_task="causal_lm_generation") == ("loss", "perplexity")

    causal_avail = metric_availability("generation", has_labels=True, hf_task="causal_lm_generation")
    assert causal_avail["train"] == ("loss", "perplexity")
    assert causal_avail["eval"] == ("loss", "perplexity")

    avail = metric_availability("generation", task_tag="summarization", has_labels=True)
    assert avail["train"] == ("loss", "perplexity")
    assert avail["eval"][:3] == ("rouge1", "rouge2", "rougeL")

    avail_infer_only = metric_availability("generation", task_tag="translation", has_labels=False)
    assert avail_infer_only["train"] == tuple()
    assert avail_infer_only["eval"] == ("sacrebleu",)


def test_image_classification_secondary_metric_is_f1():
    assert canonical_metric_names("classification", "metric", hf_task="image_classification") == ("accuracy", "f1")


def test_run_manifest_preserves_explainability_fields():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf",
            "hf_task": "text_generation",
            "hf_model_id": "distilgpt2",
            "dataset_name": "wikitext",
            "explainability_enabled": True,
            "explainability_method": "integrated_gradients",
            "explainability_target": "generated_token",
        }
    )

    resolved = _resolve_row(row, {})
    dataset_args = _build_dataset_args(resolved)

    assert resolved["explainability_enabled"] is True
    assert resolved["explainability_method"] == "integrated_gradients"
    assert resolved["explainability_target"] == "generated_token"
    assert dataset_args["explainability_enabled"] is True
    assert dataset_args["explainability_method"] == "integrated_gradients"
    assert dataset_args["explainability_target"] == "generated_token"


def test_run_manifest_allows_image_finetune_gpu_eligible_rows():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf_finetune",
            "hf_task": "image_classification",
            "hf_model_id": "apple/mobilevit-small",
            "dataset_name": "beans",
            "modality": "image",
            "run_regime": "finetune_transfer",
        }
    )

    validation = runner._validate_row(_resolve_row(row, {}))

    assert validation.ok


def test_database_run_id_prefers_manifest_external_run_id():
    assert _resolve_database_run_id({"external_run_id": " hf_imgcls_000016 "}, fallback_run_id="generated") == "hf_imgcls_000016"
    assert _resolve_database_run_id({"external_run_id": ""}, fallback_run_id="generated") == "generated"
    assert _resolve_database_run_id({}, fallback_run_id="generated") == "generated"


def test_rocm_miopen_runtime_error_detection():
    assert is_rocm_miopen_runtime_error("RuntimeError('miopenStatusUnknownError')")
    assert is_rocm_miopen_runtime_error("HIPRTC_ERROR_COMPILATION while building MIOpenBatchNorm")
    assert not is_rocm_miopen_runtime_error("RuntimeError('out of memory')")


def test_sqlite_writer_abort_rolls_back_run_rows(tmp_path):
    db_path = tmp_path / "federated.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.seed_metrics()
    writer.write_run(
        {
            "run_id": "skip-me",
            "dataset": "hf",
            "task_type": "classification",
            "model_type": "hf_finetune",
            "num_clients": 1,
            "num_rounds": 1,
        }
    )
    writer.abort()

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", ("skip-me",)).fetchone()[0]

    assert count == 0


def test_retrieval_and_vqa_metric_availability():
    retrieval = metric_availability("retrieval", has_labels=True)
    assert retrieval["eval"] == ("r@1", "r@5", "r@10")

    vqa = metric_availability("vqa", has_labels=True)
    assert vqa["eval"] == ("exact_match",)


def test_canonical_task_family_accepts_manifest_task_types():
    assert canonical_task_family("detection", None) == "detection"
    assert canonical_task_family("segmentation", None) == "segmentation"
    assert canonical_task_family("generation", None) == "generation"
    assert canonical_task_family("retrieval", None) == "retrieval"
    assert canonical_task_family("image_classification", None) == "classification"


def test_causal_lm_metrics_report_loss_as_primary_and_perplexity_as_secondary():
    spec = CausalLMGenerationSpec()
    out = spec.metrics(
        np.asarray([[1, 2, 3]]),
        np.asarray([[1, 4, 3]]),
        y_extra={"loss_mean": 0.5},
    )

    assert np.isclose(out["primary"], 0.5)
    assert np.isclose(out["secondary"], np.exp(0.5))
    assert np.isclose(out["named_metrics"]["cross_entropy_loss"], 0.5)
    assert np.isclose(out["named_metrics"]["perplexity"], np.exp(0.5))
    assert np.isclose(out["named_metrics"]["token_accuracy"], 2 / 3)
