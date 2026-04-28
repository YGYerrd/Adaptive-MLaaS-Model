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
from mlaas_data_generator.models.adapters.hf_task import CausalLMGenerationSpec, Seq2SeqGenerationSpec
from mlaas_data_generator.registry import DATASET_REGISTRY, MODEL_REGISTRY
from mlaas_data_generator.runtime_compat import is_rocm_miopen_runtime_error
from mlaas_data_generator.storage.writer import make_writer


def test_generation_registries_cover_curated_tasks_and_metadata():
    generation_datasets = {
        dataset_id: spec for dataset_id, spec in DATASET_REGISTRY.items() if spec["task_key"] in {"text_generation", "text2text_generation"}
    }
    assert sum(spec["task_key"] == "text_generation" for spec in generation_datasets.values()) == 10
    assert sum(spec["task_key"] == "text2text_generation" for spec in generation_datasets.values()) == 10
    assert "wikitext2_text2text" not in generation_datasets
    assert generation_datasets["wikitext2_lm"]["pipeline_tag"] == "text-generation"
    assert generation_datasets["wikitext2_lm"]["loader_template"] == "hf_causal_lm"
    assert generation_datasets["wikitext2_lm"]["explainability"]["supports_feature_attribution"] is True
    assert generation_datasets["wikitext2_lm"]["explainability"]["preferred_methods"] == ["integrated_gradients", "token_saliency"]

    assert generation_datasets["cnn_dailymail"]["pipeline_tag"] == "text2text-generation"
    assert generation_datasets["cnn_dailymail"]["dataset_name"] == "cnn_dailymail"
    assert generation_datasets["cnn_dailymail"]["explainability"]["supports_example_level_rationales"] is True
    assert generation_datasets["cnn_dailymail"]["explainability"]["target_type"] == "summary_token"
    assert generation_datasets["xsum"]["dataset_name"] == "EdinburghNLP/xsum"
    assert generation_datasets["xsum"]["label_column"] == "summary"
    assert generation_datasets["samsum"]["text_column"] == "dialogue"
    assert generation_datasets["arxiv_summarization"]["label_column"] == "abstract"

    generation_models = {
        model_id: spec for model_id, spec in MODEL_REGISTRY.items() if spec["task_key"] in {"text_generation", "text2text_generation"}
    }
    assert sum(spec["task_key"] == "text_generation" for spec in generation_models.values()) == 10
    assert sum(spec["task_key"] == "text2text_generation" for spec in generation_models.values()) == 10
    assert {"distilgpt2", "gpt2", "google/flan-t5-small", "t5-small"}.issubset(
        {spec["hf_model_id"] for spec in generation_models.values()}
    )
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
    assert set(df["hf_task"]) == {"causal_lm_generation", "seq2seq_generation"}
    assert set(df["run_regime"]) == {"inference_only"}
    assert set(df["model_role"]) == {"service"}
    assert df["sample_size"].notna().all()
    assert (df["sample_size"].astype(int) == df["max_samples"].astype(int)).all()
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
    assert canonical_metric_names("generation", "metric", hf_task="seq2seq_generation", task_tag="summarization") == ("rouge1", "rouge2")

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


def test_run_manifest_uses_manifest_max_samples_before_config_sample_size():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf",
            "sample_size": "N/A",
            "max_samples": 64,
        }
    )

    resolved = _resolve_row(row, {})

    assert resolved["sample_size"] == 64
    assert resolved["max_samples"] == 64
    assert _build_dataset_args(resolved)["max_samples"] == 64


def test_run_manifest_falls_back_to_config_sample_size_when_manifest_samples_missing():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf",
            "sample_size": "N/A",
            "max_samples": "N/A",
        }
    )

    resolved = _resolve_row(row, {})

    assert resolved["sample_size"] == runner.CONFIG["sample_size"]


def test_run_manifest_preserves_experiment_service_metadata():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf_finetune",
            "hf_task": "sequence_classification",
            "hf_model_id": "bert-base-uncased",
            "dataset_name": "glue",
            "run_regime": "finetune_transfer",
            "service_source": "huggingface_hub",
            "modality": "text",
            "input_schema": "single_text",
            "model_role": "task_head",
            "fit_decision": "selected",
            "fit_reason": "selected score=0.95",
            "realism_score": 0.95,
            "domain_alignment": "benchmark",
            "hf_pipeline_tag": "text-classification",
            "hf_downloads": 123,
            "hf_likes": 45,
            "hf_author": "google",
            "hf_url": "https://huggingface.co/bert-base-uncased",
            "hf_service_meta_json": '{"service_variant_index": 0}',
        }
    )

    dataset_args = _build_dataset_args(_resolve_row(row, {}))

    assert dataset_args["run_regime"] == "finetune_transfer"
    assert dataset_args["service_source"] == "huggingface_hub"
    assert dataset_args["input_schema"] == "single_text"
    assert dataset_args["fit_reason"] == "selected score=0.95"
    assert dataset_args["realism_score"] == 0.95
    assert dataset_args["hf_service_meta_json"] == '{"service_variant_index": 0}'


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


def test_run_manifest_rejects_seq2seq_single_text_dataset_rows():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf",
            "hf_task": "seq2seq_generation",
            "hf_model_id": "google/flan-t5-small",
            "dataset_name": "wikitext",
            "text_column": "text",
            "label_column": "text",
        }
    )

    validation = runner._validate_row(_resolve_row(row, {}))

    assert not validation.ok
    assert "distinct source and target columns" in validation.error


def test_run_manifest_accepts_seq2seq_source_target_dataset_rows():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf",
            "hf_task": "seq2seq_generation",
            "hf_model_id": "google/flan-t5-small",
            "dataset_name": "cnn_dailymail",
            "text_column": "article",
            "label_column": "highlights",
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


def test_sqlite_writer_persists_run_failures_and_safe_large_params(tmp_path):
    db_path = tmp_path / "federated.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.seed_metrics()
    writer.write_run(
        {
            "run_id": "big-param-run",
            "dataset": "hf",
            "task_type": "generation",
            "model_type": "hf",
            "num_clients": 1,
            "num_rounds": 1,
        }
    )
    writer.write_run_param("big-param-run", "runner", "params_count", 2**80)
    writer.write_run_failure(
        external_run_id="hf_failed_001",
        row_index=7,
        case_name="failed_case",
        run_group_id="group-1",
        failure_stage="runtime_exception",
        error_message="boom",
        resolved_config_json='{"external_run_id": "hf_failed_001"}',
        traceback_text="Traceback...",
    )
    writer.finish()

    with sqlite3.connect(db_path) as conn:
        failure = conn.execute(
            "SELECT external_run_id, failure_stage, error_message FROM run_failures"
        ).fetchone()
        metric_type = conn.execute(
            "SELECT domain, data_type FROM metrics WHERE name = 'fail_reason_category'"
        ).fetchone()
        value = conn.execute(
            """
            SELECT value_int, value_num, value_text
            FROM run_params
            WHERE run_id = 'big-param-run' AND key = 'params_count'
            """
        ).fetchone()

    assert failure == ("hf_failed_001", "runtime_exception", "boom")
    assert metric_type == ("reliability", "text")
    assert value[0] is None
    assert value[1] is not None or value[2] is not None


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


class _TinyDecodeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    _vocab = {
        1: "the",
        2: "cat",
        3: "sat",
        4: "on",
        5: "mat",
    }

    def batch_decode(self, rows, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        decoded = []
        for row in rows:
            words = []
            for token_id in list(row):
                token_id = int(token_id)
                if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                    continue
                words.append(self._vocab.get(token_id, f"tok{token_id}"))
            decoded.append(" ".join(words))
        return decoded


def test_seq2seq_summarization_metrics_decode_text_before_rouge():
    spec = Seq2SeqGenerationSpec()
    out = spec.metrics(
        np.asarray([[1, 2, 3, 4, 5, -100]]),
        np.asarray([[2, 3, 5, 0, 0, 0]]),
        y_extra={
            "task_tag": "summarization",
            "loss_mean": 0.5,
            "ignore_index": -100,
            "tokenizer": _TinyDecodeTokenizer(),
        },
    )

    assert np.isclose(out["primary"], 0.75)
    assert np.isclose(out["named_metrics"]["rouge1"], 0.75)
    assert np.isclose(out["named_metrics"]["rouge2"], 1 / 3)
    assert np.isclose(out["named_metrics"]["rougel"], 0.75)
    assert np.isclose(out["named_metrics"]["perplexity"], np.exp(0.5))
