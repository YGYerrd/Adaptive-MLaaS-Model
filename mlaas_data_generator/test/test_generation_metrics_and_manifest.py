from mlaas_data_generator.cli.manifest.hf_manifest_builder import MANIFEST_COLUMNS, build_hf_manifest
from mlaas_data_generator.federated.strategies.base import canonical_generation_metrics, metric_availability
from mlaas_data_generator.registry import DATASET_REGISTRY, MODEL_REGISTRY


def test_generation_registries_cover_curated_tasks_and_metadata():
    generation_datasets = {
        dataset_id: spec for dataset_id, spec in DATASET_REGISTRY.items() if spec["task_key"] in {"text_generation", "text2text_generation"}
    }
    assert set(generation_datasets) == {"wikitext2_lm", "cnn_dailymail"}
    assert generation_datasets["wikitext2_lm"]["pipeline_tag"] == "text-generation"
    assert generation_datasets["wikitext2_lm"]["loader_template"] == "hf_causal_lm"
    assert generation_datasets["wikitext2_lm"]["explainability"]["supports_feature_attribution"] is True

    assert generation_datasets["cnn_dailymail"]["pipeline_tag"] == "text2text-generation"
    assert generation_datasets["cnn_dailymail"]["dataset_name"] == "cnn_dailymail"
    assert generation_datasets["cnn_dailymail"]["explainability"]["supports_example_level_rationales"] is True

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
    assert generation_models["flan-t5-small_text2text"]["pipeline_tag"] == "text2text-generation"
    assert generation_models["flan-t5-small_text2text"]["family"] == "t5"
    assert generation_models["flan-t5-small_text2text"]["explainability"]["supports_attention_rollout"] is True

    assert "task_tag" in MANIFEST_COLUMNS
    assert not any(spec["task_key"] == "image_classification" for spec in DATASET_REGISTRY.values())
    assert not any(spec["task_key"] == "visual_question_answering" for spec in MODEL_REGISTRY.values())


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


def test_generation_metric_availability_by_subtype():
    assert canonical_generation_metrics("summarization", has_labels=True) == ("rouge1", "rouge2", "rougeL", "perplexity")
    assert canonical_generation_metrics("translation", has_labels=True) == ("sacrebleu", "perplexity")
    assert canonical_generation_metrics("translation", has_labels=False) == ("sacrebleu",)
    assert canonical_generation_metrics("captioning", has_labels=True) == ("cider", "bleu", "perplexity")

    avail = metric_availability("generation", task_tag="summarization", has_labels=True)
    assert avail["train"] == ("loss", "perplexity")
    assert avail["eval"][:3] == ("rouge1", "rouge2", "rougeL")

    avail_infer_only = metric_availability("generation", task_tag="translation", has_labels=False)
    assert avail_infer_only["train"] == tuple()
    assert avail_infer_only["eval"] == ("sacrebleu",)


def test_retrieval_and_vqa_metric_availability():
    retrieval = metric_availability("retrieval", has_labels=True)
    assert retrieval["eval"] == ("r@1", "r@5", "r@10")

    vqa = metric_availability("vqa", has_labels=True)
    assert vqa["eval"] == ("exact_match",)
