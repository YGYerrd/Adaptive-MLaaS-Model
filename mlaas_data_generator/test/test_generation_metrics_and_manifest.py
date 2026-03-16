from mlaas_data_generator.cli.manifest.hf_manifest_builder import TASK_SPECS, SUPPORTED_DATASETS, MANIFEST_COLUMNS
from mlaas_data_generator.federated.strategies.base import canonical_generation_metrics, metric_availability


def test_manifest_registers_generation_tasks_and_task_tag_column():
    assert "text_generation" in TASK_SPECS
    assert TASK_SPECS["text_generation"].pipeline_tag == "text-generation"
    assert TASK_SPECS["text_generation"].hf_task == "causal_lm_generation"

    assert "text2text_generation" in TASK_SPECS
    assert TASK_SPECS["text2text_generation"].pipeline_tag == "text2text-generation"
    assert TASK_SPECS["text2text_generation"].hf_task == "seq2seq_generation"

    assert "text-generation" in SUPPORTED_DATASETS
    assert "text2text-generation" in SUPPORTED_DATASETS
    assert "task_tag" in MANIFEST_COLUMNS


def test_generation_metric_availability_by_subtype():
    assert canonical_generation_metrics("summarization", has_labels=True) == ("rouge1", "rouge2", "rougeL", "perplexity")
    assert canonical_generation_metrics("translation", has_labels=True) == ("sacrebleu", "perplexity")
    assert canonical_generation_metrics("translation", has_labels=False) == ("sacrebleu",)

    avail = metric_availability("generation", task_tag="summarization", has_labels=True)
    assert avail["train"] == ("loss", "perplexity")
    assert avail["eval"][:3] == ("rouge1", "rouge2", "rougeL")

    avail_infer_only = metric_availability("generation", task_tag="translation", has_labels=False)
    assert avail_infer_only["train"] == tuple()
    assert avail_infer_only["eval"] == ("sacrebleu",)
