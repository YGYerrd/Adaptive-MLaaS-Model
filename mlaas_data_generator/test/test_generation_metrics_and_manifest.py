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

    assert "image_classification" in TASK_SPECS
    assert TASK_SPECS["image_classification"].hf_task == "image_classification"
    assert "object_detection" in TASK_SPECS
    assert TASK_SPECS["object_detection"].hf_task == "image_detection"
    assert "image_segmentation" in TASK_SPECS
    assert TASK_SPECS["image_segmentation"].hf_task == "image_segmentation"

    assert "image-classification" in SUPPORTED_DATASETS
    assert "object-detection" in SUPPORTED_DATASETS
    assert "image-segmentation" in SUPPORTED_DATASETS

    assert "image_captioning" in TASK_SPECS
    assert TASK_SPECS["image_captioning"].hf_task == "image_captioning"
    assert "text_image_retrieval" in TASK_SPECS
    assert TASK_SPECS["text_image_retrieval"].hf_task == "text_image_retrieval"
    assert "visual_question_answering" in TASK_SPECS
    assert TASK_SPECS["visual_question_answering"].hf_task == "visual_question_answering"

    assert "image-to-text" in SUPPORTED_DATASETS
    assert "zero-shot-image-classification" in SUPPORTED_DATASETS
    assert "visual-question-answering" in SUPPORTED_DATASETS


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
