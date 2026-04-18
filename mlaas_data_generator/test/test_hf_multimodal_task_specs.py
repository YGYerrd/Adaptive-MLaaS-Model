import numpy as np

from mlaas_data_generator.models.adapters.hf_task import (
    ImageCaptioningSpec,
    TextImageRetrievalSpec,
    VQASpec,
)


def test_image_captioning_metrics_cider_and_bleu():
    spec = ImageCaptioningSpec()
    y_true = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    y_pred = np.array([[1, 2, 9, 4], [5, 0, 7, 8]])
    out = spec.metrics(y_true, y_pred)
    assert "cider" in out["named_metrics"]
    assert "bleu" in out["named_metrics"]
    assert out["primary"] >= out["secondary"]


def test_retrieval_metrics_from_statistics_recall_at_k():
    spec = TextImageRetrievalSpec()
    out = spec.metrics_from_statistics({"r1_correct": 3, "r5_correct": 7, "r10_correct": 9, "total": 10})
    assert np.isclose(out["named_metrics"]["accuracy"], 0.3)
    assert np.isclose(out["named_metrics"]["top1_accuracy"], 0.3)
    assert np.isclose(out["named_metrics"]["r@1"], 0.3)
    assert np.isclose(out["named_metrics"]["r@5"], 0.7)
    assert np.isclose(out["named_metrics"]["r@10"], 0.9)


def test_vqa_metrics_exact_match_with_normalization():
    spec = VQASpec()
    y_true = np.array(["The cat", "an apple", "blue"], dtype=object)
    y_pred = np.array(["cat", "apple!", "red"], dtype=object)
    out = spec.metrics(y_true, y_pred)
    assert np.isclose(out["named_metrics"]["exact_match"], 2 / 3)
