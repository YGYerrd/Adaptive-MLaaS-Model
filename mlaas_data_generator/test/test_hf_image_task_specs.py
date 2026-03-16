import numpy as np

from mlaas_data_generator.models.adapters.hf_task import (
    ImageClassificationSpec,
    ObjectDetectionSpec,
    ImageSegmentationSpec,
)


def test_image_classification_metrics_from_statistics_topk():
    spec = ImageClassificationSpec()
    out = spec.metrics_from_statistics({"top1_correct": 7, "top5_correct": 9, "total": 10})
    assert np.isclose(out["primary"], 0.7)
    assert np.isclose(out["secondary"], 0.9)
    assert np.isclose(out["named_metrics"]["top1_accuracy"], 0.7)
    assert np.isclose(out["named_metrics"]["top5_accuracy"], 0.9)


def test_object_detection_metrics_from_statistics_map_summary():
    spec = ObjectDetectionSpec()
    out = spec.metrics_from_statistics(
        {
            "gt": 20,
            "tp_0.5": 12,
            "fp_0.5": 6,
            "tp_0.75": 10,
            "fp_0.75": 8,
            "tp_0.95": 6,
            "fp_0.95": 10,
        }
    )
    assert "map" in out["named_metrics"]
    assert "map@0.5" in out["named_metrics"]
    assert out["primary"] <= out["secondary"]


def test_segmentation_metrics_from_statistics_iou_and_dice():
    spec = ImageSegmentationSpec()
    out = spec.metrics_from_statistics(
        {"intersection": 30, "pred_total": 40, "target_total": 50, "union": 60}
    )
    assert np.isclose(out["primary"], 0.5)
    assert np.isclose(out["secondary"], 2 * 30 / (40 + 50))
