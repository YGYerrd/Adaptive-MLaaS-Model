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


def test_image_specs_do_not_require_tokenizer():
    assert ImageClassificationSpec.requires_tokenizer is False
    assert ObjectDetectionSpec.requires_tokenizer is False
    assert ImageSegmentationSpec.requires_tokenizer is False


def test_object_detection_does_not_require_num_labels_and_builds_without_it():
    class _AutoModelForObjectDetection:
        called_kwargs = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.called_kwargs = kwargs
            return {"model_id": model_id, "kwargs": kwargs}

    class _Transformers:
        AutoModelForObjectDetection = _AutoModelForObjectDetection

    spec = ObjectDetectionSpec()
    assert spec.requires_num_labels is False
    model = spec.build_model(_Transformers, "fake/model", num_labels=None)
    assert model["model_id"] == "fake/model"
    assert "num_labels" not in _AutoModelForObjectDetection.called_kwargs
