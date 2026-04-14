import numpy as np
import torch

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


def test_object_detection_encode_batch_converts_absolute_xywh_to_normalized_cxcywh():
    spec = ObjectDetectionSpec()
    xb = {"pixel_values": [np.zeros((3, 100, 200), dtype=np.float32)]}
    yb = [{"boxes": [[20, 10, 40, 30]], "classes": [2]}]

    _, labels_t, _ = spec.encode_batch(
        tokenizer=None,
        xb=xb,
        yb=yb,
        max_length=0,
        torch=torch,
        device=torch.device("cpu"),
    )

    boxes = labels_t[0]["boxes"].detach().cpu().numpy()
    # xywh absolute [20,10,40,30] on (h=100,w=200) -> xyxy norm [0.1,0.1,0.3,0.4] -> cxcywh [0.2,0.25,0.2,0.3]
    assert np.allclose(boxes, np.asarray([[0.2, 0.25, 0.2, 0.3]], dtype=np.float32), atol=1e-6)


def test_object_detection_batch_metric_statistics_from_outputs_counts_true_positive():
    spec = ObjectDetectionSpec(score_threshold=0.05)
    labels_t = [
        {
            "class_labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
        }
    ]

    class _Outputs:
        logits = torch.tensor([[[0.1, 5.0, -4.0]]], dtype=torch.float32)  # class 1 is top score, final index is no-object
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]], dtype=torch.float32)

    stats = spec.batch_metric_statistics_from_outputs(torch, _Outputs(), labels_t, {"score_threshold": 0.05})
    assert np.isclose(stats["gt"], 1.0)
    assert np.isclose(stats["tp_0.5"], 1.0)
    assert np.isclose(stats["tp_0.75"], 1.0)
    assert np.isclose(stats["tp_0.95"], 1.0)
