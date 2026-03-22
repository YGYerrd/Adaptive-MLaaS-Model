import sys
import types

import numpy as np

from mlaas_data_generator.data.preprocessors.hf import preprocess_hf


class DummySplit:
    def __init__(self, rows):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [r.get(item) for r in self._rows]
        return self._rows[item]


class FakeImageProcessor:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, image, return_tensors=None, do_resize=True, do_normalize=True, do_augment=False):
        arr = np.asarray(image, dtype=np.float32)
        if do_normalize and arr.max() > 1:
            arr = arr / 255.0
        if do_augment:
            arr = arr + 0.1
        # return HWC to exercise channel-order conversion.
        return {"pixel_values": arr}


def _install_fake_transformers():
    fake_mod = types.SimpleNamespace(AutoImageProcessor=FakeImageProcessor)
    sys.modules["transformers"] = fake_mod


def test_image_classification_routing_and_deterministic_eval():
    _install_fake_transformers()
    train_rows = [
        {"image": np.zeros((4, 4, 3), dtype=np.uint8), "label": 0},
        {"image": np.ones((4, 4, 3), dtype=np.uint8) * 255, "label": 1},
    ]
    test_rows = [{"image": np.ones((4, 4, 3), dtype=np.uint8), "label": 1}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "sequence_classification", "modality": "image", "task_type": "classification", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        training_augmentations=True,
        eval_augmentations=False,
    )

    x_train, y_train = train
    x_test, y_test = test

    assert meta["hf_task"] == "image_classification"
    assert x_train["pixel_values"].shape == (2, 3, 4, 4)
    assert x_test["pixel_values"].shape == (1, 3, 4, 4)
    assert np.isclose(float(x_train["pixel_values"][0, 0, 0, 0]), 0.1)
    assert not np.isclose(float(x_test["pixel_values"][0, 0, 0, 0]), 0.1)
    assert y_train.dtype == np.int64
    assert y_test.dtype == np.int64


def test_image_decode_error_skip_and_report():
    _install_fake_transformers()
    train_rows = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 0},
        {"image": object(), "label": 1},
    ]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "label": 0}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "sequence_classification", "modality": "image", "task_type": "classification", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        on_decode_error="skip",
        report_decode_errors=True,
    )

    x_train, y_train = train
    assert x_train["pixel_values"].shape[0] == 1
    assert y_train.shape[0] == 1
    assert meta["decode_report"]["train"]["failed"] == 1


def test_image_detection_schema_passthrough():
    _install_fake_transformers()
    train_rows = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [[0, 0, 1, 1]], "classes": [2]},
    ]
    test_rows = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [], "classes": []},
    ]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "sequence_classification", "modality": "image", "task_type": "detection", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        boxes_column="boxes",
        classes_column="classes",
    )

    _, y_train = train
    assert meta["schema"]["detection"]["boxes_column"] == "boxes"
    assert y_train[0]["boxes"].shape == (1, 4)
    assert y_train[0]["classes"].shape == (1,)


def test_image_task_dispatch_uses_hf_task_when_modality_missing():
    _install_fake_transformers()
    train_rows = [{"image": np.zeros((3, 3, 3), dtype=np.uint8), "label": 1}]
    test_rows = [{"image": np.ones((3, 3, 3), dtype=np.uint8), "label": 0}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "image_classification", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
    )

    x_train, y_train = train
    x_test, y_test = test
    assert meta["modality"] == "image"
    assert meta["task_type"] == "classification"
    assert meta["hf_task"] == "image_classification"
    assert x_train["pixel_values"].shape == (1, 3, 3, 3)
    assert x_test["pixel_values"].shape == (1, 3, 3, 3)
    assert y_train.tolist() == [1]
    assert y_test.tolist() == [0]


def test_image_task_dispatch_normalizes_detection_alias_with_wrong_modality():
    _install_fake_transformers()
    train_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [[0, 0, 1, 1]], "classes": [2]}]
    test_rows = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "boxes": [], "classes": []}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "object_detection", "modality": "text", "hf_id": "dummy"},
        hf_model_id="dummy/vision",
        boxes_column="boxes",
        classes_column="classes",
    )

    _, y_train = train
    _, y_test = test
    assert meta["modality"] == "image"
    assert meta["task_type"] == "detection"
    assert meta["hf_task"] == "image_detection"
    assert y_train[0]["boxes"].shape == (1, 4)
    assert y_test[0]["boxes"].shape == (0, 4)
