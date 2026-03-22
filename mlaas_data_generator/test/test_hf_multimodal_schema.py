import sys
import types

import numpy as np

from mlaas_data_generator.data.sources.huggingface import load_huggingface_source
from mlaas_data_generator.data.preprocessors.hf_multimodal import preprocess_hf_multimodal


class DummyDS:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self.rows]
        return self.rows[key]

    def select(self, idxs):
        return DummyDS([self.rows[i] for i in idxs])

    def train_test_split(self, test_size=0.2, seed=42, shuffle=True):
        n_test = max(1, int(len(self.rows) * test_size))
        return {"train": DummyDS(self.rows[:-n_test]), "test": DummyDS(self.rows[-n_test:])}


def test_hf_source_multimodal_pair_drop(monkeypatch):
    train_ds = DummyDS([
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "text": "a", "label": 0},
        {"image": None, "text": "b", "label": 1},
    ])
    test_ds = DummyDS([
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "text": "c", "label": 1},
    ])
    fake_mod = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: train_ds if kwargs.get("split") == "train" else test_ds
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    (train, _), (test, _), meta = load_huggingface_source(
        dataset_name="dummy",
        modality="multimodal",
        image_column="image",
        text_column="text",
        label_column="label",
        missing_pair_handling="drop",
    )

    assert len(train) + len(test) == 2
    assert meta["schema"]["text_column"] == "text"
    assert meta["schema"]["pair_validation"]["missing_pair_handling"] == "drop"
    assert meta["accounting"]["raw_record_count"] == 2
    assert meta["accounting"]["post_filter_record_count"] == 1


def test_hf_multimodal_preprocessor_contract(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "hello", "label": 1},
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "world", "label": 0},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "test", "label": 1},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, y_train), (_, _), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {"task_type": "classification"},
        hf_model_id="dummy/model",
        image_column="image",
        text_column="text",
        label_column="label",
        max_length=4,
    )

    assert set(x_train.keys()) == {"input_ids", "attention_mask", "pixel_values"}
    assert x_train["input_ids"].shape[0] == x_train["pixel_values"].shape[0] == len(y_train)
    assert meta["schema"]["batch_contract"]["combined_keys"] == ["input_ids", "attention_mask", "pixel_values"]
    assert meta["accounting"]["sequence_count"] == 2



def test_preprocess_hf_dispatches_vqa_without_multimodal_metadata(monkeypatch):
    from mlaas_data_generator.data.preprocessors import hf as hf_preprocessors

    captured = {}

    def _stub(train, test, meta, **kwargs):
        captured["meta"] = dict(meta)
        captured["kwargs"] = dict(kwargs)
        return (
            {
                "input_ids": np.array([[1, 2]]),
                "attention_mask": np.array([[1, 1]]),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
            },
            np.array([1]),
        ), (
            {
                "input_ids": np.array([[3, 4]]),
                "attention_mask": np.array([[1, 1]]),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
            },
            np.array([0]),
        ), meta

    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_multimodal", _stub)

    (_, _), (_, _), meta = hf_preprocessors.preprocess_hf(
        (DummyDS([{"image": np.ones((2, 2, 3), dtype=np.uint8), "question": "q", "answer": "a"}]), None),
        (DummyDS([{"image": np.ones((2, 2, 3), dtype=np.uint8), "question": "q2", "answer": "a2"}]), None),
        {"hf_task": "visual_question_answering"},
        hf_model_id="dummy/model",
    )

    assert meta["modality"] == "multimodal"
    assert meta["hf_task"] == "visual_question_answering"
    assert captured["kwargs"]["hf_task"] == "visual_question_answering"



def test_preprocess_hf_dispatches_retrieval_without_multimodal_metadata(monkeypatch):
    from mlaas_data_generator.data.preprocessors import hf as hf_preprocessors

    captured = {}

    def _stub(train, test, meta, **kwargs):
        captured["meta"] = dict(meta)
        captured["kwargs"] = dict(kwargs)
        return (
            {
                "input_ids": np.array([[1, 2]]),
                "attention_mask": np.array([[1, 1]]),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
            },
            np.array([0]),
        ), (
            {
                "input_ids": np.array([[3, 4]]),
                "attention_mask": np.array([[1, 1]]),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
            },
            np.array([0]),
        ), meta

    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_multimodal", _stub)

    (_, _), (_, _), meta = hf_preprocessors.preprocess_hf(
        (DummyDS([{"image": np.ones((2, 2, 3), dtype=np.uint8), "text": "caption"}]), None),
        (DummyDS([{"image": np.ones((2, 2, 3), dtype=np.uint8), "text": "caption 2"}]), None),
        {"hf_task": "text_image_retrieval"},
        hf_model_id="dummy/model",
    )

    assert meta["modality"] == "multimodal"
    assert meta["hf_task"] == "text_image_retrieval"
    assert captured["kwargs"]["hf_task"] == "text_image_retrieval"



def test_hf_multimodal_vqa_defaults(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "what?", "answer": "cat"},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "where?", "answer": "home"},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (_, y_train), (_, y_test), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/model",
        hf_task="visual_question_answering",
    )

    assert y_train.tolist() == ["cat"]
    assert y_test.tolist() == ["home"]
    assert meta["text_column"] == "question"
    assert meta["label_column"] == "answer"
