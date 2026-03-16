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
    ds = DummyDS([
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "text": "a", "label": 0},
        {"image": None, "text": "b", "label": 1},
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "text": "c", "label": 1},
    ])
    fake_mod = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: ds)
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
