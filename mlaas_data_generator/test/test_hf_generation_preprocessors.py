import sys
import types

import numpy as np
import pytest

from mlaas_data_generator.data.preprocessors.hf import preprocess_hf


class DummySplit:
    def __init__(self, rows):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        return [r.get(key) for r in self._rows]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    padding_side = "right"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def _encode_one(self, text):
        base = [min(50, max(3, len(tok))) for tok in str(text).split()]
        return base or [3]

    def __call__(self, texts=None, text_target=None, truncation=True, padding=False, max_length=8, add_special_tokens=True, return_attention_mask=True, **kwargs):
        seqs = text_target if text_target is not None else texts
        if isinstance(seqs, str):
            seqs = [seqs]
        ids = []
        masks = []
        for t in seqs:
            token_ids = self._encode_one(t)
            if add_special_tokens:
                token_ids = [1] + token_ids
            token_ids = token_ids[: int(max_length)]
            ids.append(token_ids)
            masks.append([1] * len(token_ids))
        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = masks
        return out


def _install_fake_transformers():
    fake_mod = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)
    sys.modules["transformers"] = fake_mod


def test_causal_lm_generation_preprocessor_prompt_completion_mapping():
    _install_fake_transformers()
    train_rows = [
        {"prompt": "Write a haiku", "completion": "Soft rain at dusk"},
        {"prompt": "Translate hello", "completion": "bonjour"},
    ]
    test_rows = [{"prompt": "Say hi", "completion": "hi"}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 12, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        source_max_length=6,
        target_max_length=6,
        dynamic_padding=True,
    )

    x_train, y_train = train
    assert set(x_train.keys()) >= {"input_ids", "attention_mask"}
    assert x_train["input_ids"].shape == y_train.shape
    assert meta["column_mapping"]["prompt"] == "prompt"
    assert meta["column_mapping"]["target"] == "completion"
    assert meta["accounting"]["sequence_count"] == 2
    assert meta["accounting"]["supervised_token_count"] > 0


def test_seq2seq_generation_preprocessor_source_target_mapping():
    _install_fake_transformers()
    train_rows = [
        {"source_text": "summarize this long article", "target_text": "short summary"},
        {"source_text": "another source", "target_text": "target output with extra words"},
    ]
    test_rows = [{"source_text": "src", "target_text": "tgt"}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        source_max_length=7,
        target_max_length=5,
        dynamic_padding=True,
    )

    x_train, y_train = train
    x_test, y_test = test

    assert set(x_train.keys()) >= {"input_ids", "attention_mask"}
    assert x_train["input_ids"].shape[1] <= 7
    assert y_train.shape[1] <= 5
    assert x_train["input_ids"].shape[0] == y_train.shape[0]
    assert x_test["input_ids"].shape[0] == y_test.shape[0]
    assert np.any(y_train == -100)
    assert meta["column_mapping"]["source"] == "source_text"
    assert meta["column_mapping"]["target"] == "target_text"


def test_causal_lm_generation_preprocessor_single_text_column():
    _install_fake_transformers()
    train_rows = [
        {"text": "The quick brown fox"},
        {"text": "Jumps over lazy dogs"},
    ]
    test_rows = [{"text": "Single column inference text"}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 9, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        dynamic_padding=True,
    )

    x_train, y_train = train
    x_test, y_test = test

    assert set(x_train.keys()) >= {"input_ids", "attention_mask"}
    assert x_train["input_ids"].shape == y_train.shape
    assert x_test["input_ids"].shape == y_test.shape
    assert meta["generation_mode"] == "single_text"
    assert meta["column_mapping"] == {"text": "text"}

    non_pad_train = x_train["attention_mask"] == 1
    non_pad_test = x_test["attention_mask"] == 1
    assert np.array_equal(y_train[non_pad_train], x_train["input_ids"][non_pad_train])
    assert np.array_equal(y_test[non_pad_test], x_test["input_ids"][non_pad_test])
    assert np.all(y_train[~non_pad_train] == -100)
    assert np.all(y_test[~non_pad_test] == -100)


def test_seq2seq_generation_preprocessor_article_highlights_mapping():
    _install_fake_transformers()
    train_rows = [
        {"article": "Long article body", "highlights": "Short summary"},
        {"article": "Another document", "highlights": "Another summary"},
    ]
    test_rows = [{"article": "Held-out article", "highlights": "Held-out summary"}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        source_max_length=7,
        target_max_length=5,
        dynamic_padding=True,
    )

    x_train, y_train = train
    x_test, y_test = test

    assert x_train["input_ids"].shape[0] == y_train.shape[0]
    assert x_test["input_ids"].shape[0] == y_test.shape[0]
    assert meta["column_mapping"] == {"source": "article", "target": "highlights"}


def test_seq2seq_generation_preprocessor_column_mapping_overrides_heuristics():
    _install_fake_transformers()
    train_rows = [
        {"article": "Heuristic source", "highlights": "Heuristic target", "src": "Mapped source", "tgt": "Mapped target"},
    ]
    test_rows = [
        {"article": "Heuristic source test", "highlights": "Heuristic target test", "src": "Mapped source test", "tgt": "Mapped target test"},
    ]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        column_mapping={"source": "src", "target": "tgt"},
        source_max_length=7,
        target_max_length=5,
        dynamic_padding=True,
    )

    assert meta["column_mapping"] == {"source": "src", "target": "tgt"}


def test_seq2seq_generation_preprocessor_raises_without_plausible_target():
    _install_fake_transformers()
    train_rows = [{"article": "Long article body", "document": "Duplicate candidate source"}]
    test_rows = [{"article": "Held-out article", "document": "Held-out duplicate source"}]

    with pytest.raises(ValueError, match="Could not resolve target column"):
        preprocess_hf(
            (DummySplit(train_rows), None),
            (DummySplit(test_rows), None),
            {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
            hf_model_id="dummy/model",
            dynamic_padding=True,
        )


def test_causal_lm_generation_preprocessor_sets_left_padding_and_meta():
    _install_fake_transformers()
    train_rows = [{"text": "left pad me"}]
    test_rows = [{"text": "and me too"}]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 8, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        dynamic_padding=True,
    )

    assert meta["padding_side"] == "left"
    assert meta["pad_token_id"] == 0
