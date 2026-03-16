import numpy as np
from unittest.mock import patch

from mlaas_data_generator.data.preprocessors.hf import preprocess_hf


class MiniDataset:
    def __init__(self, rows):
        self._rows = rows
        self.column_names = list(rows.keys())
        self.features = {k: None for k in self.column_names}

    def __getitem__(self, key):
        return self._rows[key]

    def __len__(self):
        first_col = self.column_names[0]
        return len(self._rows[first_col])


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "</s>"
    vocab_size = 256

    def __call__(self, texts, truncation=True, max_length=16, padding=False, add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        for txt in texts:
            toks = [((ord(c) % 50) + 3) for c in str(txt)]
            if add_special_tokens:
                toks = toks[: max(0, max_length - 1)] + [self.eos_token_id]
            else:
                toks = toks[:max_length]
            ids.append(toks)
        return {"input_ids": ids}


def _meta():
    return {"hf_id": "dummy", "max_length": 12, "seed": 123, "modality": "text"}


@patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
def test_causal_lm_prompt_target_column_mapping(_):
    train_ds = MiniDataset({"instruction": ["Hi", "Bye"], "output": ["there", "now"]})
    test_ds = MiniDataset({"instruction": ["Go"], "output": ["home"]})

    (train, test, meta2) = preprocess_hf(
        (train_ds, None),
        (test_ds, None),
        {**_meta(), "hf_task": "causal_lm"},
        hf_model_id="dummy-model",
        column_mapping={"prompt_column": "instruction", "target_column": "output"},
        source_max_length=8,
        target_max_length=6,
        dynamic_padding=True,
    )

    x_train, y_train = train
    assert set(x_train.keys()) == {"input_ids", "attention_mask"}
    assert x_train["input_ids"].shape == y_train.shape
    assert np.any(y_train == -100), "prompt tokens should be masked for causal LM"
    assert meta2["hf_task"] == "causal_lm"


@patch("transformers.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
def test_seq2seq_source_target_mapping_and_label_masking(_):
    train_ds = MiniDataset({"input": ["question1", "question2"], "label": ["answer1", "answer2"]})
    test_ds = MiniDataset({"input": ["question3"], "label": ["answer3"]})

    (train, test, meta2) = preprocess_hf(
        (train_ds, None),
        (test_ds, None),
        {**_meta(), "hf_task": "seq2seq"},
        hf_model_id="dummy-model",
        column_mapping={"source_column": "input", "target_column": "label"},
        source_max_length=10,
        target_max_length=7,
    )

    x_train, y_train = train
    assert set(x_train.keys()) == {"input_ids", "attention_mask"}
    assert y_train.ndim == 2
    assert np.any(y_train == -100), "target padding should be masked"
    assert meta2["column_mapping"]["source"] == "input"
    assert meta2["column_mapping"]["target"] == "label"
