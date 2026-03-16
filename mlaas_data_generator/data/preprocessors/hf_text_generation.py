import numpy as np

from .label_schema import attach_label_schema


PROMPT_COLUMN_CANDIDATES = ("prompt", "instruction")
TARGET_COLUMN_CANDIDATES = ("completion", "response", "output")
SOURCE_COLUMN_CANDIDATES = ("source_text", "input")
TARGET_TEXT_COLUMN_CANDIDATES = ("target_text", "label")
TEXT_COLUMN_CANDIDATES = ("text",)


def _pick_column(explicit_value, cols, candidates, column_role):
    if explicit_value:
        if explicit_value not in cols:
            raise ValueError(f"Missing {column_role} '{explicit_value}'. Available={sorted(cols)}")
        return explicit_value

    for c in candidates:
        if c in cols:
            return c

    raise ValueError(
        f"Could not infer {column_role} from columns {sorted(cols)}. "
        f"Set an explicit column_mapping/{column_role} in dataset args."
    )


def _load_tokenizer(hf_model_id):
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise ImportError(
            "HF generation preprocessing requires 'transformers'. Install with: pip install transformers"
        ) from e

    try:
        tok = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    return tok


def _pad_sequences(rows, pad_value, dynamic_padding=False, max_length=None):
    if not rows:
        return np.zeros((0, 0), dtype="int32")

    if dynamic_padding:
        target_len = max(len(r) for r in rows)
    else:
        target_len = int(max_length) if max_length else max(len(r) for r in rows)

    out = np.full((len(rows), target_len), int(pad_value), dtype="int32")
    for i, row in enumerate(rows):
        cur = row[:target_len]
        out[i, : len(cur)] = np.asarray(cur, dtype="int32")
    return out


def _tokenize_texts(tokenizer, texts, max_length):
    return tokenizer(
        list(texts),
        truncation=True,
        max_length=int(max_length),
        padding=False,
        add_special_tokens=True,
    )


def preprocess_hf_text_causal_lm(
    train,
    test,
    meta,
    *,
    hf_model_id,
    prompt_column=None,
    target_column=None,
    text_column=None,
    source_max_length=None,
    target_max_length=None,
    dynamic_padding=False,
    label_pad_value=-100,
    label_strategy="target_only",
):
    ds_train, _ = train
    ds_test, _ = test

    cols = set(ds_train.column_names)
    tokenizer = _load_tokenizer(hf_model_id)

    label_pad_value = int(label_pad_value)
    source_max_length = int(source_max_length or meta.get("max_length", 128))
    target_max_length = int(target_max_length or meta.get("max_length", 128))
    dynamic_padding = bool(dynamic_padding)
    label_strategy = str(label_strategy or "target_only").strip().lower()

    resolved_text_column = None
    resolved_prompt_column = None
    resolved_target_column = None

    if text_column:
        resolved_text_column = _pick_column(text_column, cols, TEXT_COLUMN_CANDIDATES, "text_column")
    elif prompt_column or target_column:
        resolved_prompt_column = _pick_column(prompt_column, cols, PROMPT_COLUMN_CANDIDATES, "prompt_column")
        resolved_target_column = _pick_column(target_column, cols, TARGET_COLUMN_CANDIDATES, "target_column")
    else:
        if any(c in cols for c in PROMPT_COLUMN_CANDIDATES) and any(c in cols for c in TARGET_COLUMN_CANDIDATES):
            resolved_prompt_column = _pick_column(None, cols, PROMPT_COLUMN_CANDIDATES, "prompt_column")
            resolved_target_column = _pick_column(None, cols, TARGET_COLUMN_CANDIDATES, "target_column")
        else:
            resolved_text_column = _pick_column(None, cols, TEXT_COLUMN_CANDIDATES, "text_column")

    def _build_features(ds):
        input_rows = []
        label_rows = []

        if resolved_text_column:
            toks = _tokenize_texts(tokenizer, ds[resolved_text_column], max_length=source_max_length)
            for token_ids in toks["input_ids"]:
                ids = list(token_ids)
                if label_strategy == "target_only":
                    labels = list(ids)
                elif label_strategy == "full_text":
                    labels = list(ids)
                else:
                    raise ValueError("causal_lm label_strategy must be one of: target_only, full_text")

                input_rows.append(ids)
                label_rows.append(labels)

        else:
            prompt_tok = _tokenize_texts(tokenizer, ds[resolved_prompt_column], max_length=source_max_length)
            target_tok = _tokenize_texts(tokenizer, ds[resolved_target_column], max_length=target_max_length)

            eos_id = tokenizer.eos_token_id
            for p_ids, t_ids in zip(prompt_tok["input_ids"], target_tok["input_ids"]):
                p = list(p_ids)
                t = list(t_ids)
                if eos_id is not None and (not t or t[-1] != eos_id):
                    t = t + [int(eos_id)]

                merged = p + t
                labels = ([label_pad_value] * len(p)) + list(t)
                if label_strategy == "full_text":
                    labels = list(merged)

                input_rows.append(merged)
                label_rows.append(labels)

        input_ids = _pad_sequences(
            input_rows,
            pad_value=tokenizer.pad_token_id,
            dynamic_padding=dynamic_padding,
            max_length=(source_max_length + target_max_length),
        )
        attn_mask = (input_ids != int(tokenizer.pad_token_id)).astype("int32")

        labels = _pad_sequences(
            label_rows,
            pad_value=label_pad_value,
            dynamic_padding=dynamic_padding,
            max_length=input_ids.shape[1],
        )
        labels[attn_mask == 0] = label_pad_value

        return {"input_ids": input_ids, "attention_mask": attn_mask}, labels

    X_train, y_train = _build_features(ds_train)
    X_test, y_test = _build_features(ds_test)

    max_input_len = int(X_train["input_ids"].shape[1]) if X_train["input_ids"].ndim == 2 else 0

    meta2 = dict(meta)
    meta2.update({
        "hf_task": "causal_lm",
        "modality": "text",
        "x_format": "dict",
        "x_keys": ["input_ids", "attention_mask"],
        "label_granularity": "token",
        "num_classes": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "label_pad_value": label_pad_value,
        "hf_model_id": hf_model_id,
        "source_max_length": source_max_length,
        "target_max_length": target_max_length,
        "dynamic_padding": dynamic_padding,
        "label_strategy": label_strategy,
        "input_shape": (max_input_len,),
        "column_mapping": {
            "text": resolved_text_column,
            "prompt": resolved_prompt_column,
            "target": resolved_target_column,
        },
        "hf_model_required": "AutoModelForCausalLM",
    })

    meta2 = attach_label_schema(meta2, y_train, default_num_labels=meta2["num_classes"], ignore_index=label_pad_value)
    return (X_train, y_train), (X_test, y_test), meta2


def preprocess_hf_text_seq2seq(
    train,
    test,
    meta,
    *,
    hf_model_id,
    source_column=None,
    target_column=None,
    source_max_length=None,
    target_max_length=None,
    dynamic_padding=False,
    label_pad_value=-100,
):
    ds_train, _ = train
    ds_test, _ = test

    cols = set(ds_train.column_names)
    tokenizer = _load_tokenizer(hf_model_id)

    source_column = _pick_column(source_column, cols, SOURCE_COLUMN_CANDIDATES, "source_column")
    target_column = _pick_column(target_column, cols, TARGET_TEXT_COLUMN_CANDIDATES, "target_column")

    source_max_length = int(source_max_length or meta.get("max_length", 128))
    target_max_length = int(target_max_length or meta.get("max_length", 128))
    dynamic_padding = bool(dynamic_padding)
    label_pad_value = int(label_pad_value)

    def _build_features(ds):
        src_tok = _tokenize_texts(tokenizer, ds[source_column], max_length=source_max_length)
        tgt_tok = _tokenize_texts(tokenizer, ds[target_column], max_length=target_max_length)

        input_ids = _pad_sequences(
            src_tok["input_ids"],
            pad_value=tokenizer.pad_token_id,
            dynamic_padding=dynamic_padding,
            max_length=source_max_length,
        )
        attention_mask = (input_ids != int(tokenizer.pad_token_id)).astype("int32")

        target_ids = _pad_sequences(
            tgt_tok["input_ids"],
            pad_value=tokenizer.pad_token_id,
            dynamic_padding=dynamic_padding,
            max_length=target_max_length,
        )
        labels = target_ids.astype("int32")
        labels[labels == int(tokenizer.pad_token_id)] = label_pad_value

        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels

    X_train, y_train = _build_features(ds_train)
    X_test, y_test = _build_features(ds_test)

    meta2 = dict(meta)
    meta2.update({
        "hf_task": "seq2seq",
        "modality": "text",
        "x_format": "dict",
        "x_keys": ["input_ids", "attention_mask"],
        "label_granularity": "token",
        "num_classes": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "label_pad_value": label_pad_value,
        "hf_model_id": hf_model_id,
        "source_max_length": source_max_length,
        "target_max_length": target_max_length,
        "dynamic_padding": dynamic_padding,
        "input_shape": (int(X_train["input_ids"].shape[1]),),
        "column_mapping": {
            "source": source_column,
            "target": target_column,
        },
        "hf_model_required": "AutoModelForSeq2SeqLM",
    })

    meta2 = attach_label_schema(meta2, y_train, default_num_labels=meta2["num_classes"], ignore_index=label_pad_value)
    return (X_train, y_train), (X_test, y_test), meta2
