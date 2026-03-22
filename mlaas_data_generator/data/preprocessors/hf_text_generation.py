import numpy as np


# Seq2seq auto-mapping priority is ordered from the most explicit generic schema
# names to common summarization/translation aliases. The first match wins so the
# behavior stays deterministic whenever multiple plausible columns are present.
SEQ2SEQ_SOURCE_CANDIDATES = [
    "source_text",
    "input",
    "prompt",
    "instruction",
    "text",
    "source",
    "article",
    "document",
    "context",
    "question",
    "src",
    "input_text",
]

SEQ2SEQ_TARGET_CANDIDATES = [
    "target_text",
    "label",
    "output",
    "completion",
    "response",
    "target",
    "highlights",
    "summary",
    "answer",
    "tgt",
    "output_text",
]


def _first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def resolve_generation_columns(ds_train, *, column_mapping=None, hf_task="causal_lm_generation"):
    cols = set(ds_train.column_names)
    mapping = dict(column_mapping or {})

    if hf_task == "seq2seq_generation":
        source_col = mapping.get("source") or _first_existing(cols, SEQ2SEQ_SOURCE_CANDIDATES)
        target_col = mapping.get("target") or _first_existing(cols, SEQ2SEQ_TARGET_CANDIDATES)
        mode = "source_target"
    else:
        prompt_col = mapping.get("prompt") or _first_existing(cols, [
            "prompt", "instruction", "input", "source_text", "text", "source",
        ])
        target_col = mapping.get("target") or _first_existing(cols, [
            "completion", "response", "output", "target_text", "label", "target",
        ])
        source_col = prompt_col
        mode = "prompt_target"

        if target_col is None:
            single_text_col = _first_existing(cols, [
                "text", "content", "document", "body", "passage", "article",
            ])
            if single_text_col is None and len(cols) == 1:
                single_text_col = next(iter(cols))
            if single_text_col is not None:
                source_col = single_text_col
                target_col = single_text_col
                mode = "single_text"

    if not source_col:
        raise ValueError(
            f"Could not resolve source/prompt column for task '{hf_task}'. "
            f"Provide dataset_args.column_mapping. Available columns: {sorted(cols)}"
        )
    if not target_col:
        raise ValueError(
            f"Could not resolve target column for task '{hf_task}'. "
            f"Provide dataset_args.column_mapping. Available columns: {sorted(cols)}"
        )

    return {"source": source_col, "target": target_col, "mode": mode}


def _to_text_list(values):
    out = []
    for v in values:
        if v is None:
            out.append("")
        else:
            out.append(str(v))
    return out


def _pad_encodings(enc, pad_to):
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    padded_ids = []
    padded_mask = []
    for ids, mask in zip(input_ids, attention_mask):
        ids = list(ids)[:pad_to]
        mask = list(mask)[:pad_to]
        pad_len = max(0, pad_to - len(ids))
        padded_ids.append(ids + [0] * pad_len)
        padded_mask.append(mask + [0] * pad_len)
    return np.asarray(padded_ids, dtype="int32"), np.asarray(padded_mask, dtype="int32")


def preprocess_hf_text_causal_lm_generation(
    train,
    test,
    meta,
    *,
    hf_model_id,
    column_mapping=None,
    max_length=None,
    source_max_length=None,
    target_max_length=None,
    prompt_loss_only=True,
    ignore_index=-100,
    dynamic_padding=False,
):
    ds_train, _ = train
    ds_test, _ = test

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    resolved = resolve_generation_columns(ds_train, column_mapping=column_mapping, hf_task="causal_lm_generation")
    prompt_col, target_col = resolved["source"], resolved["target"]
    generation_mode = resolved.get("mode", "prompt_target")

    max_len = int(max_length or meta.get("max_length", 512))
    src_max = int(source_max_length or max_len)
    tgt_max = int(target_max_length or max_len)

    def _encode_split(ds):
        pad_id = int(tokenizer.pad_token_id)
        eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id)

        ids_list, mask_list, label_list = [], [], []
        if generation_mode == "single_text":
            texts = _to_text_list(ds[prompt_col])
            tok = tokenizer(texts, truncation=True, padding=False, max_length=max_len, add_special_tokens=True)

            for ids in tok["input_ids"]:
                full_ids = list(ids)[:max_len]
                if eos_id is not None and full_ids:
                    if len(full_ids) < max_len and full_ids[-1] != eos_id:
                        full_ids = full_ids + [eos_id]
                    elif full_ids[-1] != eos_id and len(full_ids) == max_len:
                        full_ids[-1] = eos_id
                labels = full_ids.copy()
                ids_list.append(full_ids)
                mask_list.append([1] * len(full_ids))
                label_list.append(labels)
        else:
            prompts = _to_text_list(ds[prompt_col])
            targets = _to_text_list(ds[target_col])

            p_tok = tokenizer(prompts, truncation=True, padding=False, max_length=src_max, add_special_tokens=True)
            t_tok = tokenizer(targets, truncation=True, padding=False, max_length=tgt_max, add_special_tokens=False)

            for p_ids, t_ids in zip(p_tok["input_ids"], t_tok["input_ids"]):
                full_ids = (list(p_ids) + list(t_ids) + [eos_id])[:max_len]
                labels = full_ids.copy()
                if prompt_loss_only:
                    prompt_len = min(len(p_ids), len(full_ids))
                    labels[:prompt_len] = [int(ignore_index)] * prompt_len

                ids_list.append(full_ids)
                mask_list.append([1] * len(full_ids))
                label_list.append(labels)

        pad_to = max(len(x) for x in ids_list) if dynamic_padding else max_len
        padded_ids, padded_mask, padded_labels = [], [], []
        for ids, mask, labels in zip(ids_list, mask_list, label_list):
            ids = ids[:pad_to]
            mask = mask[:pad_to]
            labels = labels[:pad_to]
            pad_len = max(0, pad_to - len(ids))
            padded_ids.append(ids + [pad_id] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
            padded_labels.append(labels + [int(ignore_index)] * pad_len)

        X = {
            "input_ids": np.asarray(padded_ids, dtype="int32"),
            "attention_mask": np.asarray(padded_mask, dtype="int32"),
        }
        y = np.asarray(padded_labels, dtype="int32")
        return X, y, pad_to

    X_train, y_train, train_len = _encode_split(ds_train)
    X_test, y_test, test_len = _encode_split(ds_test)

    pad_len = max(train_len, test_len) if dynamic_padding else max_len
    if dynamic_padding and train_len != test_len:
        # keep shapes aligned for downstream tensorization consistency
        X_train["input_ids"], X_train["attention_mask"] = _pad_encodings(X_train, pad_len)
        X_test["input_ids"], X_test["attention_mask"] = _pad_encodings(X_test, pad_len)
        y_train = np.pad(y_train, ((0, 0), (0, pad_len - y_train.shape[1])), constant_values=int(ignore_index))
        y_test = np.pad(y_test, ((0, 0), (0, pad_len - y_test.shape[1])), constant_values=int(ignore_index))

    meta2 = dict(meta)
    meta2.update({
        "hf_task": "causal_lm_generation",
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": list(X_train.keys()),
        "input_shape": (pad_len,),
        "label_granularity": "token",
        "modality": "text",
        "num_classes": int(meta.get("num_classes", 1) or 1),
        "column_mapping": (
            {"text": prompt_col}
            if generation_mode == "single_text"
            else {"prompt": prompt_col, "target": target_col}
        ),
        "generation_mode": generation_mode,
        "prompt_loss_only": bool(prompt_loss_only),
        "ignore_index": int(ignore_index),
        "source_max_length": src_max,
        "target_max_length": tgt_max,
        "dynamic_padding": bool(dynamic_padding),
        "pad_token_id": int(tokenizer.pad_token_id),
        "padding_side": str(getattr(tokenizer, "padding_side", "right")),
    })
    return (X_train, y_train), (X_test, y_test), meta2


def preprocess_hf_text_seq2seq_generation(
    train,
    test,
    meta,
    *,
    hf_model_id,
    column_mapping=None,
    max_length=None,
    source_max_length=None,
    target_max_length=None,
    ignore_index=-100,
    dynamic_padding=False,
):
    ds_train, _ = train
    ds_test, _ = test

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    resolved = resolve_generation_columns(ds_train, column_mapping=column_mapping, hf_task="seq2seq_generation")
    source_col, target_col = resolved["source"], resolved["target"]

    max_len = int(max_length or meta.get("max_length", 512))
    src_max = int(source_max_length or max_len)
    tgt_max = int(target_max_length or max_len)

    def _encode_split(ds):
        src = _to_text_list(ds[source_col])
        tgt = _to_text_list(ds[target_col])

        source_enc = tokenizer(src, truncation=True, padding=False, max_length=src_max, return_attention_mask=True)
        target_enc = tokenizer(text_target=tgt, truncation=True, padding=False, max_length=tgt_max, return_attention_mask=False)

        src_pad_to = max(len(x) for x in source_enc["input_ids"]) if dynamic_padding else src_max
        tgt_pad_to = max(len(x) for x in target_enc["input_ids"]) if dynamic_padding else tgt_max

        pad_id = int(tokenizer.pad_token_id)

        X_ids, X_mask, y = [], [], []
        for s_ids, s_mask, t_ids in zip(source_enc["input_ids"], source_enc["attention_mask"], target_enc["input_ids"]):
            s_ids = list(s_ids)[:src_pad_to]
            s_mask = list(s_mask)[:src_pad_to]
            t_ids = list(t_ids)[:tgt_pad_to]

            s_pad = max(0, src_pad_to - len(s_ids))
            t_pad = max(0, tgt_pad_to - len(t_ids))

            X_ids.append(s_ids + [pad_id] * s_pad)
            X_mask.append(s_mask + [0] * s_pad)
            y.append(t_ids + [int(ignore_index)] * t_pad)

        X = {
            "input_ids": np.asarray(X_ids, dtype="int32"),
            "attention_mask": np.asarray(X_mask, dtype="int32"),
        }
        labels = np.asarray(y, dtype="int32")
        return X, labels, src_pad_to, tgt_pad_to

    X_train, y_train, train_src_len, train_tgt_len = _encode_split(ds_train)
    X_test, y_test, test_src_len, test_tgt_len = _encode_split(ds_test)

    src_pad_to = max(train_src_len, test_src_len) if dynamic_padding else src_max
    tgt_pad_to = max(train_tgt_len, test_tgt_len) if dynamic_padding else tgt_max
    if dynamic_padding:
        if X_train["input_ids"].shape[1] != src_pad_to:
            X_train["input_ids"], X_train["attention_mask"] = _pad_encodings(X_train, src_pad_to)
        if X_test["input_ids"].shape[1] != src_pad_to:
            X_test["input_ids"], X_test["attention_mask"] = _pad_encodings(X_test, src_pad_to)

        if y_train.shape[1] != tgt_pad_to:
            y_train = np.pad(y_train, ((0, 0), (0, tgt_pad_to - y_train.shape[1])), constant_values=int(ignore_index))
        if y_test.shape[1] != tgt_pad_to:
            y_test = np.pad(y_test, ((0, 0), (0, tgt_pad_to - y_test.shape[1])), constant_values=int(ignore_index))

    meta2 = dict(meta)
    meta2.update({
        "hf_task": "seq2seq_generation",
        "hf_model_id": hf_model_id,
        "x_format": "dict",
        "x_keys": list(X_train.keys()),
        "input_shape": (src_pad_to,),
        "label_granularity": "token",
        "modality": "text",
        "num_classes": int(meta.get("num_classes", 1) or 1),
        "column_mapping": {"source": source_col, "target": target_col},
        "ignore_index": int(ignore_index),
        "source_max_length": src_max,
        "target_max_length": tgt_max,
        "dynamic_padding": bool(dynamic_padding),
        "target_shape": (tgt_pad_to,),
    })

    return (X_train, y_train), (X_test, y_test), meta2
