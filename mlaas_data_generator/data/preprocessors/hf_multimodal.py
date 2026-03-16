import numpy as np


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _validate_pair_alignment(ds, *, image_column, text_column, split_name, missing_pair_handling):
    valid_indices = []
    missing_pairs = []

    for idx in range(len(ds)):
        row = ds[idx]
        has_image = _has_value(row.get(image_column))
        has_text = _has_value(row.get(text_column))
        if has_image and has_text:
            valid_indices.append(idx)
            continue
        if has_image != has_text:
            missing_pairs.append(idx)

    if missing_pairs and missing_pair_handling == "error":
        raise ValueError(
            f"HF multimodal split '{split_name}' has {len(missing_pairs)} rows with broken image/text pairs. "
            "Use missing_pair_handling='drop' to filter rows with missing counterparts."
        )

    filtered = ds if not missing_pairs or missing_pair_handling == "error" else ds.select(valid_indices)
    report = {
        "split": split_name,
        "missing_pair_rows": len(missing_pairs),
        "aligned_rows": len(valid_indices),
        "output_rows": len(filtered),
        "policy": missing_pair_handling,
    }
    return filtered, report


def _encode_split(
    ds,
    *,
    tokenizer,
    image_processor,
    image_column,
    text_column,
    label_column,
    max_length,
):
    input_ids = []
    attention_masks = []
    pixel_values = []
    labels = []

    for idx in range(len(ds)):
        row = ds[idx]
        text_val = row.get(text_column)
        image_val = row.get(image_column)

        if not _has_value(text_val) or image_val is None:
            continue

        text_enc = tokenizer(
            str(text_val),
            truncation=True,
            padding="max_length",
            max_length=int(max_length),
            return_attention_mask=True,
            return_tensors=None,
        )
        image_enc = image_processor(
            image_val,
            return_tensors=None,
            do_resize=True,
            do_normalize=True,
        )

        ids = np.asarray(text_enc["input_ids"], dtype=np.int64)
        mask = np.asarray(text_enc["attention_mask"], dtype=np.int64)
        pix = np.asarray(image_enc.get("pixel_values", image_enc), dtype=np.float32)

        if ids.ndim == 2:
            ids = ids[0]
        if mask.ndim == 2:
            mask = mask[0]
        if pix.ndim == 4:
            pix = pix[0]
        if pix.ndim == 3 and pix.shape[0] != 3 and pix.shape[-1] == 3:
            pix = np.transpose(pix, (2, 0, 1))

        if pix.ndim != 3 or pix.shape[0] != 3:
            raise ValueError(f"Multimodal image encoding must produce CHW with 3 channels, got shape={pix.shape}")

        input_ids.append(ids)
        attention_masks.append(mask)
        pixel_values.append(pix)
        if label_column is not None:
            labels.append(row.get(label_column))

    x = {
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "attention_mask": np.asarray(attention_masks, dtype=np.int64),
        "pixel_values": np.asarray(pixel_values, dtype=np.float32),
    }
    y = np.asarray(labels) if label_column is not None else np.zeros((len(input_ids),), dtype=np.int64)

    if len(x["input_ids"]) != len(x["pixel_values"]):
        raise ValueError("Multimodal alignment check failed: token and image batch lengths differ")

    if label_column is not None and len(y) != len(x["input_ids"]):
        raise ValueError("Multimodal alignment check failed: label length does not match paired inputs")

    return x, y


def preprocess_hf_multimodal(
    train,
    test,
    meta,
    *,
    hf_model_id,
    image_column="image",
    text_column="text",
    label_column=None,
    max_length=128,
    missing_pair_handling="drop",
):
    try:
        from transformers import AutoTokenizer, AutoImageProcessor
    except Exception as e:
        raise ImportError("HF multimodal preprocessing requires transformers[vision]") from e

    policy = str(missing_pair_handling or "drop").strip().lower()
    if policy not in {"drop", "error"}:
        raise ValueError("missing_pair_handling must be one of ['drop', 'error']")

    ds_train, _ = train
    ds_test, _ = test

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    image_processor = AutoImageProcessor.from_pretrained(hf_model_id)

    ds_train, train_report = _validate_pair_alignment(
        ds_train,
        image_column=image_column,
        text_column=text_column,
        split_name="train",
        missing_pair_handling=policy,
    )
    ds_test, test_report = _validate_pair_alignment(
        ds_test,
        image_column=image_column,
        text_column=text_column,
        split_name="test",
        missing_pair_handling=policy,
    )

    x_train, y_train = _encode_split(
        ds_train,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_column=image_column,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    x_test, y_test = _encode_split(
        ds_test,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_column=image_column,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )

    meta.update(
        {
            "modality": "multimodal",
            "hf_processor": hf_model_id,
            "image_column": image_column,
            "text_column": text_column,
            "label_column": label_column,
            "x_keys": ["input_ids", "attention_mask", "pixel_values"],
            "schema": {
                "image_column": image_column,
                "text_column": text_column,
                "label_column": label_column,
                "pair_validation": {
                    "missing_pair_handling": policy,
                    "train": train_report,
                    "test": test_report,
                },
                "batch_contract": {
                    "text_keys": ["input_ids", "attention_mask"],
                    "image_keys": ["pixel_values"],
                    "combined_keys": ["input_ids", "attention_mask", "pixel_values"],
                },
            },
        }
    )

    return (x_train, y_train), (x_test, y_test), meta
