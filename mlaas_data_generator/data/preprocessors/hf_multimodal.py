from ..accounting import append_accounting_stage, finalize_accounting
from ..multimodal_columns import resolve_existing_column
import numpy as np


_IMAGE_COLUMN_ALIASES = ("image", "img", "images", "pixel_values")
_TEXT_COLUMN_ALIASES = (
    "text",
    "caption",
    "captions",
    "sentence",
    "sentences",
    "description",
    "descriptions",
    "question",
)

_TASK_COLUMN_DEFAULTS = {
    "visual_question_answering": {"text_column": "question", "label_column": "answer"},
    "text_image_retrieval": {"text_column": "caption", "label_column": None},
    "image_captioning": {"text_column": "caption", "label_column": None},
}


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _is_pil_image(value):
    try:
        from PIL import Image
    except Exception:
        return False
    return isinstance(value, Image.Image)


def _coerce_image_input(value):
    if _is_pil_image(value):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("array") is not None:
            return np.asarray(value.get("array"))
        if value.get("bytes") is not None:
            return value.get("bytes")
        if value.get("path"):
            return value.get("path")
    return value


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
    hf_task,
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

    def _process_image(image_value):
        candidate_kwargs = [
            {"return_tensors": None, "do_resize": True, "do_normalize": True},
            {"return_tensors": None},
            {},
        ]
        last_err = None
        image_input = _coerce_image_input(image_value)
        for kwargs in candidate_kwargs:
            try:
                return image_processor(image_input, **kwargs)
            except TypeError as e:
                last_err = e
                continue
            except ValueError as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        return image_processor(image_input)

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
        image_enc = _process_image(image_val)

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
        if hf_task == "image_captioning":
            label_text = row.get(label_column) if label_column is not None else text_val
            if not _has_value(label_text):
                label_text = text_val
            label_enc = tokenizer(
                str(label_text),
                truncation=True,
                padding="max_length",
                max_length=int(max_length),
                return_attention_mask=True,
                return_tensors=None,
            )
            label_ids = np.asarray(label_enc["input_ids"], dtype=np.int64)
            label_mask = np.asarray(label_enc["attention_mask"], dtype=np.int64)
            if label_ids.ndim == 2:
                label_ids = label_ids[0]
            if label_mask.ndim == 2:
                label_mask = label_mask[0]
            label_ids = label_ids.copy()
            label_ids[label_mask == 0] = -100
            labels.append(label_ids)
        elif label_column is not None:
            labels.append(row.get(label_column))

    x = {
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "attention_mask": np.asarray(attention_masks, dtype=np.int64),
        "pixel_values": np.asarray(pixel_values, dtype=np.float32),
    }
    y = np.asarray(labels) if labels else np.zeros((len(input_ids),), dtype=np.int64)

    if len(x["input_ids"]) != len(x["pixel_values"]):
        raise ValueError("Multimodal alignment check failed: token and image batch lengths differ")

    if label_column is not None and len(y) != len(x["input_ids"]):
        raise ValueError("Multimodal alignment check failed: label length does not match paired inputs")

    return x, y, len(input_ids)


def _resolve_multimodal_columns(
    hf_task,
    image_column,
    text_column,
    label_column,
    question_column,
    answer_column,
    ranking_label_column,
):
    task = str(hf_task or "multimodal").strip().lower().replace("-", "_")
    defaults = _TASK_COLUMN_DEFAULTS.get(task, {})

    resolved_text_column = text_column
    resolved_label_column = label_column

    if task == "visual_question_answering":
        resolved_text_column = question_column or (
            defaults.get("text_column", "question") if text_column in {None, "", "text"} else text_column
        )
        resolved_label_column = answer_column or (
            defaults.get("label_column", "answer") if label_column in {None, "", "label"} else label_column
        )
    elif task == "text_image_retrieval":
        resolved_text_column = text_column or defaults.get("text_column", "text")
        resolved_label_column = ranking_label_column
    elif task == "image_captioning":
        resolved_text_column = text_column or defaults.get("text_column", "text")
        resolved_label_column = label_column

    return task, image_column, resolved_text_column, resolved_label_column


def preprocess_hf_multimodal(
    train,
    test,
    meta,
    *,
    hf_model_id,
    hf_task="multimodal",
    image_column="image",
    text_column="text",
    label_column=None,
    max_length=128,
    missing_pair_handling="drop",
    question_column=None,
    answer_column=None,
    ranking_label_column=None,
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

    hf_task, image_column, text_column, label_column = _resolve_multimodal_columns(
        hf_task,
        image_column,
        text_column,
        label_column,
        question_column,
        answer_column,
        ranking_label_column,
    )
    train_columns = list(getattr(ds_train, "column_names", []) or [])
    train_column_set = set(train_columns)
    if train_columns:
        text_column = resolve_existing_column(
            text_column,
            train_columns,
            aliases=_TEXT_COLUMN_ALIASES,
            numbered_alias_bases=("caption", "captions", "sentence", "sentences", "description", "descriptions"),
        )
        image_column = resolve_existing_column(image_column, train_columns, aliases=_IMAGE_COLUMN_ALIASES)
        if hf_task == "image_captioning":
            label_column = resolve_existing_column(
                label_column,
                train_columns,
                aliases=(text_column,),
                numbered_alias_bases=("caption", "captions", "sentence", "sentences", "description", "descriptions"),
            )
            if label_column not in train_column_set:
                label_column = text_column
        elif label_column is not None:
            label_column = resolve_existing_column(label_column, train_columns)
            if label_column not in train_column_set:
                label_column = None

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

    x_train, y_train, train_survived = _encode_split(
        ds_train,
        hf_task=hf_task,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_column=image_column,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    x_test, y_test, test_survived = _encode_split(
        ds_test,
        hf_task=hf_task,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_column=image_column,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )

    meta = append_accounting_stage(
        meta,
        stage="hf_multimodal",
        split="train",
        input_record_count=train_report.get("aligned_rows", len(ds_train)),
        post_filter_record_count=train_survived,
        tokenized_record_count=train_survived,
        emitted_record_count=train_survived,
        sequence_count=train_survived,
        metric_instance_count=len(y_train),
    )
    meta = append_accounting_stage(
        meta,
        stage="hf_multimodal",
        split="test",
        input_record_count=test_report.get("aligned_rows", len(ds_test)),
        post_filter_record_count=test_survived,
        tokenized_record_count=test_survived,
        emitted_record_count=test_survived,
        sequence_count=test_survived,
        metric_instance_count=len(y_test),
    )
    meta = finalize_accounting(meta)
    inferred_input_shape = ()
    pixel_values = x_train.get("pixel_values") if isinstance(x_train, dict) else None
    if pixel_values is not None and len(pixel_values) > 0:
        inferred_input_shape = tuple(getattr(pixel_values[0], "shape", ()))
    meta.update(
        {
            "input_shape": inferred_input_shape,
            "modality": "multimodal",
            "hf_task": hf_task,
            "hf_processor": hf_model_id,
            "image_column": image_column,
            "text_column": text_column,
            "label_column": label_column,
            "x_keys": ["input_ids", "attention_mask", "pixel_values"],
            "schema": {
                "image_column": image_column,
                "text_column": text_column,
                "label_column": label_column,
                "task": hf_task,
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
