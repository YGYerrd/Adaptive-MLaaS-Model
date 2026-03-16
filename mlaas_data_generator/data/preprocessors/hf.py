from .hf_text_sequence import preprocess_hf_text_sequence
from .hf_text_similarity import preprocess_hf_text_similarity
from .hf_text_fill_mask import preprocess_hf_text_fill_mask
from .hf_text_token import preprocess_hf_text_token
from .hf_text_generation import (
    preprocess_hf_text_causal_lm_generation,
    preprocess_hf_text_seq2seq_generation,
)
from .hf_image import preprocess_hf_image


_EXPECTED_BATCH_KEYS = {
    "sequence_classification": {"input_ids", "attention_mask"},
    "token_classification": {"input_ids", "attention_mask"},
    "sentence_similarity": {"input_ids", "attention_mask"},
    "fill_mask": {"input_ids", "attention_mask"},
    "causal_lm_generation": {"input_ids", "attention_mask"},
    "seq2seq_generation": {"input_ids", "attention_mask"},
    "image_classification": {"pixel_values"},
    "image_detection": {"pixel_values"},
    "image_segmentation": {"pixel_values"},
}


def _validate_hf_preprocessor_output(train, test, meta):
    x_train, y_train = train
    x_test, y_test = test
    hf_task = str(meta.get("hf_task", "")).strip().lower().replace("-", "_")

    if not isinstance(x_train, dict) or not isinstance(x_test, dict):
        raise TypeError(f"HF task '{hf_task}' requires dict features; got {type(x_train)} / {type(x_test)}")

    expected = _EXPECTED_BATCH_KEYS.get(hf_task, {"input_ids", "attention_mask"})
    missing_train = sorted(expected - set(x_train.keys()))
    missing_test = sorted(expected - set(x_test.keys()))
    if missing_train or missing_test:
        raise ValueError(
            f"HF preprocessor output validation failed for task '{hf_task}'. "
            f"Missing train keys={missing_train}, test keys={missing_test}."
        )

    if y_train is None or y_test is None:
        raise ValueError(f"HF preprocessor output validation failed for task '{hf_task}': missing labels.")

    train_count = len(next(iter(x_train.values())))
    test_count = len(next(iter(x_test.values())))
    if train_count != len(y_train) or test_count != len(y_test):
        raise ValueError(
            f"HF preprocessor output validation failed for task '{hf_task}': feature/label batch mismatch."
        )

    meta["x_keys"] = list(x_train.keys())
    return train, test, meta


def preprocess_hf(train, test, meta, **dataset_args):
    modality = str(meta.get("modality", "text")).strip().lower()
    hf_task = str(meta.get("hf_task", "sequence_classification")).strip().lower().replace("-", "_")
    if hf_task in {"mlm", "masked_lm"}:
        hf_task = "fill_mask"

    hf_model_id = dataset_args.get("hf_model_id")
    if not hf_model_id:
        raise ValueError("HF preprocessing requires hf_model_id in dataset_args")

    if modality == "image":
        task_type = str(meta.get("task_type", "classification")).strip().lower()
        hf_task = f"image_{task_type}"
        meta["hf_task"] = hf_task
        out = preprocess_hf_image(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            image_column=dataset_args.get("image_column", "image"),
            label_column=dataset_args.get("label_column", "label"),
            boxes_column=dataset_args.get("boxes_column"),
            classes_column=dataset_args.get("classes_column"),
            mask_column=dataset_args.get("mask_column"),
            task_type=task_type,
            training_augmentations=dataset_args.get("training_augmentations", True),
            eval_augmentations=dataset_args.get("eval_augmentations", False),
            on_decode_error=dataset_args.get("on_decode_error", "skip"),
            report_decode_errors=dataset_args.get("report_decode_errors", True),
        )
        return _validate_hf_preprocessor_output(*out)

    if modality != "text":
        raise NotImplementedError(f"HF modality '{modality}' not implemented")

    if hf_task == "sequence_classification":
        out = preprocess_hf_text_sequence(
            train, test, meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", "text"),
            label_column=dataset_args.get("label_column", "label"),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "token_classification":
        out = preprocess_hf_text_token(
            train, test, meta,
            hf_model_id=hf_model_id,
            tokens_column=dataset_args.get("tokens_column") or dataset_args.get("text_column"),
            label_column=dataset_args.get("label_column"),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "sentence_similarity":
        out = preprocess_hf_text_similarity(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", ["sentence1", "sentence2"]),
            label_column=dataset_args.get("label_column", "label"),
            label_mode=dataset_args.get("label_mode", "auto"),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "fill_mask":
        out = preprocess_hf_text_fill_mask(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", "text"),
            mlm_probability=dataset_args.get("mlm_probability", 0.15),
            label_pad_value=dataset_args.get("label_pad_value", -100),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "causal_lm_generation":
        out = preprocess_hf_text_causal_lm_generation(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            column_mapping=dataset_args.get("column_mapping"),
            max_length=dataset_args.get("max_length"),
            source_max_length=dataset_args.get("source_max_length"),
            target_max_length=dataset_args.get("target_max_length"),
            prompt_loss_only=dataset_args.get("prompt_loss_only", True),
            ignore_index=dataset_args.get("label_pad_value", -100),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "seq2seq_generation":
        out = preprocess_hf_text_seq2seq_generation(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            column_mapping=dataset_args.get("column_mapping"),
            max_length=dataset_args.get("max_length"),
            source_max_length=dataset_args.get("source_max_length"),
            target_max_length=dataset_args.get("target_max_length"),
            ignore_index=dataset_args.get("label_pad_value", -100),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    raise ValueError(f"Unsupported HF text task: {hf_task}")
