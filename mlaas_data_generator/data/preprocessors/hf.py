from .hf_text_sequence import preprocess_hf_text_sequence
from .hf_text_similarity import preprocess_hf_text_similarity
from .hf_text_fill_mask import preprocess_hf_text_fill_mask
from .hf_text_token import preprocess_hf_text_token
from .hf_text_generation import (
    preprocess_hf_text_causal_lm_generation,
    preprocess_hf_text_seq2seq_generation,
)
from .hf_image import preprocess_hf_image
from .hf_multimodal import preprocess_hf_multimodal


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
    "multimodal": {"input_ids", "attention_mask", "pixel_values"},
}

_DATASET_TEMPLATE_TO_TASK = {
    "hf_text_classification": "sequence_classification",
    "hf_token_classification": "token_classification",
    "hf_sentence_pair_classification": "sentence_similarity",
    "hf_masked_lm": "fill_mask",
    "hf_causal_lm": "causal_lm_generation",
    "hf_seq2seq": "seq2seq_generation",
    # backwards compatibility
    "hf_text_sequence": "sequence_classification",
    "hf_text_token": "token_classification",
    "hf_text_similarity": "sentence_similarity",
    "hf_text_fill_mask": "fill_mask",
    "hf_text_generation": None,
}


def _normalize_hf_task(hf_task):
    task = str(hf_task or "sequence_classification").strip().lower().replace("-", "_")
    aliases = {
        "text_classification": "sequence_classification",
        "seq_cls": "sequence_classification",
        "token_cls": "token_classification",
        "masked_lm": "fill_mask",
        "mlm": "fill_mask",
        "text_generation": "causal_lm_generation",
        "causal_lm": "causal_lm_generation",
        "text2text": "seq2seq_generation",
        "text2text_generation": "seq2seq_generation",
        "vision_classification": "image_classification",
        "image_cls": "image_classification",
        "object_detection": "image_detection",
        "detection": "image_detection",
        "semantic_segmentation": "image_segmentation",
        "segmentation": "image_segmentation",
    }
    return aliases.get(task, task)


def _resolve_dataset_loader_template(meta, dataset_args):
    dataset_meta = meta.get("dataset_args") if isinstance(meta.get("dataset_args"), dict) else {}
    template = dataset_args.get("loader_template") or meta.get("loader_template") or dataset_meta.get("loader_template")
    return str(template).strip().lower() if template else None


def _resolve_text_hf_task(meta, dataset_args):
    template = _resolve_dataset_loader_template(meta, dataset_args)
    if template == "hf_text_generation":
        task_tag = str(meta.get("task_tag") or dataset_args.get("task_tag") or "").strip().lower().replace("-", "_")
        pipeline_tag = str(meta.get("pipeline_tag") or dataset_args.get("pipeline_tag") or "").strip().lower()
        if task_tag in {"summarization", "translation", "seq2seq", "text2text"} or pipeline_tag == "text2text-generation":
            return "seq2seq_generation"
        return "causal_lm_generation"
    mapped = _DATASET_TEMPLATE_TO_TASK.get(template)
    if mapped:
        return mapped
    return _normalize_hf_task(meta.get("hf_task", dataset_args.get("hf_task", "sequence_classification")))


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

    if modality == "multimodal":
        meta["hf_task"] = "multimodal"
        out = preprocess_hf_multimodal(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            image_column=dataset_args.get("image_column", "image"),
            text_column=dataset_args.get("text_column", "text"),
            label_column=dataset_args.get("label_column"),
            max_length=dataset_args.get("max_length", meta.get("max_length", 128)),
            missing_pair_handling=dataset_args.get("missing_pair_handling", "drop"),
        )
        return _validate_hf_preprocessor_output(*out)

    if modality != "text":
        raise NotImplementedError(f"HF modality '{modality}' not implemented")

    hf_task = _resolve_text_hf_task(meta, dataset_args)
    meta["hf_task"] = hf_task

    if hf_task == "sequence_classification":
        out = preprocess_hf_text_sequence(
            train, test, meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", "text"),
            label_column=dataset_args.get("label_column", "label"),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
        )
        return _validate_hf_preprocessor_output(*out)

    if hf_task == "token_classification":
        out = preprocess_hf_text_token(
            train, test, meta,
            hf_model_id=hf_model_id,
            tokens_column=dataset_args.get("tokens_column") or dataset_args.get("text_column"),
            label_column=dataset_args.get("label_column"),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
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
            dynamic_padding=dataset_args.get("dynamic_padding", False),
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
            dynamic_padding=dataset_args.get("dynamic_padding", False),
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
