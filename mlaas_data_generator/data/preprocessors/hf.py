from .hf_text_sequence import preprocess_hf_text_sequence
from .hf_text_similarity import preprocess_hf_text_similarity
from .hf_text_fill_mask import preprocess_hf_text_fill_mask
from .hf_text_token import preprocess_hf_text_token
from .hf_text_generation import preprocess_hf_text_causal_lm, preprocess_hf_text_seq2seq


TASK_EXPECTED_BATCH_KEYS = {
    "sequence_classification": {"input_ids", "attention_mask"},
    "token_classification": {"input_ids", "attention_mask"},
    "sentence_similarity": {"input_ids", "attention_mask"},
    "fill_mask": {"input_ids", "attention_mask"},
    "causal_lm": {"input_ids", "attention_mask"},
    "seq2seq": {"input_ids", "attention_mask"},
}


def _resolve_column_arg(dataset_args, config_key, default=None):
    mapping = dataset_args.get("column_mapping") or {}
    if config_key in mapping:
        return mapping[config_key]
    return dataset_args.get(config_key, default)


def _validate_hf_preprocessor_output(train, test, hf_task):
    x_train, y_train = train
    x_test, y_test = test

    if not isinstance(x_train, dict) or not isinstance(x_test, dict):
        raise ValueError(f"HF task '{hf_task}' requires dict feature outputs from preprocessor")

    expected = TASK_EXPECTED_BATCH_KEYS.get(hf_task, set())
    missing_train = expected.difference(x_train.keys())
    missing_test = expected.difference(x_test.keys())

    if missing_train or missing_test:
        raise ValueError(
            f"Preprocessor output keys do not match HF task '{hf_task}'. "
            f"Missing train={sorted(missing_train)} test={sorted(missing_test)}"
        )

    if y_train is None or y_test is None:
        raise ValueError(f"HF task '{hf_task}' requires non-null labels")

def preprocess_hf(train, test, meta, **dataset_args):
    modality = meta.get("modality", "text")
    hf_task = str(meta.get("hf_task", "sequence_classification")).strip().lower().replace("-", "_")
    if hf_task in {"mlm", "masked_lm"}:
        hf_task = "fill_mask"

    if modality != "text":
        raise NotImplementedError(f"HF modality '{modality}' not implemented")

    hf_model_id = dataset_args.get("hf_model_id")
    if not hf_model_id:
        raise ValueError("HF preprocessing requires hf_model_id in dataset_args")

    if hf_task == "sequence_classification":
        out = preprocess_hf_text_sequence(
            train, test, meta,
            hf_model_id=hf_model_id,
            text_column=_resolve_column_arg(dataset_args, "text_column", "text"),
            label_column=_resolve_column_arg(dataset_args, "label_column", "label"),
        )
        _validate_hf_preprocessor_output(out[0], out[1], "sequence_classification")
        return out

    if hf_task == "token_classification":
        out = preprocess_hf_text_token(
            train, test, meta,
            hf_model_id=hf_model_id,
            tokens_column=_resolve_column_arg(dataset_args, "tokens_column") or _resolve_column_arg(dataset_args, "text_column"),
            label_column=_resolve_column_arg(dataset_args, "label_column"),
        )
        _validate_hf_preprocessor_output(out[0], out[1], "token_classification")
        return out
    
    if hf_task == "sentence_similarity":
        out = preprocess_hf_text_similarity(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            text_column=_resolve_column_arg(dataset_args, "text_column", ["sentence1", "sentence2"]),
            label_column=_resolve_column_arg(dataset_args, "label_column", "label"),
            label_mode=dataset_args.get("label_mode", "auto"),
        )
        _validate_hf_preprocessor_output(out[0], out[1], "sentence_similarity")
        return out

    if hf_task == "fill_mask":
        out = preprocess_hf_text_fill_mask(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            text_column=_resolve_column_arg(dataset_args, "text_column", "text"),
            mlm_probability=dataset_args.get("mlm_probability", 0.15),
            label_pad_value=dataset_args.get("label_pad_value", -100),
        )
        _validate_hf_preprocessor_output(out[0], out[1], "fill_mask")
        return out

    if hf_task in {"causal_lm", "causal_language_modeling", "causal_language_model"}:
        out = preprocess_hf_text_causal_lm(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            prompt_column=_resolve_column_arg(dataset_args, "prompt_column"),
            target_column=_resolve_column_arg(dataset_args, "target_column"),
            text_column=_resolve_column_arg(dataset_args, "text_column"),
            source_max_length=dataset_args.get("source_max_length"),
            target_max_length=dataset_args.get("target_max_length"),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
            label_pad_value=dataset_args.get("label_pad_value", -100),
            label_strategy=dataset_args.get("label_strategy", "target_only"),
        )
        _validate_hf_preprocessor_output(out[0], out[1], "causal_lm")
        return out

    if hf_task in {"seq2seq", "text2text", "translation", "summarization"}:
        out = preprocess_hf_text_seq2seq(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            source_column=_resolve_column_arg(dataset_args, "source_column"),
            target_column=_resolve_column_arg(dataset_args, "target_column"),
            source_max_length=dataset_args.get("source_max_length"),
            target_max_length=dataset_args.get("target_max_length"),
            dynamic_padding=dataset_args.get("dynamic_padding", False),
            label_pad_value=dataset_args.get("label_pad_value", -100),
        )
        _validate_hf_preprocessor_output(out[0], out[1], "seq2seq")
        return out

    raise ValueError(f"Unsupported HF text task: {hf_task}")
