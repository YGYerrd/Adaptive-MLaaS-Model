from .hf_text_sequence import preprocess_hf_text_sequence
from .hf_text_similarity import preprocess_hf_text_similarity
from .hf_text_token import preprocess_hf_text_token

def preprocess_hf(train, test, meta, **dataset_args):
    modality = meta.get("modality", "text")
    hf_task = meta.get("hf_task", "sequence_classification")

    if modality != "text":
        raise NotImplementedError(f"HF modality '{modality}' not implemented")

    hf_model_id = dataset_args.get("hf_model_id")
    if not hf_model_id:
        raise ValueError("HF preprocessing requires hf_model_id in dataset_args")

    if hf_task == "sequence_classification":
        return preprocess_hf_text_sequence(
            train, test, meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", "text"),
            label_column=dataset_args.get("label_column", "label"),
        )

    if hf_task == "token_classification":
        return preprocess_hf_text_token(
            train, test, meta,
            hf_model_id=hf_model_id,
            tokens_column=dataset_args.get("tokens_column"),
            label_column=dataset_args.get("label_column"),
        )
    
    if hf_task == "sentence_similarity":
        return preprocess_hf_text_similarity(
            train,
            test,
            meta,
            hf_model_id=hf_model_id,
            text_column=dataset_args.get("text_column", ["sentence1", "sentence2"]),
            label_column=dataset_args.get("label_column", "label"),
            label_mode=dataset_args.get("label_mode", "auto"),
        )


    raise ValueError(f"Unsupported HF text task: {hf_task}")