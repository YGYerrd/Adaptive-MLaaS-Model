from __future__ import annotations

MODEL_REGISTRY: dict[str, list[dict[str, object]]] = {
    "text_classification": [
        {
            "model_id": "distilbert-base-uncased",
            "dataset_keys": ["glue_sst2", "ag_news", "imdb"],
        },
        {
            "model_id": "roberta-base",
            "dataset_keys": ["ag_news", "imdb"],
        },
        {
            "model_id": "bert-base-uncased",
            "dataset_keys": ["glue_sst2", "ag_news"],
        },
    ],
    "token_classification": [
        {
            "model_id": "bert-base-cased",
            "dataset_keys": ["conll2003", "wnut_17"],
        },
        {
            "model_id": "distilbert-base-cased",
            "dataset_keys": ["conll2003"],
        },
        {
            "model_id": "distilbert-base-uncased",
            "dataset_keys": ["wnut_17"],
        },
    ],
    "sentence_similarity": [
        {
            "model_id": "distilbert-base-uncased",
            "dataset_keys": ["glue_stsb", "glue_mrpc"],
        },
        {
            "model_id": "roberta-base",
            "dataset_keys": ["glue_stsb", "glue_mrpc"],
        },
        {
            "model_id": "microsoft/MiniLM-L12-H384-uncased",
            "dataset_keys": ["glue_stsb"],
        },
    ],
    "fill_mask": [
        {
            "model_id": "distilroberta-base",
            "dataset_keys": ["wikitext2", "ag_news_fillmask"],
        },
        {
            "model_id": "bert-base-uncased",
            "dataset_keys": ["wikitext2", "ag_news_fillmask"],
        },
    ],
    "text_generation": [
        {
            "model_id": "distilgpt2",
            "dataset_keys": ["wikitext2_lm"],
        },
        {
            "model_id": "gpt2",
            "dataset_keys": ["wikitext2_lm"],
        },
    ],
    "text2text_generation": [
        {
            "model_id": "google/flan-t5-small",
            "dataset_keys": ["cnn_dailymail"],
        },
        {
            "model_id": "t5-small",
            "dataset_keys": ["cnn_dailymail"],
        },
    ],
}
