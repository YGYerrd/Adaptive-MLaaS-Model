import pandas as pd

from mlaas_data_generator.cli.manifest.hf_manifest_builder import build_hf_manifest
from mlaas_data_generator.cli.run_manifest import _build_dataset_args, _resolve_row, _validate_row
from mlaas_data_generator.federated.strategies.factory import make_task_strategy
from mlaas_data_generator.federated.strategies.hf_strategy import HFStrategy


def _manifest_for_requested_matrix() -> pd.DataFrame:
    return build_hf_manifest(
        task_keys=[
            "text_classification",
            "token_classification",
            "sentence_similarity",
            "text_generation",
            "text2text_generation",
            "fill_mask",
            "image_classification",
            "object_detection",
            "image_segmentation",
            "text_image_retrieval",
            "visual_question_answering",
            "keras_image_classification",
            "sklearn_image_classification",
            "tabular_regression",
            "clustering",
        ],
        models_per_task=10,
        datasets_per_model=3,
        run_regimes=["finetune_transfer", "inference_only"],
        variants_per_pair=1,
        seed=13,
        manifest_profile="test",
        avg_sample_size=128,
    )


def test_manifest_builder_covers_requested_task_matrix():
    df = _manifest_for_requested_matrix()

    def has(**expected):
        mask = pd.Series(True, index=df.index)
        for column, value in expected.items():
            mask &= df[column].astype(str) == str(value)
        return bool(mask.any())

    assert has(hf_model_id="bert-base-uncased", dataset_name="glue", dataset_config="sst2")
    assert has(hf_model_id="bert-base-uncased", dataset_name="ag_news")
    assert has(hf_model_id="bert-base-uncased", dataset_name="imdb")
    assert has(hf_model_id="bert-base-cased", dataset_name="conll2003")
    assert has(hf_model_id="bert-base-cased", dataset_name="wnut_17")
    assert has(hf_model_id="microsoft/MiniLM-L12-H384-uncased", dataset_name="glue", dataset_config="stsb")
    assert has(hf_model_id="microsoft/MiniLM-L12-H384-uncased", dataset_name="glue", dataset_config="mrpc")
    assert has(hf_model_id="gpt2", dataset_name="wikitext")
    assert has(hf_model_id="t5-small", dataset_name="wikitext", task_tag="language-modeling")
    assert has(hf_model_id="t5-small", dataset_name="cnn_dailymail", task_tag="summarization")
    assert has(hf_model_id="bert-base-uncased", dataset_name="wikitext", hf_task="fill_mask")
    assert has(dataset="cifar10", model_type="cnn", task_type="classification")
    assert has(dataset="cifar10", model_type="randomforest", task_type="classification")
    assert has(hf_model_id="google/vit-base-patch16-224", dataset_name="cifar10")
    assert has(hf_model_id="google/vit-base-patch16-224", dataset_name="beans")
    assert has(dataset="synthetic", task_type="regression", model_type="mlp")
    assert has(dataset="uci_wine_quality", task_type="regression", model_type="randomforest")
    assert has(dataset="synthetic", task_type="clustering", model_type="kmeans")
    assert has(hf_model_id="facebook/detr-resnet-50", dataset_name="detection-datasets/coco")
    assert has(hf_model_id="nvidia/segformer-b0-finetuned-ade-512-512", dataset_name="zhoubolei/scene_parse_150")
    assert has(hf_model_id="openai/clip-vit-base-patch32", dataset_name="jxie/flickr8k")
    assert has(dataset_name="HuggingFaceM4/VQAv2", task_tag="vqa")


def test_manifest_rows_validate_and_pass_generic_task_to_loader():
    df = _manifest_for_requested_matrix()
    failures = []
    for _, row in df.iterrows():
        resolved = _resolve_row(row, {})
        validation = _validate_row(resolved)
        if not validation.ok:
            failures.append((row["case_name"], validation.error))

    assert failures == []

    clustering_row = df[df["task_type"] == "clustering"].iloc[0]
    clustering_args = _build_dataset_args(_resolve_row(clustering_row, {}))
    assert clustering_args["task"] == "clustering"


def test_manifest_participation_rate_does_not_shrink_samples():
    row = pd.Series(
        {
            "dataset": "synthetic",
            "model_type": "mlp",
            "task_type": "regression",
            "num_clients": 2,
            "client_participation_rate": 0.5,
            "sample_size": 1200,
        }
    )

    resolved = _resolve_row(row, {})

    assert resolved["client_participation_rate"] == 0.5
    assert resolved["sample_frac"] is None


def test_hf_strategy_routes_all_hf_manifest_families():
    cases = [
        ("classification", "sequence_classification"),
        ("classification", "token_classification"),
        ("classification", "fill_mask"),
        ("regression", "sentence_similarity"),
        ("classification", "sentence_similarity"),
        ("detection", "image_detection"),
        ("segmentation", "image_segmentation"),
        ("generation", "causal_lm_generation"),
        ("generation", "seq2seq_generation"),
        ("retrieval", "text_image_retrieval"),
        ("vqa", "visual_question_answering"),
    ]

    for task_type, hf_task in cases:
        strategy = make_task_strategy(
            task_type=task_type,
            meta={"task_type": task_type, "input_shape": (8,), "num_classes": 2},
            knobs={"batch_size": 1},
            config={"model_type": "hf_finetune", "hf_task": hf_task},
            x_test=None,
            y_test=None,
            metric_key="metric",
            save_weights=False,
        )
        assert isinstance(strategy, HFStrategy)
