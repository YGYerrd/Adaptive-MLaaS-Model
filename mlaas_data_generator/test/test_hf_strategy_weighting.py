import math

from mlaas_data_generator.federated.strategies.base import ClientOutcome
from mlaas_data_generator.federated.strategies.hf_strategy import HFStrategy


class DummyAdapter:
    def __init__(self):
        self._weights = [1.0]

    def get_weights(self):
        return list(self._weights)

    def set_weights(self, weights):
        self._weights = list(weights)

    def evaluate(self, x_test, y_test):
        return 0.25, 0.75, 0.5, {"eval_sequence_count": len(y_test)}


def make_strategy(hf_task: str) -> HFStrategy:
    return HFStrategy(
        meta={"accounting": {"sequence_count": 10, "supervised_token_count": 100}},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={"model_type": "hf_finetune", "dataset_args": {"hf_task": hf_task}},
        x_test=[0, 1],
        y_test=[0, 1],
        metric_key="accuracy",
        save_weights=False,
    )


def make_outcome(*, payload, samples_count=4, sequence_count=None, supervised_token_count=None, unit=None, value=None):
    return ClientOutcome(
        participated=True,
        fail_reason="",
        samples_count=samples_count,
        duration=1.0,
        loss=0.1,
        metric_value=0.9,
        metric_score=0.9,
        extra_metric=0.8,
        rounds_so_far=1,
        comm_down=0,
        comm_up=0,
        cpu_time_s=None,
        cpu_utilization=None,
        memory_used_mb=None,
        memory_utilization=None,
        gpu_utilization=None,
        gpu_memory_utilization=None,
        gpu_memory_used_mb=None,
        peak_vram_mb=None,
        avg_vram_mb=None,
        peak_host_ram_mb=None,
        avg_host_ram_mb=None,
        payload=payload,
        extras={},
        sequence_count=sequence_count,
        supervised_token_count=supervised_token_count,
        aggregation_weight_unit=unit,
        aggregation_weight_value=value,
    )


def test_train_client_uses_supervised_token_weighting_for_token_tasks(monkeypatch):
    strategy = make_strategy("token_classification")

    def fake_train_eval(adapter, x_train, y_train):
        train_qos = {
            "train_sequence_count": 3,
            "train_supervised_token_count": 17,
        }
        eval_qos = {"eval_sequence_count": 2, "eval_supervised_token_count": 9}
        return 0.2, 0.8, 0.7, train_qos, eval_qos

    monkeypatch.setattr(strategy, "_get_client_adapter", lambda client_id: DummyAdapter())
    monkeypatch.setattr(strategy, "_train_eval", fake_train_eval)

    outcome = strategy.train_client(
        client_id="c1",
        x=[1, 2, 3],
        y=[0, 1, 0],
        global_model=DummyAdapter(),
        round_idx=1,
        rounds_so_far=1,
        comm_down=0,
    )

    assert outcome.sequence_count == 3
    assert outcome.supervised_token_count == 17
    assert outcome.aggregation_weight_unit == "supervised_token_count"
    assert outcome.aggregation_weight_value == 17.0


def test_aggregate_and_eval_uses_sequence_weights_for_image_classification(monkeypatch):
    strategy = make_strategy("image_classification")
    adapter = DummyAdapter()
    captured = {}

    def fake_aggregate(payloads, weights=None):
        captured["payloads"] = payloads
        captured["weights"] = weights
        return [sum(weights)]

    monkeypatch.setattr("mlaas_data_generator.federated.strategies.hf_strategy.aggregate_state_dict", fake_aggregate)

    outcomes = [
        make_outcome(payload=[1.0], sequence_count=5, unit="sequence_count", value=5.0),
        make_outcome(payload=[3.0], sequence_count=9, unit="sequence_count", value=9.0),
    ]

    loss, primary, score, secondary = strategy.aggregate_and_eval(
        global_model=adapter,
        client_payloads=[o.payload for o in outcomes],
        client_outcomes=outcomes,
        round_idx=1,
        x_train=[],
        x_test=[0, 1],
        y_test=[0, 1],
    )

    assert captured["weights"] == [5.0, 9.0]
    assert adapter.get_weights() == [14.0]
    assert math.isclose(loss, 0.25)
    assert math.isclose(primary, 0.75)
    assert math.isclose(score, 0.75)
    assert math.isclose(secondary, 0.5)


def test_loggable_run_params_image_schema_omits_text_defaults():
    strategy = HFStrategy(
        meta={"modality": "image"},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={
            "model_type": "hf_finetune",
            "dataset_args": {
                "hf_task": "image_detection",
                "dataset_name": "dummy",
                "image_column": "image",
                "label_column": "label",
                "boxes_column": "boxes",
                "classes_column": "classes",
            },
        },
        x_test=[],
        y_test=[],
        metric_key="accuracy",
        save_weights=False,
    )

    params = strategy.loggable_run_params()
    assert "padding_mode" not in params["adapter"]
    assert params["dataset"]["image_column"] == "image"
    assert params["dataset"]["boxes_column"] == "boxes"
    assert params["dataset"]["classes_column"] == "classes"
    assert "text_column" not in params["dataset"]
    assert "tokens_column" not in params["dataset"]
    assert "padding_mode" not in params["dataset"]


def test_loggable_run_params_multimodal_includes_pair_integrity_and_both_schemas():
    strategy = HFStrategy(
        meta={"modality": "multimodal"},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={
            "model_type": "hf_finetune",
            "dataset_args": {
                "hf_task": "visual_question_answering",
                "dataset_name": "dummy",
                "image_column": "image",
                "text_column": "question",
                "label_column": "answer",
                "missing_pair_handling": "drop",
                "dynamic_padding": True,
            },
        },
        x_test=[],
        y_test=[],
        metric_key="accuracy",
        save_weights=False,
    )

    params = strategy.loggable_run_params()
    assert params["adapter"]["padding_mode"] == "dynamic"
    assert params["dataset"]["image_column"] == "image"
    assert params["dataset"]["text_column"] == "question"
    assert params["dataset"]["label_column"] == "answer"
    assert params["dataset"]["missing_pair_handling"] == "drop"
    assert params["dataset"]["padding_mode"] == "dynamic"


def test_loggable_run_params_infers_image_modality_from_hf_task():
    strategy = HFStrategy(
        meta={},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={
            "model_type": "hf_finetune",
            "dataset_args": {
                "hf_task": "image_segmentation",
                "dataset_name": "dummy",
                "image_column": "image",
                "mask_column": "mask",
            },
        },
        x_test=[],
        y_test=[],
        metric_key="accuracy",
        save_weights=False,
    )

    params = strategy.loggable_run_params()
    assert params["dataset"]["image_column"] == "image"
    assert params["dataset"]["mask_column"] == "mask"
    assert "text_column" not in params["dataset"]
    assert "padding_mode" not in params["adapter"]
