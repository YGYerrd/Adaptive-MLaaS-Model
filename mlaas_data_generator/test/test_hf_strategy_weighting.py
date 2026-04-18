import math
import numpy as np

from mlaas_data_generator.federated.strategies.base import ClientOutcome
from mlaas_data_generator.federated.strategies.hf_strategy import HFStrategy
from mlaas_data_generator.models.adapters.hf_adapter import TransformersTextClassifierAdapter


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


def test_hf_inference_adapter_delegates_model_interface_to_core():
    class FakeCore:
        def __init__(self):
            self.weights = {"layer": np.asarray([1.0])}

        def count_params(self):
            return 123

        def get_weights(self):
            return self.weights

        def set_weights(self, weights):
            self.weights = weights

    adapter = TransformersTextClassifierAdapter.__new__(TransformersTextClassifierAdapter)
    adapter.core = FakeCore()

    assert adapter.count_params() == 123
    assert adapter.get_weights()["layer"][0] == 1.0

    replacement = {"layer": np.asarray([2.0])}
    adapter.set_weights(replacement)
    assert adapter.core.weights is replacement


def test_hf_strategy_caps_segmentation_batch_size():
    strategy = HFStrategy(
        meta={},
        knobs={"batch_size": 32, "local_epochs": 1, "learning_rate": 1e-4},
        config={"model_type": "hf", "dataset_args": {"hf_task": "image_segmentation"}},
        x_test=[0, 1],
        y_test=[0, 1],
        metric_key="iou",
        save_weights=False,
    )

    params = strategy.loggable_run_params()["adapter"]
    assert strategy.knobs["batch_size"] == 2
    assert strategy.knobs["requested_batch_size"] == 32
    assert params["requested_batch_size"] == 32
    assert "capped image_segmentation batch_size 32->2" in params["runtime_adjustments"]


def test_hf_strategy_forces_detr_cpu_without_explicit_device():
    strategy = HFStrategy(
        meta={},
        knobs={"batch_size": 4, "local_epochs": 1, "learning_rate": 1e-4},
        config={
            "model_type": "hf",
            "dataset_args": {
                "hf_task": "image_detection",
                "hf_model_id": "facebook/detr-resnet-50",
            },
        },
        x_test=[0, 1],
        y_test=[0, 1],
        metric_key="map",
        save_weights=False,
    )

    params = strategy.loggable_run_params()["adapter"]
    assert strategy.config["device"] == "cpu"
    assert strategy.config["dataset_args"]["device"] == "cpu"
    assert params["device"] == "cpu"
    assert "forced facebook/detr object detection device=cpu" in params["runtime_adjustments"]


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


def test_hf_strategy_task_type_uses_segmentation_meta():
    strategy = HFStrategy(
        meta={"task_type": "segmentation"},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={"model_type": "hf_finetune", "dataset_args": {"hf_task": "image_segmentation"}},
        x_test=[],
        y_test=[],
        metric_key="iou",
        save_weights=False,
    )

    assert strategy.task_type() == "segmentation"


def test_segmentation_global_metrics_use_classwise_statistics():
    strategy = HFStrategy(
        meta={"task_type": "segmentation"},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={"model_type": "hf_finetune", "dataset_args": {"hf_task": "image_segmentation"}},
        x_test=[],
        y_test=[],
        metric_key="iou",
        save_weights=False,
    )

    primary, secondary = strategy._metrics_from_stats(
        {
            "class_0_intersection": 3,
            "class_0_pred_total": 4,
            "class_0_target_total": 5,
            "class_1_intersection": 2,
            "class_1_pred_total": 3,
            "class_1_target_total": 3,
        }
    )

    assert np.isclose(primary, np.mean([3 / 6, 2 / 4]))
    assert np.isclose(secondary, np.mean([(2 * 3) / 9, (2 * 2) / 6]))


def test_image_classification_global_metrics_use_macro_f1_secondary():
    strategy = make_strategy("image_classification")

    primary, secondary = strategy._metrics_from_stats(
        {
            "top1_correct": 7,
            "top5_correct": 10,
            "total": 10,
            "class_0_tp": 3,
            "class_0_pred_total": 4,
            "class_0_target_total": 5,
            "class_1_tp": 4,
            "class_1_pred_total": 6,
            "class_1_target_total": 5,
        }
    )

    assert np.isclose(primary, 0.7)
    assert np.isclose(secondary, np.mean([(2 * 3) / 9, (2 * 4) / 11]))


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


def test_train_client_inference_image_classification_backfills_accuracy_from_qos(monkeypatch):
    strategy = HFStrategy(
        meta={"accounting": {"sequence_count": 10, "supervised_token_count": 100}},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={"model_type": "hf", "dataset_args": {"hf_task": "image_classification"}},
        x_test=[0, 1],
        y_test=[0, 1],
        metric_key="accuracy",
        save_weights=False,
    )

    class _Adapter:
        def evaluate(self, x, y, inference_only=True, max_eval_time_s=None, progress_log_interval=None):
            return np.nan, np.nan, np.nan, {"top1_accuracy": 0.42, "top5_accuracy": 0.84, "eval_sequence_count": len(y)}

    monkeypatch.setattr(strategy, "build_model", lambda: _Adapter())

    outcome = strategy.train_client(
        client_id="c1",
        x={"pixel_values": [0, 1, 2]},
        y=[0, 1, 0],
        global_model=None,
        round_idx=1,
        rounds_so_far=1,
        comm_down=0,
    )

    assert outcome.participated is True
    assert math.isclose(outcome.metric_value, 0.42)
    assert math.isclose(outcome.extra_metric, 0.84)
    assert math.isclose(outcome.metric_score, 0.42)


def test_train_client_inference_uses_heldout_eval_set(monkeypatch):
    strategy = HFStrategy(
        meta={"accounting": {"sequence_count": 10, "supervised_token_count": 100}},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={"model_type": "hf", "dataset_args": {"hf_task": "image_detection"}},
        x_test={"pixel_values": ["heldout"]},
        y_test=["heldout-label"],
        metric_key="map",
        save_weights=False,
    )
    seen = {}

    class _Adapter:
        def evaluate(self, x, y, inference_only=True, max_eval_time_s=None, progress_log_interval=None):
            seen["x"] = x
            seen["y"] = y
            return 0.1, 0.2, 0.3, {"eval_sequence_count": len(y)}

    monkeypatch.setattr(strategy, "build_model", lambda: _Adapter())

    outcome = strategy.train_client(
        client_id="c1",
        x={"pixel_values": ["client-train"]},
        y=["client-label"],
        global_model=None,
        round_idx=1,
        rounds_so_far=1,
        comm_down=0,
    )

    assert outcome.participated is True
    assert seen["x"] == strategy.x_test
    assert seen["y"] == strategy.y_test
    assert outcome.sequence_count == 1


def test_train_eval_clamps_unsafe_detection_finetune_learning_rate(monkeypatch):
    strategy = HFStrategy(
        meta={"accounting": {"sequence_count": 10}, "hf_task": "image_detection"},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-3},
        config={"model_type": "hf_finetune", "dataset_args": {"hf_task": "image_detection"}},
        x_test=["heldout-x"],
        y_test=["heldout-y"],
        metric_key="map",
        save_weights=False,
    )
    seen = {}

    class _Adapter:
        def fit(self, x, y, epochs=1, lr=5e-5, max_train_time_s=60):
            seen["fit_lr"] = lr
            return {}

        def evaluate(self, x, y):
            seen["eval_x"] = x
            seen["eval_y"] = y
            return np.nan, 0.2, 0.3, {}

    loss, primary, secondary, train_qos, eval_qos = strategy._train_eval(_Adapter(), ["train-x"], ["train-y"])

    assert np.isnan(loss)
    assert np.isclose(primary, 0.2)
    assert np.isclose(secondary, 0.3)
    assert np.isclose(seen["fit_lr"], 1e-4)
    assert seen["eval_x"] == strategy.x_test
    assert seen["eval_y"] == strategy.y_test
    assert train_qos["learning_rate_adjusted"] is True
    assert np.isclose(train_qos["requested_learning_rate"], 1e-3)
    assert np.isclose(train_qos["effective_learning_rate"], 1e-4)
    assert eval_qos == {}


def test_aggregate_and_eval_inference_image_classification_uses_qos_accuracy_when_metric_values_nan():
    strategy = HFStrategy(
        meta={"accounting": {"sequence_count": 10, "supervised_token_count": 100}},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={"model_type": "hf", "dataset_args": {"hf_task": "image_classification"}},
        x_test=[0, 1],
        y_test=[0, 1],
        metric_key="accuracy",
        save_weights=False,
    )

    outcomes = [
        ClientOutcome(
            participated=True,
            fail_reason="",
            samples_count=4,
            duration=1.0,
            loss=0.1,
            metric_value=np.nan,
            metric_score=np.nan,
            extra_metric=np.nan,
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
            payload=None,
            extras={"accuracy": 0.25, "top5_accuracy": 0.75},
            sequence_count=2,
            supervised_token_count=None,
            aggregation_weight_unit="sequence_count",
            aggregation_weight_value=2.0,
        ),
        ClientOutcome(
            participated=True,
            fail_reason="",
            samples_count=4,
            duration=1.0,
            loss=0.3,
            metric_value=np.nan,
            metric_score=np.nan,
            extra_metric=np.nan,
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
            payload=None,
            extras={"top1_accuracy": 0.75, "top5_accuracy": 1.0},
            sequence_count=6,
            supervised_token_count=None,
            aggregation_weight_unit="sequence_count",
            aggregation_weight_value=6.0,
        ),
    ]

    loss, primary, score, secondary = strategy.aggregate_and_eval(
        global_model=None,
        client_payloads=[],
        client_outcomes=outcomes,
        round_idx=1,
        x_train=[],
        x_test=[],
        y_test=[],
    )

    assert math.isclose(loss, 0.25)
    assert math.isclose(primary, (0.25 * 2 + 0.75 * 6) / 8.0)
    assert math.isclose(score, primary)
    assert math.isclose(secondary, (0.75 * 2 + 1.0 * 6) / 8.0)


def test_build_model_infers_num_labels_from_y_test_for_image_classification(monkeypatch):
    captured = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("mlaas_data_generator.models.builders.create_model", fake_create_model)

    strategy = HFStrategy(
        meta={"input_shape": (3, 32, 32), "num_classes": None},
        knobs={
            "batch_size": 2,
            "local_epochs": 1,
            "learning_rate": 1e-4,
            "hidden_layers": [16],
            "activation": "relu",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
        },
        config={
            "model_type": "hf_finetune",
            "hf_task": "image_classification",
            "hf_model_id": "google/vit-base-patch16-224",
        },
        x_test={"pixel_values": np.zeros((4, 3, 32, 32), dtype=np.float32)},
        y_test=np.asarray([0, 1, 2, 1], dtype=np.int64),
        metric_key="accuracy",
        save_weights=False,
    )

    strategy.build_model()
    assert captured["num_classes"] == 3


def test_build_model_passes_meta_and_infers_num_labels_for_image_segmentation(monkeypatch):
    captured = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("mlaas_data_generator.models.builders.create_model", fake_create_model)

    strategy = HFStrategy(
        meta={"input_shape": (3, 512, 512), "label_pad_value": 255, "ignore_index": 255},
        knobs={
            "batch_size": 2,
            "local_epochs": 1,
            "learning_rate": 1e-4,
            "hidden_layers": [16],
            "activation": "relu",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
        },
        config={
            "model_type": "hf_finetune",
            "hf_task": "image_segmentation",
            "hf_model_id": "nvidia/segformer-b0-finetuned-ade-512-512",
        },
        x_test={"pixel_values": np.zeros((2, 3, 512, 512), dtype=np.float32)},
        y_test=[
            np.asarray([[255, 0], [1, 2]], dtype=np.int64),
            np.asarray([[2, 1], [0, 255]], dtype=np.int64),
        ],
        metric_key="iou",
        save_weights=False,
    )

    strategy.build_model()
    assert captured["num_classes"] == 3
    assert captured["meta"]["label_pad_value"] == 255


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


def test_loggable_run_params_reports_effective_detection_learning_rate():
    strategy = HFStrategy(
        meta={"modality": "image"},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-3},
        config={
            "model_type": "hf_finetune",
            "dataset_args": {
                "hf_task": "image_detection",
                "dataset_name": "dummy",
                "image_column": "image",
                "label_column": "objects",
            },
        },
        x_test=[],
        y_test=[],
        metric_key="map",
        save_weights=False,
    )

    params = strategy.loggable_run_params()
    assert np.isclose(params["adapter"]["lr"], 1e-4)
    assert np.isclose(params["adapter"]["requested_lr"], 1e-3)
    assert params["adapter"]["learning_rate_adjusted"] is True
    assert params["adapter"]["aggregation_weight_unit"] == "sequence_count"


def test_loggable_run_params_reports_resolved_hf_splits():
    strategy = HFStrategy(
        meta={"modality": "image", "train_split": "train", "test_split": "validation"},
        knobs={"batch_size": 2, "local_epochs": 1, "learning_rate": 1e-4},
        config={
            "model_type": "hf_finetune",
            "dataset_args": {
                "hf_task": "image_classification",
                "dataset_name": "dummy",
                "train_split": "train",
                "test_split": "test",
                "image_column": "image",
                "label_column": "label",
            },
        },
        x_test=[],
        y_test=[],
        metric_key="accuracy",
        save_weights=False,
    )

    params = strategy.loggable_run_params()
    assert params["dataset"]["train_split"] == "train"
    assert params["dataset"]["test_split"] == "validation"
    assert params["dataset"]["requested_test_split"] == "test"
    assert "requested_train_split" not in params["dataset"]


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
