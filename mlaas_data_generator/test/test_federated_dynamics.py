import sqlite3

import numpy as np

from mlaas_data_generator.federated import orchestrator as orch
from mlaas_data_generator.federated.dynamics import (
    evaluate_run_dynamics,
    repeated_round_metrics,
    weight_delta,
)
from mlaas_data_generator.federated.strategies.base import ClientOutcome
from mlaas_data_generator.storage.writer import SQLiteWriter


def _outcome(*, payload, round_idx=1):
    return ClientOutcome(
        participated=True,
        fail_reason="",
        samples_count=2,
        duration=0.01,
        loss=0.1,
        metric_value=0.8,
        metric_score=0.8,
        extra_metric=0.7,
        rounds_so_far=round_idx,
        comm_down=8,
        comm_up=8,
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
    )


def test_weight_delta_compares_keras_payload_dict_to_model_weight_list():
    before = [np.asarray([1.0, 2.0]), np.asarray([[3.0]])]
    after = {"layer_0": np.asarray([1.0, 4.0]), "layer_1": np.asarray([[6.0]])}

    delta = weight_delta(before, after)

    assert delta["available"] is True
    assert delta["changed"] is True
    assert np.isclose(delta["l2"], np.sqrt(13.0))
    assert np.isclose(delta["max_abs"], 3.0)
    assert delta["keys_compared"] == 2


def test_count_model_params_uses_nested_core_model():
    class FakeParam:
        def __init__(self, size):
            self.size = size

        def numel(self):
            return self.size

    class FakeTorchModel:
        def parameters(self):
            return [FakeParam(5), FakeParam(7)]

    class FakeCore:
        model = FakeTorchModel()

    class FakeAdapter:
        core = FakeCore()

    assert orch._count_model_params(FakeAdapter()) == 12


def test_repeated_rounds_are_expected_for_non_updating_regimes():
    previous = {"loss": 0.1, "metric": 0.9, "score": 0.9, "extra": 0.8}
    current = dict(previous)

    values = repeated_round_metrics(
        previous,
        current,
        expected_update=False,
        global_weights_changed=False,
    )

    assert values["round_repeated_global_metrics_flag"] is True
    assert values["round_repetition_expected_flag"] is True
    assert values["round_redundant_flag"] is False


def test_repeated_rounds_are_redundant_when_update_expected_but_weights_static():
    previous = {"loss": 0.1, "metric": 0.9, "score": 0.9, "extra": 0.8}
    current = dict(previous)

    values = repeated_round_metrics(
        previous,
        current,
        expected_update=True,
        global_weights_changed=False,
    )

    assert values["round_repeated_global_metrics_flag"] is True
    assert values["round_repetition_expected_flag"] is False
    assert values["round_redundant_flag"] is True


def test_evaluate_run_dynamics_reports_redundant_rounds(tmp_path):
    db_path = tmp_path / "dynamics.db"
    writer = SQLiteWriter(str(db_path))
    writer.start()
    writer.seed_metrics()
    writer.write_run(
        {
            "run_id": "run-1",
            "dataset": "dummy",
            "task_type": "classification",
            "model_type": "mlp",
            "num_clients": 2,
            "num_rounds": 2,
        }
    )
    for round_idx in (1, 2):
        writer.write_measurements(
            run_id="run-1",
            round=round_idx,
            client_id=None,
            values={
                "global_loss": 0.2,
                "global_accuracy": 0.8,
                "global_metric_score": 0.8,
                "global_aux_metric": 0.7,
                "federated_update_expected_flag": True,
                "round_global_weight_changed_flag": round_idx == 1,
                "round_repeated_global_metrics_flag": round_idx == 2,
                "round_redundant_flag": round_idx == 2,
            },
        )
    writer.finish()

    summary = evaluate_run_dynamics(db_path, run_id="run-1")[0]

    assert summary["redundant_rounds"] == [2]
    assert summary["update_expected_rounds"] == [1, 2]
    assert summary["issues"]


def test_orchestrator_records_client_server_and_carry_forward_dynamics(monkeypatch, tmp_path):
    db_path = tmp_path / "orchestrator.db"

    def fake_load_dataset(dataset, **kwargs):
        x = np.arange(8, dtype=np.float32).reshape(4, 2)
        y = np.asarray([0, 1, 0, 1], dtype=np.int64)
        return (x, y), (x, y), {"input_shape": (2,), "num_classes": 2, "task_type": "classification"}

    class FakeModel:
        def __init__(self):
            self._weights = [np.asarray([0.0], dtype=np.float64)]

        def get_weights(self):
            return [np.array(w, copy=True) for w in self._weights]

        def set_weights(self, weights):
            self._weights = [np.asarray(w, dtype=np.float64).copy() for w in weights]

        def count_params(self):
            return 1

    class FakeStrategy:
        inference_only = False

        def __init__(self):
            self.hf_task = "unknown"

        def build_model(self):
            return FakeModel()

        def comm_down_bytes(self, global_model):
            return 8

        def loggable_run_params(self):
            return {"aggregator": {"strategy": "fedavg_uniform"}}

        def summary_lines(self):
            return []

        def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down):
            current = global_model.get_weights()[0]
            payload = {"layer_0": current + float(round_idx)}
            return _outcome(payload=payload, round_idx=round_idx)

        def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
            mean_weight = np.mean([payload["layer_0"] for payload in client_payloads], axis=0)
            global_model.set_weights([mean_weight])
            return 1.0 / float(round_idx), float(round_idx), float(round_idx), 0.5

    monkeypatch.setattr(orch, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(orch, "make_task_strategy", lambda **kwargs: FakeStrategy())

    gen = orch.FederatedDataGenerator(
        config={
            "db_path": str(db_path),
            "dataset": "dummy",
            "task_type": "classification",
            "model_type": "mlp",
            "num_clients": 2,
            "num_rounds": 2,
            "local_epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.1,
            "distribution_type": "iid",
            "distribution_param": None,
            "custom_distributions": None,
            "sample_size": None,
            "sample_frac": None,
            "client_dropout_rate": 0.0,
            "save_weights": False,
            "verbose_progress": False,
            "seed": 7,
        },
        dataset="dummy",
        task_type="classification",
        model_type="mlp",
    )
    result = gen.run()

    assert result["rounds"] == 2

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT m.round, m.client_id, md.name, m.value_num, m.value_bool
            FROM measurements m
            JOIN metrics md ON md.metric_id = m.metric_id
            WHERE m.run_id = ? AND md.name IN (
                'client_update_changed_flag',
                'round_global_weight_changed_flag',
                'global_weights_carried_forward_flag',
                'round_redundant_flag'
            )
            ORDER BY m.round, m.client_id, md.name
            """,
            (result["run_id"],),
        ).fetchall()

    by_metric = {(row[0], row[1], row[2]): row[4] for row in rows}
    assert by_metric[(1, None, "round_global_weight_changed_flag")] == 1
    assert by_metric[(2, None, "round_global_weight_changed_flag")] == 1
    assert by_metric[(2, None, "global_weights_carried_forward_flag")] == 1
    assert by_metric[(2, None, "round_redundant_flag")] == 0
    assert any(
        row[2] == "client_update_changed_flag" and row[4] == 1
        for row in rows
        if row[1] is not None
    )


def test_orchestrator_persists_fetched_hf_model_metadata(monkeypatch, tmp_path):
    db_path = tmp_path / "hf-metadata.db"

    def fake_load_dataset(dataset, **kwargs):
        x = np.arange(4, dtype=np.float32).reshape(2, 2)
        y = np.asarray([0, 1], dtype=np.int64)
        return (
            (x, y),
            (x, y),
            {
                "input_shape": (2,),
                "num_classes": 2,
                "task_type": "classification",
                "hf_task": "sequence_classification",
            },
        )

    class FakeModel:
        def __init__(self):
            self._weights = [np.asarray([0.0], dtype=np.float64)]

        def get_weights(self):
            return [np.array(w, copy=True) for w in self._weights]

        def set_weights(self, weights):
            self._weights = [np.asarray(w, dtype=np.float64).copy() for w in weights]

        def count_params(self):
            return 1

    class FakeStrategy:
        inference_only = False

        def __init__(self):
            self.hf_task = "sequence_classification"

        def build_model(self):
            return FakeModel()

        def comm_down_bytes(self, global_model):
            return 8

        def loggable_run_params(self):
            return {"adapter": {"hf_model_id": "org/test-model"}}

        def summary_lines(self):
            return []

        def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down):
            return _outcome(payload={"layer_0": np.asarray([1.0])}, round_idx=round_idx)

        def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
            global_model.set_weights([np.asarray([1.0])])
            return 0.1, 0.9, 0.8, 0.01

    calls = []

    def fake_fetch_hf_model_meta(hf_model_id):
        calls.append(hf_model_id)
        return {
            "hf_model_id": hf_model_id,
            "hf_pipeline_tag": "text-classification",
            "hf_downloads": 12345,
            "hf_likes": 678,
            "hf_author": "org",
            "hf_url": f"https://huggingface.co/{hf_model_id}",
            "hf_service_meta_json": '{"provider": "test"}',
        }

    monkeypatch.setattr(orch, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(orch, "make_task_strategy", lambda **kwargs: FakeStrategy())
    monkeypatch.setattr(orch, "fetch_hf_model_meta", fake_fetch_hf_model_meta)

    gen = orch.FederatedDataGenerator(
        config={
            "db_path": str(db_path),
            "dataset": "hf",
            "task_type": "classification",
            "model_type": "hf_finetune",
            "num_clients": 1,
            "num_rounds": 1,
            "local_epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.1,
            "distribution_type": "iid",
            "distribution_param": None,
            "custom_distributions": None,
            "sample_size": None,
            "sample_frac": None,
            "client_dropout_rate": 0.0,
            "save_weights": False,
            "verbose_progress": False,
            "seed": 7,
        },
        dataset="hf",
        task_type="classification",
        model_type="hf_finetune",
        dataset_args={
            "dataset_name": "dummy-hf-dataset",
            "hf_model_id": "org/test-model",
            "hf_task": "sequence_classification",
        },
    )
    result = gen.run()

    assert calls == ["org/test-model"]

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT key, value_int, value_num, value_text
            FROM run_params
            WHERE run_id = ?
              AND scope = 'dataset'
              AND key IN ('hf_downloads', 'hf_likes', 'hf_pipeline_tag', 'hf_url')
            """,
            (result["run_id"],),
        ).fetchall()

    values = {
        key: value_int if value_int is not None else value_num if value_num is not None else value_text
        for key, value_int, value_num, value_text in rows
    }
    assert values["hf_downloads"] == 12345
    assert values["hf_likes"] == 678
    assert values["hf_pipeline_tag"] == "text-classification"
    assert values["hf_url"] == "https://huggingface.co/org/test-model"
