import sqlite3

import numpy as np

from mlaas_data_generator.federated import orchestrator as orch
from mlaas_data_generator.federated.strategies.base import ClientOutcome
from mlaas_data_generator.federated.update_signature import (
    compute_and_store_update_signature,
    compute_composition_mus,
    load_update_signature_vector,
)


def test_update_signature_persists_normalised_compressed_delta(tmp_path):
    before = {"layer": np.asarray([1.0, 1.0, 1.0])}
    after = {"layer": np.asarray([2.0, 1.0, 1.0])}

    metadata = compute_and_store_update_signature(
        before,
        after,
        output_dir=tmp_path,
        run_id="run-a",
        round_idx=1,
        client_id="client_1",
        dim=8,
        seed=7,
    )

    assert metadata["update_signature_id"]
    assert metadata["signature_dim"] == 8
    assert np.isclose(metadata["signature_norm"], 1.0)

    vector = load_update_signature_vector(metadata)
    assert vector is not None
    assert vector.shape == (8,)
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_composition_mus_uses_selected_service_signature_alignment(tmp_path):
    aligned_a = compute_and_store_update_signature(
        {"w": np.asarray([0.0, 0.0])},
        {"w": np.asarray([1.0, 0.0])},
        output_dir=tmp_path,
        run_id="run-a",
        round_idx=1,
        client_id="client_1",
        dim=16,
        seed=3,
    )
    aligned_b = compute_and_store_update_signature(
        {"w": np.asarray([0.0, 0.0])},
        {"w": np.asarray([2.0, 0.0])},
        output_dir=tmp_path,
        run_id="run-b",
        round_idx=1,
        client_id="client_1",
        dim=16,
        seed=3,
    )
    opposing = compute_and_store_update_signature(
        {"w": np.asarray([0.0, 0.0])},
        {"w": np.asarray([-1.0, 0.0])},
        output_dir=tmp_path,
        run_id="run-c",
        round_idx=1,
        client_id="client_1",
        dim=16,
        seed=3,
    )

    aligned_score = compute_composition_mus([aligned_a, aligned_b])
    conflicting_score = compute_composition_mus([aligned_a, opposing])

    assert aligned_score > 0.99
    assert conflicting_score < 0.01


def test_orchestrator_records_update_signature_metadata(monkeypatch, tmp_path):
    db_path = tmp_path / "signatures.db"

    def fake_load_dataset(dataset, **kwargs):
        x = np.arange(8, dtype=np.float32).reshape(4, 2)
        y = np.asarray([0, 1, 0, 1], dtype=np.int64)
        return (x, y), (x, y), {"input_shape": (2,), "num_classes": 2, "task_type": "classification"}

    class FakeModel:
        def __init__(self):
            self._weights = [np.asarray([0.0, 0.0], dtype=np.float64)]

        def get_weights(self):
            return [np.array(w, copy=True) for w in self._weights]

        def set_weights(self, weights):
            self._weights = [np.asarray(w, dtype=np.float64).copy() for w in weights]

        def count_params(self):
            return 2

    class FakeStrategy:
        inference_only = False

        def __init__(self):
            self.hf_task = "unknown"

        def build_model(self):
            return FakeModel()

        def comm_down_bytes(self, global_model):
            return 16

        def loggable_run_params(self):
            return {"aggregator": {"strategy": "fedavg_uniform"}}

        def summary_lines(self):
            return []

        def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down):
            payload = {"layer_0": global_model.get_weights()[0] + np.asarray([1.0, 0.0])}
            return ClientOutcome(
                participated=True,
                fail_reason="",
                samples_count=2,
                duration=0.01,
                loss=0.1,
                metric_value=0.8,
                metric_score=0.8,
                extra_metric=0.7,
                rounds_so_far=rounds_so_far,
                comm_down=comm_down,
                comm_up=16,
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

        def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
            global_model.set_weights([client_payloads[0]["layer_0"]])
            return 0.1, 0.8, 0.8, 0.7

    monkeypatch.setattr(orch, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(orch, "make_task_strategy", lambda **kwargs: FakeStrategy())

    gen = orch.FederatedDataGenerator(
        config={
            "db_path": str(db_path),
            "dataset": "dummy",
            "task_type": "classification",
            "model_type": "mlp",
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
            "update_signature_dim": 12,
        },
        dataset="dummy",
        task_type="classification",
        model_type="mlp",
    )
    result = gen.run()

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT md.name, m.value_int, m.value_num, m.value_text
            FROM measurements m
            JOIN metrics md ON md.metric_id = m.metric_id
            WHERE m.run_id = ?
              AND m.client_id = 'client_1'
              AND md.name IN ('update_signature_id', 'signature_dim', 'signature_norm', 'update_signature_path')
            """,
            (result["run_id"],),
        ).fetchall()

    values = {
        name: value_text if value_text is not None else value_int if value_int is not None else value_num
        for name, value_int, value_num, value_text in rows
    }
    assert values["signature_dim"] == 12
    assert np.isclose(values["signature_norm"], 1.0)
    assert values["update_signature_id"]
    assert load_update_signature_vector(values["update_signature_path"]) is not None
