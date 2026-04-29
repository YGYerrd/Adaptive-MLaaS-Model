import sqlite3

import numpy as np

from mlaas_data_generator.services import runner


class DummyModel:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, x, y, epochs=1, batch_size=32, verbose=0):
        self.fit_calls += 1

    def evaluate(self, x, y, verbose=0):
        return [0.2, 0.75]

    def predict(self, x, verbose=0):
        return np.asarray([[0.1, 0.9], [0.8, 0.2]])

    def count_params(self):
        return 128

    def get_weights(self):
        return [np.asarray([1.0, 2.0])]


def test_service_runner_writes_one_service_record(monkeypatch, tmp_path):
    db_path = tmp_path / "services.db"

    def fake_load_dataset(name, **kwargs):
        x_train = np.asarray([[0.0], [1.0]])
        y_train = np.asarray([0, 1])
        x_test = np.asarray([[0.0], [1.0]])
        y_test = np.asarray([1, 0])
        meta = {"input_shape": (1,), "num_classes": 2, "task_type": "classification", "input_schema": "tabular_features"}
        return (x_train, y_train), (x_test, y_test), meta

    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(runner, "create_model", lambda **kwargs: DummyModel())
    monkeypatch.setattr(runner, "capture_hardware_snapshot", lambda: {"platform": "test"})

    result = runner.execute_service(
        {
            "service_id": "svc_smoke",
            "db_path": str(db_path),
            "dataset": "synthetic",
            "dataset_name": "synthetic",
            "model_type": "mlp",
            "task_type": "classification",
            "modality": "tabular",
            "training_regime": "generic",
            "training_epochs": 1,
            "batch_size": 2,
        }
    )

    assert result.status == "success"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM services WHERE service_id='svc_smoke'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM service_metrics WHERE service_id='svc_smoke'").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM service_split_provenance WHERE service_id='svc_smoke'").fetchone()[0] == 2
        old_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('rounds','clients','service_client_distributions')"
            )
        }
        assert old_tables == set()
