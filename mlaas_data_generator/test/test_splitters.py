import numpy as np

from mlaas_data_generator.data.splitters import _shrink_dataset, split_data


def test_shrink_dataset_uses_stricter_bound_when_size_and_frac_provided():
    x = np.arange(100)
    y = np.arange(100)

    x2, y2 = _shrink_dataset(x, y, sample_size=10, sample_frac=0.8, rng=123)

    assert len(x2) == 10
    assert len(y2) == 10


def test_dirichlet_falls_back_to_iid_for_token_level_labels():
    x = np.arange(20)
    y = np.tile(np.arange(4), (20, 1))

    clients, resolved = split_data(x, y, num_clients=2, strategy="dirichlet", distribution_param=0.5, rng=123)

    assert set(clients.keys()) == {"client_1", "client_2"}
    assert resolved["strategy"] == "iid"
    assert "fallback_reason" in resolved


def test_iid_with_fewer_samples_than_clients_preserves_all_samples_without_crash():
    x = np.arange(3)
    y = np.arange(3)

    clients, resolved = split_data(x, y, num_clients=5, strategy="iid", rng=123)

    assert resolved["strategy"] == "iid"
    assert len(clients) == 5
    assigned = sum(len(client["x"]) for client in clients.values())
    non_empty = sum(1 for client in clients.values() if len(client["x"]) > 0)
    assert assigned == len(x)
    assert non_empty >= 1


def test_iid_with_remainder_does_not_drop_records():
    x = np.arange(10)
    y = np.arange(10)

    clients, _ = split_data(x, y, num_clients=3, strategy="iid", rng=123)

    assigned_indices = np.concatenate([client["x"] for client in clients.values()])
    assert len(assigned_indices) == len(x)
    assert set(assigned_indices.tolist()) == set(x.tolist())


def test_iid_total_assigned_equals_original_sample_count():
    x = np.arange(17)
    y = np.arange(17)

    clients, _ = split_data(x, y, num_clients=4, strategy="iid", rng=123)

    total_x = sum(len(client["x"]) for client in clients.values())
    total_y = sum(len(client["y"]) for client in clients.values())
    assert total_x == len(x)
    assert total_y == len(y)
