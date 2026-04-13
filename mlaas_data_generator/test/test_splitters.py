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
