import numpy as np

from mlaas_data_generator.data.splitters import _shrink_dataset


def test_shrink_dataset_uses_stricter_bound_when_size_and_frac_provided():
    x = np.arange(100)
    y = np.arange(100)

    x2, y2 = _shrink_dataset(x, y, sample_size=10, sample_frac=0.8, rng=123)

    assert len(x2) == 10
    assert len(y2) == 10
