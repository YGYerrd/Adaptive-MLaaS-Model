import numpy as np

from mlaas_data_generator.data.distributions import get_token_label_stats


def test_get_token_label_stats_ignores_ignore_and_pad_tokens():
    y = np.asarray([
        [-100, 11, 11, 0],
        [-100, 13, 0, 13],
    ])

    stats = get_token_label_stats(y, ignore_index=-100, pad_token_id=0)

    assert stats["total_tokens"] == 8
    assert stats["supervised_tokens"] == 4
    assert stats["unique_supervised_token_ids"] == 2
    assert stats["top_supervised_token_ids"] == {13: 2, 11: 2} or stats["top_supervised_token_ids"] == {11: 2, 13: 2}
    assert stats["supervised_ratio"] == 0.5
