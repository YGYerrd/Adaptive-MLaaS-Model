import numpy as np

from mlaas_data_generator.data.distributions import get_data_distribution, get_token_label_stats


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


def test_get_data_distribution_detection_dict_targets():
    y = [
        {"boxes": [[0, 0, 10, 10], [10, 10, 20, 20]], "labels": [1, 2]},
        {"boxes": [[0, 0, 8, 8]], "labels": [2]},
    ]

    stats = get_data_distribution(y, num_classes=None)

    assert stats["samples"] == 2
    assert stats["total_boxes"] == 3
    assert stats["avg_boxes_per_sample"] == 1.5
    assert stats["class_counts"] == {1: 1, 2: 2}


def test_get_data_distribution_detection_supports_classes_and_class_labels():
    y = [
        {"boxes": [[0, 0, 10, 10]], "classes": [5]},
        {"bbox": [[1, 1, 8, 8]], "class_labels": [7]},
    ]

    stats = get_data_distribution(y, num_classes=None)

    assert stats["samples"] == 2
    assert stats["total_boxes"] == 2
    assert stats["class_counts"] == {5: 1, 7: 1}


def test_get_data_distribution_detection_supports_nested_annotation_schemas():
    y = [
        {"annotation": {"objects": {"bbox": [[0, 0, 10, 10]], "category": [3]}}},
        {"annotation": {"annotations": {"boxes": [[2, 2, 6, 6]], "category_id": [4]}}},
    ]

    stats = get_data_distribution(y, num_classes=None)

    assert stats["samples"] == 2
    assert stats["total_boxes"] == 2
    assert stats["class_counts"] == {3: 1, 4: 1}


def test_get_data_distribution_counts_object_labels_when_num_classes_unknown():
    stats = get_data_distribution(["cat", "cat", "home", None], num_classes=None, bins=2)

    assert stats["cat"] == 2
    assert stats["home"] == 1
