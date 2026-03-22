import numpy as np

def get_mlm_masked_token_stats(y, *, ignore_index=-100, top_k=10):
    """Return MLM masking-oriented stats instead of class histograms."""
    if y is None:
        return {
            "total_tokens": 0,
            "masked_tokens": 0,
            "masked_ratio": 0.0,
            "unique_masked_token_ids": 0,
            "top_masked_token_ids": {},
        }

    y_arr = np.asarray(y)
    if y_arr.size == 0:
        return {
            "total_tokens": 0,
            "masked_tokens": 0,
            "masked_ratio": 0.0,
            "unique_masked_token_ids": 0,
            "top_masked_token_ids": {},
        }

    y_flat = y_arr.reshape(-1)
    total_tokens = int(y_flat.size)
    mask = y_flat != int(ignore_index)
    masked = y_flat[mask]
    masked_tokens = int(masked.size)
    masked_ratio = float(masked_tokens / max(1, total_tokens))

    if masked_tokens == 0:
        return {
            "total_tokens": total_tokens,
            "masked_tokens": 0,
            "masked_ratio": masked_ratio,
            "unique_masked_token_ids": 0,
            "top_masked_token_ids": {},
        }

    token_ids, counts = np.unique(masked.astype("int64", copy=False), return_counts=True)
    order = np.argsort(counts)[::-1][: int(top_k)]
    top = {int(token_ids[i]): int(counts[i]) for i in order}

    return {
        "total_tokens": total_tokens,
        "masked_tokens": masked_tokens,
        "masked_ratio": masked_ratio,
        "unique_masked_token_ids": int(token_ids.size),
        "top_masked_token_ids": top,
    }



def get_token_label_stats(y, *, ignore_index=-100, pad_token_id=None, top_k=10):
    """Return summary stats for token-label tasks like generation."""
    if y is None:
        return {
            "total_tokens": 0,
            "supervised_tokens": 0,
            "supervised_ratio": 0.0,
            "unique_supervised_token_ids": 0,
            "top_supervised_token_ids": {},
        }

    y_arr = np.asarray(y)
    if y_arr.size == 0:
        return {
            "total_tokens": 0,
            "supervised_tokens": 0,
            "supervised_ratio": 0.0,
            "unique_supervised_token_ids": 0,
            "top_supervised_token_ids": {},
        }

    y_flat = y_arr.reshape(-1)
    total_tokens = int(y_flat.size)

    mask = np.ones(total_tokens, dtype=bool)
    if ignore_index is not None:
        mask &= (y_flat != int(ignore_index))
    if pad_token_id is not None:
        mask &= (y_flat != int(pad_token_id))

    supervised = y_flat[mask]
    supervised_tokens = int(supervised.size)
    supervised_ratio = float(supervised_tokens / max(1, total_tokens))

    if supervised_tokens == 0:
        return {
            "total_tokens": total_tokens,
            "supervised_tokens": 0,
            "supervised_ratio": supervised_ratio,
            "unique_supervised_token_ids": 0,
            "top_supervised_token_ids": {},
        }

    try:
        supervised = supervised.astype("int64", copy=False)
    except Exception:
        cleaned = []
        for token in supervised:
            if token is None:
                continue
            try:
                cleaned.append(int(token))
            except Exception:
                continue
        supervised = np.asarray(cleaned, dtype="int64")
        supervised_tokens = int(supervised.size)
        supervised_ratio = float(supervised_tokens / max(1, total_tokens))
        if supervised_tokens == 0:
            return {
                "total_tokens": total_tokens,
                "supervised_tokens": 0,
                "supervised_ratio": supervised_ratio,
                "unique_supervised_token_ids": 0,
                "top_supervised_token_ids": {},
            }

    token_ids, counts = np.unique(supervised, return_counts=True)
    order = np.argsort(counts)[::-1][: int(top_k)]
    top = {int(token_ids[i]): int(counts[i]) for i in order}

    return {
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "supervised_ratio": supervised_ratio,
        "unique_supervised_token_ids": int(token_ids.size),
        "top_supervised_token_ids": top,
    }

def get_data_distribution(
    y,
    num_classes=None,
    bins=None,
    value_range=None,
    label_pad_value=-100,
):
    # -------------------------
    # Regression path
    # -------------------------
    if num_classes is None:
        if y is None:
            if bins is None:
                bins = 10
            return {f"bin_{i}": 0 for i in range(int(bins))}

        if bins is None:
            bins = 10

        y_arr = np.asarray(y, dtype="float32").reshape(-1)
        if value_range is not None:
            hist, _ = np.histogram(y_arr, bins=int(bins), range=value_range)
        else:
            hist, _ = np.histogram(y_arr, bins=int(bins))

        return {f"bin_{i}": int(hist[i]) for i in range(len(hist))}

    # -------------------------
    # Classification path
    # -------------------------
    num_classes = int(num_classes)
    distribution = {i: 0 for i in range(num_classes)}

    if y is None:
        return distribution

    try:
        y_arr = np.asarray(y)
    except Exception:
        return distribution

    y_flat = y_arr.reshape(-1)

    try:
        y_flat = y_flat.astype("int64", copy=False)
    except Exception:
        cleaned = []
        for t in y_flat:
            if t is None:
                continue
            try:
                cleaned.append(int(t))
            except Exception:
                continue
        if not cleaned:
            return distribution
        y_flat = np.asarray(cleaned, dtype="int64")

    if label_pad_value is not None:
        y_flat = y_flat[y_flat != int(label_pad_value)]
    y_flat = y_flat[y_flat >= 0]

    if y_flat.size == 0:
        return distribution
    counts = np.bincount(y_flat, minlength=num_classes)

    for i in range(num_classes):
        distribution[i] = int(counts[i])

    return distribution


def _generate_regular_distribution(num_clients: int, start_client: int = 1, num_labels: int = 10, samples_per_label: int = 100):
    regular_distributions = {}
    for i in range(start_client, num_clients + 1):
        regular_distributions[f"client_{i}"] = {
            label: samples_per_label for label in range(num_labels)
        }
    return regular_distributions


def prepare_client_distributions(custom_distributions: dict | None, num_clients: int):
    """Validate and extend custom distributions to match num_clients.

    If fewer distributions are provided than num_clients, regular distributions
    are generated for the remaining clients. If more are provided, the extra
    distributions are discarded. A warning is printed in both cases.
    """
    if custom_distributions is None:
        return None

    custom_distributions = {
        client: {int(label): count for label, count in dist.items()}
        for client, dist in custom_distributions.items()
    }

    num_custom = len(custom_distributions)
    if num_custom != num_clients:
        print(
            f"Warning: Provided distributions for {num_custom} clients, "
            f"but {num_clients} clients expected."
        )
        if num_custom < num_clients:
            start = num_custom + 1
            regular = _generate_regular_distribution(num_clients, start)
            custom_distributions.update(regular)
        else:
            allowed = sorted(custom_distributions.keys())[:num_clients]
            custom_distributions = {k: custom_distributions[k] for k in allowed}

    return custom_distributions