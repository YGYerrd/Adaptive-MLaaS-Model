import numpy as np
from numpy.random import default_rng, Generator


def _take(x, idx):
    if isinstance(x, dict):
        return {k: _take(v, idx) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[idx]
    if isinstance(x, (list, tuple)):
        idx_list = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
        return [x[i] for i in idx_list]
    x_arr = np.asarray(x, dtype=object)
    return x_arr[idx]

def _num_samples(x):
    """Return the number of examples in ``x`` across supported container shapes."""
    if isinstance(x, dict):
        if not x:
            return 0
        first = next(iter(x.values()))
        return int(len(first))
    return int(len(x))


def _is_scalar_label_vector(y):
    """True when labels are a 1D vector of per-example scalar class ids."""
    arr = np.asarray(y, dtype=object)
    if arr.ndim != 1:
        return False
    if arr.size == 0:
        return True
    sample = arr[0]
    if isinstance(sample, (list, tuple, dict, np.ndarray)):
        return False
    return True

def _build_clients_from_indices(x, y, indices_by_client: dict):
    clients = {}
    for cid, idx in indices_by_client.items():
        clients[cid] = {"x": _take(x, idx), "y": _take(y, idx)}
    return clients


def _seed(rng):
    return rng if isinstance(rng, Generator) else default_rng()


def _split_iid(x, y, num_clients, rng=None):
    n = _num_samples(x)
    seed = _seed(rng)

    idx = seed.permutation(n)
    splits = np.array_split(idx, num_clients)
    indices_by_client = {
        f"client_{i+1}": split for i, split in enumerate(splits)
    }
    return _build_clients_from_indices(x, y, indices_by_client)


def _split_quantity_skew(x, y, num_clients, alpha, rng=None):
    """Split data with IID label distribution but uneven sample counts.
    `alpha` controls the difference in client sizes.
    Larger `alpha` results in more balanced client sizes.
    """
    n = _num_samples(x)
    seed = _seed(rng)

    proportions = seed.dirichlet([alpha] * num_clients)
    counts = (proportions * n).astype(int)

    # Fix rounding
    diff = n - counts.sum()
    for i in range(abs(diff)):
        counts[i % num_clients] += 1 if diff > 0 else -1

    idx = seed.permutation(n)
    indices_by_client = {}
    start = 0
    for i, count in enumerate(counts):
        end = start + count
        indices_by_client[f"client_{i+1}"] = idx[start:end]
        start = end
    return _build_clients_from_indices(x, y, indices_by_client)


def _split_dirichlet_label_skew(x, y, num_clients, alpha, rng=None):
    """Split data so that each class is distributed via Dirichlet over clients."""
    seed = _seed(rng)
    num_classes = len(np.unique(y))
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]

    indices_by_client = {f"client_{i+1}": [] for i in range(num_clients)}

    for indices in class_indices:
        seed.shuffle(indices)
        proportions = seed.dirichlet([alpha] * num_clients)
        counts = (proportions * len(indices)).astype(int)
        diff = len(indices) - counts.sum()
        for i in range(abs(diff)):
            counts[i % num_clients] += 1 if diff > 0 else -1

        start = 0
        for i, count in enumerate(counts):
            end = start + max(count, 0)
            if end > start:
                indices_by_client[f"client_{i+1}"].extend(indices[start:end].tolist())
            start = end

    indices_by_client = {cid: np.asarray(idxs, dtype=int) for cid, idxs in indices_by_client.items()}
    return _build_clients_from_indices(x, y, indices_by_client)


def _split_shard_based(x, y, num_clients, shards_per_client, rng=None):
    """Sort by label, split into shards, randomly assign shards to clients."""
    seed = _seed(rng)
    num_shards = num_clients * shards_per_client
    idx_sorted = np.argsort(y, kind="stable")
    shards = np.array_split(idx_sorted, num_shards)
    seed.shuffle(shards)
    indices_by_client = {}
    for i in range(num_clients):
        shard_indices = np.concatenate(shards[i * shards_per_client : (i + 1) * shards_per_client])
        indices_by_client[f"client_{i+1}"] = shard_indices
    return _build_clients_from_indices(x, y, indices_by_client)


def _split_label_per_client(x, y, num_clients, k, rng=None):
    """Each client receives data from only k labels (chosen uniformly without replacement)."""
    seed = _seed(rng)
    num_classes = int(np.max(y)) + 1
    class_indices = {c: np.where(y == c)[0] for c in range(num_classes)}
    clients_labels = {i: seed.choice(num_classes, k, replace=False) for i in range(num_clients)}

    indices_by_client = {f"client_{i+1}": [] for i in range(num_clients)}
    for label, idxs in class_indices.items():
        recipients = [cid for cid, labels in clients_labels.items() if label in labels]
        if not recipients:
            continue
        seed.shuffle(idxs)
        splits = np.array_split(idxs, len(recipients))
        for cid, split in zip(recipients, splits):
            if len(split) > 0:
                indices_by_client[f"client_{cid+1}"].extend(split.tolist())

    indices_by_client = {cid: np.asarray(idxs, dtype=int) for cid, idxs in indices_by_client.items()}
    return _build_clients_from_indices(x, y, indices_by_client)


def _split_custom_data(x, y, client_distributions: dict, rng=None):
    """Split ``(x, y)`` according to ``client_distributions``."""
    seed = _seed(rng)
    num_classes = int(np.max(y)) + 1

    # Build a mutable pool of available indices per label
    pool_by_label = {}
    for lbl in range(num_classes):
        idxs = np.where(y == lbl)[0]
        seed.shuffle(idxs)
        pool_by_label[lbl] = idxs

    # Allocate indices to clients based on requested counts
    indices_by_client = {cid: [] for cid in client_distributions.keys()}
    for cid, dist in client_distributions.items():
        for label_raw, count in dist.items():
            lbl = int(label_raw)
            if lbl not in pool_by_label:
                continue
            pool = pool_by_label[lbl]
            if len(pool) == 0 or count <= 0:
                continue
            take = min(int(count), len(pool))
            chosen, remaining = pool[:take], pool[take:]
            indices_by_client[cid].extend(chosen.tolist())
            pool_by_label[lbl] = remaining  # shrink pool

    indices_by_client = {cid: np.asarray(idxs, dtype=int) for cid, idxs in indices_by_client.items()}
    return _build_clients_from_indices(x, y, indices_by_client)


def _shrink_dataset(x, y, sample_size=None, sample_frac=None, rng=None):
    seed = _seed(rng)
    n = _num_samples(x)
    if sample_size is None and sample_frac is None:
        return x, y
    requested_size = None if sample_size is None else int(sample_size)
    if sample_frac is not None:
        frac_size = int(round(n * float(sample_frac)))
        sample_size = frac_size if requested_size is None else min(requested_size, frac_size)
    sample_size = max(0, min(n, int(sample_size)))
    idx = seed.choice(n, size=sample_size, replace=False)

    if isinstance(y, np.ndarray):
        y_subset = y[idx]
    else:
        y_subset = [y[i] for i in idx]

    return _take(x, idx), y_subset


def split_data(x, y, num_clients, strategy = "iid", distribution_param = None, custom_distributions=None, sample_size=None, sample_frac=None, rng=None):
    strategy = strategy.lower()
    if sample_size or sample_frac:
        x,y = _shrink_dataset(x=x, y=y, sample_frac=sample_frac, sample_size=sample_size, rng=rng)
    
    resolved = {"strategy": strategy, "distribution_param": None}
    requires_scalar_labels = {"dirichlet", "shard", "label_per_client", "custom"}
    if strategy in requires_scalar_labels and not _is_scalar_label_vector(y):
        resolved["strategy"] = "iid"
        resolved["fallback_reason"] = (
            f"strategy='{strategy}' requires scalar class labels; "
            "falling back to iid split for structured/token labels"
        )
        return _split_iid(x, y, num_clients, rng=rng), resolved

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    if strategy == "iid":
        return _split_iid(x, y, num_clients, rng=rng), resolved

    if strategy == "quantity_skew":
        alpha = float(distribution_param) if distribution_param is not None else 1.0
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for quantity_skew.")
        resolved["distribution_param"] = alpha
        return _split_quantity_skew(x, y, num_clients, alpha, rng=rng), resolved

    if strategy == "dirichlet":
        alpha = float(distribution_param) if distribution_param is not None else 0.5
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for dirichlet.")
        resolved["distribution_param"] = alpha
        return _split_dirichlet_label_skew(x, y, num_clients, alpha, rng=rng), resolved

    if strategy == "shard":
        shards_per_client = int(distribution_param) if distribution_param is not None else 2
        if shards_per_client <= 0:
            raise ValueError("shards_per_client must be > 0 for shard.")
        resolved["distribution_param"] = shards_per_client
        return _split_shard_based(x, y, num_clients, shards_per_client, rng=rng), resolved

    if strategy == "label_per_client":
        k = int(distribution_param) if distribution_param is not None else 1
        if not (1 <= k <= int(np.max(y)) + 1):
            raise ValueError("k must be in [1, num_classes] for label_per_client.")
        resolved["distribution_param"] = k
        return _split_label_per_client(x, y, num_clients, k, rng=rng), resolved

    if strategy == "custom":
        from .distributions import prepare_client_distributions
        if not custom_distributions:
                raise ValueError("custom_distributions must be provided for 'custom' strategy.'")
        adjusted = prepare_client_distributions(custom_distributions, num_clients)
        return _split_custom_data(x, y, adjusted, rng=rng), resolved

    raise ValueError(f"Unknown data split strategy: {strategy}")
