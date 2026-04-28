from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..federated.update_signature import compute_composition_mus


SERVICE_REQUIRED_COLUMNS = [
    "service_id",
    "run_id",
    "task_family",
    "model_type",
    "modality",
    "metric_score",
    "latency",
    "data_volume",
    "resource_cost_score",
    "data_distribution",
    "model_update_signature",
    "update_signature_path",
    "signature_dim",
    "signature_norm",
    "computation_time",
    "batch_size",
    "reliability_score",
    "trust_score",
    "explainability_score",
    "run_regime",
    "weights_exported",
    "latency_supported",
    "explainability_supported",
    "trust_supported",
    "missing_runtime_fields",
]


def _normalise_service_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, source_row in raw.iterrows():
        source = _row_dict(source_row)

        run_id = _text(_pick(source, "run_id", "Run ID"), default=f"run_{idx}")
        service_id = _text(_pick(source, "service_id", "Service ID"), default=run_id)
        task_family = _text(
            _pick(source, "task_family", "task_type", "Task family", "Task type"),
            default="unknown",
        ).lower()
        model_type = _text(_pick(source, "model_type", "Model type"), default="unknown")

        primary_metric_name = _text(
            _pick(source, "metric_primary_name", "Primary metric name", "primary_metric_name"),
            default="metric",
        ).lower()
        metric_value = _number(_pick(source, "metric_score", "Metric score", "Primary metric", "primary_metric"))
        if not _is_finite(metric_value):
            metric_value = 0.0
        if primary_metric_name in {"rmse", "mae", "loss", "perplexity"}:
            metric_score = 1.0 / (1.0 + max(0.0, float(metric_value)))
        else:
            metric_score = _bounded(metric_value)

        computation_time = _number(
            _pick(source, "computation_time", "Mean compute time", "compute_time_s", "train_time_s")
        )
        if not _is_finite(computation_time):
            computation_time = 0.0

        latency_raw = _pick(source, "latency", "Latency", "inference_latency_s")
        latency = _number(latency_raw)
        latency_supported = _is_finite(latency)
        missing_runtime_fields = []
        if not latency_supported:
            latency = computation_time
            missing_runtime_fields.append("latency_proxy")

        explainability_score = _number(
            _pick(source, "explainability_score", "Explainability score", "explainability_self_faithfulness_score")
        )
        explainability_supported = _is_finite(explainability_score)
        if not explainability_supported:
            explainability_score = 0.5
            missing_runtime_fields.append("explainability_score")

        trust_score = _number(_pick(source, "trust_score", "Trust score"))
        trust_supported = _is_finite(trust_score)
        if not trust_supported:
            trust_score = 0.5
            missing_runtime_fields.append("trust_score")

        reliability_score = _number(_pick(source, "reliability_score", "Reliability score", "participation_rate", "Participation rate"))
        if not _is_finite(reliability_score):
            reliability_score = trust_score

        row = {
            "service_id": service_id,
            "run_id": run_id,
            "task_family": task_family,
            "model_type": model_type,
            "modality": _text(_pick(source, "modality", "Modality"), default="unknown").lower(),
            "metric_score": _bounded(metric_score),
            "latency": float(max(0.0, latency)),
            "data_volume": float(max(0.0, _number(_pick(source, "data_volume", "Dataset size", "dataset_size"), 0.0))),
            "resource_cost_score": _bounded(_number(_pick(source, "resource_cost_score", "Resource cost score"), 0.5)),
            "data_distribution": _text(_pick(source, "data_distribution", "Data distribution"), default="unknown").lower(),
            "model_update_signature": _pick(source, "model_update_signature", "Model update signature"),
            "update_signature_path": _text(
                _pick(source, "update_signature_path", "signature_path", "Compressed vector path"),
                default="",
            ),
            "signature_dim": _int_or_none(_pick(source, "signature_dim", "Signature dim")),
            "signature_norm": _number(_pick(source, "signature_norm", "Signature norm")),
            "computation_time": float(max(0.0, computation_time)),
            "batch_size": int(max(1, _number(_pick(source, "batch_size", "Batch size"), 1.0))),
            "reliability_score": _bounded(reliability_score),
            "trust_score": _bounded(trust_score),
            "explainability_score": _bounded(explainability_score),
            "run_regime": _text(_pick(source, "run_regime", "Run regime"), default="generic"),
            "weights_exported": _bool(_pick(source, "weights_exported", "Weights exported"), default=False),
            "latency_supported": bool(latency_supported),
            "explainability_supported": bool(explainability_supported),
            "trust_supported": bool(trust_supported),
            "missing_runtime_fields": missing_runtime_fields,
        }
        rows.append(row)

    services = pd.DataFrame(rows)
    for column in SERVICE_REQUIRED_COLUMNS:
        if column not in services.columns:
            services[column] = None
    return services[SERVICE_REQUIRED_COLUMNS]


def generate_service_requests(
    services: pd.DataFrame,
    *,
    request_count: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    services = _normalise_if_needed(services)
    rng = np.random.default_rng(seed)
    task_families = sorted(str(v) for v in services["task_family"].dropna().unique())
    if not task_families:
        task_families = ["unknown"]

    rows = []
    for idx in range(int(request_count)):
        task = str(rng.choice(task_families))
        pool = services[services["task_family"] == task]
        if pool.empty:
            pool = services

        pool_size = int(len(pool))
        max_workflow = max(1, min(3, pool_size))
        min_workflow = 1 if pool_size == 1 else 2
        workflow_length = int(rng.integers(min_workflow, max_workflow + 1))

        rows.append(
            {
                "request_id": f"req_{idx:06d}",
                "task_family": task,
                "workflow_length": workflow_length,
                "min_quality": float(rng.uniform(0.45, 0.8)),
                "max_latency": float(max(0.01, np.nanpercentile(pool["latency"], 75) * rng.uniform(1.0, 1.8))),
                "max_resource_cost": float(rng.uniform(0.55, 1.0)),
            }
        )
    return pd.DataFrame(rows)


def generate_compositions(
    services: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    candidates_per_request: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    services = _normalise_if_needed(services)
    rng = np.random.default_rng(seed)
    rows = []

    for _, request_row in requests.iterrows():
        request = _row_dict(request_row)
        request_id = str(request.get("request_id"))
        task_family = str(request.get("task_family") or "").lower()
        pool = services[services["task_family"].astype(str).str.lower() == task_family]
        if pool.empty:
            pool = services

        workflow_length = int(request.get("workflow_length") or min(2, len(pool)))
        workflow_length = max(1, min(workflow_length, len(pool)))

        combinations = list(itertools.combinations(range(len(pool)), workflow_length))
        rng.shuffle(combinations)
        selected_combinations = combinations[: int(candidates_per_request)]

        if len(selected_combinations) < int(candidates_per_request):
            seen = {combo for combo in selected_combinations}
            attempts = 0
            while len(selected_combinations) < int(candidates_per_request) and attempts < 1000:
                combo = tuple(sorted(rng.choice(len(pool), size=workflow_length, replace=False).tolist()))
                if combo not in seen:
                    selected_combinations.append(combo)
                    seen.add(combo)
                attempts += 1

        for candidate_idx, combo in enumerate(selected_combinations):
            selected = pool.iloc[list(combo)].reset_index(drop=True)
            component_scores = _composition_components(selected, request)
            composability_score = _weighted_composability(component_scores)
            penalty_adjusted_score = _penalty_adjusted_score(composability_score, selected, request)
            service_ids = [str(value) for value in selected["service_id"].tolist()]
            rows.append(
                {
                    "request_id": request_id,
                    "candidate_id": f"{request_id}_cand_{candidate_idx:03d}",
                    "service_ids": json.dumps(service_ids),
                    "workflow_length": int(len(service_ids)),
                    **component_scores,
                    "composability_score": float(composability_score),
                    "penalty_adjusted_score": float(penalty_adjusted_score),
                    "selected_flag": False,
                }
            )

    compositions = pd.DataFrame(rows)
    if compositions.empty:
        return compositions

    for request_id, group in compositions.groupby("request_id"):
        best_idx = group["penalty_adjusted_score"].idxmax()
        compositions.loc[best_idx, "selected_flag"] = True

    score_columns = ["dhs", "mus", "shs", "ses", "hsq", "srs", "composability_score", "penalty_adjusted_score"]
    for column in score_columns:
        compositions[column] = compositions[column].astype(float).clip(0.0, 1.0)
    return compositions.reset_index(drop=True)


def load_services_from_db(db_path: str | Path) -> pd.DataFrame:
    """Load one service row per run from a normalised MLaaS SQLite DB."""
    query = """
    WITH latest_round AS (
      SELECT run_id, MAX(round) AS round
      FROM measurements
      WHERE round IS NOT NULL
      GROUP BY run_id
    ),
    metric_values AS (
      SELECT
        m.run_id,
        m.round,
        m.client_id,
        md.name,
        COALESCE(m.value_num, m.value_int, m.value_bool) AS numeric_value,
        m.value_text
      FROM measurements m
      JOIN metrics md ON md.metric_id = m.metric_id
    ),
    run_params_pivot AS (
      SELECT
        run_id,
        MAX(CASE WHEN scope = 'runner' AND key = 'run_regime' THEN COALESCE(value_text, CAST(value_int AS TEXT), CAST(value_num AS TEXT), CAST(value_bool AS TEXT)) END) AS run_regime,
        MAX(CASE WHEN scope = 'runner' AND key = 'weights_exported' THEN value_bool END) AS weights_exported,
        MAX(CASE WHEN scope = 'adapter' AND key = 'modality' THEN value_text END) AS modality,
        MAX(CASE WHEN scope = 'dataset' AND key = 'modality' THEN value_text END) AS dataset_modality
      FROM run_params
      GROUP BY run_id
    )
    SELECT
      r.run_id,
      r.run_id AS service_id,
      r.task_type AS task_family,
      r.model_type,
      COALESCE(rp.modality, rp.dataset_modality, 'unknown') AS modality,
      MAX(CASE WHEN mv.name = 'global_metric_score' THEN mv.numeric_value END) AS metric_score,
      AVG(CASE WHEN mv.name = 'inference_latency_s' THEN mv.numeric_value END) AS latency,
      AVG(CASE WHEN mv.name = 'compute_time_s' THEN mv.numeric_value END) AS computation_time,
      MAX(CASE WHEN mv.name = 'train_set_size' THEN mv.numeric_value END) AS data_volume,
      AVG(CASE WHEN mv.name = 'resource_cost_score' THEN mv.numeric_value END) AS resource_cost_score,
      MAX(CASE WHEN mv.name = 'update_signature_path' THEN mv.value_text END) AS update_signature_path,
      MAX(CASE WHEN mv.name = 'signature_dim' THEN mv.numeric_value END) AS signature_dim,
      MAX(CASE WHEN mv.name = 'signature_norm' THEN mv.numeric_value END) AS signature_norm,
      AVG(CASE WHEN mv.name = 'trust_score' THEN mv.numeric_value END) AS trust_score,
      AVG(CASE WHEN mv.name = 'explainability_score' THEN mv.numeric_value END) AS explainability_score,
      AVG(CASE WHEN mv.name = 'participated_flag' THEN mv.numeric_value END) AS reliability_score,
      rp.run_regime,
      rp.weights_exported
    FROM runs r
    LEFT JOIN latest_round lr ON lr.run_id = r.run_id
    LEFT JOIN metric_values mv ON mv.run_id = r.run_id
      AND (mv.round = lr.round OR mv.round IS NULL)
    LEFT JOIN run_params_pivot rp ON rp.run_id = r.run_id
    GROUP BY r.run_id, r.task_type, r.model_type
    """
    with sqlite3.connect(str(db_path)) as conn:
        raw = pd.read_sql_query(query, conn)
    return _normalise_service_rows(raw)


def _composition_components(selected: pd.DataFrame, request: Mapping[str, Any]) -> dict[str, float]:
    records = selected.to_dict(orient="records")
    return {
        "dhs": _data_homogeneity_score(selected),
        "mus": compute_composition_mus(records),
        "shs": _schema_homogeneity_score(selected),
        "ses": _service_efficiency_score(selected),
        "hsq": float(np.nanmean(selected["metric_score"].astype(float))),
        "srs": _service_reliability_score(selected),
    }


def _weighted_composability(scores: Mapping[str, float]) -> float:
    weights = {
        "dhs": 0.15,
        "mus": 0.25,
        "shs": 0.15,
        "ses": 0.15,
        "hsq": 0.20,
        "srs": 0.10,
    }
    total = sum(weights.values())
    value = sum(float(scores.get(key, 0.0)) * weight for key, weight in weights.items()) / total
    return float(np.clip(value, 0.0, 1.0))


def _penalty_adjusted_score(score: float, selected: pd.DataFrame, request: Mapping[str, Any]) -> float:
    penalty = 0.0
    min_quality = _number(request.get("min_quality"))
    max_latency = _number(request.get("max_latency"))
    max_resource_cost = _number(request.get("max_resource_cost"))

    quality = float(np.nanmean(selected["metric_score"].astype(float)))
    latency = float(np.nanmean(selected["latency"].astype(float)))
    resource_cost = float(np.nanmean(selected["resource_cost_score"].astype(float)))

    if _is_finite(min_quality) and quality < min_quality:
        penalty += float(min_quality - quality)
    if _is_finite(max_latency) and latency > max_latency:
        penalty += min(0.5, float((latency - max_latency) / max(max_latency, 1e-9)))
    if _is_finite(max_resource_cost) and resource_cost > max_resource_cost:
        penalty += float(resource_cost - max_resource_cost)

    return float(np.clip(score - penalty, 0.0, 1.0))


def _data_homogeneity_score(selected: pd.DataFrame) -> float:
    values = [str(v) for v in selected["data_distribution"].fillna("unknown")]
    if not values:
        return 0.5
    dominant = max(values.count(value) for value in set(values))
    return float(dominant / len(values))


def _schema_homogeneity_score(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.5
    task_score = _dominant_fraction(selected["task_family"])
    modality_score = _dominant_fraction(selected["modality"])
    model_score = _dominant_fraction(selected["model_type"])
    return float(np.mean([task_score, modality_score, model_score]))


def _service_efficiency_score(selected: pd.DataFrame) -> float:
    latency = selected["latency"].astype(float).to_numpy()
    compute = selected["computation_time"].astype(float).to_numpy()
    cost = selected["resource_cost_score"].astype(float).to_numpy()
    latency_score = 1.0 / (1.0 + float(np.nanmean(np.maximum(latency, 0.0))))
    compute_score = 1.0 / (1.0 + float(np.nanmean(np.maximum(compute, 0.0))))
    cost_score = 1.0 - float(np.nanmean(np.clip(cost, 0.0, 1.0)))
    return float(np.clip(np.mean([latency_score, compute_score, cost_score]), 0.0, 1.0))


def _service_reliability_score(selected: pd.DataFrame) -> float:
    reliability = selected["reliability_score"].astype(float).to_numpy()
    trust = selected["trust_score"].astype(float).to_numpy()
    explainability = selected["explainability_score"].astype(float).to_numpy()
    return float(np.clip(np.nanmean([np.nanmean(reliability), np.nanmean(trust), np.nanmean(explainability)]), 0.0, 1.0))


def _dominant_fraction(values: Sequence[Any]) -> float:
    items = [str(value) for value in values if value is not None]
    if not items:
        return 0.5
    dominant = max(items.count(item) for item in set(items))
    return float(dominant / len(items))


def _normalise_if_needed(services: pd.DataFrame) -> pd.DataFrame:
    if set(SERVICE_REQUIRED_COLUMNS).issubset(set(services.columns)):
        return services.copy()
    return _normalise_service_rows(services)


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def _pick(source: Mapping[str, Any], *keys: str) -> Any:
    lowered = {str(key).strip().lower(): key for key in source.keys()}
    for key in keys:
        actual = lowered.get(str(key).strip().lower())
        if actual is not None:
            return source.get(actual)
    return None


def _text(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and not np.isfinite(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "not available", "n/a"}:
        return default
    return text


def _number(value: Any, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text or text.lower() in {"nan", "none", "null", "not available", "n/a"}:
            return float(default)
        value = text
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _int_or_none(value: Any) -> int | None:
    parsed = _number(value)
    if not _is_finite(parsed):
        return None
    return int(parsed)


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none", "null", "not available", "n/a"}:
            return bool(default)
        return text in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _bounded(value: Any, default: float = 0.0) -> float:
    parsed = _number(value, default)
    if not _is_finite(parsed):
        parsed = default
    return float(np.clip(parsed, 0.0, 1.0))


def _handle(args: argparse.Namespace) -> None:
    if args.input_csv:
        services = _normalise_service_rows(pd.read_csv(args.input_csv))
    else:
        services = load_services_from_db(args.db)
    requests = generate_service_requests(services, request_count=args.requests, seed=args.seed)
    compositions = generate_compositions(
        services,
        requests,
        candidates_per_request=args.candidates_per_request,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    services.to_csv(output_dir / "services.csv", index=False)
    requests.to_csv(output_dir / "requests.csv", index=False)
    compositions.to_csv(output_dir / "compositions.csv", index=False)
    print(f"Wrote service, request, and composition datasets to {output_dir}")


def register_export_finetune_dataset(subparsers) -> None:
    parser = subparsers.add_parser("export-finetune-dataset", help="Export service/request/composition datasets")
    parser.add_argument("--db", default="outputs/federated.db", help="SQLite run database")
    parser.add_argument("--input-csv", default=None, help="Optional pre-flattened service CSV")
    parser.add_argument("--output-dir", default="outputs/finetune_dataset", help="Output directory")
    parser.add_argument("--requests", type=int, default=25, help="Number of synthetic service requests")
    parser.add_argument("--candidates-per-request", type=int, default=10, help="Candidate compositions per request")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(_handler=_handle)
