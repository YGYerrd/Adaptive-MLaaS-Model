from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from ..config import CONFIG
from ..data.accounting import finalize_accounting
from ..data.distributions import (
    get_data_distribution,
    get_mlm_masked_token_stats,
    get_retrieval_pair_stats,
    get_token_label_stats,
    get_vqa_answer_stats,
)
from ..models.label_schema import infer_label_format, infer_num_labels
from ..models.train_eval import evaluate_model, train_local_model
from ..storage.writer import make_writer
from .perturbation import run_perturbation_stage
from .system_metrics import ResourceTracker, capture_hardware_snapshot
from .taxonomy import canonical_label_format, canonical_metric_names, canonical_task_family, metric_domain, metric_score_value

load_dataset = None
create_model = None


@dataclass
class ServiceExecutionResult:
    service_id: str
    status: str
    db_path: str
    metrics: dict[str, Any]
    error: str | None = None


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "na", "n/a", "nan", "null", "none", "not applicable", "not_applicable"}
    return False


def resolve_service_id(config: Mapping[str, Any]) -> str:
    explicit = config.get("service_id")
    if not _is_blank(explicit):
        return str(explicit).strip()

    parts = [
        config.get("task_type"),
        config.get("hf_task"),
        config.get("hf_model_id") or config.get("model_type"),
        config.get("dataset_name") or config.get("dataset"),
        config.get("dataset_config"),
        config.get("training_regime"),
        config.get("dataset_variant"),
        config.get("split_variant"),
        config.get("knob_variant"),
    ]
    raw = "|".join("" if value is None else str(value).strip().lower() for value in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    prefix = str(config.get("task") or config.get("task_type") or "service").strip().lower().replace(" ", "_")
    prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix).strip("_") or "service"
    return f"{prefix}_{digest}"


class ServiceRunner:
    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(CONFIG)
        self.config.update(dict(config or {}))
        self.config["device"] = _normalize_requested_device(self.config.get("device"))
        self.service_id = resolve_service_id(self.config)

    def run(self) -> ServiceExecutionResult:
        service_id = self.service_id
        db_path = str(self.config.get("db_path") or CONFIG["db_path"])
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        writer = make_writer("sqlite", db_path=db_path)
        writer.start()

        try:
            record, metrics, artifacts, split_provenance_rows = self._execute_service(started_at=started_at)
            writer.write_service(record)
            writer.write_service_metrics(service_id, metrics)
            for row in split_provenance_rows:
                writer.write_service_split_provenance(service_id, **row)
            for artifact in artifacts:
                writer.write_service_artifact(service_id, **artifact)
            writer.finish()
            return ServiceExecutionResult(service_id=service_id, status="success", db_path=db_path, metrics=metrics)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            failure_record = self._service_record(
                status="failed",
                started_at=started_at,
                metadata={
                    "runtime_total_s": elapsed,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            try:
                writer.write_service(failure_record)
                writer.write_service_failure(
                    service_id=service_id,
                    row_index=_safe_int(self.config.get("row_index")),
                    case_name=_safe_str(self.config.get("case_name")),
                    manifest_group_id=_safe_str(self.config.get("manifest_group_id")),
                    failure_stage="service_execution",
                    error_message=str(exc),
                    resolved_config_json=json.dumps(self.config, default=str),
                    traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip(),
                )
                writer.finish()
            except Exception:
                writer.abort()
            return ServiceExecutionResult(service_id=service_id, status="failed", db_path=db_path, metrics={}, error=str(exc))

    def _execute_service(self, *, started_at: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        service_start = time.perf_counter()
        dataset_args = dict(self.config.get("dataset_args") or {})
        training_regime = str(self.config.get("training_regime") or "finetune_transfer").strip().lower()
        dataset_args.setdefault("inference_only", training_regime in {"inference_only", "inference"})
        train, test, meta = _load_dataset(self.config.get("dataset", "hf"), **dataset_args)
        (x_train, y_train), (x_test, y_test) = train, test
        meta = dict(meta or {})
        meta.setdefault("hf_model_id", self.config.get("hf_model_id"))
        meta.setdefault("hf_task", self.config.get("hf_task"))
        meta.setdefault("task_tag", self.config.get("task_tag"))
        hf_metadata = _fetch_hf_metadata(self.config, meta)
        if hf_metadata:
            meta.setdefault("hf_metadata", hf_metadata)
            for key, value in hf_metadata.items():
                if value is not None:
                    meta.setdefault(key, value)

        split_info = self._resolve_service_split(x_train, y_train, meta)
        x_train, y_train = split_info["x_train"], split_info["y_train"]
        split_provenance_rows = list(split_info["provenance_rows"])

        task_family = canonical_task_family(self.config.get("task_type") or meta.get("task_type"), self.config.get("hf_task") or meta.get("hf_task"))
        primary_name, secondary_name = canonical_metric_names(
            task_family,
            self.config.get("metric_key"),
            hf_task=self.config.get("hf_task") or meta.get("hf_task"),
            task_tag=self.config.get("task_tag") or meta.get("task_tag"),
        )
        model_build_start = time.perf_counter()
        model = self._build_model(meta=meta, task_family=task_family)
        model_build_s = time.perf_counter() - model_build_start
        resolved_device = _resolve_execution_device(model)
        gpu_fallback_warning = _gpu_fallback_warning(self.config.get("device"), resolved_device)
        self._print_run_summary(
            task_family=task_family,
            split_info=split_info,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            resolved_device=resolved_device,
        )

        tracker = ResourceTracker()
        tracker.start()
        train_metrics: dict[str, Any] = {}
        eval_qos: dict[str, Any] = {}
        loss = primary = secondary = math.nan
        train_runtime_s = 0.0

        workload_start = time.perf_counter()
        if training_regime in {"inference_only", "inference"}:
            eval_start = time.perf_counter()
            loss, primary, secondary, eval_qos = self._evaluate_model(model, x_test, y_test, inference_only=True, task_family=task_family)
            eval_runtime_s = time.perf_counter() - eval_start
        else:
            train_start = time.perf_counter()
            train_metrics = self._train_model(model, x_train, y_train, task_family=task_family)
            train_runtime_s = time.perf_counter() - train_start
            eval_start = time.perf_counter()
            loss, primary, secondary, eval_qos = self._evaluate_model(model, x_test, y_test, inference_only=False, task_family=task_family)
            eval_runtime_s = time.perf_counter() - eval_start
            train_metrics.setdefault("train_runtime_s", train_runtime_s)

        workload_runtime_s = time.perf_counter() - workload_start
        primary_score = metric_score_value(task_family, primary_name, primary)
        model_size = _count_model_params(model)
        benchmark_samples = _sample_count(x_test, y_test)
        train_samples = _sample_count(x_train, y_train)
        explainability, perturbation_artifact = _service_perturbation_metrics(
            model,
            x_test,
            y_test,
            config={**self.config, "service_id": self.service_id},
            meta=meta,
            task_family=task_family,
        )
        tracked_runtime_s = time.perf_counter() - workload_start
        usage = tracker.stop(tracked_runtime_s)
        runtime_total_s = time.perf_counter() - service_start
        perf_metrics = _performance_alias_metrics(
            eval_qos=eval_qos,
            train_runtime_s=train_runtime_s,
            eval_runtime_s=eval_runtime_s,
            runtime_total_s=runtime_total_s,
            benchmark_samples=benchmark_samples,
        )
        resource_metrics = _resource_metrics(
            runtime_total_s=runtime_total_s,
            workload_runtime_s=workload_runtime_s,
            train_metrics=train_metrics,
            eval_qos=eval_qos,
            model_size=model_size,
            usage=usage,
        )
        reliability = _reliability_metrics(status="completed", eval_qos=eval_qos)
        hardware_snapshot = capture_hardware_snapshot() if _config_bool(self.config.get("measure_system_metrics"), True) else None
        accounting_meta = finalize_accounting(meta, batch_size=_safe_int(self.config.get("batch_size")))
        accounting = accounting_meta.get("accounting", {}) if isinstance(accounting_meta, Mapping) else {}
        train_distribution = _distribution_summary(x_train, y_train, meta=meta, task_family=task_family, hf_task=self.config.get("hf_task") or meta.get("hf_task"))
        benchmark_distribution = _distribution_summary(x_test, y_test, meta=meta, task_family=task_family, hf_task=self.config.get("hf_task") or meta.get("hf_task"))

        metrics: dict[str, Any] = {
            "loss": _metric(loss, "quality", direction="lower_better"),
            primary_name: _metric(primary, "quality", direction=_quality_direction(primary_name)),
            "metric_score": _metric(primary_score, "quality", direction="higher_better"),
            "primary_metric_name": _metric(primary_name, "metadata"),
            "auxiliary_metric_name": _metric(secondary_name, "metadata"),
            "model_build_s": _metric(model_build_s, "runtime", "s", "lower_better"),
            "workload_runtime_s": _metric(workload_runtime_s, "runtime", "s", "lower_better"),
            "evaluation_runtime_s": _metric(eval_runtime_s, "runtime", "s", "lower_better"),
            "eval_runtime_s": _metric(eval_runtime_s, "runtime", "s", "lower_better"),
            "training_runtime_s": _metric(train_runtime_s, "runtime", "s", "lower_better"),
            "train_runtime_s": _metric(train_runtime_s, "runtime", "s", "lower_better"),
            "runtime_total_s": _metric(runtime_total_s, "runtime", "s", "lower_better"),
            "runtime_s": _metric(runtime_total_s, "runtime", "s", "lower_better"),
            "service_runtime_s": _metric(runtime_total_s, "runtime", "s", "lower_better"),
            "compute_time_s": _metric(train_runtime_s + eval_runtime_s, "runtime", "s", "lower_better"),
            "model_params_count": _metric(model_size, "resource", "parameters", "lower_better"),
            "params_count": _metric(model_size, "resource", "parameters", "lower_better"),
            "model_size": _metric(model_size, "resource", "parameters", "lower_better"),
            "train_set_size": _metric(train_samples, "metadata", "samples", "neutral"),
            "benchmark_set_size": _metric(benchmark_samples, "metadata", "samples", "neutral"),
            "dataset_size": _metric(train_samples, "metadata", "samples", "neutral"),
            "task_family": _metric(task_family, "metadata"),
            "label_format": _metric(infer_label_format(meta, task_type=task_family) or canonical_label_format(task_family), "metadata"),
            "num_labels": _metric(infer_num_labels(meta, fallback=meta.get("num_classes")), "metadata"),
            "split_strategy": _metric(split_info["resolved"].get("strategy"), "metadata"),
            "data_distribution": _metric(split_info["resolved"].get("strategy"), "metadata"),
            "dataset_distribution_json": _metric(split_info["distribution_map"], "metadata"),
            "split_provenance_json": _metric(split_info["distribution_map"], "metadata"),
            "train_distribution_json": _metric(train_distribution, "metadata"),
            "benchmark_distribution_json": _metric(benchmark_distribution, "metadata"),
            "batch_size": _metric(_safe_int(self.config.get("batch_size")), "metadata", "samples", "neutral"),
            "learning_rate": _metric(_to_float(self.config.get("learning_rate")), "metadata", None, "neutral"),
            "epochs": _metric(_safe_int(self.config.get("training_epochs", self.config.get("epochs"))), "metadata", None, "neutral"),
            "device": _metric(resolved_device, "metadata"),
            "gpu_requested_cpu_fallback_flag": _metric(bool(gpu_fallback_warning), "reliability", direction="lower_better"),
        }
        if gpu_fallback_warning:
            metrics["gpu_fallback_warning"] = _metric(gpu_fallback_warning, "reliability")
        metrics.update(_hf_metadata_metrics(hf_metadata))
        metrics.update(perf_metrics)
        if secondary_name:
            metrics[secondary_name] = _metric(secondary, "quality", direction=_quality_direction(secondary_name))
        metrics.update(_specify_metric_dict(train_metrics))
        metrics.update(_specify_metric_dict(eval_qos))
        metrics.update(resource_metrics)
        metrics.update(explainability)
        metrics.update(reliability)
        if accounting:
            metrics["dataset_accounting"] = _metric(accounting, "metadata")

        functional_attributes = {
            "task_family": task_family,
            "label_format": infer_label_format(meta, task_type=task_family) or canonical_label_format(task_family),
            "primary_metric": primary_name,
            "secondary_metric": secondary_name,
            "metric_score": primary_score,
            "input_schema": self.config.get("input_schema") or meta.get("input_schema"),
            "output_schema": self.config.get("output_schema") or meta.get("label_format"),
        }
        metadata = {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "hardware_snapshot": hardware_snapshot,
            "loader_meta": _jsonable(meta),
            "hf_metadata": hf_metadata,
            "split_resolution": split_info["resolved"],
            "split_provenance": split_info["distribution_map"],
            "resolved_device": resolved_device,
            "gpu_fallback_warning": gpu_fallback_warning,
            "service_source": self.config.get("service_source"),
            "fit_decision": self.config.get("fit_decision"),
            "fit_reason": self.config.get("fit_reason"),
        }
        record = self._service_record(
            status="completed",
            started_at=started_at,
            functional_attributes=functional_attributes,
            metadata=metadata,
            task_family=task_family,
            registry_metadata={**_registry_metadata(self.config), **hf_metadata},
        )
        artifacts = []
        if perturbation_artifact:
            artifacts.append(perturbation_artifact)
        split_provenance_rows.append(
            {
                "split_name": "benchmark",
                "samples_count": benchmark_samples,
                "data_distribution": benchmark_distribution,
                "split_config": {
                    "source": self.config.get("benchmark_split") or self.config.get("test_split"),
                    "role": "benchmark",
                },
            }
        )
        return record, metrics, artifacts, split_provenance_rows

    def _resolve_service_split(self, x_train, y_train, meta: Mapping[str, Any]) -> dict[str, Any]:
        strategy = (
            self.config.get("split_strategy")
            or self.config.get("distribution_type")
            or self.config.get("data_distribution")
            or "iid"
        )
        strategy = _canonical_split_strategy(strategy)
        rng = np.random.default_rng(_safe_int(self.config.get("seed")) or 42)
        original_train_samples = _sample_count(x_train, y_train)
        sample_cap = self.config.get("sample_size")
        if sample_cap is None:
            sample_cap = self.config.get("max_samples")
        x_train, y_train, sample_info = _sample_service_rows(
            x_train,
            y_train,
            sample_size=sample_cap,
            sample_frac=self.config.get("sample_frac"),
            rng=rng,
        )
        hf_task = self.config.get("hf_task") or meta.get("hf_task")
        task_family = canonical_task_family(self.config.get("task_type") or meta.get("task_type"), hf_task)
        train_distribution = _distribution_summary(x_train, y_train, meta=meta, task_family=task_family, hf_task=hf_task)
        resolved = {
            "strategy": strategy,
            "distribution_type": self.config.get("distribution_type") or strategy,
            "distribution_param": self.config.get("distribution_param"),
            "sample_frac": self.config.get("sample_frac"),
            "requested_sample_size_total": sample_info.get("requested_sample_size_total"),
            "original_train_samples": original_train_samples,
            "effective_train_samples": _sample_count(x_train, y_train),
            "provenance_only": True,
        }
        distribution_map = {"train": train_distribution}
        return {
            "x_train": x_train,
            "y_train": y_train,
            "resolved": resolved,
            "distribution_map": distribution_map,
            "provenance_rows": [
                {
                    "split_name": "train",
                    "samples_count": _sample_count(x_train, y_train),
                    "data_distribution": train_distribution,
                    "split_config": resolved,
                }
            ],
        }

    def _print_run_summary(
        self,
        *,
        task_family: str,
        split_info: Mapping[str, Any],
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        resolved_device: str,
    ) -> None:
        summary = self._build_run_summary(
            task_family=task_family,
            split_info=split_info,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            resolved_device=resolved_device,
        )
        print("========== SERVICE RUN SUMMARY ==========")
        for key, value in summary.items():
            print(f"{key}: {_format_summary_value(value)}")
        print("=========================================")

    def _build_run_summary(
        self,
        *,
        task_family: str,
        split_info: Mapping[str, Any],
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        resolved_device: str,
    ) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "case_name": self.config.get("case_name"),
            "task_family": task_family,
            "task_type": self.config.get("task_type"),
            "dataset_name": self.config.get("dataset_name") or _nested_get(self.config, "dataset_args", "dataset_name"),
            "dataset_config": self.config.get("dataset_config") or _nested_get(self.config, "dataset_args", "dataset_config"),
            "model_id": self.config.get("hf_model_id") or self.config.get("model_id") or getattr(model, "model_id", None),
            "training_regime": self.config.get("training_regime"),
            "dataset_variant": self.config.get("dataset_variant"),
            "split_variant": self.config.get("split_variant"),
            "knob_variant": self.config.get("knob_variant"),
            "split_strategy": _nested_get(split_info, "resolved", "strategy"),
            "split provenance": split_info.get("distribution_map"),
            "train samples": _sample_count(x_train, y_train),
            "benchmark samples": _sample_count(x_test, y_test),
            "effective model input samples": _sample_count(x_train, y_train),
            "batch size": self.config.get("batch_size"),
            "learning rate": self.config.get("learning_rate"),
            "epochs": self.config.get("training_epochs", self.config.get("epochs")),
            "device": resolved_device,
            "save_weights": _config_bool(self.config.get("save_weights"), False),
            "explainability_enabled": _config_bool(
                self.config.get("enable_perturbation_metrics", self.config.get("explainability_enabled")),
                True,
            ),
        }

    def _build_model(self, *, meta: Mapping[str, Any], task_family: str):
        input_shape = meta.get("input_shape")
        if input_shape is not None:
            input_shape = tuple(input_shape)
        return _create_model(
            input_shape=input_shape,
            num_classes=meta.get("num_classes"),
            hidden_layers=self.config.get("hidden_layers", [64]),
            learning_rate=float(self.config.get("learning_rate", 0.001) or 0.001),
            activation=self.config.get("activation", "relu"),
            weight_decay=float(self.config.get("weight_decay", 0.0) or 0.0),
            optimizer=self.config.get("optimizer", "adam"),
            task_type=task_family if task_family in {"classification", "regression", "clustering"} else self.config.get("task_type", task_family),
            model_type=self.config.get("model_type"),
            meta=dict(meta),
            hf_model_id=self.config.get("hf_model_id"),
            hf_task=self.config.get("hf_task"),
            max_length=self.config.get("max_length"),
            device=_device_arg_for_model(self.config.get("device")),
            batch_size=int(self.config.get("batch_size", 16) or 16),
            task_tag=self.config.get("task_tag"),
            clustering_k=self.config.get("clustering_k"),
            clustering_init=self.config.get("clustering_init", "k-means++"),
            clustering_n_init=self.config.get("clustering_n_init", 10),
            clustering_max_iter=self.config.get("clustering_max_iter", 300),
            clustering_tol=self.config.get("clustering_tol", 1e-4),
            seed=self.config.get("seed", 42),
        )

    def _train_model(self, model, x_train, y_train, *, task_family: str) -> dict[str, Any]:
        epochs = int(self.config.get("training_epochs", self.config.get("epochs", 1)) or 1)
        batch_size = int(self.config.get("batch_size", 32) or 32)
        learning_rate = float(self.config.get("learning_rate", 0.001) or 0.001)
        if task_family == "clustering" and hasattr(model, "fit"):
            model.fit(x_train)
            return {"training_epochs": epochs}
        if hasattr(model, "fit") and model.__class__.__name__.lower().startswith("transformers"):
            qos = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                lr=learning_rate,
                max_train_time_s=self.config.get("max_train_time_s", 60),
            )
            return dict(qos or {}, training_epochs=epochs)
        train_local_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size, lr=learning_rate)
        return {"training_epochs": epochs}

    def _evaluate_model(self, model, x_test, y_test, *, inference_only: bool, task_family: str) -> tuple[float, float, float, dict[str, Any]]:
        if hasattr(model, "evaluate") and model.__class__.__name__.lower().startswith("transformers"):
            loss, primary, secondary, qos = model.evaluate(
                x_test,
                y_test,
                inference_only=inference_only,
                max_eval_time_s=self.config.get("max_eval_time_s"),
                progress_log_interval=self.config.get("eval_progress_log_interval", 10),
            )
            return loss, primary, secondary, dict(qos or {})
        if task_family == "clustering" and hasattr(model, "evaluate"):
            loss, primary, secondary = model.evaluate(x_test, y_test)
            return loss, primary, secondary, {}
        loss, primary, secondary = evaluate_model(model, x_test, y_test, task_type=("regression" if task_family == "regression" else "classification"))
        return loss, primary, secondary, {}

    def _service_record(
        self,
        *,
        status: str,
        started_at: str,
        functional_attributes: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        task_family: str | None = None,
        registry_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        training_regime = self.config.get("training_regime")
        return {
            "service_id": self.service_id,
            "status": status,
            "case_name": self.config.get("case_name"),
            "task_family": task_family or self.config.get("task_type"),
            "task_type": self.config.get("task_type"),
            "modality": self.config.get("modality"),
            "input_schema": self.config.get("input_schema"),
            "output_schema": self.config.get("output_schema"),
            "dataset": self.config.get("dataset"),
            "dataset_name": self.config.get("dataset_name"),
            "dataset_config": self.config.get("dataset_config"),
            "train_split": self.config.get("train_split"),
            "benchmark_split": self.config.get("benchmark_split") or self.config.get("test_split"),
            "model_type": self.config.get("model_type"),
            "model_id": self.config.get("hf_model_id") or self.config.get("model_id"),
            "hf_task": self.config.get("hf_task"),
            "training_regime": training_regime,
            "dataset_variant": self.config.get("dataset_variant"),
            "split_variant": self.config.get("split_variant"),
            "knob_variant": self.config.get("knob_variant"),
            "service_config_json": self.config.get("service_config") or _service_config(self.config),
            "registry_metadata_json": registry_metadata if registry_metadata is not None else _registry_metadata(self.config),
            "functional_attributes_json": functional_attributes or {},
            "metadata_json": metadata or {"started_at": started_at},
        }


def execute_service(config: Mapping[str, Any]) -> ServiceExecutionResult:
    return ServiceRunner(config).run()


def _normalize_requested_device(value: Any) -> Any:
    if value is None:
        return "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "nan", "auto", "gpu", "auto_gpu", "cuda_auto"}:
            return "auto"
        return value.strip()
    return value


def _device_arg_for_model(value: Any) -> Any:
    return None if _normalize_requested_device(value) == "auto" else value


def _gpu_fallback_warning(requested_device: Any, resolved_device: Any) -> str | None:
    requested = str(_normalize_requested_device(requested_device)).strip().lower()
    resolved = str(resolved_device or "").strip().lower()
    if requested in {"auto", "gpu", "auto_gpu", "cuda"} and resolved == "cpu":
        return "gpu_requested_but_cpu_resolved"
    return None


def _resolve_execution_device(model) -> str:
    candidates = [model, getattr(model, "core", None), getattr(model, "model", None)]
    for obj in candidates:
        if obj is None:
            continue
        device = getattr(obj, "device", None)
        if device is not None:
            return str(device)
    for obj in candidates:
        parameters = getattr(obj, "parameters", None) if obj is not None else None
        if callable(parameters):
            try:
                return str(next(parameters()).device)
            except Exception:
                pass
    return "unknown"


def _canonical_split_strategy(value: Any) -> str:
    strategy = str(value or "iid").strip().lower().replace("-", "_")
    aliases = {
        "niid": "dirichlet",
        "non_iid": "dirichlet",
        "non_iid_dirichlet": "dirichlet",
        "shards": "shard",
        "quantity": "quantity_skew",
        "quantity_skewed": "quantity_skew",
    }
    return aliases.get(strategy, strategy)


def _parse_jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return value
    return value


def _sample_service_rows(x, y, *, sample_size: Any, sample_frac: Any, rng) -> tuple[Any, Any, dict[str, Any]]:
    total = _sample_count(x, y)
    requested = _safe_int(sample_size)
    if requested is None and sample_frac is not None:
        try:
            requested = int(round(total * float(sample_frac)))
        except Exception:
            requested = None
    if requested is None:
        return x, y, {"requested_sample_size_total": None}
    requested = max(0, min(total, int(requested)))
    if requested == total:
        return x, y, {"requested_sample_size_total": requested}
    idx = rng.choice(total, size=requested, replace=False) if requested > 0 else np.asarray([], dtype=int)
    return _take_rows(x, idx), _take_rows(y, idx), {"requested_sample_size_total": requested}


def _take_rows(value, idx):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _take_rows(child, idx) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        return value[idx]
    if isinstance(value, tuple):
        idx_list = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
        return tuple(value[i] for i in idx_list)
    if isinstance(value, list):
        idx_list = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
        return [value[i] for i in idx_list]
    try:
        return np.asarray(value, dtype=object)[idx]
    except Exception:
        return value


def _distribution_summary(x, y, *, meta: Mapping[str, Any], task_family: str | None, hf_task: str | None) -> Any:
    hf_task_norm = str(hf_task or "").strip().lower()
    try:
        if hf_task_norm in {"fill_mask", "masked_lm"}:
            return get_mlm_masked_token_stats(y, ignore_index=int(meta.get("ignore_index", -100)))
        if task_family == "retrieval" or hf_task_norm == "text_image_retrieval":
            return get_retrieval_pair_stats(x)
        if task_family == "vqa" or hf_task_norm == "visual_question_answering":
            return get_vqa_answer_stats(y, ignore_index=int(meta.get("ignore_index", -100)))
        if task_family == "generation" or hf_task_norm in {"token_classification", "causal_lm_generation", "seq2seq_generation"}:
            return get_token_label_stats(
                y,
                ignore_index=int(meta.get("ignore_index", -100)),
                pad_token_id=meta.get("pad_token_id"),
            )
        return get_data_distribution(
            y,
            num_classes=meta.get("num_classes"),
            bins=meta.get("distribution_bins") or 10,
            value_range=meta.get("distribution_range"),
            label_pad_value=int(meta.get("ignore_index", -100)),
        )
    except Exception as exc:
        return {"error": type(exc).__name__, "message": str(exc)}


def _format_summary_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return "" if value is None else value


def _fetch_hf_metadata(config: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    hf_model_id = config.get("hf_model_id") or meta.get("hf_model_id")
    hf_dataset_id = config.get("dataset_name") or meta.get("dataset_name") or config.get("dataset")
    result: dict[str, Any] = {
        "hf_model_id": hf_model_id,
        "hf_dataset_id": hf_dataset_id,
    }
    if not hf_model_id and not hf_dataset_id:
        return result
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        result["hf_metadata_error"] = f"huggingface_hub import failed: {exc}"
        return result

    api = HfApi()
    if hf_model_id:
        try:
            model_info = api.model_info(str(hf_model_id))
            model_payload = _hf_info_payload(model_info)
            result.update(_normalise_hf_info(model_payload, model_info, prefix="hf_model"))
            result["downloads"] = result.get("hf_model_downloads")
            result["likes"] = result.get("hf_model_likes")
            result["pipeline_tag"] = result.get("hf_model_pipeline_tag")
            result["library_name"] = result.get("hf_model_library_name")
            result["license"] = result.get("hf_model_license")
            result["tags"] = result.get("hf_model_tags")
            result["last_modified"] = result.get("hf_model_last_modified")
            size = _extract_hf_model_size(model_payload, model_info)
            result["model_size"] = size
            result["params_count"] = size
        except Exception as exc:
            result["hf_model_metadata_error"] = str(exc)
    if hf_dataset_id:
        try:
            dataset_info = api.dataset_info(str(hf_dataset_id))
            dataset_payload = _hf_info_payload(dataset_info)
            result.update(_normalise_hf_info(dataset_payload, dataset_info, prefix="hf_dataset"))
            if result.get("downloads") is None:
                result["downloads"] = result.get("hf_dataset_downloads")
            if result.get("likes") is None:
                result["likes"] = result.get("hf_dataset_likes")
        except Exception as exc:
            result["hf_dataset_metadata_error"] = str(exc)
    result["hf_service_meta_json"] = json.dumps(result, ensure_ascii=False, default=str)
    return result


def _hf_info_payload(info: Any) -> dict[str, Any]:
    if info is None:
        return {}
    to_dict = getattr(info, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    data = getattr(info, "__dict__", None)
    return dict(data) if isinstance(data, dict) else {}


def _normalise_hf_info(payload: Mapping[str, Any], info: Any, *, prefix: str) -> dict[str, Any]:
    card_data = payload.get("cardData") or payload.get("card_data") or getattr(info, "cardData", None) or {}
    if not isinstance(card_data, Mapping):
        card_data = {}
    last_modified = payload.get("last_modified") if payload.get("last_modified") is not None else getattr(info, "last_modified", None)
    tags = payload.get("tags") if payload.get("tags") is not None else getattr(info, "tags", None)
    return {
        f"{prefix}_downloads": payload.get("downloads") if payload.get("downloads") is not None else getattr(info, "downloads", None),
        f"{prefix}_likes": payload.get("likes") if payload.get("likes") is not None else getattr(info, "likes", None),
        f"{prefix}_pipeline_tag": payload.get("pipeline_tag") or getattr(info, "pipeline_tag", None),
        f"{prefix}_library_name": payload.get("library_name") or getattr(info, "library_name", None),
        f"{prefix}_license": payload.get("license") or card_data.get("license") or getattr(info, "license", None),
        f"{prefix}_tags": tags,
        f"{prefix}_last_modified": _to_iso8601(last_modified),
    }


def _extract_hf_model_size(payload: Mapping[str, Any], info: Any) -> int | None:
    candidates = [
        payload.get("safetensors"),
        getattr(info, "safetensors", None),
        payload.get("cardData"),
        payload.get("card_data"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        for key in ("total", "parameters", "params", "params_count", "model_size"):
            value = candidate.get(key)
            parsed = _safe_int(value)
            if parsed is not None and parsed > 0:
                return parsed
    return None


def _to_iso8601(value: Any) -> Any:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    return value


def _hf_metadata_metrics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    fields = {
        "downloads": ("metadata", None, "neutral"),
        "likes": ("metadata", None, "neutral"),
        "model_size": ("resource", "parameters", "lower_better"),
        "params_count": ("resource", "parameters", "lower_better"),
        "pipeline_tag": ("metadata", None, "neutral"),
        "library_name": ("metadata", None, "neutral"),
        "license": ("metadata", None, "neutral"),
        "tags": ("metadata", None, "neutral"),
        "last_modified": ("metadata", None, "neutral"),
        "hf_model_id": ("metadata", None, "neutral"),
        "hf_dataset_id": ("metadata", None, "neutral"),
    }
    for key, (domain, unit, direction) in fields.items():
        if metadata.get(key) is not None:
            metrics[key] = _metric(metadata.get(key), domain, unit, direction)
    for key, value in metadata.items():
        if key.startswith("hf_model_") or key.startswith("hf_dataset_"):
            metrics[key] = _metric(value, metric_domain(key))
    return metrics


def _performance_alias_metrics(
    *,
    eval_qos: Mapping[str, Any],
    train_runtime_s: float,
    eval_runtime_s: float,
    runtime_total_s: float,
    benchmark_samples: int,
) -> dict[str, Any]:
    latency = _first_number(
        eval_qos,
        "inference_latency_s",
        "latency_s_mean",
        "inference_latency_s_mean",
        "eval_latency_s_mean",
    )
    if latency is None:
        ms = _first_number(eval_qos, "inference_latency_ms_mean", "eval_latency_ms_mean")
        latency = None if ms is None else ms / 1000.0
    if latency is None and benchmark_samples > 0 and eval_runtime_s > 0:
        latency = float(eval_runtime_s) / float(benchmark_samples)

    tail_latency = _first_number(eval_qos, "inference_latency_s_p95", "tail_latency", "latency_s_p95")
    if tail_latency is None:
        ms = _first_number(eval_qos, "inference_latency_ms_p95", "eval_latency_ms_p95")
        tail_latency = None if ms is None else ms / 1000.0
    if tail_latency is None:
        tail_latency = latency

    throughput = _first_number(eval_qos, "throughput", "throughput_samples_s", "throughput_eps", "eval_throughput_eps", "examples_per_second")
    if throughput is None and benchmark_samples > 0 and eval_runtime_s > 0:
        throughput = float(benchmark_samples) / float(eval_runtime_s)

    compute_time_s = float(train_runtime_s or 0.0) + float(eval_runtime_s or 0.0)
    return {
        "latency": _metric(latency, "latency", "s", "lower_better"),
        "tail_latency": _metric(tail_latency, "latency", "s", "lower_better"),
        "inference_latency_s": _metric(latency, "latency", "s", "lower_better"),
        "inference_latency_s_p95": _metric(tail_latency, "latency", "s", "lower_better"),
        "throughput": _metric(throughput, "runtime", "samples/s", "higher_better"),
        "throughput_samples_s": _metric(throughput, "runtime", "samples/s", "higher_better"),
        "compute_time_s": _metric(compute_time_s, "runtime", "s", "lower_better"),
        "runtime_s": _metric(runtime_total_s, "runtime", "s", "lower_better"),
    }


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping:
            value = _to_float(mapping.get(key))
            if not math.isnan(value):
                return value
    return None


def _service_perturbation_metrics(model, x_eval, y_eval, *, config: Mapping[str, Any], meta: Mapping[str, Any], task_family: str):
    enabled = _config_bool(config.get("enable_perturbation_metrics", config.get("explainability_enabled")), True)
    if not enabled:
        return {
            "perturbation_enabled_flag": _metric(False, "explainability"),
            "explainability_supported_flag": _metric(False, "explainability"),
            "explainability_score": _metric(0.0, "explainability", "score", "higher_better"),
        }, None
    log_stage = _config_bool(config.get("perturbation_stage_logging"), True)
    start = time.perf_counter()
    if log_stage:
        print(
            f"[Perturbation] service stage starts | service_id={config.get('service_id') or 'unknown'} "
            f"| task={task_family or 'unknown'} | samples={_sample_count(x_eval, y_eval)}",
            flush=True,
        )
    try:
        values = run_perturbation_stage(
            model,
            x_eval,
            y_eval,
            task_family=task_family,
            hf_task=config.get("hf_task") or meta.get("hf_task"),
            config=dict(config),
            meta=dict(meta),
            service_id=config.get("service_id"),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        values = {
            "perturbation_enabled_flag": True,
            "perturbation_supported_flag": False,
            "explainability_supported_flag": False,
            "explainability_score": 0.0,
            "perturbation_error": f"{type(exc).__name__}: {exc}",
            "perturbation_duration_s": elapsed,
        }
    if isinstance(values, dict):
        values.setdefault("perturbation_duration_s", time.perf_counter() - start)
    if log_stage:
        print(
            f"[Perturbation] service stage ends | service_id={config.get('service_id') or 'unknown'} "
            f"| supported={(values or {}).get('perturbation_supported_flag')} "
            f"| samples={(values or {}).get('perturbation_sample_count', 0)} "
            f"| duration_s={(values or {}).get('perturbation_duration_s', time.perf_counter() - start):.2f}",
            flush=True,
        )
    artifact = None
    samples = values.pop("perturbation_samples", None) if isinstance(values, dict) else None
    if samples:
        artifact = {
            "artifact_type": "perturbation_samples",
            "artifact_uri": f"service://{config.get('service_id')}/perturbation_samples",
            "metadata": {"perturbation_samples": samples},
        }
    return {key: _metric(value, metric_domain(key)) for key, value in (values or {}).items()}, artifact


def _load_dataset(name: str, **kwargs):
    global load_dataset
    if load_dataset is None:
        from ..data.master_loader import load_dataset as imported

        load_dataset = imported
    return load_dataset(name, **kwargs)


def _create_model(**kwargs):
    global create_model
    if create_model is None:
        from ..models.builders import create_model as imported

        create_model = imported
    return create_model(**kwargs)


def _service_config(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "batch_size",
        "learning_rate",
        "training_epochs",
        "optimizer",
        "weight_decay",
        "momentum",
        "max_samples",
        "sample_size",
        "max_length",
        "device",
        "mixed_precision",
        "num_workers",
        "timeout_s",
    )
    return {key: config.get(key) for key in keys if config.get(key) is not None}


def _registry_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "service_source",
        "model_role",
        "fit_decision",
        "fit_reason",
        "realism_score",
        "domain_alignment",
        "dataset_hint",
        "hf_pipeline_tag",
        "hf_downloads",
        "hf_likes",
        "hf_model_id",
        "hf_dataset_id",
        "downloads",
        "likes",
        "model_size",
        "params_count",
        "pipeline_tag",
        "library_name",
        "license",
        "tags",
        "last_modified",
        "hf_model_downloads",
        "hf_model_likes",
        "hf_model_pipeline_tag",
        "hf_model_library_name",
        "hf_model_license",
        "hf_model_tags",
        "hf_model_last_modified",
        "hf_dataset_downloads",
        "hf_dataset_likes",
        "hf_dataset_license",
        "hf_dataset_tags",
        "hf_dataset_last_modified",
        "hf_author",
        "hf_url",
        "hf_service_meta_json",
    )
    return {key: config.get(key) for key in keys if config.get(key) is not None}


def _sample_count(x, y=None) -> int:
    source = y if y is not None else x
    if source is None:
        return 0
    if isinstance(source, Mapping):
        for value in source.values():
            try:
                return int(len(value))
            except Exception:
                continue
        return 0
    try:
        return int(len(source))
    except Exception:
        return 0


def _count_model_params(model) -> int:
    for candidate in (model, getattr(model, "core", None), getattr(model, "model", None)):
        if candidate is None:
            continue
        count_params = getattr(candidate, "count_params", None)
        if callable(count_params):
            try:
                count = int(count_params())
                if count >= 0:
                    return count
            except Exception:
                pass
        parameters = getattr(candidate, "parameters", None)
        if callable(parameters):
            try:
                return int(sum(p.numel() for p in parameters()))
            except Exception:
                pass
    return 0


def _resource_metrics(*, runtime_total_s, workload_runtime_s, train_metrics, eval_qos, model_size, usage) -> dict[str, Any]:
    memory_mb = usage.memory_used_mb or usage.peak_host_ram_mb or 0.0
    gpu_mb = usage.gpu_memory_used_mb or usage.peak_vram_mb or 0.0
    comm_bytes = float(eval_qos.get("comm_bytes", 0.0) or 0.0)
    raw_resource_cost = (
        float(runtime_total_s or 0.0)
        + float(workload_runtime_s or 0.0)
        + float(memory_mb or 0.0) / 1024.0
        + float(gpu_mb or 0.0) / 1024.0
        + float(comm_bytes or 0.0) / 1073741824.0
        + (float(model_size or 0.0) / 1_000_000.0 if model_size else 0.0)
    )
    metrics = {
        "cpu_time_s": _metric(usage.cpu_time_s, "resource", "s", "lower_better"),
        "cpu_utilization": _metric(usage.cpu_utilization, "resource", "percent", "lower_better"),
        "memory_used_mb": _metric(usage.memory_used_mb, "resource", "MB", "lower_better"),
        "memory_utilization": _metric(usage.memory_utilization, "resource", "percent", "lower_better"),
        "gpu_utilization": _metric(usage.gpu_utilization, "resource", "percent", "lower_better"),
        "gpu_memory_used_mb": _metric(usage.gpu_memory_used_mb, "resource", "MB", "lower_better"),
        "peak_vram_mb": _metric(usage.peak_vram_mb, "resource", "MB", "lower_better"),
        "avg_vram_mb": _metric(usage.avg_vram_mb, "resource", "MB", "lower_better"),
        "peak_host_ram_mb": _metric(usage.peak_host_ram_mb, "resource", "MB", "lower_better"),
        "avg_host_ram_mb": _metric(usage.avg_host_ram_mb, "resource", "MB", "lower_better"),
        "raw_resource_cost": _metric(raw_resource_cost, "cost", None, "lower_better"),
        "resource_cost_score": _metric(1.0 / (1.0 + raw_resource_cost), "cost", "score", "higher_better"),
    }
    return metrics


def _explainability_metrics(model, config: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    if not _config_bool(config.get("explainability_enabled"), True):
        return {"explainability_supported_flag": _metric(False, "explainability"), "explainability_score": _metric(0.0, "explainability")}
    declared = config.get("explainability_method") or _nested_get(meta, "explainability", "preferred_methods")
    supported = _nested_get(meta, "explainability", "supported")
    if supported is None:
        supported = _has_importance_signal(model)
    score = _importance_proxy_score(model) if supported else 0.0
    return {
        "explainability_supported_flag": _metric(bool(supported), "explainability", direction="higher_better"),
        "explainability_method": _metric(declared[0] if isinstance(declared, list) and declared else declared or "metadata_or_importance_proxy", "explainability"),
        "explainability_score": _metric(score, "explainability", "score", "higher_better"),
    }


def _has_importance_signal(model) -> bool:
    return any(hasattr(model, attr) for attr in ("feature_importances_", "coef_", "cluster_centers_", "get_weights"))


def _importance_proxy_score(model) -> float:
    arrays = []
    for attr in ("feature_importances_", "coef_", "cluster_centers_"):
        if hasattr(model, attr):
            try:
                arrays.append(np.asarray(getattr(model, attr), dtype="float64").reshape(-1))
            except Exception:
                pass
    get_weights = getattr(model, "get_weights", None)
    if callable(get_weights):
        try:
            arrays.extend(np.asarray(w, dtype="float64").reshape(-1) for w in get_weights())
        except Exception:
            pass
    arrays = [arr for arr in arrays if arr.size]
    if not arrays:
        return 0.0
    values = np.abs(np.concatenate(arrays))
    values = values[np.isfinite(values)]
    if values.size == 0 or float(values.sum()) <= 0.0:
        return 0.0
    probs = values / float(values.sum())
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(values.size)) if values.size > 1 else 1.0
    return float(np.clip(1.0 - (entropy / max(max_entropy, 1e-12)), 0.0, 1.0))


def _reliability_metrics(*, status: str, eval_qos: Mapping[str, Any]) -> dict[str, Any]:
    failed = status != "completed"
    reliability_score = 0.0 if failed else 1.0
    if eval_qos.get("label_space_mismatch"):
        reliability_score = min(reliability_score, 0.75)
    if eval_qos.get("truncation_rate") is not None:
        try:
            reliability_score = min(reliability_score, max(0.0, 1.0 - float(eval_qos["truncation_rate"])))
        except Exception:
            pass
    return {
        "service_success_flag": _metric(not failed, "reliability", direction="higher_better"),
        "reliability_score": _metric(reliability_score, "reliability", "score", "higher_better"),
    }


def _specify_metric_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if value is None:
            continue
        normalized = str(key).strip().lower()
        if normalized.endswith("_ms_mean"):
            out[normalized.replace("_ms_mean", "_s_mean")] = _metric(_to_float(value) / 1000.0, "latency", "s", "lower_better")
        elif normalized.endswith("_ms_p95"):
            out[normalized.replace("_ms_p95", "_s_p95")] = _metric(_to_float(value) / 1000.0, "latency", "s", "lower_better")
        else:
            out[normalized] = _metric(value, metric_domain(normalized))
    return out


def _metric(value: Any, domain: str, unit: str | None = None, direction: str = "neutral") -> dict[str, Any]:
    return {"value": value, "domain": domain, "unit": unit, "direction": direction}


def _quality_direction(metric_name: str) -> str:
    return "lower_better" if str(metric_name).lower() in {"loss", "rmse", "mae", "perplexity"} else "higher_better"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except Exception:
        return str(value)
