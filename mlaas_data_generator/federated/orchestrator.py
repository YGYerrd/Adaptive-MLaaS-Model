# orchestrator.py
from __future__ import annotations
import os, uuid, json
import time
from datetime import datetime, timezone
import numpy as np
from numbers import Number

from ..config import CONFIG
from ..data.master_loader import load_dataset
from ..data.splitters import split_data
from ..data.distributions import get_data_distribution, get_mlm_masked_token_stats
from ..data.sources.hf_meta import fetch_hf_model_meta
from ..storage.writer import make_writer
from .strategies.factory import make_task_strategy
from .strategies.base import canonical_task_family, canonical_label_format, canonical_metric_names, normalize_hf_task, metric_availability
from .system_metrics import capture_hardware_snapshot, summarize_round_usage
from ..models.label_schema import infer_label_format, infer_num_labels


class FederatedDataGenerator:
    """Generate MLaaS client records using a simple federated-learning loop."""
    def __init__(
        self,
        config: dict | None = None,
        dataset: str | None = None,
        task_type: str | None = None,
        model_type: str | None = None,
        dataset_args: dict | None = None,
    ):
        self.config = CONFIG.copy()
        if config:
            self.config.update(config)

        self.dataset = dataset or self.config.get("dataset", "fashion_mnist")
        self.model_type = model_type or self.config.get("model_type", "CNN")
        self.config["dataset"] = self.dataset
        self.config["model_type"] = self.model_type

        # dataset args
        self.dataset_args = {}
        config_dataset_args = self.config.get("dataset_args") or {}
        if isinstance(config_dataset_args, dict):
            self.dataset_args.update(config_dataset_args)
        if dataset_args:
            self.dataset_args.update(dataset_args)
        if self.dataset_args:
            self.config["dataset_args"] = dict(self.dataset_args)

        # load data
        train, test, meta = load_dataset(self.dataset, **self.dataset_args)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train, test
        self.meta = meta
        self.input_shape = tuple(meta["input_shape"])
        self.num_classes = meta.get("num_classes")

        # task type resolution
        requested_task = task_type or self.config.get("task_type")
        meta_task = meta.get("task_type", "classification")
        self.task_type = requested_task or meta_task
        if requested_task != meta_task:
            print(f"Warning: overriding dataset task type '{meta_task}' with requested '{self.task_type}'.")

        # metric keys
        if self.task_type == "clustering":
            self.metric_key = "silhouette"
            self.metric_label = "Silhouette"
        elif self.task_type == "classification":
            self.metric_key = "accuracy"
            self.metric_label = "Accuracy"
        else:
            self.metric_key = "rmse"
            self.metric_label = "RMSE"

        self.target_scaler = meta.get("target_scaler")
        self.save_weights = bool(self.config.get("save_weights", True))
        self.distribution_bins = int(self.config.get("distribution_bins", 10) or 10)

        # Regression: set value range for distribution summaries
        if self.task_type == "regression" or self.num_classes is None:
            if len(self.y_train) > 0:
                y_min = float(np.min(self.y_train))
                y_max = float(np.max(self.y_train))
                if y_min == y_max:
                    y_min -= 0.5
                    y_max += 0.5
                self.distribution_range = (y_min, y_max)
            else:
                self.distribution_range = (0.0, 1.0)
        else:
            self.distribution_range = None

        # knobs
        hidden_layers = self.config.get("hidden_layers", [self.config.get("reduced_neurons", 64)])
        if hidden_layers is None:
            hidden_layers = [self.config.get("reduced_neurons", 64)]
        self.hidden_layers = list(hidden_layers)

        self.knobs = {
            "num_clients": int(self.config["num_clients"]),
            "num_rounds": int(self.config["num_rounds"]),
            "local_epochs": int(self.config["local_epochs"]),
            "batch_size": self.config["batch_size"],
            "learning_rate": self.config["learning_rate"],
            "hidden_layers": self.hidden_layers,
            "activation": self.config.get("activation", "relu"),
            "dropout": float(self.config.get("dropout", 0.0) or 0.0),
            "weight_decay": float(self.config.get("weight_decay", 0.0) or 0.0),
            "optimizer": self.config.get("optimizer", "adam"),
            "distribution_type": self.config.get("distribution_type", "iid"),
            "distribution_param": self.config.get("distribution_param", None),
            "custom_distributions": self.config.get("custom_distributions", None),
            "sample_size": self.config.get("sample_size", None),
            "sample_frac": self.config.get("sample_frac", None),
            "distribution_bins": self.distribution_bins,
            "early_stopping_patience": self.config.get("early_stopping_patience"),
        }

        self.rng = np.random.default_rng(self.config.get("seed", 42))
        self.hf_task = normalize_hf_task(
            self.dataset_args.get("hf_task") or self.config.get("hf_task") or self.meta.get("hf_task")
        )
        self.task_family = canonical_task_family(self.task_type, self.hf_task)

        # strategy encapsulates build/train/eval details
        self.strategy = make_task_strategy(
            task_type=self.task_type,
            meta=self.meta,
            knobs=self.knobs,
            config=self.config,
            x_test=self.x_test,
            y_test=self.y_test,
            metric_key=self.metric_key,
            save_weights=self.save_weights,
        )

        # Disable multi-round training for non-federated models
        if self.task_type == "clustering" or (self.model_type or "").lower() == "randomforest":
            print(f"Non-federated model detected ({self.model_type}); forcing single-round training.")
            self.knobs["num_rounds"] = 1
    
    def _early_stopping_patience(self):
        patience_cfg = self.config.get("early_stopping_patience")
        if patience_cfg in (None, "", False):
            return None
        try:
            p = int(patience_cfg)
        except (TypeError, ValueError):
            return None
        if p <= 0:
            return None
        return p

    def _resolve_execution_device(self, model):
        """Best-effort device string for run summaries (CPU/CUDA/DirectML/etc)."""
        candidates = [
            model,
            getattr(model, "core", None),
            getattr(model, "model", None),
        ]

        for obj in candidates:
            if obj is None:
                continue
            device = getattr(obj, "device", None)
            if device is not None:
                return str(device)

        torch_model = getattr(model, "model", None)
        if torch_model is not None:
            try:
                return str(next(torch_model.parameters()).device)
            except Exception:
                pass

        return "unknown"
    

    def _canonical_run_metadata(self):
        hf_task = normalize_hf_task(getattr(self.strategy, "hf_task", self.hf_task))
        task_family = canonical_task_family(self.task_type, hf_task)
        label_format = infer_label_format(self.meta, task_type=self.task_type) or canonical_label_format(task_family)
        task_tag = str(self.config.get("task_tag") or self.config.get("dataset_args", {}).get("task_tag") or "").strip().lower() or None
        metric_primary_name, metric_secondary_name = canonical_metric_names(task_family, self.metric_key)
        has_labels = self.y_test is not None
        availability = metric_availability(task_family, task_tag=task_tag, has_labels=has_labels)
        if task_family == "generation":
            eval_metrics = availability.get("eval", tuple())
            if eval_metrics:
                metric_primary_name = eval_metrics[0]
                metric_secondary_name = eval_metrics[1] if len(eval_metrics) > 1 else None
        return {
            "task_family": task_family,
            "task_tag": task_tag,
            "label_format": label_format,
            "metric_primary_name": metric_primary_name,
            "metric_secondary_name": metric_secondary_name,
            "train_metric_names": list(availability.get("train", tuple())),
            "eval_metric_names": list(availability.get("eval", tuple())),
            "num_labels": infer_num_labels(self.meta, fallback=self.num_classes),
            "train_set_size": int(len(self.y_train)),
            "eval_set_size": int(len(self.y_test)),
        }

    def _extract_dynamic_metrics(self, outcome):
        extras = getattr(outcome, "extras", {}) if outcome is not None else {}

        def _pick(keys, default=None):
            for k in keys:
                if isinstance(extras, dict) and extras.get(k) is not None:
                    return extras.get(k)
            return default

        fail_reason = getattr(outcome, "fail_reason", "") or ""

        has_nan = any(
            isinstance(v, float) and np.isnan(v)
            for v in [
                getattr(outcome, "loss", np.nan),
                getattr(outcome, "metric_value", np.nan),
                getattr(outcome, "metric_score", np.nan),
                getattr(outcome, "extra_metric", np.nan),
            ]
        ) or ("nan" in fail_reason.lower())

        return {
            "effective_batch_size": int(_pick(["effective_batch_size", "batch_size", "train_batch_size"], self.knobs.get("batch_size") or 0) or 0),
            "tokens_in": int(_pick(["tokens_in", "input_tokens", "prompt_tokens", "train_tokens_in"], 0) or 0),
            "tokens_out": int(_pick(["tokens_out", "output_tokens", "completion_tokens", "train_tokens_out"], 0) or 0),
            "avg_seq_len": float(_pick(["avg_seq_len", "avg_sequence_length", "mean_seq_len"], 0.0) or 0.0),
            "truncation_rate": float(_pick(["truncation_rate", "truncated_fraction", "trunc_rate"], 0.0) or 0.0),
            "oom_count": int(_pick(["oom_count"], 0) or 0) + int("out of memory" in fail_reason.lower() or "cuda oom" in fail_reason.lower()),
            "nan_count": int(_pick(["nan_count"], 0) or 0) + int(has_nan),
            "fail_reason_category": self._categorize_fail_reason(fail_reason),
        }

    def _categorize_fail_reason(self, fail_reason: str):
        text = (fail_reason or "").strip().lower()
        if not text:
            return "none"
        if "dropout" in text:
            return "dropout"
        if "out of memory" in text or "cuda oom" in text or "oom" in text:
            return "oom"
        if "nan" in text:
            return "nan"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        return "runtime_error"
    
    def _safe_number(self, value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, Number) and not isinstance(value, bool):
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                return None
            return value
        return None

    def _safe_metric_value(self, value):
        if value is None:
            return None
        if isinstance(value, (bool, int, str, dict, list)):
            return value
        parsed_number = self._safe_number(value)
        if parsed_number is not None:
            return parsed_number
        if isinstance(value, np.bool_):
            return bool(value)
        return None

    def _normalize_outcome_extras(self, outcome):
        extras = getattr(outcome, "extras", {}) if outcome is not None else {}
        if not isinstance(extras, dict):
            return {}

        canonical = {}

        aliases = {
            "train_step_latency_ms_mean": ("seconds_per_step", 1.0 / 1000.0),
            "train_step_latency_ms_p95": ("seconds_per_step_p95", 1.0 / 1000.0),
            "eval_latency_ms_mean": ("inference_latency_s", 1.0 / 1000.0),
            "eval_latency_ms_p95": ("inference_latency_s_p95", 1.0 / 1000.0),
            "inference_latency_ms_mean": ("inference_latency_s", 1.0 / 1000.0),
            "inference_latency_ms_p95": ("inference_latency_s_p95", 1.0 / 1000.0),
            "throughput_eps": ("examples_per_second", 1.0),
            "train_throughput_eps": ("examples_per_second", 1.0),
            "eval_throughput_eps": ("examples_per_second", 1.0),
            "throughput_tps": ("tokens_per_second", 1.0),
            "tokens_per_second": ("tokens_per_second", 1.0),
            "train_tokens_per_second": ("tokens_per_second", 1.0),
            "eval_tokens_per_second": ("tokens_per_second", 1.0),
        }

        for key, value in extras.items():
            parsed = self._safe_metric_value(value)
            if parsed is None:
                continue

            if key in aliases:
                canonical_name, multiplier = aliases[key]
                numeric = self._safe_number(parsed)
                if numeric is not None:
                    canonical[canonical_name] = float(numeric) * multiplier
                continue

            canonical[key] = parsed

        train_time_s = self._safe_number(extras.get("train_time_s"))
        if train_time_s is not None:
            canonical["train_time_s"] = float(train_time_s)
            epochs = self._safe_number(extras.get("epochs"))
            if epochs is None:
                epochs = self._safe_number(extras.get("train_epochs"))
            if epochs is None:
                epochs = self._safe_number(self.knobs.get("local_epochs"))
            if epochs and epochs > 0:
                canonical["seconds_per_epoch"] = float(train_time_s) / float(epochs)

        tokens_total = self._safe_number(extras.get("tokens_total"))
        if tokens_total is None:
            tokens_total = self._safe_number(extras.get("train_tokens_total"))
        if tokens_total is None:
            tokens_total = self._safe_number(extras.get("eval_tokens_total"))
        if tokens_total is None:
            token_in = self._safe_number(extras.get("tokens_in"))
            token_out = self._safe_number(extras.get("tokens_out"))
            if token_in is not None and token_out is not None:
                tokens_total = token_in + token_out
        if tokens_total is not None:
            canonical["tokens_total"] = int(tokens_total)

        return canonical

    def _round_qos_rollups(self, records):
        qos_metrics = [
            "seconds_per_step",
            "seconds_per_epoch",
            "examples_per_second",
            "tokens_per_second",
            "inference_latency_s",
        ]
        rollups = {}
        for metric_name in qos_metrics:
            vals = [float(r[metric_name]) for r in records if metric_name in r and self._safe_number(r[metric_name]) is not None]
            if not vals:
                continue
            rollups[f"round_{metric_name}_mean"] = float(np.mean(vals))
            rollups[f"round_{metric_name}_p95"] = float(np.percentile(vals, 95))
        return rollups
    
    def run(self):
        run_start_epoch = time.time()
        run_start_ts = datetime.now(timezone.utc).isoformat()

        os.makedirs("weights", exist_ok=True)

        verbose_progress = bool(self.config.get("verbose_progress", True))
        phase_label = "inference" if bool(getattr(self.strategy, "inference_only", False)) else "training"

        early_stopping_patience = self._early_stopping_patience()

        if self.task_type == "regression" and self.knobs["distribution_type"] in {"dirichlet", "shard", "label_per_client"}:
            print("Warning: label-based partitioning not supported for regression; using 'iid'.")
            self.knobs["distribution_type"] = "iid"

        clients, split_info = split_data(
            self.x_train,
            self.y_train,
            self.knobs["num_clients"],
            strategy=self.knobs["distribution_type"],
            distribution_param=self.knobs["distribution_param"],
            custom_distributions=self.knobs["custom_distributions"],
            sample_size=self.knobs["sample_size"],
            sample_frac=self.knobs["sample_frac"],
            rng=self.rng,
        )
        
        global_model = self.strategy.build_model()
        execution_device = self._resolve_execution_device(global_model)

        print("\n========== RUN SUMMARY ==========")

        # Universal info (runner-ish)
        base = [
            ("dataset", self.dataset),
            ("task_type", self.task_type),
            ("model_type", self.model_type),
            ("num_clients", self.knobs["num_clients"]),
            ("num_rounds", self.knobs["num_rounds"]),
            ("client_dropout_rate", self.config.get("client_dropout_rate", 0.0)),
            ("seed", self.config.get("seed", 42)),
            ("save_weights", self.save_weights),
            ("input_shape", self.input_shape),
            ("num_classes", self.num_classes),
            ("execution_device", execution_device),
        ]

        # Splitter info (always relevant)
        splitter = [
            ("split.strategy", self.knobs.get("distribution_type")),
            ("split.param", self.knobs.get("distribution_param")),
            ("split.sample_size", self.knobs.get("sample_size")),
            ("split.sample_frac", self.knobs.get("sample_frac")),
            ("split.distribution_bins", self.knobs.get("distribution_bins")),
        ]

        def _print_kv(items, width=26):
            for k, v in items:
                if v is None:
                    continue
                print(f"{k:>{width}} : {v}")

        _print_kv(base)
        print("------------------------------------------------")
        _print_kv(splitter)

        # Strategy-specific (adapter/dataset/etc) — the important part
        lines = self.strategy.summary_lines()
        if lines:
            print("------------------------------------------------")
            for k, v in lines:
                if k.startswith("[") and v == "":
                    print(k)
                    continue
                if v is None:
                    continue
                print(f"{k:>26} : {v}")

        print("================================================\n")

        print(f"Execution mode: federated {phase_label}")
        if verbose_progress:
            print("Verbose progress logging is enabled.")
            print("Per-client lifecycle logs: start -> strategy call -> completion/failure.")
        print()

        print("Client data distributions before training:")
        client_distributions = {}
        is_fill_mask = self.task_family == "fill_mask"
        ignore_index = int(self.meta.get("label_pad_value", -100))
        for client_id, data in clients.items():
            if is_fill_mask:
                dist = get_mlm_masked_token_stats(
                    data["y"],
                    ignore_index=ignore_index,
                )
            else:
                dist = get_data_distribution(
                    data["y"],
                    self.num_classes,
                    bins=self.knobs.get("distribution_bins"),
                    value_range=self.distribution_range,
                    label_pad_value=ignore_index,
                )
            client_distributions[client_id] = dist
            print(f"{client_id}: {dist}")

        # build global model via strategy

        hardware_snapshot = capture_hardware_snapshot()

        try:
            params_count = int(global_model.count_params())
        except Exception:
            try:
                params_count = int(sum(p.numel() for p in global_model.model.parameters()))
            except Exception:
                params_count = 0

        run_id = str(uuid.uuid4())

        hf_model_meta = {}
        hf_model_id = (self.dataset_args.get("hf_model_id") or "").strip()
        is_hf_run = ("hf" in (self.dataset or "").lower()) and bool(hf_model_id)
        if is_hf_run:
            try:
                hf_model_meta = fetch_hf_model_meta(hf_model_id) or {}
            except Exception as exc:
                print(f"Warning: failed to fetch Hugging Face model metadata for '{hf_model_id}': {exc}")
                hf_model_meta = {}

            user_hf_task = self.dataset_args.get("hf_task") or self.config.get("hf_task")
            if not user_hf_task:
                pipeline_task = hf_model_meta.get("hf_pipeline_tag") or hf_model_meta.get("hf_task")
                normalized_pipeline_task = normalize_hf_task(pipeline_task)
                if normalized_pipeline_task and normalized_pipeline_task != "unknown":
                    self.hf_task = normalized_pipeline_task
                    self.task_family = canonical_task_family(self.task_type, self.hf_task)
                    if hasattr(self.strategy, "hf_task"):
                        self.strategy.hf_task = self.hf_task


        db_path = self.config.get("db_path", "federated2.db")
        writer = make_writer("sqlite", db_path=db_path)
        writer.start()
        try:
            # Seed metric dictionary (recommended)
            if hasattr(writer, "seed_metrics"):
                writer.seed_metrics()

            # --- runs dimension
            writer.write_run(
                {
                    "run_id": run_id,
                    "dataset": self.dataset,
                    "task_type": self.task_type,
                    "model_type": self.model_type,
                    "num_clients": self.knobs["num_clients"],
                    "num_rounds": self.knobs["num_rounds"],
                }
            )

            # --- run_params (normalised config)
            # scope suggestions: runner/dataset/adapter/aggregator/splitter
            if hasattr(writer, "write_run_param"):
                # runner level
                writer.write_run_param(run_id, "runner", "seed", self.config.get("seed", 42))
                writer.write_run_param(run_id, "runner", "client_dropout_rate", self.config.get("client_dropout_rate", 0.0))
                writer.write_run_param(run_id, "runner", "save_weights", self.save_weights)

                # benchmark identity for cross-run comparisons
                benchmark_identity = {
                    "dataset": self.dataset,
                    "model": self.model_type,
                    "clients": self.knobs.get("num_clients"),
                    "rounds": self.knobs.get("num_rounds"),
                    "batch": self.knobs.get("batch_size"),
                    "max_length": self.dataset_args.get("max_length", self.config.get("max_length", self.meta.get("max_length"))),
                    "device": execution_device,
                }
                for key, value in benchmark_identity.items():
                    writer.write_run_param(run_id, "benchmark", key, value)

                # splitter / distribution
                writer.write_run_param(run_id, "splitter", "distribution_type", self.knobs.get("distribution_type"))
                writer.write_run_param(run_id, "splitter", "distribution_param", self.knobs.get("distribution_param"))
                writer.write_run_param(run_id, "splitter", "distribution_bins", self.knobs.get("distribution_bins"))
                writer.write_run_param(run_id, "splitter", "sample_size", self.knobs.get("sample_size"))
                writer.write_run_param(run_id, "splitter", "sample_frac", self.knobs.get("sample_frac"))

                params_by_scope = self.strategy.loggable_run_params()
                for scope, kv in (params_by_scope or {}).items():
                    for k, v in (kv or {}).items():
                        writer.write_run_param(run_id, scope, k, v)

                # dataset args (store as JSON)
                writer.write_run_param(run_id, "dataset", "dataset_args", self.dataset_args)

                if is_hf_run:
                    if self.hf_task and self.hf_task != "unknown":
                        writer.write_run_param(run_id, "dataset", "hf_task", self.hf_task)
                    def _pick_hf_value(key):
                        for source in (hf_model_meta, self.meta, self.dataset_args, self.config):
                            if isinstance(source, dict) and key in source:
                                return source.get(key)
                        return None

                    def _is_present(value):
                        if value is None:
                            return False
                        if isinstance(value, str):
                            return bool(value.strip())
                        if isinstance(value, (dict, list, tuple, set)):
                            return len(value) > 0
                        return True

                    hf_metadata_keys = [
                        "hf_model_id",
                        "hf_pipeline_tag",
                        "hf_downloads",
                        "hf_likes",
                        "hf_last_modified",
                        "hf_author",
                        "hf_url",
                        "hf_service_meta_json",
                    ]
                    for key in hf_metadata_keys:
                        value = _pick_hf_value(key)
                        if _is_present(value):
                            writer.write_run_param(run_id, "dataset", key, value)


                # hardware snapshot / params count (optional)
                writer.write_run_param(run_id, "runner", "params_count", params_count)
                writer.write_run_param(run_id, "runner", "hardware_snapshot", hardware_snapshot)

            writer.write_measurements(
                run_id=run_id,
                round=None,
                client_id=None,
                values={
                    **self._canonical_run_metadata(),
                    "run_start_ts": run_start_ts,
                },
            )

            # --- clients dimension
            for client_id, data in clients.items():
                writer.write_client(
                    {
                        "run_id": run_id,
                        "client_id": client_id,
                        "data_distribution_json": json.dumps(client_distributions.get(client_id, {})),
                        "samples_count": int(len(data["y"])),
                    }
                )

            participated_counts = {cid: 0 for cid in clients.keys()}

            for round_num in range(self.knobs["num_rounds"]):
                round_idx = round_num + 1
                print(f"--- Round {round_idx} ---")

                client_payloads = []
                client_outcomes = []
                round_metrics = []
                round_qos_records = []
                skipped_clients = 0

                down_bytes = self.strategy.comm_down_bytes(global_model)
                if verbose_progress:
                    print(
                        f"Round {round_idx}: scheduled_clients={len(clients)}, "
                        f"phase={phase_label}, comm_down_bytes={int(down_bytes)}"
                    )

                # Round dimension row
                writer.write_round(
                    {
                        "run_id": run_id,
                        "round": round_idx,
                        "scheduled_clients": len(clients),
                        "attempted_clients": None,
                        "participating_clients": None,
                        "dropped_clients": None,
                    }
                )

                for client_id, data in clients.items():
                    dist = client_distributions.get(client_id)
                    n_samples = int(len(data["y"]))

                    # dropout
                    if self.rng.random() < self.config.get("client_dropout_rate", 0.0):
                        skipped_clients += 1
                        # record dropout as measurements
                        writer.write_measurements(
                            run_id=run_id,
                            round=round_idx,
                            client_id=client_id,
                            values={
                                "participated_flag": False,
                                "fail_reason": "client_dropout",
                                "samples_count": n_samples,
                                "comm_bytes_down": int(down_bytes),
                                "comm_bytes_up": 0,
                                "compute_time_s": 0.0,
                                "effective_batch_size": int(self.knobs.get("batch_size") or 0),
                                "tokens_in": 0,
                                "tokens_out": 0,
                                "avg_seq_len": 0.0,
                                "truncation_rate": 0.0,
                                "oom_count": 0,
                                "nan_count": 0,
                                "fail_reason_category": "dropout",
                            },
                        )
                        print(f"{client_id} dropped out")
                        continue

                    next_rounds_so_far = participated_counts[client_id] + 1
                    print(
                        f"{client_id} {phase_label}... "
                        f"(samples={n_samples}, round={round_idx}, participation_count={next_rounds_so_far})"
                    )
                    if verbose_progress:
                        print(f"{client_id}: invoking strategy.train_client")

                    outcome = self.strategy.train_client(
                        client_id=client_id,
                        x=data["x"],
                        y=data["y"],
                        global_model=global_model,
                        round_idx=round_idx,
                        rounds_so_far=next_rounds_so_far,
                        comm_down=down_bytes,
                    )

                    if verbose_progress:
                        status = "participated" if outcome.participated else f"skipped ({outcome.fail_reason or 'unknown reason'})"
                        print(
                            f"{client_id}: strategy completed, status={status}, "
                            f"duration_s={float(outcome.duration):.3f}, "
                            f"metric={self.metric_key}:{float(outcome.metric_value) if outcome.metric_value == outcome.metric_value else 'nan'}"
                        )

                    client_outcomes.append(outcome)

                    client_values = {
                        "participated_flag": bool(outcome.participated),
                        "fail_reason": outcome.fail_reason if not outcome.participated else None,
                        "samples_count": int(outcome.samples_count),
                        "compute_time_s": float(outcome.duration),
                        "comm_bytes_down": int(outcome.comm_down),
                        "comm_bytes_up": int(outcome.comm_up),
                        "loss": float(outcome.loss) if outcome.loss == outcome.loss else None,
                        self.metric_key: float(outcome.metric_value) if outcome.metric_value == outcome.metric_value else None,
                        "metric_score": float(outcome.metric_score) if outcome.metric_score == outcome.metric_score else None,
                        "extra_metric": float(outcome.extra_metric) if outcome.extra_metric == outcome.extra_metric else None,
                        "cpu_time_s": float(outcome.cpu_time_s) if outcome.cpu_time_s is not None else None,
                        "memory_used_mb": float(outcome.memory_used_mb) if outcome.memory_used_mb is not None else None,
                        "gpu_memory_used_mb": float(outcome.gpu_memory_used_mb) if outcome.gpu_memory_used_mb is not None else None,
                    }
                    client_values.update(self._extract_dynamic_metrics(outcome))
                    client_values.update(self._normalize_outcome_extras(outcome))

                    client_values = {
                        key: value
                        for key, value in client_values.items()
                        if self._safe_metric_value(value) is not None
                    }

                    if outcome.participated:
                        round_qos_records.append(client_values)

                    # write client-round measurements
                    writer.write_measurements(
                        run_id=run_id,
                        round=round_idx,
                        client_id=client_id,
                        values=client_values,
                    )

                    round_metrics.append(
                        {
                            "participated": bool(outcome.participated),
                            "duration": outcome.duration,
                            "cpu_utilization": outcome.cpu_utilization,
                            "memory_utilization": outcome.memory_utilization,
                            "memory_used_mb": outcome.memory_used_mb,
                            "gpu_utilization": outcome.gpu_utilization,
                            "gpu_memory_utilization": outcome.gpu_memory_utilization,
                            "gpu_memory_used_mb": outcome.gpu_memory_used_mb,
                            "cpu_time_s": outcome.cpu_time_s,
                        }
                    )

                    if outcome.participated:
                        participated_counts[client_id] = next_rounds_so_far

                    if outcome.payload is not None:
                        client_payloads.append(outcome.payload)

                # aggregate & evaluate globally
                if verbose_progress:
                    print(
                        f"Round {round_idx}: aggregating {len(client_payloads)} payload(s) from "
                        f"{sum(1 for o in client_outcomes if getattr(o, 'participated', False))} participating client(s)."
                    )
                loss, global_metric, global_score, global_extra = self.strategy.aggregate_and_eval(
                    global_model=global_model,
                    client_payloads=client_payloads,
                    client_outcomes=client_outcomes,
                    round_idx=round_idx,
                    x_train=self.x_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                )

                round_usage_summary = summarize_round_usage(
                    round_metrics,
                    scheduled_clients=len(clients),
                    skipped_clients=skipped_clients,
                )

                # update round dimension with aggregates
                writer.write_round(
                    {
                        "run_id": run_id,
                        "round": round_idx,
                        "scheduled_clients": len(clients),
                        "attempted_clients": int(len(clients) - skipped_clients),
                        "participating_clients": int(sum(1 for o in client_outcomes if getattr(o, "participated", False))),
                        "dropped_clients": int(skipped_clients),
                    }
                )

                # write round-level measurements (client_id NULL)
                writer.write_measurements(
                    run_id=run_id,
                    round=round_idx,
                    client_id=None,
                    values={
                        "global_loss": loss,
                        f"global_{self.metric_key}": global_metric,
                        "global_metric_score": global_score,
                        "global_aux_metric": global_extra,
                        "round_resource_summary": round_usage_summary,
                        **self._round_qos_rollups(round_qos_records),
                    },
                )

                if self.task_type == "regression" and self.target_scaler and self.target_scaler.get("type") == "standard":
                    rmse_std = float(global_metric)
                    rmse_orig = rmse_std * float(self.target_scaler["std"])
                    print(f"Global model {self.metric_label}: {rmse_std:.6f} (standardized) | {rmse_orig:.2f} (original units)")
                else:
                    print(f"Global model {self.metric_label}: {global_metric}")

                if self.task_type == "regression":
                    print(f"Global metric score: {global_score}")
                if global_extra is not None:
                    print(f"Global auxiliary metric: {global_extra}")
                if verbose_progress:
                    print(
                        f"Round {round_idx} complete: attempted={int(len(clients) - skipped_clients)}, "
                        f"participating={int(sum(1 for o in client_outcomes if getattr(o, 'participated', False)))}, "
                        f"dropped={int(skipped_clients)}"
                    )
                    print("----------------------------------------")

        finally:
            run_end_epoch = time.time()
            run_end_ts = datetime.now(timezone.utc).isoformat()
            writer.write_measurements(
                run_id=run_id,
                round=None,
                client_id=None,
                values={
                    "run_end_ts": run_end_ts,
                    "run_total_runtime_s": float(run_end_epoch - run_start_epoch),
                },
            )
            writer.finish()

        print("Federated Learning Process Complete!\n")
        return {
            "run_id": run_id,
            "db_path": db_path,
            "rounds": self.knobs["num_rounds"],
            "clients": self.knobs["num_clients"],
        }
