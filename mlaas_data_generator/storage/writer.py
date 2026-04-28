from __future__ import annotations
import sqlite3, os, json, numpy as np
from typing import Mapping, Any

ALLOWED_METRIC_DOMAINS = {"quality", "performance", "reliability", "cost", "resource"}
SQLITE_INT_MIN = -(2**63)
SQLITE_INT_MAX = 2**63 - 1

def make_writer(kind: str, **kwargs):
    if kind == "sqlite":
        return SQLiteWriter(**kwargs)
    raise ValueError(f"Unknown writer kind: {kind}")

class SQLiteWriter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._metric_cache = {}  # name(lower) -> metric_id

    def start(self) -> None:
        folder = os.path.dirname(self.db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        init_needed = not os.path.exists(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")

        if init_needed:
            from importlib import resources
            sql = resources.files(__package__).joinpath("schemaV2.sql").read_text(encoding="utf-8")
            self.conn.executescript(sql)
            self.conn.commit()
        else:
            self._ensure_runtime_schema()

        # Optional but helpful: faster inserts
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")

    def _ensure_runtime_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_failures (
              failure_id           INTEGER PRIMARY KEY AUTOINCREMENT,
              external_run_id      TEXT,
              row_index            INTEGER,
              case_name            TEXT,
              run_group_id         TEXT,
              failure_stage        TEXT NOT NULL,
              error_message        TEXT,
              resolved_config_json TEXT,
              traceback_text       TEXT,
              created_at           TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_run_failures_external_run_id ON run_failures(external_run_id);
            CREATE INDEX IF NOT EXISTS idx_run_failures_run_group_id ON run_failures(run_group_id);
            CREATE INDEX IF NOT EXISTS idx_run_failures_stage ON run_failures(failure_stage);
            """
        )
        self.conn.commit()

    def _ins(self, table, row) -> None:
        keys = list(row.keys())
        placeholders = ",".join(["?"] * len(keys))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(keys)}) VALUES ({placeholders})"
        self.conn.execute(sql, [row[k] for k in keys])

    # -------- Dimensions --------

    def write_run(self, row: Mapping[str, Any]) -> None:
        self._ins("runs", row)

    def write_round(self, row: Mapping[str, Any]) -> None:
        self._ins("rounds", row)

    def write_client(self, row: Mapping[str, Any]) -> None:
        self._ins("clients", row)

    def write_run_param(self, run_id: str, scope: str, key: str, value: Any) -> None:
        row = {"run_id": run_id, "scope": scope, "key": key}

        value_columns = self._coerce_value_columns(value, null_as_json=False)
        if value_columns is None:
            return
        row.update(value_columns)

        self._ins("run_params", row)

    def write_run_failure(
        self,
        *,
        external_run_id: str | None,
        row_index: int | None,
        case_name: str | None,
        run_group_id: str | None,
        failure_stage: str,
        error_message: str | None,
        resolved_config_json: str | None,
        traceback_text: str | None = None,
    ) -> None:
        self._ensure_runtime_schema()
        self._ins(
            "run_failures",
            {
                "external_run_id": external_run_id,
                "row_index": row_index,
                "case_name": case_name,
                "run_group_id": run_group_id,
                "failure_stage": failure_stage,
                "error_message": error_message,
                "resolved_config_json": resolved_config_json,
                "traceback_text": traceback_text,
            },
        )

    # -------- Metric registry --------

    def _get_metric_id(self, name: str) -> int:
        key = (name or "").strip().lower()
        if key in self._metric_cache:
            return self._metric_cache[key]

        cur = self.conn.execute("SELECT metric_id FROM metrics WHERE name = ?", (key,))
        row = cur.fetchone()
        if row:
            mid = int(row[0])
            self._metric_cache[key] = mid
            return mid

        # If not found, you either:
        #  - raise error (strict), or
        #  - auto-create with defaults (flexible).
        # I recommend strict + seed, but here’s flexible default:
        self._ensure_metric(
            name=key,
            domain="resource",
            unit=None,
            direction="neutral",
            data_type="num",
            description=None,
        )
        cur = self.conn.execute("SELECT metric_id FROM metrics WHERE name = ?", (key,))
        mid = int(cur.fetchone()[0])
        self._metric_cache[key] = mid
        return mid

    def _ensure_metric(self, name: str, domain: str, unit: str | None, direction: str, data_type: str, description: str | None) -> None:
        name = (name or "").strip().lower()
        domain = (domain or "resource").strip().lower()
        if domain not in ALLOWED_METRIC_DOMAINS:
            domain = "resource"
        self.conn.execute(
            """
            INSERT OR IGNORE INTO metrics (name, domain, unit, direction, data_type, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, domain, unit, direction, data_type, description),
        )
        self.conn.execute(
            """
            UPDATE metrics
            SET domain = ?, unit = ?, direction = ?, data_type = ?, description = ?
            WHERE name = ?
            """,
            (domain, unit, direction, data_type, description, name),
        )

    def seed_metrics(self) -> None:
        # Seed your core metric set once per DB. Expand as you add QoS.
        core = [
            ("accuracy", "quality", "proportion", "higher_better", "num", "Classification accuracy"),
            ("f1", "quality", "proportion", "higher_better", "num", "F1 score"),
            ("loss", "quality", None, "lower_better", "num", "Loss"),
            ("participated_flag", "reliability", "bool", "higher_better", "bool", "Client participated in round"),
            ("fail_reason", "reliability", None, "neutral", "text", "Failure reason / exception"),
            ("fail_reason_category", "reliability", None, "neutral", "text", "Normalised failure reason category"),
            ("compute_time_s", "performance", "s", "lower_better", "num", "Client round compute duration"),
            ("comm_bytes_up", "resource", "bytes", "lower_better", "int", "Upload communication bytes"),
            ("comm_bytes_down", "resource", "bytes", "lower_better", "int", "Download communication bytes"),
            ("samples_count", "resource", "samples", "neutral", "int", "Legacy example-count field kept for backward compatibility"),
            ("aggregation_weight_unit", "metadata", None, "neutral", "text", "Unit used to weight this client during federated aggregation"),
            ("aggregation_weight_value", "resource", None, "neutral", "num", "Numeric weight value used during federated aggregation"),
            ("cpu_time_s", "resource", "s", "lower_better", "num", "CPU time"),
            ("memory_used_mb", "resource", "MB", "lower_better", "num", "Memory used"),
            ("gpu_memory_used_mb", "resource", "MB", "lower_better", "num", "GPU memory used"),
            ("task_family", "metadata", None, "neutral", "text", "Canonical task family for this run"),
            ("label_format", "metadata", None, "neutral", "text", "Canonical label format for this run"),
            ("num_labels", "metadata", None, "neutral", "int", "Number of target labels/classes for this run"),
            ("metric_primary_name", "metadata", None, "neutral", "text", "Primary evaluation metric name"),
            ("metric_secondary_name", "metadata", None, "neutral", "text", "Secondary evaluation metric name"),
            ("eval_set_size", "metadata", "samples", "neutral", "int", "Evaluation set size"),
            ("train_set_size", "metadata", "samples", "neutral", "int", "Training set size"),
            ("raw_record_count", "metadata", "records", "neutral", "int", "Dataset rows loaded before split filtering and truncation"),
            ("post_filter_record_count", "metadata", "records", "neutral", "int", "Dataset rows remaining after filtering and truncation"),
            ("tokenized_record_count", "metadata", "records", "neutral", "int", "Dataset rows tokenized or tensorized"),
            ("sequence_count", "metadata", "sequences", "neutral", "int", "Model-ready sequences or tensors emitted"),
            ("supervised_token_count", "metadata", "tokens", "neutral", "int", "Supervised target tokens available for loss and metrics"),
            ("batch_count", "metadata", "batches", "neutral", "int", "Approximate training batches implied by dataset accounting"),
            ("metric_instance_count", "metadata", "metric_instances", "neutral", "int", "Examples contributing metric instances"),
            ("effective_batch_size", "resource", "samples", "neutral", "int", "Effective per-step batch size"),
            ("tokens_in", "resource", "tokens", "neutral", "int", "Input token count"),
            ("tokens_out", "resource", "tokens", "neutral", "int", "Output token count"),
            ("avg_seq_len", "resource", "tokens", "neutral", "num", "Average sequence length"),
            ("truncation_rate", "reliability", "proportion", "lower_better", "num", "Fraction of truncated inputs"),
            ("oom_count", "reliability", "count", "lower_better", "int", "Out-of-memory event count"),
            ("nan_count", "reliability", "count", "lower_better", "int", "NaN event count"),
            ("train_time_s", "performance", "s", "lower_better", "num", "Client-side training time"),
            ("seconds_per_step", "performance", "s/step", "lower_better", "num", "Average train step duration"),
            ("seconds_per_step_p95", "performance", "s/step", "lower_better", "num", "P95 train step duration"),
            ("seconds_per_epoch", "performance", "s/epoch", "lower_better", "num", "Average epoch duration"),
            ("examples_per_second", "performance", "examples/s", "higher_better", "num", "Processed examples per second"),
            ("tokens_per_second", "performance", "tokens/s", "higher_better", "num", "Processed tokens per second"),
            ("tokens_total", "resource", "tokens", "neutral", "int", "Total tokens consumed"),
            ("inference_latency_s", "performance", "s", "lower_better", "num", "Mean inference latency"),
            ("inference_latency_s_p95", "performance", "s", "lower_better", "num", "P95 inference latency"),
            ("round_seconds_per_step_mean", "performance", "s/step", "lower_better", "num", "Round mean seconds_per_step across participating clients"),
            ("round_seconds_per_step_p95", "performance", "s/step", "lower_better", "num", "Round p95 seconds_per_step across participating clients"),
            ("round_seconds_per_epoch_mean", "performance", "s/epoch", "lower_better", "num", "Round mean of client seconds_per_epoch"),
            ("round_seconds_per_epoch_p95", "performance", "s/epoch", "lower_better", "num", "Round p95 of client seconds_per_epoch"),
            ("round_examples_per_second_mean", "performance", "examples/s", "higher_better", "num", "Round mean of client examples_per_second"),
            ("round_examples_per_second_p95", "performance", "examples/s", "higher_better", "num", "Round p95 of client examples_per_second"),
            ("round_tokens_per_second_mean", "performance", "tokens/s", "higher_better", "num", "Round mean of client tokens_per_second"),
            ("round_tokens_per_second_p95", "performance", "tokens/s", "higher_better", "num", "Round p95 of client tokens_per_second"),
            ("round_inference_latency_s_mean", "performance", "s", "lower_better", "num", "Round mean of client inference_latency_s"),
            ("round_inference_latency_s_p95", "performance", "s", "lower_better", "num", "Round p95 of client inference_latency_s"),
            ("peak_vram_mb", "resource", "MB", "lower_better", "num", "Peak GPU memory sampled during client execution"),
            ("avg_vram_mb", "resource", "MB", "lower_better", "num", "Average GPU memory sampled during client execution"),
            ("peak_host_ram_mb", "resource", "MB", "lower_better", "num", "Peak host RAM sampled during client execution"),
            ("avg_host_ram_mb", "resource", "MB", "lower_better", "num", "Average host RAM sampled during client execution"),
            ("retry_count", "reliability", "count", "lower_better", "int", "Retry attempts consumed by operation"),
            ("dropout_events", "reliability", "count", "lower_better", "int", "Dropout events observed"),
            ("cold_start_time", "performance", "s", "lower_better", "num", "Model and tokenizer cold-start load time"),
            ("seconds_per_step_steady", "performance", "s/step", "lower_better", "num", "Steady-state train step duration excluding first step"),
            ("seconds_per_step_steady_p95", "performance", "s/step", "lower_better", "num", "P95 steady-state train step duration"),
            ("inference_latency_s_steady", "performance", "s", "lower_better", "num", "Steady-state inference latency excluding first batch"),
            ("inference_latency_s_steady_p95", "performance", "s", "lower_better", "num", "P95 steady-state inference latency"),
            ("round_seconds_per_step_steady_mean", "performance", "s/step", "lower_better", "num", "Round mean steady-state seconds_per_step across participating clients"),
            ("round_seconds_per_step_steady_p95", "performance", "s/step", "lower_better", "num", "Round p95 steady-state seconds_per_step across participating clients"),
            ("round_inference_latency_s_steady_mean", "performance", "s", "lower_better", "num", "Round mean steady-state inference latency"),
            ("round_inference_latency_s_steady_p95", "performance", "s", "lower_better", "num", "Round p95 steady-state inference latency"),
            ("federated_update_expected_flag", "federated_dynamics", "bool", "neutral", "bool", "Round is expected to perform model weight updates"),
            ("aggregation_payload_count", "federated_dynamics", "payloads", "neutral", "int", "Client weight payloads submitted for aggregation"),
            ("client_update_l2", "federated_dynamics", None, "higher_better", "num", "L2 norm between incoming global weights and client-updated payload"),
            ("client_update_max_abs", "federated_dynamics", None, "higher_better", "num", "Max absolute element change between incoming global weights and client payload"),
            ("client_update_changed_flag", "federated_dynamics", "bool", "higher_better", "bool", "Client payload differs from incoming global weights beyond tolerance"),
            ("client_update_layer_count", "federated_dynamics", "layers", "neutral", "int", "Comparable client payload layers"),
            ("update_signature_id", "resource", None, "neutral", "text", "Identifier for the compressed local model update signature"),
            ("signature_dim", "resource", "dimensions", "neutral", "int", "Stored update signature vector dimension"),
            ("signature_norm", "resource", None, "neutral", "num", "L2 norm of the raw local model update before signature normalisation"),
            ("update_signature_path", "resource", None, "neutral", "text", "Path to the compressed update signature vector"),
            ("update_signature_method", "resource", None, "neutral", "text", "Compression method used for the update signature"),
            ("update_signature_source_dim", "resource", "parameters", "neutral", "int", "Number of parameter deltas used to build the signature"),
            ("update_signature_layer_count", "resource", "layers", "neutral", "int", "Number of comparable layers used to build the signature"),
            ("update_signature_error", "reliability", None, "neutral", "text", "Best-effort update signature capture failure reason"),
            ("round_global_weight_delta_l2", "federated_dynamics", None, "higher_better", "num", "L2 norm between pre-aggregation and post-aggregation global weights"),
            ("round_global_weight_delta_max_abs", "federated_dynamics", None, "higher_better", "num", "Max absolute global weight change after aggregation"),
            ("round_global_weight_changed_flag", "federated_dynamics", "bool", "higher_better", "bool", "Global weights changed after server aggregation beyond tolerance"),
            ("round_global_weight_layer_count", "federated_dynamics", "layers", "neutral", "int", "Comparable global model layers"),
            ("round_start_global_delta_l2", "federated_dynamics", None, "lower_better", "num", "L2 norm between previous round final weights and current round start weights"),
            ("round_start_global_delta_max_abs", "federated_dynamics", None, "lower_better", "num", "Max absolute difference between previous final and current start weights"),
            ("global_weights_carried_forward_flag", "federated_dynamics", "bool", "higher_better", "bool", "Current round started from the previous round's final global weights"),
            ("round_repeated_global_metrics_flag", "federated_dynamics", "bool", "neutral", "bool", "Round global metrics exactly match the previous round within tolerance"),
            ("round_repetition_expected_flag", "federated_dynamics", "bool", "neutral", "bool", "Repeated global metrics are expected because no model update is expected"),
            ("round_redundant_flag", "federated_dynamics", "bool", "lower_better", "bool", "Repeated metrics and unchanged global weights in a round expected to update"),
            ("perturbation_enabled_flag", "quality", "bool", "neutral", "bool", "Post-evaluation perturbation stage was enabled"),
            ("perturbation_supported_flag", "quality", "bool", "higher_better", "bool", "Perturbation probe produced valid sample-level results"),
            ("perturbation_sample_count", "quality", "samples", "neutral", "int", "Evaluation samples used by perturbation probe"),
            ("perturbation_baseline_confidence_mean", "quality", "confidence", "neutral", "num", "Mean baseline prediction confidence across perturbed samples"),
            ("perturbation_duration_s", "performance", "s", "lower_better", "num", "Runtime of post-evaluation perturbation probe"),
            ("perturbation_error", "quality", None, "neutral", "text", "Best-effort perturbation probe failure reason"),
            ("perturbation_samples", "quality", None, "neutral", "json", "Structured per-sample perturbation records"),
            ("explainability_supported_flag", "quality", "bool", "higher_better", "bool", "Task-faithfulness explainability probe produced valid sample-level results"),
            ("explainability_task_family", "quality", None, "neutral", "text", "Task family used by task-specific explainability scoring"),
            ("explainability_method", "quality", None, "neutral", "text", "Attribution and scoring method used for explainability"),
            ("explainability_quality_metric", "quality", None, "neutral", "text", "Task-specific model-output quality quantity degraded by perturbation"),
            ("explainability_budget_fractions", "quality", None, "neutral", "json", "Fractions of meaningful input units removed during explainability scoring"),
            ("explainability_confidence_drop_mean", "quality", "confidence_delta", "higher_better", "num", "Mean confidence drop after masking influential input units"),
            ("explainability_confidence_drop_std", "quality", "confidence_delta", "lower_better", "num", "Standard deviation of confidence drop after masking influential input units"),
            ("explainability_confidence_drop_p50", "quality", "confidence_delta", "higher_better", "num", "Median confidence drop after masking influential input units"),
            ("explainability_confidence_drop_p10", "quality", "confidence_delta", "higher_better", "num", "P10 confidence drop after masking influential input units"),
            ("explainability_confidence_drop_p90", "quality", "confidence_delta", "higher_better", "num", "P90 confidence drop after masking influential input units"),
            ("explainability_prediction_change_rate", "quality", "proportion", "higher_better", "num", "Rate at which targeted perturbations changed predictions"),
            ("explainability_unit_fraction_mean", "quality", "proportion", "lower_better", "num", "Mean fraction of meaningful input units masked by targeted perturbations"),
            ("explainability_unit_fraction_p95", "quality", "proportion", "lower_better", "num", "P95 fraction of meaningful input units masked by targeted perturbations"),
            ("explainability_targeted_degradation_mean", "quality", "proportion", "higher_better", "num", "Mean task-quality degradation after removing explanation-ranked evidence"),
            ("explainability_random_degradation_mean", "quality", "proportion", "neutral", "num", "Mean task-quality degradation after removing random evidence of equal size"),
            ("explainability_self_faithfulness_score", "quality", "score", "higher_better", "num", "Self-faithfulness score from model-ranked evidence removal versus random evidence"),
            ("explainability_self_faithfulness_score_p10", "quality", "score", "higher_better", "num", "P10 self-faithfulness score"),
            ("explainability_semantic_supported_flag", "quality", "bool", "higher_better", "bool", "Expected semantic evidence was available for explainability scoring"),
            ("explainability_semantic_supported_rate", "quality", "proportion", "higher_better", "num", "Fraction of perturbation samples with expected semantic evidence"),
            ("explainability_semantic_target_source", "quality", None, "neutral", "text", "Source used to select expected semantic evidence"),
            ("explainability_semantic_degradation_mean", "quality", "proportion", "higher_better", "num", "Mean task-quality degradation after removing expected semantic evidence"),
            ("explainability_semantic_random_degradation_mean", "quality", "proportion", "neutral", "num", "Mean task-quality degradation after removing random evidence matched to semantic evidence size"),
            ("explainability_semantic_behavior_score", "quality", "score", "higher_better", "num", "Combined semantic sensitivity and selectivity score"),
            ("explainability_semantic_behavior_score_p10", "quality", "score", "higher_better", "num", "P10 combined semantic sensitivity and selectivity score"),
            ("explainability_semantic_sensitivity_score", "quality", "score", "higher_better", "num", "Absolute degradation score after masking expected semantic evidence"),
            ("explainability_semantic_sensitivity_score_p10", "quality", "score", "higher_better", "num", "P10 semantic sensitivity score"),
            ("explainability_semantic_selectivity_score", "quality", "score", "higher_better", "num", "Excess degradation from expected semantic evidence removal versus random evidence"),
            ("explainability_semantic_selectivity_score_p10", "quality", "score", "higher_better", "num", "P10 semantic selectivity score"),
            ("explainability_semantic_alignment_score", "quality", "score", "higher_better", "num", "Overlap between model-ranked important evidence and expected semantic evidence"),
            ("explainability_semantic_alignment_score_p10", "quality", "score", "higher_better", "num", "P10 semantic alignment score"),
            ("explainability_meaningful_drop_threshold", "quality", "proportion", "neutral", "num", "Targeted degradation threshold treated as a fully meaningful confidence or quality drop"),
            ("explainability_selectivity_floor", "quality", "score", "neutral", "num", "Minimum semantic behavior credit retained when expected evidence causes an absolute drop but not excess degradation over random"),
            ("explainability_score", "quality", "score", "higher_better", "num", "Hybrid explainability score combining semantic sensitivity, semantic alignment, and self-faithfulness"),
            ("explainability_score_p10", "quality", "score", "higher_better", "num", "P10 hybrid explainability score"),
            ("trust_confidence_delta_mean", "reliability", "confidence_delta", "lower_better", "num", "Mean confidence movement under benign perturbations"),
            ("trust_confidence_delta_std", "reliability", "confidence_delta", "lower_better", "num", "Standard deviation of confidence movement under benign perturbations"),
            ("trust_confidence_delta_p95", "reliability", "confidence_delta", "lower_better", "num", "P95 confidence movement under benign perturbations"),
            ("trust_confidence_delta_max", "reliability", "confidence_delta", "lower_better", "num", "Maximum confidence movement under benign perturbations"),
            ("trust_prediction_stability", "reliability", "proportion", "higher_better", "num", "Prediction stability under benign perturbations"),
            ("trust_prediction_stability_min", "reliability", "proportion", "higher_better", "num", "Minimum per-sample prediction stability under benign perturbations"),
            ("trust_confidence_stability", "reliability", "score", "higher_better", "num", "Confidence stability score under benign perturbations"),
            ("trust_score", "reliability", "score", "higher_better", "num", "Aggregate prediction and confidence stability score"),
            ("trust_score_p05", "reliability", "score", "higher_better", "num", "P05 per-sample aggregate prediction and confidence stability score"),
            ("trust_score_min", "reliability", "score", "higher_better", "num", "Minimum per-sample aggregate prediction and confidence stability score"),
        ]
        for name, domain, unit, direction, dtype, desc in core:
            self._ensure_metric(name, domain, unit, direction, dtype, desc)

        self.conn.commit()

    # -------- Facts --------

    def write_measurements(self, run_id: str, round: int | None, client_id: str | None, values: Mapping[str, Any]) -> None:
        for metric_name, value in (values or {}).items():
            if value is None:
                continue

            metric_id = self._get_metric_id(metric_name)

            row = {
                "run_id": run_id,
                "round": round,
                "client_id": client_id,
                "metric_id": metric_id,
            }

            row.update(self._coerce_measurement_value(value))

            for k in ["value_num","value_int","value_bool","value_text","value_json"]:
                if row.get(k) is None:
                    row.pop(k, None)
            value_keys = [k for k in row if k.startswith("value_")]
            if len(value_keys) != 1:
                print("CHECK VIOLATION ABOUT TO HAPPEN:", value_keys, row)

            #cur = self.conn.execute("SELECT data_type FROM metrics WHERE metric_id = ?", (metric_id,))
            #dtype = cur.fetchone()[0]
            #print("METRIC:", metric_name, "declared type:", dtype, "value keys:", value_keys)
            self._ins("measurements", row)

    def finish(self) -> None:
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None
            self._metric_cache = {}

    def abort(self) -> None:
        if self.conn is not None:
            self.conn.rollback()
            self.conn.close()
            self.conn = None
            self._metric_cache = {}

    def _coerce_measurement_value(self, v):
        out = self._coerce_value_columns(v, null_as_json=True)
        return {
            "value_num": out.get("value_num"),
            "value_int": out.get("value_int"),
            "value_bool": out.get("value_bool"),
            "value_text": out.get("value_text"),
            "value_json": out.get("value_json"),
        }

    def _coerce_value_columns(self, v, *, null_as_json: bool):
        out = {
            "value_num": None,
            "value_int": None,
            "value_bool": None,
            "value_text": None,
            "value_json": None,
        }

        # Normalise numpy scalars early
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)

        if v is None:
            if not null_as_json:
                return None
            out["value_json"] = json.dumps(None)
            return out

        if isinstance(v, bool):
            out["value_bool"] = 1 if v else 0
            return out

        if isinstance(v, int):
            if SQLITE_INT_MIN <= int(v) <= SQLITE_INT_MAX:
                out["value_int"] = v
            else:
                try:
                    as_float = float(v)
                except OverflowError:
                    out["value_text"] = str(v)
                else:
                    if np.isfinite(as_float):
                        out["value_num"] = as_float
                    else:
                        out["value_text"] = str(v)
            return out

        if isinstance(v, float):
            if np.isnan(v):
                out["value_json"] = json.dumps(None)
            else:
                out["value_num"] = v
            return out

        if isinstance(v, str):
            out["value_text"] = v
            return out

        # dict / list / anything else -> JSON
        try:
            out["value_json"] = json.dumps(v, ensure_ascii=False, default=str)
        except Exception:
            out["value_text"] = str(v)

        return out
