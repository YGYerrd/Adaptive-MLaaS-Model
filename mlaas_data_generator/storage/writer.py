from __future__ import annotations
import sqlite3, os, json, numpy as np
from typing import Mapping, Any

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

        # Optional but helpful: faster inserts
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")

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

        # Exactly one typed column
        if isinstance(value, bool):
            row["value_bool"] = 1 if value else 0
        elif isinstance(value, int) and not isinstance(value, bool):
            row["value_int"] = value
        elif isinstance(value, float):
            row["value_num"] = value
        elif value is None:
            # run_params schema forbids all NULL value_*; so skip Nones.
            return
        elif isinstance(value, (dict, list)):
            row["value_json"] = json.dumps(value)
        else:
            row["value_text"] = str(value)

        self._ins("run_params", row)

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
        self.conn.execute(
            """
            INSERT OR IGNORE INTO metrics (name, domain, unit, direction, data_type, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, domain, unit, direction, data_type, description),
        )

    def seed_metrics(self) -> None:
        # Seed your core metric set once per DB. Expand as you add QoS.
        core = [
            ("accuracy", "quality", "proportion", "higher_better", "num", "Classification accuracy"),
            ("f1", "quality", "proportion", "higher_better", "num", "F1 score"),
            ("loss", "quality", None, "lower_better", "num", "Loss"),
            ("participated_flag", "reliability", "bool", "higher_better", "bool", "Client participated in round"),
            ("fail_reason", "reliability", None, "neutral", "text", "Failure reason / exception"),
            ("compute_time_s", "performance", "s", "lower_better", "num", "Client round compute duration"),
            ("comm_bytes_up", "resource", "bytes", "lower_better", "int", "Upload communication bytes"),
            ("comm_bytes_down", "resource", "bytes", "lower_better", "int", "Download communication bytes"),
            ("samples_count", "resource", "samples", "neutral", "int", "Samples used by client this round"),
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

    def _coerce_measurement_value(self, v):
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

        # Ensure we never violate the CHECK constraint
        if v is None:
            out["value_json"] = json.dumps(None)
            return out

        if isinstance(v, bool):
            out["value_bool"] = 1 if v else 0
            return out

        if isinstance(v, int):
            out["value_int"] = v
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