# hf_strategy.py
import time
import math
import numpy as np

from .base import TaskStrategy, ClientOutcome, _nanmean, weights_size, metric_score_value
from ..system_metrics import ResourceTracker
from ...models.train_eval import aggregate_state_dict


class HFStrategy(TaskStrategy):
    """
    Single HF strategy that covers:
      - inference-only sequence classification (hf / transformers)
      - fine-tune sequence classification (hf_finetune / transformers_finetune)
      - fine-tune token classification (hf_task=token_classification)
    Behaviour is driven by config + dataset_args.
    """

    def task_type(self):
        return "classification"
        
    def __init__(self, meta, knobs, config, x_test, y_test, metric_key, save_weights):
        super().__init__(meta, knobs, config, x_test, y_test, metric_key, save_weights)

        mt = (self.config.get("model_type") or "").lower()
        self.inference_only = mt in ("hf", "hf_text", "transformers")

        ds_args = self.config.get("dataset_args", {}) or {}
        self.hf_task = (ds_args.get("hf_task") or self.config.get("hf_task") or "sequence_classification").lower()

    # -------------------------
    # Logging + scoring policies
    # -------------------------
    def _weighting_policy(self):
        task = str(self.hf_task or "").lower()
        token_weighted_tasks = {
            "token_classification",
            "token-cls",
            "ner",
            "fill_mask",
            "masked_lm",
            "mlm",
            "causal_lm_generation",
            "causal-lm",
            "language-modeling",
            "language_modeling",
            "seq2seq_generation",
        }
        sequence_weighted_tasks = {
            "sequence_classification",
            "text_classification",
            "sentence_similarity",
            "image_classification",
        }
        if task in token_weighted_tasks:
            return "supervised_token_count"
        if task in sequence_weighted_tasks:
            return "sequence_count"

        accounting = (self.meta or {}).get("accounting") if isinstance(self.meta, dict) else {}
        if accounting.get("supervised_token_count"):
            return "supervised_token_count"
        return "sequence_count"

    def _resolve_client_weighting(self, samples_count, extras=None):
        extras = extras if isinstance(extras, dict) else {}

        def _intish(*keys):
            for key in keys:
                value = extras.get(key)
                if value is None:
                    continue
                try:
                    return int(value)
                except Exception:
                    continue
            return None

        sequence_count = _intish(
            "train_sequence_count",
            "eval_sequence_count",
            "sequence_count",
            "eval_samples",
            "train_samples",
        )
        if sequence_count is None:
            sequence_count = int(samples_count)

        supervised_token_count = _intish(
            "train_supervised_token_count",
            "eval_supervised_token_count",
            "supervised_token_count",
            "tokens_total",
            "train_loss_denominator_count",
        )

        weight_unit = self._weighting_policy()
        if weight_unit == "supervised_token_count":
            weight_value = supervised_token_count
        else:
            weight_unit = "sequence_count"
            weight_value = sequence_count

        if weight_value is None or weight_value <= 0:
            weight_unit = "sequence_count"
            weight_value = sequence_count if sequence_count and sequence_count > 0 else int(samples_count)

        return {
            "sequence_count": int(sequence_count) if sequence_count is not None else None,
            "supervised_token_count": int(supervised_token_count) if supervised_token_count is not None else None,
            "aggregation_weight_unit": weight_unit,
            "aggregation_weight_value": float(weight_value),
        }

    def _metric_score(self, primary_metric_value):
        if primary_metric_value != primary_metric_value:
            return np.nan

        if self.hf_task in ("causal_lm_generation", "causal-lm", "language-modeling", "language_modeling"):
            return metric_score_value("regression", float(primary_metric_value))

        # token classification primary is typically F1 already in [0,1]
        if self.hf_task in ("token_classification", "token-cls", "ner"):
            return float(primary_metric_value)

        # sequence classification primary is typically accuracy
        return metric_score_value("classification", float(primary_metric_value))

    def loggable_run_params(self):
        ds_args = self.config.get("dataset_args", {}) or {}

        hf_model_id = ds_args.get("hf_model_id") or self.config.get("hf_model_id")
        max_length  = ds_args.get("max_length") or self.config.get("max_length")
        device      = ds_args.get("device") or self.config.get("device")

        adapter = {
            "inference_only": self.inference_only,
            "fine_tune": (not self.inference_only),
            "hf_task": self.hf_task,
            "hf_model_id": hf_model_id,
            "max_length": max_length,
            "device": device,
            "batch_size": self.knobs.get("batch_size"),
            "local_epochs": self.knobs.get("local_epochs"),
            "lr": self.knobs.get("learning_rate"),
            "max_new_tokens": ds_args.get("max_new_tokens") or self.config.get("max_new_tokens"),
            "num_beams": ds_args.get("num_beams") or self.config.get("num_beams"),
            "do_sample": ds_args.get("do_sample") if ds_args.get("do_sample") is not None else self.config.get("do_sample"),
            "temperature": ds_args.get("temperature") or self.config.get("temperature"),
            "top_k": ds_args.get("top_k") or self.config.get("top_k"),
            "top_p": ds_args.get("top_p") or self.config.get("top_p"),
            "length_penalty": ds_args.get("length_penalty") or self.config.get("length_penalty"),
            "max_train_time_s": self.knobs.get("max_train_time_s", self.config.get("max_train_time_s", 60)),
            "padding_mode": ("dynamic" if ds_args.get("dynamic_padding") else "max_length"),
            "aggregation_weight_unit": self._weighting_policy(),
        }

        dataset = {
            "dataset_name": ds_args.get("dataset_name"),
            "dataset_config": ds_args.get("dataset_config"),
            "train_split": ds_args.get("train_split"),
            "test_split": ds_args.get("test_split"),
            "text_column": ds_args.get("text_column"),
            "tokens_column": ds_args.get("tokens_column"),
            "label_column": ds_args.get("label_column"),
            "max_samples": ds_args.get("max_samples"),
            "dynamic_padding": ds_args.get("dynamic_padding"),
            "padding_mode": ("dynamic" if ds_args.get("dynamic_padding") else "max_length"),
        }

        adapter = {k: v for k, v in adapter.items() if v is not None}
        dataset = {k: v for k, v in dataset.items() if v is not None}
        return {"adapter": adapter, "dataset": dataset}

    # -------------------------
    # Federation-safe metric stats
    # -------------------------

    def _collect_metric_stats(self, outcomes):
        stats = {}
        for o in outcomes:
            extras = getattr(o, "extras", {}) or {}
            if not isinstance(extras, dict):
                continue
            for k, v in extras.items():
                if not str(k).startswith("metric_stat_"):
                    continue
                key = str(k)[12:]
                try:
                    stats[key] = float(stats.get(key, 0.0)) + float(v)
                except Exception:
                    continue
        return stats

    def _metrics_from_stats(self, stats):
        if not stats:
            return None
        task = str(self.hf_task or "").lower()

        if task in {"image_classification"}:
            total = float(stats.get("total", 0.0))
            if total <= 0:
                return None
            top1 = float(stats.get("top1_correct", 0.0)) / total
            top5 = float(stats.get("top5_correct", 0.0)) / total
            return top1, top5

        if task in {"image_detection", "object_detection"}:
            gt = float(stats.get("gt", 0.0))
            if gt <= 0:
                return None
            vals = []
            for thr in (0.5, 0.75, 0.95):
                tp = float(stats.get(f"tp_{thr}", 0.0))
                fp = float(stats.get(f"fp_{thr}", 0.0))
                vals.append(tp / max(gt + fp, 1e-9))
            return float(np.mean(vals)), float(vals[0])

        if task in {"image_segmentation", "semantic_segmentation"}:
            inter = float(stats.get("intersection", 0.0))
            union = float(stats.get("union", 0.0))
            pred_total = float(stats.get("pred_total", 0.0))
            target_total = float(stats.get("target_total", 0.0))
            if union <= 0 or (pred_total + target_total) <= 0:
                return None
            iou = inter / union
            dice = (2.0 * inter) / max(pred_total + target_total, 1e-9)
            return iou, dice

        return None

    # -------------------------
    # Model/adapter management
    # -------------------------
    def comm_down_bytes(self, global_model):
        # In inference mode you currently treat comms as 0 (no payload exchange)
        if self.inference_only:
            return 0

        try:
            w = global_model.get_weights()
            return weights_size(w)
        except Exception:
            return 0

    def _get_client_adapter(self, client_id):
        local_adapter = getattr(self, "_client_adapters", {}).get(client_id)
        if local_adapter is None:
            if not hasattr(self, "_client_adapters"):
                self._client_adapters = {}
            local_adapter = self.build_model()
            self._client_adapters[client_id] = local_adapter
        return local_adapter

    # -------------------------
    # Train/eval logic
    # -------------------------
    def _train_eval(self, adapter, x_train, y_train):
        """
        Expected adapter API:
          - fit(x, y, epochs, lr) -> dict qos
          - evaluate(x, y) -> (loss, primary, secondary, qos)

        For token classification: primary is assumed F1, secondary assumed accuracy (or similar).
        """
        train_qos = adapter.fit(
            x_train,
            y_train,
            epochs=self.knobs.get("local_epochs", 1),
            lr=self.knobs.get("learning_rate", 5e-5),
            max_train_time_s=self.knobs.get("max_train_time_s", self.config.get("max_train_time_s", 60)),
        )
        loss, primary, secondary, eval_qos = adapter.evaluate(self.x_test, self.y_test)
        return loss, primary, secondary, train_qos, eval_qos

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down):
        samples_count = len(y)

        start = time.time()
        tracker = ResourceTracker()
        tracker.start()

        try:
            if self.inference_only:
                adapter = global_model if global_model is not None else self.build_model()
                loss, primary, secondary, qos = adapter.evaluate(
                    x,
                    y,
                    inference_only=True,
                    max_eval_time_s=self.knobs.get("max_eval_time_s", self.config.get("max_eval_time_s")),
                    progress_log_interval=self.knobs.get(
                        "eval_progress_log_interval",
                        self.config.get("eval_progress_log_interval", 10),
                    ),
                )

                duration = time.time() - start
                usage = tracker.stop(duration)

                mscore = self._metric_score(primary)
                weighting = self._resolve_client_weighting(samples_count, qos)

                return ClientOutcome(
                    participated=True,
                    fail_reason="",
                    samples_count=samples_count,
                    duration=duration,
                    loss=loss,
                    metric_value=float(primary) if primary == primary else np.nan,
                    metric_score=float(mscore) if mscore == mscore else np.nan,
                    extra_metric=float(secondary) if secondary == secondary else np.nan,
                    rounds_so_far=rounds_so_far,
                    comm_down=0,
                    comm_up=0,
                    cpu_time_s=usage.cpu_time_s,
                    cpu_utilization=usage.cpu_utilization,
                    memory_used_mb=usage.memory_used_mb,
                    memory_utilization=usage.memory_utilization,
                    gpu_utilization=usage.gpu_utilization,
                    gpu_memory_utilization=usage.gpu_memory_utilization,
                    gpu_memory_used_mb=usage.gpu_memory_used_mb,
                    peak_vram_mb=usage.peak_vram_mb,
                    avg_vram_mb=usage.avg_vram_mb,
                    peak_host_ram_mb=usage.peak_host_ram_mb,
                    avg_host_ram_mb=usage.avg_host_ram_mb,
                    payload=None,
                    extras=qos if isinstance(qos, dict) else {},
                    **weighting,
                )

            # fine-tune mode
            local_adapter = self._get_client_adapter(client_id)

            if global_model is not None:
                local_adapter.set_weights(global_model.get_weights())

            loss, primary, secondary, train_qos, eval_qos = self._train_eval(local_adapter, x, y)

            duration = time.time() - start
            usage = tracker.stop(duration)

            payload = local_adapter.get_weights()
            mscore = self._metric_score(primary)

            extras = {}
            if isinstance(train_qos, dict):
                extras.update(train_qos)
            if isinstance(eval_qos, dict):
                extras.update(eval_qos)
            weighting = self._resolve_client_weighting(samples_count, extras)

            return ClientOutcome(
                participated=True,
                fail_reason="",
                samples_count=samples_count,
                duration=duration,
                loss=loss,
                metric_value=float(primary) if primary == primary else np.nan,
                metric_score=float(mscore) if mscore == mscore else np.nan,
                extra_metric=float(secondary) if secondary == secondary else np.nan,
                rounds_so_far=rounds_so_far,
                comm_down=comm_down,
                comm_up=weights_size(payload),
                cpu_time_s=usage.cpu_time_s,
                cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb,
                memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization,
                gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=payload,
                extras=extras,
                **weighting,
            )

        except Exception as e:
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)

            weighting = self._resolve_client_weighting(samples_count, {})

            return ClientOutcome(
                participated=False,
                fail_reason=repr(e),
                samples_count=samples_count,
                duration=duration,
                loss=np.nan,
                metric_value=np.nan,
                metric_score=np.nan,
                extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1,
                comm_down=(0 if self.inference_only else comm_down),
                comm_up=0,
                cpu_time_s=usage.cpu_time_s,
                cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb,
                memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization,
                gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=None,
                extras={},
                **weighting,
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]
        if not participated:
            return np.nan, np.nan, np.nan, np.nan

        if self.inference_only:
            weights = []
            for o in participated:
                value = getattr(o, "aggregation_weight_value", None)
                if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                    value = getattr(o, "sequence_count", None) or getattr(o, "samples_count", 1)
                weights.append(float(value))

            def _weighted(values, ws):
                pairs = [(float(v), float(w)) for v, w in zip(values, ws) if v == v and w > 0]
                if not pairs:
                    return np.nan
                vals, wts = zip(*pairs)
                return float(np.average(vals, weights=wts))

            loss = _weighted([o.loss for o in participated], weights)
            stats = self._collect_metric_stats(participated)
            derived = self._metrics_from_stats(stats)
            if derived is not None:
                primary, secondary = derived
            else:
                primary = _weighted([o.metric_value for o in participated], weights)
                secondary = _weighted([o.extra_metric for o in participated], weights)
            mscore = self._metric_score(primary)
            return loss, primary, mscore, secondary

        adapter = global_model if global_model is not None else self.build_model()

        payloads = [o.payload for o in participated if o.payload is not None]
        weights = []
        for o in participated:
            if o.payload is None:
                continue
            value = getattr(o, "aggregation_weight_value", None)
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                value = getattr(o, "sequence_count", None) or getattr(o, "samples_count", 1)
            weights.append(float(value))

        if payloads:
            agg = aggregate_state_dict(payloads, weights=weights)
            adapter.set_weights(agg)

        loss, primary, secondary, _qos = adapter.evaluate(x_test, y_test)
        mscore = self._metric_score(primary)
        return loss, primary, mscore, secondary
