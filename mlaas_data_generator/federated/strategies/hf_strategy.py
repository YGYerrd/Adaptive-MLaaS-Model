# hf_strategy.py
import time
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
    def _metric_score(self, primary_metric_value):
        if primary_metric_value != primary_metric_value:
            return np.nan

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
        }

        adapter = {k: v for k, v in adapter.items() if v is not None}
        dataset = {k: v for k, v in dataset.items() if v is not None}
        return {"adapter": adapter, "dataset": dataset}

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
                loss, primary, secondary, qos = adapter.evaluate(x, y)

                duration = time.time() - start
                usage = tracker.stop(duration)

                mscore = self._metric_score(primary)

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
            )

        except Exception as e:
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)

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
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]
        if not participated:
            return np.nan, np.nan, np.nan, np.nan

        if self.inference_only:
            if self.hf_task == "fill_mask":
                weights = []
                for o in participated:
                    extras = getattr(o, "extras", {}) if hasattr(o, "extras") else {}
                    token_w = 0
                    if isinstance(extras, dict):
                        token_w = int(extras.get("tokens_total") or 0)
                    weights.append(float(max(1, token_w)))

                def _weighted(values, ws):
                    pairs = [(float(v), float(w)) for v, w in zip(values, ws) if v == v and w > 0]
                    if not pairs:
                        return np.nan
                    vals, wts = zip(*pairs)
                    return float(np.average(vals, weights=wts))

                loss = _weighted([o.loss for o in participated], weights)
                primary = _weighted([o.metric_value for o in participated], weights)
                secondary = _weighted([o.extra_metric for o in participated], weights)
            else:
                loss = _nanmean([o.loss for o in participated])
                primary = _nanmean([o.metric_value for o in participated])
                secondary = _nanmean([o.extra_metric for o in participated])
            mscore = self._metric_score(primary)
            return loss, primary, mscore, secondary

        adapter = global_model if global_model is not None else self.build_model()

        payloads = [o.payload for o in participated if o.payload is not None]
        weights = [float(o.samples_count) for o in participated if o.payload is not None]

        if payloads:
            agg = aggregate_state_dict(payloads, weights=weights)
            adapter.set_weights(agg)

        loss, primary, secondary, _qos = adapter.evaluate(x_test, y_test)
        mscore = self._metric_score(primary)
        return loss, primary, mscore, secondary