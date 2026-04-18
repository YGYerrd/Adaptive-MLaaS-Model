# classification_keras.py
from ...models.train_eval import train_local_model, evaluate_model, aggregate_weights
from ..system_metrics import ResourceTracker
from .base import TaskStrategy, ClientOutcome, metric_score_value, _nanmean, _is_keras_like, weights_size
import time, json
import numpy as np


class ClassificationStrategy(TaskStrategy):
    def task_type(self) -> str: return "classification"

    def loggable_run_params(self):
        params = super().loggable_run_params()
        mt = (self.config.get("model_type") or "").lower()
        params["aggregator"] = {
            "strategy": "client_metric_average_no_weight_updates" if mt == "randomforest" else "fedavg_uniform",
            "aggregation_weight_unit": "client_uniform",
        }
        return params

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome:
        local_model = self.build_model()
        samples_count = len(y)

        if _is_keras_like(local_model) and _is_keras_like(global_model):
            try:
                local_model.set_weights(global_model.get_weights())
            except Exception:
                pass
        
        start = time.time()
        tracker = ResourceTracker()
        tracker.start()

        try:
            weights = train_local_model(
                local_model, x, y,
                epochs=self.knobs["local_epochs"],
                batch_size=self.knobs["batch_size"],
            )
            duration = time.time() - start
            usage = tracker.stop(duration)
            loss, metric_value, extra_metric = evaluate_model(local_model, self.x_test, self.y_test, task_type="classification")
            mscore = metric_score_value("classification", metric_value)

            if self.save_weights and weights is not None:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=4)

            extras = self.perturbation_metrics(local_model, client_id=client_id, round_idx=round_idx)

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=metric_value, metric_score=mscore, extra_metric=extra_metric,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=weights_size(weights),
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=weights, extras=extras,  # accuracy/f1 added in records builder
            )
        except Exception as e:
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)
            return ClientOutcome(
                participated=False, fail_reason=repr(e), samples_count=samples_count, duration=duration,
                loss=np.nan, metric_value=np.nan, metric_score=np.nan, extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1, comm_down=comm_down, comm_up=0,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=None, extras={},
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test,):
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]
        if client_payloads:
            new_global_weights = aggregate_weights(client_payloads)
            # Keep parity with your existing set_weights(list_ordered)
            global_model.set_weights([new_global_weights[f"layer_{i}"] for i in range(len(new_global_weights))])
            if self.save_weights:
                with open(f"weights/global_round_{round_idx}.json", "w") as f:
                    json.dump({k: np.asarray(v).tolist() for k, v in new_global_weights.items()}, f, indent=4)
        else:
            print("No participating clients provided weights; using client metrics fallback.")
            if participated:
                loss = _nanmean([o.loss for o in participated])
                metric_value = _nanmean([o.metric_value for o in participated])
                extra_metric = _nanmean([o.extra_metric for o in participated])
                mscore = metric_score_value("classification", metric_value)
                return loss, metric_value, mscore, extra_metric

        loss, metric_value, extra_metric = evaluate_model(global_model, x_test, y_test, task_type="classification")
        mscore = metric_score_value("classification", metric_value)
        return loss, metric_value, mscore, extra_metric


class RegressionStrategy(TaskStrategy):
    def task_type(self) -> str: return "regression"

    def loggable_run_params(self):
        params = super().loggable_run_params()
        mt = (self.config.get("model_type") or "").lower()
        params["aggregator"] = {
            "strategy": "client_metric_average_no_weight_updates" if mt == "randomforest" else "fedavg_uniform",
            "aggregation_weight_unit": "client_uniform",
        }
        return params

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome:
        local_model = self.build_model()
        samples_count = len(y)
        if _is_keras_like(local_model) and _is_keras_like(global_model):
            try:
                local_model.set_weights(global_model.get_weights())
            except Exception:
                pass
        start = time.time()
        tracker = ResourceTracker()
        tracker.start()
        try:
            weights = train_local_model(
                local_model, x, y,
                epochs=self.knobs["local_epochs"],
                batch_size=self.knobs["batch_size"],
            )
            duration = time.time() - start
            usage = tracker.stop(duration)
            loss, metric_value, extra_metric = evaluate_model(local_model, self.x_test, self.y_test, task_type="regression")
            mscore = metric_score_value("regression", metric_value)

            if self.save_weights and weights is not None:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: np.asarray(v).tolist() for k, v in weights.items()}, f, indent=4)

            extras = self.perturbation_metrics(local_model, client_id=client_id, round_idx=round_idx)

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=metric_value, metric_score=mscore, extra_metric=extra_metric,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=weights_size(weights),
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=weights, extras=extras,
            )
        except Exception:
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)
            return ClientOutcome(
                participated=False, fail_reason="error", samples_count=samples_count, duration=duration,
                loss=np.nan, metric_value=np.nan, metric_score=np.nan, extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1, comm_down=comm_down, comm_up=0,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=None, extras={},
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test,):
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]
        if client_payloads:
            new_global_weights = aggregate_weights(client_payloads)
            # Keep parity with your existing set_weights(list_ordered)
            global_model.set_weights([new_global_weights[f"layer_{i}"] for i in range(len(new_global_weights))])
            if self.save_weights:
                with open(f"weights/global_round_{round_idx}.json", "w") as f:
                    json.dump({k: np.asarray(v).tolist() for k, v in new_global_weights.items()}, f, indent=4)
        else:
            print("No participating clients provided weights; using client metrics fallback.")
            if participated:
                loss = _nanmean([o.loss for o in participated])
                metric_value = _nanmean([o.metric_value for o in participated])
                extra_metric = _nanmean([o.extra_metric for o in participated])
                mscore = metric_score_value("regression", metric_value)
                return loss, metric_value, mscore, extra_metric
        
        print(f"[DEBUG] Model compiled: {hasattr(global_model, 'optimizer') and global_model.optimizer is not None}")

        loss, metric_value, extra_metric = evaluate_model(global_model, x_test, y_test, task_type="regression")
        mscore = metric_score_value("regression", metric_value)
        return loss, metric_value, mscore, extra_metric
