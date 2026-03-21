import time
import numpy as np

from .hf_task import SequenceClassificationSpec
from .hf_cache import get_cached_tokenizer


class HFCore:
    """
    Framework-agnostic HF training loop wrapper.

    Loader schema support:
      - xs can be dict-of-arrays (preferred, from loader preprocessors)
      - xs can be list of raw texts or list-of-token-lists (legacy)
    """

    def __init__(
        self,
        model_id,
        num_labels=None,
        max_length=128,
        batch_size=16,
        device=None,
        task_spec=None,
        label_pad_value=-100,
        generation_config=None,
        task_tag=None,
    ):
        try:
            import torch
            import transformers
        except Exception as e:
            raise ImportError(
                "HF adapters require 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from e

        self.torch = torch
        self.transformers = transformers

        self.model_id = model_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.label_pad_value = int(label_pad_value)

        self.device = self._resolve_device(device)

        self.task_spec = task_spec or SequenceClassificationSpec()
        self.generation_config = self._resolve_generation_config(generation_config)
        self.task_tag = (task_tag or "").strip().lower().replace("-", "_") or None
        self.tokenizer, self.tokenizer_load_s, self.tokenizer_cache_hit = get_cached_tokenizer(
            hf_model_id=model_id,
            task=getattr(self.task_spec, "name", None),
            device=self.device,
            transformers_module=transformers,
        )

        self.model = None
        self.weight_format = None
        self.model_load_s = 0.0
        self.model_cache_hit = False
        needs_num_labels = bool(getattr(self.task_spec, "requires_num_labels", True))
        if num_labels is not None or not needs_num_labels:
            model_load_start = time.time()
            self.model = self.task_spec.build_model(transformers, model_id, num_labels)
            self.model_load_s = float(time.time() - model_load_start)
            self.weight_format = getattr(self.task_spec, "weight_format", None)
            self.model.to(self.device)

    def _qos_startup(self):
        return {
            "tokenizer_load_s": float(self.tokenizer_load_s),
            "model_load_s": float(self.model_load_s),
            "cold_start_time": float(self.tokenizer_load_s + self.model_load_s),
            "tokenizer_cache_hit": bool(self.tokenizer_cache_hit),
            "model_cache_hit": bool(self.model_cache_hit),
        }

    def _resolve_generation_config(self, generation_config):
        defaults = {
            "max_new_tokens": 64,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "length_penalty": 1.0,
        }
        cfg = dict(defaults)
        if isinstance(generation_config, dict):
            cfg.update({k: generation_config[k] for k in defaults.keys() if k in generation_config and generation_config[k] is not None})
        cfg["max_new_tokens"] = int(cfg["max_new_tokens"])
        cfg["num_beams"] = int(cfg["num_beams"])
        cfg["do_sample"] = bool(cfg["do_sample"])
        cfg["temperature"] = float(cfg["temperature"])
        cfg["top_k"] = int(cfg["top_k"])
        cfg["top_p"] = float(cfg["top_p"])
        cfg["length_penalty"] = float(cfg["length_penalty"])

        if not cfg["do_sample"]:
            cfg["temperature"] = 1.0
        if cfg["num_beams"] > 1:
            cfg["do_sample"] = False

        return cfg

    def _resolve_device(self, device):
        torch = self.torch

        if device is not None:
            return device

        if torch.cuda.is_available():
            return "cuda"

        try:
            import torch_directml

            return torch_directml.device()
        except Exception:
            return "cpu"

    def _batch_iter(self, xs, ys):
        bs = self.batch_size

        if isinstance(xs, dict):
            n = len(next(iter(xs.values())))
            for i in range(0, n, bs):
                xb = {k: v[i:i + bs] for k, v in xs.items()}
                yb = None if ys is None else ys[i:i + bs]
                yield xb, yb
            return

        n = len(xs)
        for i in range(0, n, bs):
            xb = xs[i:i + bs]
            yb = None if ys is None else ys[i:i + bs]
            yield xb, yb

    def _debug_shape(self, value):
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                return tuple(int(dim) for dim in shape)
            except Exception:
                return shape
        try:
            arr = np.asarray(value, dtype=object)
            return tuple(int(dim) for dim in arr.shape)
        except Exception:
            return None

    def _debug_preview(self, value, max_items=12):
        if value is None:
            return None

        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()

        sample = value
        while isinstance(sample, (list, tuple)) and sample:
            sample = sample[0]

        if isinstance(sample, (list, tuple)):
            return list(sample[:max_items])
        return sample

    def debug_first_processed_batch(self, xs, ys, inference_only=False):
        torch = self.torch
        batch_iter = self._batch_iter(xs, ys)
        xb, yb = next(batch_iter)
        enc, labels_t, _ = self.task_spec.encode_batch(
            self.tokenizer,
            xb,
            yb,
            self.max_length,
            torch,
            self.device,
            ignore_index=self.label_pad_value,
            inference_only=bool(inference_only),
        )

        input_ids = enc.get("input_ids") if isinstance(enc, dict) else None
        attention_mask = enc.get("attention_mask") if isinstance(enc, dict) else None

        finite_ok = True
        finite_details = {}
        if isinstance(enc, dict):
            for key, tensor in enc.items():
                if hasattr(tensor, "dtype") and hasattr(torch, "is_floating_point") and torch.is_floating_point(tensor):
                    finite_value = bool(torch.isfinite(tensor).all().detach().cpu().item())
                    finite_details[key] = finite_value
                    finite_ok = finite_ok and finite_value
        if labels_t is not None and hasattr(labels_t, "dtype") and torch.is_floating_point(labels_t):
            labels_finite = bool(torch.isfinite(labels_t).all().detach().cpu().item())
            finite_details["labels"] = labels_finite
            finite_ok = finite_ok and labels_finite

        nested_object_keys = []
        if isinstance(xb, dict):
            for key, value in xb.items():
                arr = np.asarray(value, dtype=object)
                if arr.dtype == object:
                    nested_object_keys.append(str(key))

        token_source = xb.get("tokens") if isinstance(xb, dict) and "tokens" in xb else input_ids
        tag_source = xb.get("ner_tags") if isinstance(xb, dict) and "ner_tags" in xb else (yb if yb is not None else labels_t)

        return {
            "input_ids_shape": self._debug_shape(input_ids),
            "attention_mask_shape": self._debug_shape(attention_mask),
            "labels_shape": self._debug_shape(labels_t),
            "token_example": self._debug_preview(token_source),
            "ner_tags_example": self._debug_preview(tag_source),
            "finite_ok": bool(finite_ok),
            "finite_details": finite_details,
            "nested_object_keys": nested_object_keys,
        }

    def count_params(self):
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters()))

    def get_weights(self):
        sd = self.model.state_dict()
        out = {}
        for k, v in sd.items():
            out[k] = v.detach().cpu().numpy()
        return out

    def set_weights(self, weights_dict):
        torch = self.torch
        sd = self.model.state_dict()
        new_sd = {}
        for k, v in sd.items():
            if k in weights_dict:
                new_sd[k] = torch.tensor(weights_dict[k], device="cpu")
            else:
                new_sd[k] = v.detach().cpu()
        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(self.device)

    def _training_timed_out(self, train_start_ts, max_train_time_s):
        if max_train_time_s is None:
            return False
        return (time.time() - float(train_start_ts)) > float(max_train_time_s)

    def finetune(self, xs, ys, epochs=1, lr=5e-5, max_train_time_s=60):
        torch = self.torch

        def _count_supervised_tokens(labels_t, ignore_index):
            if labels_t is None or labels_t.ndim < 2:
                return 0
            return int((labels_t != int(ignore_index)).sum().detach().cpu().item())

        y_local = ys

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(lr))

        total_loss = 0.0
        total_loss_weight = 0
        total_tokens = 0
        step_lat_ms = []
        t_start = time.time()
        timeout_hit = False

        for _ in range(int(epochs)):
            for xb, yb in self._batch_iter(xs, y_local):
                if self._training_timed_out(t_start, max_train_time_s):
                    timeout_hit = True
                    break

                t0 = time.time()

                enc, labels_t, extra = self.task_spec.encode_batch(
                    self.tokenizer,
                    xb,
                    yb,
                    self.max_length,
                    torch,
                    self.device,
                    ignore_index=self.label_pad_value,
                    inference_only=False,
                )

                optimizer.zero_grad(set_to_none=True)
                model_inputs = self.task_spec.build_forward_inputs(enc, labels_t=labels_t, inference_only=False)
                outputs = self.model(**model_inputs)
                logits = outputs.logits
                loss = self.task_spec.extract_loss(torch, outputs, logits, labels_t, extra)
                if loss is None:
                    raise ValueError("Supervised fine-tune mode requires labels/loss-capable batch")
                loss.backward()
                optimizer.step()

                total_tokens += _count_supervised_tokens(labels_t, self.label_pad_value)

                if isinstance(labels_t, torch.Tensor) and labels_t.ndim >= 2:
                    w = _count_supervised_tokens(labels_t, self.label_pad_value)
                elif isinstance(xb, dict):
                    w = len(next(iter(xb.values())))
                else:
                    w = len(xb)

                total_loss += float(loss.detach().cpu().item()) * float(max(1, w))
                total_loss_weight += int(max(1, w))

                step_lat_ms.append((time.time() - t0) * 1000.0)
            
            if timeout_hit:
                break

        duration_s = time.time() - t_start
        self.model.eval()

        step_mean = float(np.mean(step_lat_ms)) if step_lat_ms else np.nan
        step_p95 = float(np.percentile(step_lat_ms, 95)) if step_lat_ms else np.nan
        steady_steps = step_lat_ms[1:] if len(step_lat_ms) > 1 else []
        steady_step_mean = float(np.mean(steady_steps)) if steady_steps else np.nan
        steady_step_p95 = float(np.percentile(steady_steps, 95)) if steady_steps else np.nan

        train_loss = float(total_loss / max(1, total_loss_weight))
        train_throughput = float(total_loss_weight / max(duration_s, 1e-9))
        token_throughput = float(total_tokens / max(duration_s, 1e-9)) if total_tokens > 0 else np.nan

        return {
            "train_loss": train_loss,
            "train_time_s": float(duration_s),
            "train_step_latency_ms_mean": step_mean,
            "train_step_latency_ms_p95": step_p95,
            "train_step_latency_ms_steady_mean": steady_step_mean,
            "train_step_latency_ms_steady_p95": steady_step_p95,
            "train_throughput_eps": train_throughput,
            **self._qos_startup(),
            "train_samples": int(total_loss_weight),
            "tokens_total": int(total_tokens),
            "tokens_per_second": token_throughput,
            "batch_size": int(self.batch_size),
            "device": str(self.device),
            "hf_model_id": self.model_id,
            "max_length": int(self.max_length),
            "hf_task": getattr(self.task_spec, "name", None),
            "label_pad_value": int(self.label_pad_value),
            "hf_weights_format": self.weight_format,
            "train_timeout_s": (None if max_train_time_s is None else float(max_train_time_s)),
            "train_stopped_early": bool(timeout_hit),
        }

    def eval(self, xs, ys, inference_only=False, max_eval_time_s=None, progress_log_interval=None):
        torch = self.torch

        def _count_supervised_tokens(labels_t, ignore_index):
            if labels_t is None or labels_t.ndim < 2:
                return 0
            return int((labels_t != int(ignore_index)).sum().detach().cpu().item())

        y_true = ys
        self.model.eval()

        latencies_ms = []
        total_loss = 0.0
        total_loss_weight = 0
        total_tokens = 0

        preds_all = []
        labels_all = []
        stats_accum = {}

        if isinstance(xs, dict):
            n_eval = len(next(iter(xs.values())))
        else:
            n_eval = len(xs)
        total_batches = int(np.ceil(float(n_eval) / float(max(1, self.batch_size)))) if n_eval else 0

        t_start = time.time()

        last_extra = {}
        total_batches = int((n_eval + max(1, self.batch_size) - 1) / max(1, self.batch_size))
        progress_log_interval = int(progress_log_interval) if progress_log_interval is not None else 0
        max_eval_time_s = float(max_eval_time_s) if max_eval_time_s is not None else None
        progress_log_every = max(1, min(25, total_batches // 4 if total_batches > 4 else total_batches or 1))

        print(
            f"[HFCore.eval] dataloader creation starts | inference_only={bool(inference_only)} "
            f"| batch_size={self.batch_size} | eval_samples={n_eval} | total_batches={total_batches}"
        )
        first_batch_logged = False

        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(self._batch_iter(xs, y_true), start=1):
                if not first_batch_logged:
                    print("[HFCore.eval] first batch pulled")
                if max_eval_time_s is not None and (time.time() - t_start) > max_eval_time_s:
                    raise TimeoutError(
                        f"HF evaluation exceeded max_eval_time_s={max_eval_time_s} "
                        f"after batch {batch_idx - 1}/{total_batches}"
                    )
                t0 = time.time()
                labels_recorded = False

                enc, labels_t, extra = self.task_spec.encode_batch(
                    self.tokenizer,
                    xb,
                    yb,
                    self.max_length,
                    torch,
                    self.device,
                    ignore_index=self.label_pad_value,
                    inference_only=bool(inference_only),
                )

                last_extra = dict(extra or {})

                if bool(inference_only) and bool(getattr(self.task_spec, "supports_generation", False)):
                    if not first_batch_logged:
                        print("[HFCore.eval] model forward starts")
                    pred_t = self.task_spec.generate_predictions(
                        self.model,
                        enc,
                        self.tokenizer,
                        torch,
                        self.generation_config,
                    )
                    if not first_batch_logged:
                        print("[HFCore.eval] first batch forward ends")
                    preds_all.append(pred_t.detach().cpu().numpy())
                else:
                    model_inputs = self.task_spec.build_forward_inputs(enc, labels_t=labels_t, inference_only=bool(inference_only))
                    if not first_batch_logged:
                        print("[HFCore.eval] model forward starts")
                    outputs = self.model(**model_inputs)
                    logits = outputs.logits
                    pred_t = self.task_spec.preds_from_logits(torch, logits, extra)
                    if not first_batch_logged:
                        print("[HFCore.eval] first batch forward ends")
                    preds_all.append(pred_t.detach().cpu().numpy())
                    
                    if not bool(inference_only):
                        stat = self.task_spec.batch_metric_statistics(torch, logits, labels_t, extra)
                        if stat:
                            for k, v in stat.items():
                                stats_accum[k] = float(stats_accum.get(k, 0.0)) + float(v)

                        stat_out = self.task_spec.batch_metric_statistics_from_outputs(torch, outputs, labels_t, extra)
                        if stat_out:
                            for k, v in stat_out.items():
                                stats_accum[k] = float(stats_accum.get(k, 0.0)) + float(v)

                        loss = self.task_spec.extract_loss(torch, outputs, logits, labels_t, extra)
                        if loss is not None and labels_t is not None:
                            labels_all.append(labels_t.detach().cpu().numpy())
                            labels_recorded = True
                            total_tokens += _count_supervised_tokens(labels_t, self.label_pad_value)

                            if labels_t.ndim >= 2:
                                w = _count_supervised_tokens(labels_t, self.label_pad_value)
                            elif isinstance(xb, dict):
                                w = len(next(iter(xb.values())))
                            else:
                                w = len(xb)

                            total_loss += float(loss.detach().cpu().item()) * float(max(1, w))
                            total_loss_weight += int(max(1, w))

                if labels_t is not None and bool(inference_only) and yb is not None and not labels_recorded:
                    labels_all.append(labels_t.detach().cpu().numpy())

                latencies_ms.append((time.time() - t0) * 1000.0)
                if total_batches and (
                    batch_idx == 1
                    or batch_idx == total_batches
                    or batch_idx % progress_log_every == 0
                ):
                    print(
                        "[HFCore.eval] progress | "
                        f"batch={batch_idx}/{total_batches} | "
                        f"samples_done={min(batch_idx * self.batch_size, n_eval)}/{n_eval} | "
                        f"last_batch_ms={latencies_ms[-1]:.2f}"
                    )
                first_batch_logged = True

        duration_s = time.time() - t_start

        y_true_np = np.concatenate(labels_all, axis=0) if labels_all else np.asarray([], dtype="int64")
        y_pred_np = np.concatenate(preds_all, axis=0) if preds_all else np.asarray([], dtype="int64")

        named_metrics = None
        loss_mean = float(total_loss / max(1, total_loss_weight)) if total_loss_weight > 0 else np.nan

        m_stats = self.task_spec.metrics_from_statistics(stats_accum) if stats_accum else None
        if isinstance(m_stats, dict) and m_stats:
            print("[HFCore.eval] metric computation starts")
            primary = float(m_stats.get("primary", np.nan))
            secondary = float(m_stats.get("secondary", np.nan))
            named_metrics = m_stats.get("named_metrics") if isinstance(m_stats, dict) else None
        elif y_true_np.size == 0 or y_pred_np.size == 0:
            primary = np.nan
            secondary = np.nan
        else:
            print("[HFCore.eval] metric computation starts")
            metrics_extra = dict(last_extra or {})
            metrics_extra["task_tag"] = self.task_tag
            metrics_extra["loss_mean"] = loss_mean
            m = self.task_spec.metrics(y_true_np, y_pred_np, y_extra=metrics_extra)
            primary = float(m.get("primary", np.nan))
            secondary = float(m.get("secondary", np.nan))
            named_metrics = m.get("named_metrics") if isinstance(m, dict) else None

            if getattr(self.task_spec, "name", None) == "fill_mask" and loss_mean == loss_mean:
                try:
                    secondary = float(np.exp(np.clip(loss_mean, a_min=-50.0, a_max=50.0)))
                except Exception:
                    secondary = np.nan

        lat_mean = float(np.mean(latencies_ms)) if latencies_ms else np.nan
        lat_p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else np.nan
        steady_lat = latencies_ms[1:] if len(latencies_ms) > 1 else []
        lat_steady_mean = float(np.mean(steady_lat)) if steady_lat else np.nan
        lat_steady_p95 = float(np.percentile(steady_lat, 95)) if steady_lat else np.nan
        throughput = float(n_eval / max(duration_s, 1e-9))
        token_throughput = float(total_tokens / max(duration_s, 1e-9)) if total_tokens > 0 else np.nan

        qos = {
            "eval_latency_ms_mean": lat_mean,
            "eval_latency_ms_p95": lat_p95,
            "eval_latency_ms_steady_mean": lat_steady_mean,
            "eval_latency_ms_steady_p95": lat_steady_p95,
            "eval_throughput_eps": throughput,
            **self._qos_startup(),
            "eval_samples": int(n_eval),
            "tokens_total": int(total_tokens),
            "tokens_per_second": token_throughput,
            "batch_size": int(self.batch_size),
            "device": str(self.device),
            "hf_model_id": self.model_id,
            "max_length": int(self.max_length),
            "hf_task": getattr(self.task_spec, "name", None),
            "label_pad_value": int(self.label_pad_value),
            "hf_weights_format": self.weight_format,
            "inference_only": bool(inference_only),
        }
        if named_metrics and isinstance(named_metrics, dict):
            for mk, mv in named_metrics.items():
                if mv is not None and not (isinstance(mv, float) and np.isnan(mv)):
                    qos[str(mk).lower()] = float(mv)

        if stats_accum:
            for sk, sv in stats_accum.items():
                qos[f"metric_stat_{sk}"] = float(sv)

        if getattr(self.task_spec, "supports_generation", False):
            qos.update({f"generation_{k}": v for k, v in self.generation_config.items()})

        return loss_mean, primary, secondary, qos
    
