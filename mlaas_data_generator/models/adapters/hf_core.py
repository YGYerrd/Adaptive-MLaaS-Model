import time
import numpy as np

from .hf_task import SequenceClassificationSpec


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
        cold_start_begin = time.time()
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=False)

        self.model = None
        self.weight_format = None
        needs_num_labels = bool(getattr(self.task_spec, "requires_num_labels", True))
        if num_labels is not None or not needs_num_labels:
            self.model = self.task_spec.build_model(transformers, model_id, num_labels)
            self.weight_format = getattr(self.task_spec, "weight_format", None)
            self.model.to(self.device)
        self.cold_start_time = float(time.time() - cold_start_begin)


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

        # New loader path: dict of arrays
        if isinstance(xs, dict):
            n = len(next(iter(xs.values())))
            for i in range(0, n, bs):
                xb = {k: v[i:i + bs] for k, v in xs.items()}
                yb = None if ys is None else ys[i:i + bs]
                yield xb, yb
            return

        # Legacy path: list-like
        n = len(xs)
        for i in range(0, n, bs):
            xb = xs[i:i + bs]
            yb = None if ys is None else ys[i:i + bs]
            yield xb, yb

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
        
        # xs may be dict-of-arrays or list-like; do not force list(xs)
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
                )

                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(**enc)
                logits = outputs.logits
                loss = self.task_spec.loss_fn(torch, logits, labels_t, extra)
                loss.backward()
                optimizer.step()

                total_tokens += _count_supervised_tokens(labels_t, self.label_pad_value)

                # batch size for dict path or list path
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
            "cold_start_time": float(self.cold_start_time),
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

    def eval(self, xs, ys):
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

        # evaluation sample count
        if isinstance(xs, dict):
            n_eval = len(next(iter(xs.values())))
        else:
            n_eval = len(xs)

        t_start = time.time()

        with torch.no_grad():
            for xb, yb in self._batch_iter(xs, y_true):
                t0 = time.time()

                enc, labels_t, extra = self.task_spec.encode_batch(
                    self.tokenizer,
                    xb,
                    yb,
                    self.max_length,
                    torch,
                    self.device,
                    ignore_index=self.label_pad_value,
                )

                outputs = self.model(**enc)
                logits = outputs.logits
                pred_t = self.task_spec.preds_from_logits(torch, logits, extra)

                preds_all.append(pred_t.detach().cpu().numpy())

                if labels_t is not None:
                    labels_all.append(labels_t.detach().cpu().numpy())

                    loss = self.task_spec.loss_fn(torch, logits, labels_t, extra)
                    total_tokens += _count_supervised_tokens(labels_t, self.label_pad_value)

                    if labels_t.ndim >= 2:
                        w = _count_supervised_tokens(labels_t, self.label_pad_value)
                    elif isinstance(xb, dict):
                        w = len(next(iter(xb.values())))
                    else:
                        w = len(xb)

                    total_loss += float(loss.detach().cpu().item()) * float(max(1, w))
                    total_loss_weight += int(max(1, w))

                latencies_ms.append((time.time() - t0) * 1000.0)

        duration_s = time.time() - t_start

        if labels_all:
            y_true_np = np.concatenate(labels_all, axis=0)
        else:
            y_true_np = np.asarray([], dtype="int64")

        if preds_all:
            y_pred_np = np.concatenate(preds_all, axis=0)
        else:
            y_pred_np = np.asarray([], dtype="int64")

        if y_true_np.size == 0 or y_pred_np.size == 0:
            primary = np.nan
            secondary = np.nan
            loss_mean = np.nan
        else:
            m = self.task_spec.metrics(y_true_np, y_pred_np, y_extra=extra)

            primary = float(m.get("primary", np.nan))
            secondary = float(m.get("secondary", np.nan))
            loss_mean = float(total_loss / max(1, total_loss_weight))

            if getattr(self.task_spec, "name", None) == "fill_mask":
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
            "cold_start_time": float(self.cold_start_time),
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
        }

        return loss_mean, primary, secondary, qos