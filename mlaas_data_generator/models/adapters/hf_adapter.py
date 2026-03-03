from .hf_core import HFCore
from .hf_task import SequenceClassificationSpec, TokenClassificationSpec
import time



class TransformersTextFineTuneAdapter:
    """
    Wrapper around HFCore.

    Supports loader schema:
      - x is dict-of-arrays: {"input_ids": ..., "attention_mask": ...}
      - x can still be legacy raw texts or token lists (kept for backwards compatibility)
    """
    def __init__(
        self,
        model_id,
        num_labels,
        max_length=128,
        batch_size=16,
        device=None,
        hf_task="sequence_classification",
        label_pad_value=-100,
        multilabel=False,
        label_format="single_index",
    ):
        if hf_task == "token_classification":
            spec = TokenClassificationSpec(multilabel=multilabel, label_format=label_format)
        else:
            spec = TokenClassificationSpec(multilabel=multilabel, label_format=label_format)

        self.core = HFCore(
            model_id=model_id,
            num_labels=int(num_labels),
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            task_spec=spec,
            label_pad_value=int(label_pad_value),
        )

        self.model_id = model_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = self.core.device

    def count_params(self):
        return self.core.count_params()

    def get_weights(self):
        return self.core.get_weights()

    def set_weights(self, weights_dict):
        self.core.set_weights(weights_dict)

    def fit(self, x, y, epochs=1, lr=5e-5):
        # x may be dict-of-arrays (new loader schema) or legacy list-like
        return self.core.finetune(x, y, epochs=epochs, lr=lr)

    def evaluate(self, x, y):
        # core returns (loss, primary, secondary, qos)
        loss, primary, secondary, qos = self.core.eval(x, y)

        acc = primary
        f1 = secondary
        return loss, acc, f1, qos


class TransformersTextClassifierAdapter:
    """
    Inference-style adapter (loads an already-finetuned sequence classification model_id).
    Uses HFCore batching/eval utilities, including dict-of-arrays support.
    """
    def __init__(
        self,
        model_id,
        max_length=128,
        batch_size=16,
        device=None,
    ):
        core = HFCore(
            model_id=model_id,
            num_labels=None,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            task_spec=SequenceClassificationSpec(),
        )

        transformers = core.transformers
        model_load_start = time.time()
        core.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)
        core.model.to(core.device)
        core.model.eval()
        core.cold_start_time += float(time.time() - model_load_start)

        self.core = core
        self.model_id = model_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = self.core.device

    def evaluate(self, x, y):
        loss, primary, secondary, qos = self.core.eval(x, y)
        acc = primary
        f1 = secondary

        qos = dict(qos)
        if "eval_latency_ms_mean" in qos:
            qos["inference_latency_ms_mean"] = qos.pop("eval_latency_ms_mean")
        if "eval_latency_ms_p95" in qos:
            qos["inference_latency_ms_p95"] = qos.pop("eval_latency_ms_p95")
        if "eval_latency_ms_steady_mean" in qos:
            qos["inference_latency_ms_steady_mean"] = qos.pop("eval_latency_ms_steady_mean")
        if "eval_latency_ms_steady_p95" in qos:
            qos["inference_latency_ms_steady_p95"] = qos.pop("eval_latency_ms_steady_p95")
        if "eval_throughput_eps" in qos:
            qos["throughput_eps"] = qos.pop("eval_throughput_eps")

        return loss, acc, f1, qos