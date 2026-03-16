from .hf_core import HFCore
from .hf_task import (
    SequenceClassificationSpec,
    SentenceSimilaritySpec,
    TokenClassificationSpec,
    FillMaskSpec,
    CausalLMGenerationSpec,
    Seq2SeqGenerationSpec,
)
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
        generation_config=None,
        task_tag=None,
    ):
        if hf_task == "token_classification":
            spec = TokenClassificationSpec(multilabel=multilabel, label_format=label_format)
        elif hf_task == "sentence_similarity":
            spec = SentenceSimilaritySpec(is_regression=(str(label_format).lower() == "continuous"))
        elif hf_task == "fill_mask":
            spec = FillMaskSpec()
        elif hf_task in {"causal_lm_generation", "causal_lm", "text_generation"}:
            spec = CausalLMGenerationSpec()
        elif hf_task in {"seq2seq_generation", "text2text_generation", "text2text"}:
            spec = Seq2SeqGenerationSpec()
        else:
            spec = SequenceClassificationSpec(multilabel=multilabel, label_format=label_format)

        self.core = HFCore(
            model_id=model_id,
            num_labels=(None if num_labels is None else int(num_labels)),
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            task_spec=spec,
            label_pad_value=int(label_pad_value),
            generation_config=generation_config,
            task_tag=task_tag,
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

    def fit(self, x, y, epochs=1, lr=5e-5, max_train_time_s=60):
        return self.core.finetune(
            x,
            y,
            epochs=epochs,
            lr=lr,
            max_train_time_s=max_train_time_s,
        )

    def evaluate(self, x, y, inference_only=False):
        loss, primary, secondary, qos = self.core.eval(x, y, inference_only=inference_only)
        return loss, primary, secondary, qos


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
        hf_task="sequence_classification",
        generation_config=None,
        task_tag=None,
    ):
        task = str(hf_task or "sequence_classification").lower().replace("-", "_")
        if task in {"causal_lm_generation", "causal_lm", "text_generation"}:
            spec = CausalLMGenerationSpec()
        elif task in {"seq2seq_generation", "text2text_generation", "text2text"}:
            spec = Seq2SeqGenerationSpec()
        elif task == "fill_mask":
            spec = FillMaskSpec()
        elif task == "token_classification":
            spec = TokenClassificationSpec()
        elif task == "sentence_similarity":
            spec = SentenceSimilaritySpec()
        else:
            spec = SequenceClassificationSpec()

        core = HFCore(
            model_id=model_id,
            num_labels=None,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            task_spec=spec,
            generation_config=generation_config,
            task_tag=task_tag,
        )

        transformers = core.transformers
        model_load_start = time.time()
        if core.model is None:
            if task in {"causal_lm_generation", "causal_lm", "text_generation"}:
                core.model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
            elif task in {"seq2seq_generation", "text2text_generation", "text2text"}:
                core.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
            elif task == "fill_mask":
                core.model = transformers.AutoModelForMaskedLM.from_pretrained(model_id)
            elif task == "token_classification":
                core.model = transformers.AutoModelForTokenClassification.from_pretrained(model_id)
            else:
                core.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)

        core.model.to(core.device)
        core.model.eval()
        core.cold_start_time += float(time.time() - model_load_start)

        self.core = core
        self.model_id = model_id
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = self.core.device

    def evaluate(self, x, y, inference_only=True):
        loss, primary, secondary, qos = self.core.eval(x, y, inference_only=inference_only)

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

        return loss, primary, secondary, qos
