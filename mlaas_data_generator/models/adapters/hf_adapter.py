from .hf_core import HFCore
from .hf_cache import get_cached_model
from .hf_task import (
    SequenceClassificationSpec,
    SentenceSimilaritySpec,
    TokenClassificationSpec,
    FillMaskSpec,
    CausalLMGenerationSpec,
    Seq2SeqGenerationSpec,
    ImageClassificationSpec,
    ObjectDetectionSpec,
    ImageSegmentationSpec,
    ImageCaptioningSpec,
    TextImageRetrievalSpec,
    VQASpec,
)


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
            resolved_num_labels = None if num_labels is None else int(num_labels)
            is_regression = (str(label_format).lower() == "continuous") or (resolved_num_labels == 1)
            spec = SentenceSimilaritySpec(is_regression=is_regression)
        elif hf_task == "fill_mask":
            spec = FillMaskSpec()
        elif hf_task in {"causal_lm_generation", "causal_lm", "text_generation"}:
            spec = CausalLMGenerationSpec()
        elif hf_task in {"seq2seq_generation", "text2text_generation", "text2text"}:
            spec = Seq2SeqGenerationSpec()
        elif hf_task in {"image_classification", "vision_classification"}:
            spec = ImageClassificationSpec()
        elif hf_task in {"object_detection", "image_detection", "detection"}:
            spec = ObjectDetectionSpec()
        elif hf_task in {"image_segmentation", "semantic_segmentation", "segmentation"}:
            spec = ImageSegmentationSpec()
        elif hf_task in {"image_captioning", "image_to_text"}:
            spec = ImageCaptioningSpec()
        elif hf_task in {"text_image_retrieval", "image_text_retrieval"}:
            spec = TextImageRetrievalSpec()
        elif hf_task in {"visual_question_answering", "vqa"}:
            spec = VQASpec()
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
            spec = SentenceSimilaritySpec(is_regression=True)
        elif task in {"image_classification", "vision_classification"}:
            spec = ImageClassificationSpec()
        elif task in {"object_detection", "image_detection", "detection"}:
            spec = ObjectDetectionSpec()
        elif task in {"image_segmentation", "semantic_segmentation", "segmentation"}:
            spec = ImageSegmentationSpec()
        elif task in {"image_captioning", "image_to_text"}:
            spec = ImageCaptioningSpec()
        elif task in {"text_image_retrieval", "image_text_retrieval"}:
            spec = TextImageRetrievalSpec()
        elif task in {"visual_question_answering", "vqa"}:
            spec = VQASpec()
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
        if core.model is None:
            def _load_model():
                if task in {"causal_lm_generation", "causal_lm", "text_generation"}:
                    return transformers.AutoModelForCausalLM.from_pretrained(model_id)
                if task in {"seq2seq_generation", "text2text_generation", "text2text"}:
                    return transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
                if task == "fill_mask":
                    return transformers.AutoModelForMaskedLM.from_pretrained(model_id)
                if task == "token_classification":
                    return transformers.AutoModelForTokenClassification.from_pretrained(model_id)
                if task in {"image_classification", "vision_classification"}:
                    return transformers.AutoModelForImageClassification.from_pretrained(model_id)
                if task in {"object_detection", "image_detection", "detection"}:
                    return transformers.AutoModelForObjectDetection.from_pretrained(model_id)
                if task in {"image_segmentation", "semantic_segmentation", "segmentation"}:
                    return transformers.AutoModelForSemanticSegmentation.from_pretrained(model_id)
                if task in {"image_captioning", "image_to_text"}:
                    return transformers.AutoModelForVision2Seq.from_pretrained(model_id)
                if task in {"text_image_retrieval", "image_text_retrieval"}:
                    return transformers.AutoModel.from_pretrained(model_id)
                if task in {"visual_question_answering", "vqa"}:
                    return transformers.AutoModelForVisualQuestionAnswering.from_pretrained(model_id)
                return transformers.AutoModelForSequenceClassification.from_pretrained(model_id)

            core.model, core.model_load_s, core.model_cache_hit = get_cached_model(
                hf_model_id=model_id,
                task=task,
                device=core.device,
                loader_fn=_load_model,
            )

        core.model.to(core.device)
        core.model.eval()

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
