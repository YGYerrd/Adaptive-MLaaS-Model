#base.p
from dataclasses import dataclass
import numpy as np
from ...hf_tasks import normalize_hf_task as shared_normalize_hf_task
from ...models.label_schema import infer_num_labels



def metric_score_value(task_type: str, metric_value: float | None) -> float:
    """Map raw metric to a [0,1]-ish score. For classification/clustering, identity; for regression, 1/(1+rmse)."""
    mv = float(metric_value) if metric_value is not None else np.nan
    if mv != mv:  # NaN
        return np.nan
    if task_type == "regression":
        return 1.0 / (1.0 + mv)
    return mv

def weights_size(weights_dict_or_list) -> int:
    if not weights_dict_or_list:
        return 0
    arrays = weights_dict_or_list.values() if isinstance(weights_dict_or_list, dict) else weights_dict_or_list
    return int(sum(np.asarray(w).nbytes for w in arrays))

def _is_keras_like(m) -> bool:
    return hasattr(m, "get_weights") and callable(getattr(m, "get_weights", None)) \
        and hasattr(m, "set_weights") and callable(getattr(m, "set_weights", None))

def _nanmean(values):
    cleaned = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not cleaned:
        return np.nan
    return float(np.mean(cleaned))


def normalize_hf_task(hf_task: str | None) -> str:
    """Normalize user/provider HF task aliases to canonical names."""
    return shared_normalize_hf_task(hf_task, default="unknown", unknown="unknown")


def canonical_task_family(task_type: str | None, hf_task: str | None = None) -> str:
    """Map legacy task_type/HF-task values to a canonical taxonomy."""
    base = (task_type or "").strip().lower()
    hf = normalize_hf_task(hf_task)

    if base in {"image_classification"}:
        return "classification"
    if base in {"object_detection", "image_detection", "detection"}:
        return "detection"
    if base in {"image_segmentation", "semantic_segmentation", "segmentation"}:
        return "segmentation"
    if base in {"generation", "text_generation", "text2text_generation", "image_captioning"}:
        return "generation"
    if base in {"retrieval", "text_image_retrieval", "image_text_retrieval"}:
        return "retrieval"
    if base in {"vqa", "visual_question_answering", "visual_qa"}:
        return "vqa"

    if base == "clustering":
        return "clustering"
    if base == "regression":
        return "regression"
    if base == "classification":
        if hf == "token_classification":
            return "token_classification"
        if hf == "fill_mask":
            return "fill_mask"
        if hf in {"sequence_classification", "image_classification"} or hf == "unknown":
            return "classification"
        if hf in {"image_detection"}:
            return "detection"
        if hf in {"image_segmentation"}:
            return "segmentation"
        if hf in {"causal_lm_generation", "seq2seq_generation", "image_captioning"}:
            return "generation"
        if hf in {"text_image_retrieval"}:
            return "retrieval"
        if hf in {"visual_question_answering"}:
            return "vqa"
        return f"hf_{hf}"
    return "unknown"


def canonical_label_format(task_family: str) -> str:
    mapping = {
        "classification": "single_label",
        "token_classification": "token_labels",
        "fill_mask": "token_labels",
        "regression": "continuous",
        "generation": "token_labels",
        "clustering": "cluster_id",
        "detection": "bbox_coco",
        "segmentation": "mask",
        "retrieval": "paired_rank",
        "vqa": "answer_text",
    }
    return mapping.get(task_family, "unknown")


def canonical_metric_names(task_family: str, metric_key: str, *, hf_task: str | None = None, task_tag: str | None = None) -> tuple[str, str | None]:
    if task_family == "classification":
        if normalize_hf_task(hf_task) == "image_classification":
            return ("accuracy", "top5_accuracy")
        return ("accuracy", "f1")
    if task_family == "token_classification":
        return ("f1", "accuracy")
    if task_family == "fill_mask":
        return ("masked_accuracy", "perplexity_proxy")
    if task_family == "regression":
        return ("rmse", "mae")
    if task_family == "generation":
        hf = normalize_hf_task(hf_task)
        tag = (task_tag or "").strip().lower().replace("-", "_")
        if hf == "causal_lm_generation":
            return ("loss", "perplexity")
        if hf == "seq2seq_generation" and tag in {"", "language_modeling", "language-modeling"}:
            return ("loss", "perplexity")
        return ("perplexity", None)
    if task_family == "clustering":
        return ("silhouette", None)
    if task_family == "detection":
        return ("map", "map@0.5")
    if task_family == "segmentation":
        return ("iou", "dice")
    if task_family == "retrieval":
        return ("r@1", "r@5")
    if task_family == "vqa":
        return ("exact_match", None)
    return ((metric_key or "metric").lower(), None)


def canonical_generation_metrics(task_tag: str | None, has_labels: bool, *, hf_task: str | None = None) -> tuple[str, ...]:
    """Decode-centric metric set for generation by subtype."""
    hf = normalize_hf_task(hf_task)
    if hf == "causal_lm_generation":
        return ("loss", "perplexity") if has_labels else tuple()

    tag = (task_tag or "").strip().lower().replace("-", "_")
    if tag == "summarization":
        base = ("rouge1", "rouge2", "rougeL")
    elif tag == "translation":
        base = ("sacrebleu",)
    elif tag in {"captioning", "image_captioning"}:
        base = ("cider", "bleu")
    else:
        base = tuple()

    if has_labels:
        return base + ("perplexity",)
    return base


def metric_availability(task_family: str, task_tag: str | None = None, has_labels: bool = True, *, hf_task: str | None = None) -> dict[str, tuple[str, ...]]:
    """Return canonical train/eval metric availability by task family."""
    if task_family == "generation":
        return {
            "train": ("loss", "perplexity") if has_labels else tuple(),
            "eval": canonical_generation_metrics(task_tag=task_tag, has_labels=has_labels, hf_task=hf_task),
        }

    if task_family == "retrieval":
        return {"train": ("loss",), "eval": ("r@1", "r@5", "r@10")}

    if task_family == "vqa":
        return {"train": ("loss",), "eval": ("exact_match",)}

    if task_family == "detection":
        return {"train": ("loss",), "eval": ("map", "map@0.5")}

    if task_family == "segmentation":
        return {"train": ("loss",), "eval": ("iou", "dice")}

    return {
        "train": ("loss",),
        "eval": tuple(),
    }

@dataclass
class ClientOutcome:
    participated: bool
    fail_reason: str
    samples_count: int
    duration: float
    loss: float
    metric_value: float 
    metric_score: float 
    extra_metric: float 
    rounds_so_far: int
    comm_down: int
    comm_up: int
    cpu_time_s: float | None
    cpu_utilization: float | None
    memory_used_mb: float | None
    memory_utilization: float | None
    gpu_utilization: float | None
    gpu_memory_utilization: float | None
    gpu_memory_used_mb: float | None
    peak_vram_mb: float | None
    avg_vram_mb: float | None
    peak_host_ram_mb: float | None
    avg_host_ram_mb: float | None
    payload: dict | list | None   # weights for NN, None for clustering
    extras: dict                  # extra columns per task
    sequence_count: int | None = None
    supervised_token_count: int | None = None
    aggregation_weight_unit: str | None = None
    aggregation_weight_value: float | None = None

class TaskStrategy:
    """Base class: thin wrapper around your existing per-task logic."""
    def __init__(self, meta, knobs, config, x_test, y_test, metric_key, save_weights: bool):
        self.meta = meta
        self.knobs = knobs
        self.config = config
        self.x_test = x_test
        self.y_test = y_test
        self.metric_key = metric_key
        self.save_weights = save_weights

    def build_model(self):
        extra = {}
        if self.task_type() == "clustering":
            for key in ("clustering_k","clustering_init","clustering_n_init","clustering_max_iter","clustering_tol","seed","random_state"):
                if key in self.config:
                    extra[key] = self.config[key]
        else:
            for key in ("rf_trees", "rf_max_depth", "mobilenet_trainable", "n_estimators", "max_depth",
                        "hf_model_id", "max_length", "device", "hf_task",
                        "max_new_tokens", "num_beams", "do_sample", "temperature", "top_k", "top_p", "length_penalty", "task_tag"):
                if key in self.config:
                    extra[key] = self.config[key]
            
            if "batch_size" in self.knobs:
                extra["batch_size"] = self.knobs["batch_size"]

        ds_args = self.config.get("dataset_args", {}) or {}
 
        for key in ("hf_model_id", "max_length", "device", "hf_task",
                    "max_new_tokens", "num_beams", "do_sample", "temperature", "top_k", "top_p", "length_penalty", "task_tag"):
            if key in ds_args and key not in extra:
                extra[key] = ds_args[key]

        model_type = (self.config.get("model_type") or "").lower()

        from ...models.builders import create_model

        common = dict(
            input_shape=tuple(self.meta["input_shape"]),
            num_classes=infer_num_labels(self.meta, fallback=self.meta.get("num_classes")),
            task_type=self.task_type(),
            model_type=self.config.get("model_type"),
            **extra,
        )

        if model_type in ("hf", "hf_text", "transformers"):
            # HF inference adapter doesn't need Keras hyperparams
            return create_model(**common)

        return create_model(
            **common,
            hidden_layers=self.knobs["hidden_layers"],
            learning_rate=self.knobs["learning_rate"],
            activation=self.knobs["activation"],
            dropout=self.knobs["dropout"],
            weight_decay=self.knobs["weight_decay"],
            optimizer=self.knobs["optimizer"],
        )

    def comm_down_bytes(self, global_model):
        # number of bytes when broadcasting global weights (0 for clustering adapter if no weights)
        try:
            return weights_size(global_model.get_weights())
        except Exception:
            return 0
    
    def loggable_run_params(self):
        ds_args = self.config.get("dataset_args", {}) or {}
        accounting = (self.meta or {}).get("accounting") if isinstance(self.meta, dict) else None

        adapter = {
            "optimizer": self.knobs.get("optimizer"),
            "learning_rate": self.knobs.get("learning_rate"),
            "batch_size": self.knobs.get("batch_size"),
            "local_epochs": self.knobs.get("local_epochs"),
            "hidden_layers": self.knobs.get("hidden_layers"),
            "activation": self.knobs.get("activation"),
            "dropout": self.knobs.get("dropout"),
            "weight_decay": self.knobs.get("weight_decay"),
            "early_stopping_patience": self.knobs.get("early_stopping_patience"),
        }

        dataset = {}
        if ds_args:
            dataset["dataset_args"] = ds_args
        if isinstance(accounting, dict):
            dataset["accounting"] = accounting

        return {
            "adapter": adapter,
            "dataset": dataset,
        } 


    def summary_lines(self):
        """
        Return list of (label, value) pairs to print for this strategy.
        Uses loggable_run_params() so output matches DB logging.
        """
        params = self.loggable_run_params() or {}
        out = []

        for scope in ("dataset", "adapter", "aggregator"):
            kv = params.get(scope) or {}
            if not kv:
                continue
            out.append((f"[{scope}]", ""))  # section header
            for k in sorted(kv.keys()):
                out.append((k, kv[k]))
        return out  
    

    
    def task_type(self) -> str: ...
    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome: ...
    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
        """Return (loss, metric_value, metric_score, extra_metric)."""
        ...
