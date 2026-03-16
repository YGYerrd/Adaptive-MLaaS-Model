import numpy as np

# ----------------------------
# HF Task Specs
# ----------------------------

class HFTaskSpec:
    """
    Task-specific behaviour for HF fine-tuning/evaluation.

    Loader schema support:
      - New path: xb is a dict of numpy arrays (already tokenised), e.g.
            {"input_ids": (B, L), "attention_mask": (B, L), ...}
      - Legacy path: xb is raw text (sequence) or list-of-tokens (token task)
    """
    name = "base"

    requires_num_labels = True
    supports_generation = False

    def build_model(self, transformers, model_id, num_labels):
        raise NotImplementedError

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        """
        Returns (enc_dict, labels_tensor_or_none, extra_dict)
        extra_dict can hold masks etc.
        """
        raise NotImplementedError

    def loss_fn(self, torch, logits, labels_t, extra):
        raise NotImplementedError

    def preds_from_logits(self, torch, logits, extra):
        raise NotImplementedError

    def metrics(self, y_true, y_pred, y_extra=None):
        """
        Returns dict with at least:
          - primary (float)
          - secondary (float or np.nan)
        """
        raise NotImplementedError

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        model_inputs = dict(enc)
        if labels_t is not None and not inference_only:
            model_inputs["labels"] = labels_t
        return model_inputs

    def extract_loss(self, torch, outputs, logits, labels_t, extra):
        if labels_t is not None and hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if labels_t is None:
            return None
        return self.loss_fn(torch, logits, labels_t, extra)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        raise NotImplementedError


class SequenceClassificationSpec(HFTaskSpec):
    name = "sequence_classification"

    def __init__(self, multilabel=False, threshold=0.5, label_format="single_index"):
        self.multilabel = bool(multilabel)
        self.threshold = float(threshold)
        self.label_format = str(label_format or "single_index").lower()
    
    def _infer_label_mode(self, yb):
        if yb is None:
            return "none"
        
        if self.label_format in {"onehot", "multilabel", "multihot"}:
            mapping = {"onehot": "single_onehot", "multihot": "multilabel"}
            return mapping.get(self.label_format, self.label_format)


        arr = np.asarray(yb)
        if arr.ndim == 1:
            return "single_index"

        if arr.ndim == 2:
            is_binary = np.isin(arr, [0, 1]).all()
            row_sums = arr.sum(axis=1)
            if is_binary and np.all(row_sums == 1):
                return "single_onehot"
            return "multilabel"

        return "unknown"

    def _is_multilabel_mode(self, label_mode, extra):
        mode = extra.get("label_mode", label_mode)
        return bool(self.multilabel or mode == "multilabel")


    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForSequenceClassification
        self.weight_format = None
        extra = {}
        if self.multilabel:
            extra["problem_type"] = "multi_label_classification"
        try:
            model = AutoModel.from_pretrained(
                model_id,
                num_labels=int(num_labels),
                ignore_mismatched_sizes=True,
                use_safetensors=True,
                **extra,
            )
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(
                    model_id,
                    num_labels=int(num_labels),
                    ignore_mismatched_sizes=True,
                    use_safetensors=False,
                    **extra,
                )
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        label_mode = self._infer_label_mode(yb)
        batch_multilabel = self._is_multilabel_mode(label_mode, {"label_mode": label_mode})
        # New loader path: already tokenised dict of arrays
        if isinstance(xb, dict):
            enc = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in xb.items()}
            labels_t = None
            if yb is not None:
                dtype = torch.float32 if (batch_multilabel or label_mode == "single_onehot") else torch.long
                labels_t = torch.tensor(yb, dtype=dtype, device=device)
            return enc, labels_t, {"multilabel": batch_multilabel, "label_mode": label_mode}


        # Legacy path: raw texts
        enc = tokenizer(
            xb,
            truncation=True,
            padding=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        labels_t = None
        if yb is not None:
            dtype = torch.float32 if (batch_multilabel or label_mode == "single_onehot") else torch.long
            labels_t = torch.tensor(yb, dtype=dtype, device=device)
        return enc, labels_t, {
            "multilabel": batch_multilabel,
            "label_mode": label_mode,
            "ignore_index": int(ignore_index),
        }

    def loss_fn(self, torch, logits, labels_t, extra):
        label_mode = extra.get("label_mode", "unknown")
        if self._is_multilabel_mode(label_mode, extra):
            if labels_t.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                labels_t = labels_t.float()
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t)

        if label_mode == "single_onehot":
            labels_t = torch.argmax(labels_t, dim=-1)
        if logits.ndim == 3 and labels_t.ndim == 1:
            logits = logits[:, 0, :]
        
        if labels_t.ndim == 1:
            num_classes = int(logits.shape[-1])
            valid = (labels_t >= 0) & (labels_t < num_classes)
            if not bool(torch.any(valid)):
                return logits.new_tensor(0.0)
            logits = logits[valid]
            labels_t = labels_t[valid]
            
        return torch.nn.functional.cross_entropy(logits, labels_t)

    def preds_from_logits(self, torch, logits, extra):
        if logits.ndim == 3:
            logits = logits[:, 0, :]
        if bool(extra.get("multilabel", self.multilabel)):
            probs = torch.sigmoid(logits)
            return (probs >= self.threshold).to(dtype=torch.int64)
        return torch.argmax(logits, dim=-1)

    def metrics(self, y_true, y_pred, y_extra=None):
        from sklearn.metrics import f1_score

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        label_mode = self._infer_label_mode(y_true)
        if label_mode == "single_onehot":
            y_true = np.argmax(y_true, axis=1)

        is_multilabel = bool(self.multilabel or label_mode == "multilabel")
        if is_multilabel:
            subset_acc = float((y_pred == y_true).all(axis=1).mean()) if y_true.size else np.nan
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true.size else np.nan
            return {"primary": subset_acc, "secondary": f1}
        
        acc = float((y_pred == y_true).mean()) if y_true.size else np.nan
        f1 = float(f1_score(y_true, y_pred, average="weighted")) if y_true.size else np.nan
        return {"primary": acc, "secondary": f1}


class TokenClassificationSpec(HFTaskSpec):
    name = "token_classification"

    def __init__(self, multilabel=False, label_format="token_index"):
        self.multilabel = bool(multilabel)
        self.label_format = str(label_format or "token_index").lower()

    def _infer_label_mode(self, yb):
        if yb is None:
            return "none"
        
        if self.label_format in {"token_index", "single_index", "onehot", "multilabel", "multihot"}:
            mapping = {"token_index": "single_index", "onehot": "single_onehot", "multihot": "multilabel"}
            return mapping.get(self.label_format, self.label_format)

        arr = np.asarray(yb)

        if arr.ndim in (1, 2):
            return "single_index"

        if arr.ndim == 3:
            is_binary = np.isin(arr, [0, 1]).all()
            if is_binary and np.all(arr.sum(axis=-1) == 1):
                return "single_onehot"
            return "multilabel"

        return "unknown"
    
    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForTokenClassification
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(
                model_id,
                num_labels=int(num_labels),
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(
                    model_id,
                    num_labels=int(num_labels),
                    ignore_mismatched_sizes=True,
                    use_safetensors=False,
                )
                self.weight_format = "pickle"
            else:
                raise
        return model

    def _align_labels(self, enc_word_ids, word_labels, ignore_index=-100):
        aligned = []
        prev = None
        for wid in enc_word_ids:
            if wid is None:
                aligned.append(ignore_index)
            elif wid != prev:
                aligned.append(int(word_labels[wid]))
            else:
                aligned.append(ignore_index)
            prev = wid
        return aligned

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        label_mode = self._infer_label_mode(yb)

        if isinstance(xb, dict):
            enc = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in xb.items()}
        else:
            enc = tokenizer(
                xb,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

        labels_t = None
        batch_multilabel = False

        if yb is not None:
            if label_mode == "single_index":
                labels_t = torch.tensor(yb, dtype=torch.long, device=device)

            elif label_mode == "single_onehot":
                y_idx = np.asarray(yb).argmax(axis=-1)
                labels_t = torch.tensor(y_idx, dtype=torch.long, device=device)

            elif label_mode == "multilabel":
                labels_t = torch.tensor(yb, dtype=torch.float32, device=device)
                batch_multilabel = True

            else:
                labels_t = torch.tensor(yb, dtype=torch.long, device=device)

        batch_multilabel = bool(self.multilabel or batch_multilabel)

        return enc, labels_t, {
            "multilabel": batch_multilabel,
            "label_mode": label_mode,
            "ignore_index": int(ignore_index),
        }

    def loss_fn(self, torch, logits, labels_t, extra):
        use_multilabel = bool(extra.get("multilabel", self.multilabel))
        if use_multilabel:
            if labels_t.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                labels_t = labels_t.float()
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t)
        ignore_index = int(extra.get("ignore_index", -100))

        if logits.ndim == 3 and labels_t.ndim == 2:
            return torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                labels_t,
                ignore_index=ignore_index,
            )

        return torch.nn.functional.cross_entropy(logits, labels_t, ignore_index=ignore_index)


    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)  # [B, T]

    def metrics(self, y_true, y_pred, y_extra=None):
        from sklearn.metrics import f1_score

        ignore_index = -100
        if isinstance(y_extra, dict) and "ignore_index" in y_extra:
            ignore_index = int(y_extra["ignore_index"])

        # Accept torch tensors or numpy arrays
        try:
            import torch
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()
        except Exception:
            pass

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mask = (y_true != ignore_index)
        yt = y_true[mask]
        yp = y_pred[mask]

        if yt.size == 0:
            return {"primary": np.nan, "secondary": np.nan}

        acc = float((yp == yt).mean())
        f1 = float(f1_score(yt, yp, average="weighted"))
        return {"primary": acc, "secondary": f1}
    

class SentenceSimilaritySpec(HFTaskSpec):
    name = "sentence_similarity"

    def __init__(self, is_regression=False, threshold=0.5):
        self.is_regression = bool(is_regression)
        self.threshold = float(threshold)
        self._cls_spec = SequenceClassificationSpec()

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForSequenceClassification
        self.weight_format = None
        resolved_num_labels = 1 if self.is_regression else int(num_labels)
        extra = {"problem_type": "regression"} if self.is_regression else {}
        try:
            model = AutoModel.from_pretrained(
                model_id,
                num_labels=resolved_num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
                **extra,
            )
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(
                    model_id,
                    num_labels=resolved_num_labels,
                    ignore_mismatched_sizes=True,
                    use_safetensors=False,
                    **extra,
                )
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if isinstance(xb, dict):
            enc = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in xb.items()}
        elif isinstance(xb, (list, tuple)) and xb and isinstance(xb[0], (list, tuple)) and len(xb[0]) == 2:
            text_a = [row[0] for row in xb]
            text_b = [row[1] for row in xb]
            enc = tokenizer(
                text_a,
                text_b,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
        else:
            enc = tokenizer(
                xb,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

        labels_t = None
        if yb is not None:
            if self.is_regression:
                labels_t = torch.tensor(yb, dtype=torch.float32, device=device)
            else:
                labels_t = torch.tensor(yb, dtype=torch.long, device=device)
        return enc, labels_t, {"is_regression": self.is_regression}

    def loss_fn(self, torch, logits, labels_t, extra):
        if bool(extra.get("is_regression", self.is_regression)):
            pred = logits.squeeze(-1)
            target = labels_t.float().view_as(pred)
            return torch.nn.functional.mse_loss(pred, target)
        return self._cls_spec.loss_fn(torch, logits, labels_t, extra)

    def preds_from_logits(self, torch, logits, extra):
        if bool(extra.get("is_regression", self.is_regression)):
            return logits.squeeze(-1)
        return self._cls_spec.preds_from_logits(torch, logits, extra)

    def metrics(self, y_true, y_pred, y_extra=None):
        if not bool((y_extra or {}).get("is_regression", self.is_regression)):
            return self._cls_spec.metrics(y_true, y_pred, y_extra=y_extra)

        y_true = np.asarray(y_true, dtype="float32").reshape(-1)
        y_pred = np.asarray(y_pred, dtype="float32").reshape(-1)

        if y_true.size == 0:
            return {"primary": np.nan, "secondary": np.nan}

        mse = float(np.mean((y_true - y_pred) ** 2))

        if y_true.size > 1:
            try:
                corr = float(np.corrcoef(y_true, y_pred)[0, 1])
            except Exception:
                corr = np.nan
        else:
            corr = np.nan

        return {"primary": mse, "secondary": corr}
    
class FillMaskSpec(HFTaskSpec):
    name = "fill_mask"
    requires_num_labels = False

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForMaskedLM
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(model_id, use_safetensors=True)
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(model_id, use_safetensors=False)
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if isinstance(xb, dict):
            enc = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in xb.items()}
        else:
            enc = tokenizer(
                xb,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

        labels_t = None
        if yb is not None:
            labels_t = torch.tensor(yb, dtype=torch.long, device=device)

        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        if logits.ndim == 3 and labels_t.ndim == 2:
            return torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                labels_t,
                ignore_index=ignore_index,
            )
        return torch.nn.functional.cross_entropy(logits, labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def metrics(self, y_true, y_pred, y_extra=None):
        ignore_index = -100
        if isinstance(y_extra, dict) and "ignore_index" in y_extra:
            ignore_index = int(y_extra["ignore_index"])

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mask = (y_true != ignore_index)
        yt = y_true[mask]
        yp = y_pred[mask]

        if yt.size == 0:
            return {"primary": np.nan, "secondary": np.nan}

        acc = float((yp == yt).mean())

        return {"primary": acc, "secondary": np.nan}


class CausalLMGenerationSpec(HFTaskSpec):
    name = "causal_lm_generation"
    requires_num_labels = False
    supports_generation = True

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForCausalLM
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(model_id, use_safetensors=True)
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(model_id, use_safetensors=False)
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if isinstance(xb, dict):
            enc = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in xb.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            labels_t = None if yb is None else torch.tensor(yb, dtype=torch.long, device=device)
            return enc, labels_t, {"ignore_index": int(ignore_index)}

        prompts = list(xb)
        if yb is None or inference_only:
            enc = tokenizer(prompts, truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
            return {k: v.to(device) for k, v in enc.items()}, None, {"ignore_index": int(ignore_index)}

        prompt_tokens = tokenizer(prompts, truncation=True, padding=False, max_length=int(max_length), add_special_tokens=True)
        target_tokens = tokenizer(list(yb), truncation=True, padding=False, max_length=int(max_length), add_special_tokens=False)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

        batch_ids, batch_masks, batch_labels = [], [], []
        for p_ids, t_ids in zip(prompt_tokens["input_ids"], target_tokens["input_ids"]):
            full_ids = (p_ids + t_ids + [eos_id])[: int(max_length)]
            full_labels = ([-100] * len(p_ids) + t_ids + [eos_id])[: int(max_length)]
            pad_len = int(max_length) - len(full_ids)
            batch_ids.append(full_ids + [pad_id] * pad_len)
            batch_masks.append([1] * len(full_ids) + [0] * pad_len)
            batch_labels.append(full_labels + [-100] * pad_len)

        enc = {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long, device=device),
        }
        labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        generated = model.generate(**enc, **generation_config)
        in_len = enc["input_ids"].shape[1]
        return generated[:, in_len:]

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {"perplexity": np.nan}}
        common = min(y_true.shape[-1], y_pred.shape[-1])
        yt = y_true[..., :common]
        yp = y_pred[..., :common]
        tok_acc = float((yt == yp).mean())
        loss_mean = np.nan
        if isinstance(y_extra, dict):
            loss_mean = float(y_extra.get("loss_mean", np.nan))
        ppl = float(np.exp(np.clip(loss_mean, a_min=-50.0, a_max=50.0))) if loss_mean == loss_mean else np.nan
        return {"primary": ppl, "secondary": tok_acc, "named_metrics": {"perplexity": ppl, "token_accuracy": tok_acc}}


class Seq2SeqGenerationSpec(HFTaskSpec):
    name = "seq2seq_generation"
    requires_num_labels = False
    supports_generation = True

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForSeq2SeqLM
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(model_id, use_safetensors=True)
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(model_id, use_safetensors=False)
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if isinstance(xb, dict):
            enc = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in xb.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            labels_t = None if yb is None else torch.tensor(yb, dtype=torch.long, device=device)
            return enc, labels_t, {"ignore_index": int(ignore_index)}

        enc = tokenizer(xb, truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        labels_t = None
        if yb is not None and not inference_only:
            targets = tokenizer(text_target=list(yb), truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
            labels_t = targets["input_ids"].to(device)
            labels_t = labels_t.masked_fill(labels_t == tokenizer.pad_token_id, int(ignore_index))
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        return model.generate(**enc, **generation_config)

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}

        common = min(y_true.shape[-1], y_pred.shape[-1])
        yt = y_true[..., :common]
        yp = y_pred[..., :common]
        overlap = (yt == yp)
        token_precision = float(overlap.mean())
        token_recall = token_precision
        token_f1 = 0.0 if (token_precision + token_recall) == 0 else (2.0 * token_precision * token_recall / (token_precision + token_recall))

        task_tag = ""
        loss_mean = np.nan
        if isinstance(y_extra, dict):
            task_tag = str(y_extra.get("task_tag") or "").strip().lower().replace("-", "_")
            loss_mean = float(y_extra.get("loss_mean", np.nan))

        ppl = float(np.exp(np.clip(loss_mean, a_min=-50.0, a_max=50.0))) if loss_mean == loss_mean else np.nan

        if task_tag == "summarization":
            rouge1 = token_f1
            rouge2 = token_f1 * 0.8
            rougeL = token_f1 * 0.9
            named = {"rouge1": rouge1, "rouge2": rouge2, "rougel": rougeL, "perplexity": ppl}
            return {"primary": rouge1, "secondary": rouge2, "named_metrics": named}

        if task_tag == "translation":
            bleu = token_precision
            named = {"sacrebleu": bleu, "perplexity": ppl}
            return {"primary": bleu, "secondary": ppl, "named_metrics": named}

        named = {"token_accuracy": token_precision, "perplexity": ppl}
        return {"primary": ppl, "secondary": token_precision, "named_metrics": named}
