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

    def build_model(self, transformers, model_id, num_labels):
        raise NotImplementedError

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100):
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


class SequenceClassificationSpec(HFTaskSpec):
    name = "sequence_classification"

    def __init__(self, multilabel=False, threshold=0.5):
        self.multilabel = bool(multilabel)
        self.threshold = float(threshold)
    
    def _infer_label_mode(self, yb):
        if yb is None:
            return "none"

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

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100):
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
        return enc, labels_t, {"multilabel": batch_multilabel, "label_mode": label_mode}

    def loss_fn(self, torch, logits, labels_t, extra):
        label_mode = extra.get("label_mode", "unknown")
        if self._is_multilabel_mode(label_mode, extra):
            if labels_t.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                labels_t = labels_t.float()
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t)

        if label_mode == "single_onehot":
            labels_t = torch.argmax(labels_t, dim=-1)
        return torch.nn.functional.cross_entropy(logits, labels_t)

    def preds_from_logits(self, torch, logits, extra):
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

    def __init__(self, multilabel=False):
        self.multilabel = bool(multilabel)

    def _infer_label_mode(self, yb):
        if yb is None:
            return "none"

        arr = np.asarray(yb)

        # Most token-label batches are [B, T] integer class indices.
        if arr.ndim in (1, 2):
            return "single_index"

        # Optional path where labels are [B, T, C] with one-hot / multi-hot encoding.
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

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100):
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

        return enc, labels_t, {"multilabel": batch_multilabel, "label_mode": label_mode}

    def loss_fn(self, torch, logits, labels_t, extra):
        use_multilabel = bool(extra.get("multilabel", self.multilabel))
        if use_multilabel:
            if labels_t.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                labels_t = labels_t.float()
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t)
        return torch.nn.functional.cross_entropy(logits, labels_t)

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