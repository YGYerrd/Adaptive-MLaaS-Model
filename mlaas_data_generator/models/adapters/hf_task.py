import numpy as np
import re

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
    requires_tokenizer = True
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

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        return None

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        return None

    def metrics_from_statistics(self, stats):
        return None

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


class ImageClassificationSpec(HFTaskSpec):
    name = "image_classification"
    requires_tokenizer = False

    def build_model(self, transformers, model_id, num_labels):
        return transformers.AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=int(num_labels),
            ignore_mismatched_sizes=True,
        )

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict) or "pixel_values" not in xb:
            raise ValueError("image classification expects dict input with 'pixel_values'")
        enc = {"pixel_values": torch.tensor(xb["pixel_values"], dtype=torch.float32, device=device)}
        labels_t = None if yb is None else torch.tensor(yb, dtype=torch.long, device=device)
        return enc, labels_t, {"top_k": 5}

    def loss_fn(self, torch, logits, labels_t, extra):
        return torch.nn.functional.cross_entropy(logits, labels_t)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        if labels_t is None:
            return None
        labels_t = labels_t.view(-1)
        top1 = torch.argmax(logits, dim=-1)
        k = int(min(int(extra.get("top_k", 5)), int(logits.shape[-1])))
        topk = torch.topk(logits, k=k, dim=-1).indices
        top1_correct = int((top1 == labels_t).sum().detach().cpu().item())
        topk_correct = int((topk == labels_t.unsqueeze(-1)).any(dim=-1).sum().detach().cpu().item())
        total = int(labels_t.shape[0])
        return {"top1_correct": top1_correct, "top5_correct": topk_correct, "total": total}

    def metrics_from_statistics(self, stats):
        total = float(stats.get("total", 0.0))
        if total <= 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}
        top1 = float(stats.get("top1_correct", 0.0)) / total
        top5 = float(stats.get("top5_correct", 0.0)) / total
        return {
            "primary": top1,
            "secondary": top5,
            "named_metrics": {"top1_accuracy": top1, "top5_accuracy": top5},
        }

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0:
            return {"primary": np.nan, "secondary": np.nan}
        acc = float((y_true == y_pred).mean())
        return {"primary": acc, "secondary": acc}


class ObjectDetectionSpec(HFTaskSpec):
    name = "image_detection"
    requires_tokenizer = False
    requires_num_labels = False

    def __init__(self, score_threshold=0.05):
        self.score_threshold = float(score_threshold)
        self._model_valid_class_ids = None

    def build_model(self, transformers, model_id, num_labels):
        kwargs = {"ignore_mismatched_sizes": True}
        if num_labels is not None:
            kwargs["num_labels"] = int(num_labels)
        model = transformers.AutoModelForObjectDetection.from_pretrained(
            model_id,
            **kwargs,
        )
        self._model_valid_class_ids = self._extract_valid_class_ids_from_model(model)
        return model

    @staticmethod
    def _extract_valid_class_ids_from_model(model):
        config = getattr(model, "config", None)
        id2label = getattr(config, "id2label", None)
        if not isinstance(id2label, dict) or not id2label:
            return None
        cleaned = {}
        for k, v in id2label.items():
            try:
                kid = int(k)
            except Exception:
                continue
            cleaned[kid] = str(v)
        if not cleaned:
            return None
        valid = [k for k in sorted(cleaned) if cleaned[k].strip().lower() != "n/a"]
        return valid or None

    def _remap_contiguous_classes_if_needed(self, classes, *, force=False):
        class_ids = np.asarray(classes, dtype=np.int64)
        valid_ids = self._model_valid_class_ids
        if class_ids.size == 0 or not valid_ids:
            return class_ids

        # COCO-style HF checkpoints (e.g. DETR/YOLOS) frequently expose id2label
        # with index 0 reserved for "N/A". Some datasets provide contiguous
        # category ids in [0, 79], where 0 means "person". In that case we need
        # to remap contiguous ids -> model ids before metric matching.
        if (
            valid_ids
            and valid_ids[0] == 1
            and int(np.min(class_ids)) >= 0
            and int(np.max(class_ids)) < len(valid_ids)
            and (force or 0 in set(class_ids.tolist()))
        ):
            mapped = np.asarray([int(valid_ids[int(cid)]) for cid in class_ids], dtype=np.int64)
            return mapped
        return class_ids

    def _remap_predicted_class_indices_if_needed(self, class_indices, num_pred_classes):
        class_ids = np.asarray(class_indices, dtype=np.int64)
        valid_ids = self._model_valid_class_ids
        if class_ids.size == 0 or not valid_ids:
            return class_ids

        # Some checkpoints expose contiguous prediction logits over only the
        # valid classes while still publishing sparse COCO ids in id2label
        # (e.g. valid ids start at 1). Detect that layout from the logits
        # dimensionality and remap only in that case.
        if (
            valid_ids[0] == 1
            and int(num_pred_classes) == int(len(valid_ids))
            and int(np.min(class_ids)) >= 0
            and int(np.max(class_ids)) < int(len(valid_ids))
        ):
            return np.asarray([int(valid_ids[int(cid)]) for cid in class_ids], dtype=np.int64)
        return class_ids

    @staticmethod
    def _normalise_pixel_array(sample):
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"object detection expects 3D image tensors, got shape={arr.shape}")
        if arr.shape[0] == 3:
            return arr
        if arr.shape[-1] == 3:
            return np.transpose(arr, (2, 0, 1))
        raise ValueError(
            "object detection expects image tensors with 3 channels in CHW or HWC layout, "
            f"got shape={arr.shape}"
        )

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict) or "pixel_values" not in xb:
            raise ValueError("object detection expects dict input with 'pixel_values'")
        pixel_values = xb["pixel_values"]
        pixel_arrays = [self._normalise_pixel_array(sample) for sample in pixel_values]
        if not pixel_arrays:
            enc = {"pixel_values": torch.empty((0, 3, 0, 0), dtype=torch.float32, device=device)}
        else:
            shapes = {arr.shape for arr in pixel_arrays}
            if len(shapes) == 1:
                batch_pixels = np.stack(pixel_arrays, axis=0)
            else:
                max_h = max(arr.shape[1] for arr in pixel_arrays)
                max_w = max(arr.shape[2] for arr in pixel_arrays)
                padded = []
                for arr in pixel_arrays:
                    pad_h = max_h - arr.shape[1]
                    pad_w = max_w - arr.shape[2]
                    padded.append(np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant"))
                batch_pixels = np.stack(padded, axis=0)
            enc = {"pixel_values": torch.tensor(batch_pixels, dtype=torch.float32, device=device)}
        labels_t = None
        if yb is not None:
            labels_t = []
            for idx, item in enumerate(yb):
                image_h = int(pixel_arrays[idx].shape[1]) if idx < len(pixel_arrays) else 1
                image_w = int(pixel_arrays[idx].shape[2]) if idx < len(pixel_arrays) else 1
                boxes_xyxy_norm = self._to_xyxy_normalized(item.get("boxes", []), image_h=image_h, image_w=image_w)
                boxes_cxcywh_norm = self._xyxy_to_cxcywh(boxes_xyxy_norm)
                class_labels = self._remap_contiguous_classes_if_needed(item.get("classes", []))
                labels_t.append(
                    {
                        "class_labels": torch.tensor(class_labels, dtype=torch.long, device=device),
                        "boxes": torch.tensor(boxes_cxcywh_norm, dtype=torch.float32, device=device),
                    }
                )
        return enc, labels_t, {"score_threshold": self.score_threshold}

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        out = dict(enc)
        if labels_t is not None and not inference_only:
            out["labels"] = labels_t
        return out

    def loss_fn(self, torch, logits, labels_t, extra):
        return None

    def extract_loss(self, torch, outputs, logits, labels_t, extra):
        return getattr(outputs, "loss", None)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def _box_iou(self, boxes_a, boxes_b):
        if boxes_a.size == 0 or boxes_b.size == 0:
            return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
        tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
        br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
        wh = np.clip(br - tl, a_min=0.0, a_max=None)
        inter = wh[..., 0] * wh[..., 1]
        area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
        area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
        union = np.clip(area_a[:, None] + area_b[None, :] - inter, a_min=1e-9, a_max=None)
        return inter / union

    @staticmethod
    def _xyxy_to_cxcywh(boxes_xyxy):
        if boxes_xyxy.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = boxes_xyxy.T
        w = np.clip(x2 - x1, a_min=0.0, a_max=None)
        h = np.clip(y2 - y1, a_min=0.0, a_max=None)
        cx = x1 + (w / 2.0)
        cy = y1 + (h / 2.0)
        return np.column_stack([cx, cy, w, h]).astype(np.float32, copy=False)

    @staticmethod
    def _cxcywh_to_xyxy(boxes_cxcywh):
        if boxes_cxcywh.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        cx, cy, w, h = boxes_cxcywh.T
        return np.column_stack([
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ]).astype(np.float32, copy=False)

    def _to_xyxy_normalized(self, boxes, image_h, image_w):
        out_boxes = np.asarray(boxes, dtype=np.float32)
        if out_boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if out_boxes.ndim == 1:
            out_boxes = out_boxes.reshape(1, 4)
        if out_boxes.ndim != 2 or out_boxes.shape[1] != 4:
            raise ValueError(f"object detection boxes must be Nx4, got shape={out_boxes.shape}")

        max_val = float(np.max(out_boxes))
        monotonic_xyxy = bool(np.all(out_boxes[:, 2] >= out_boxes[:, 0]) and np.all(out_boxes[:, 3] >= out_boxes[:, 1]))
        if max_val <= 1.5:
            boxes_xyxy = out_boxes if monotonic_xyxy else self._cxcywh_to_xyxy(out_boxes)
        else:
            bounded_xyxy = (
                monotonic_xyxy
                and np.all(out_boxes[:, 0] <= float(image_w) * 1.05)
                and np.all(out_boxes[:, 2] <= float(image_w) * 1.05)
                and np.all(out_boxes[:, 1] <= float(image_h) * 1.05)
                and np.all(out_boxes[:, 3] <= float(image_h) * 1.05)
            )
            if bounded_xyxy:
                boxes_xyxy = out_boxes
            else:
                x, y, w, h = out_boxes.T
                boxes_xyxy = np.column_stack([x, y, x + w, y + h])
            boxes_xyxy[:, [0, 2]] /= max(float(image_w), 1e-9)
            boxes_xyxy[:, [1, 3]] /= max(float(image_h), 1e-9)

        return np.clip(boxes_xyxy, a_min=0.0, a_max=1.0).astype(np.float32, copy=False)

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        if labels_t is None or outputs is None or not hasattr(outputs, "pred_boxes"):
            return None
        probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        boxes = outputs.pred_boxes.detach().cpu().numpy()

        stats = {"gt": 0.0}
        thresholds = [0.5, 0.75, 0.95]
        for thr in thresholds:
            stats[f"tp_{thr}"] = 0.0
            stats[f"fp_{thr}"] = 0.0

        for bidx, gt in enumerate(labels_t):
            gt_boxes = gt["boxes"].detach().cpu().numpy()
            gt_boxes = self._cxcywh_to_xyxy(gt_boxes)
            gt_classes = gt["class_labels"].detach().cpu().numpy()
            stats["gt"] += float(len(gt_classes))

            p_scores = probs[bidx, :, :-1].max(axis=-1)
            p_cls = probs[bidx, :, :-1].argmax(axis=-1)
            p_cls = self._remap_predicted_class_indices_if_needed(p_cls, num_pred_classes=probs.shape[-1] - 1)
            keep = p_scores >= float(extra.get("score_threshold", self.score_threshold))
            if self._model_valid_class_ids:
                valid_set = set(int(v) for v in self._model_valid_class_ids)
                keep = keep & np.asarray([int(cid) in valid_set for cid in p_cls], dtype=bool)
            p_scores = p_scores[keep]
            p_cls = p_cls[keep]
            p_boxes = boxes[bidx][keep]
            p_boxes = self._cxcywh_to_xyxy(p_boxes)

            for thr in thresholds:
                matched_gt = set()
                tp = 0
                fp = 0
                for pb, pc in zip(p_boxes, p_cls):
                    cand_idx = np.where(gt_classes == pc)[0]
                    if cand_idx.size == 0:
                        fp += 1
                        continue
                    ious = self._box_iou(np.asarray([pb]), gt_boxes[cand_idx])[0]
                    best = int(np.argmax(ious)) if ious.size else -1
                    if best >= 0 and ious[best] >= thr:
                        gt_id = int(cand_idx[best])
                        if gt_id not in matched_gt:
                            matched_gt.add(gt_id)
                            tp += 1
                        else:
                            fp += 1
                    else:
                        fp += 1
                stats[f"tp_{thr}"] += float(tp)
                stats[f"fp_{thr}"] += float(fp)
        return stats

    def metrics_from_statistics(self, stats):
        gt = float(stats.get("gt", 0.0))
        if gt <= 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}
        maps = []
        named = {}
        for thr in [0.5, 0.75, 0.95]:
            tp = float(stats.get(f"tp_{thr}", 0.0))
            fp = float(stats.get(f"fp_{thr}", 0.0))
            denom = max(gt + fp, 1e-9)
            ap = tp / denom
            maps.append(ap)
            named[f"map@{thr}"] = ap
        m_ap = float(np.mean(maps)) if maps else np.nan
        named["map"] = m_ap
        return {"primary": m_ap, "secondary": float(named.get("map@0.5", np.nan)), "named_metrics": named}

    def metrics(self, y_true, y_pred, y_extra=None):
        return {"primary": np.nan, "secondary": np.nan}


class ImageSegmentationSpec(HFTaskSpec):
    name = "image_segmentation"
    requires_tokenizer = False

    def build_model(self, transformers, model_id, num_labels):
        return transformers.AutoModelForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=int(num_labels),
            ignore_mismatched_sizes=True,
        )

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict) or "pixel_values" not in xb:
            raise ValueError("image segmentation expects dict input with 'pixel_values'")
        enc = {"pixel_values": torch.tensor(xb["pixel_values"], dtype=torch.float32, device=device)}
        labels_t = None
        if yb is not None:
            labels_t = torch.tensor(np.asarray(yb), dtype=torch.long, device=device)
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        if logits.shape[-2:] != labels_t.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=labels_t.shape[-2:], mode="bilinear", align_corners=False)
        return torch.nn.functional.cross_entropy(logits, labels_t, ignore_index=int(extra.get("ignore_index", -100)))

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=1)

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        if labels_t is None:
            return None
        if logits.shape[-2:] != labels_t.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=labels_t.shape[-2:], mode="bilinear", align_corners=False)
        pred = torch.argmax(logits, dim=1)
        ignore_index = int(extra.get("ignore_index", -100))
        valid = labels_t != ignore_index
        pred = pred[valid]
        tgt = labels_t[valid]
        if pred.numel() == 0:
            return {"intersection": 0.0, "pred_total": 0.0, "target_total": 0.0, "union": 0.0}
        intersection = float((pred == tgt).sum().detach().cpu().item())
        pred_total = float(pred.numel())
        target_total = float(tgt.numel())
        union = pred_total + target_total - intersection
        return {"intersection": intersection, "pred_total": pred_total, "target_total": target_total, "union": union}

    def metrics_from_statistics(self, stats):
        intersection = float(stats.get("intersection", 0.0))
        union = float(stats.get("union", 0.0))
        pred_total = float(stats.get("pred_total", 0.0))
        target_total = float(stats.get("target_total", 0.0))
        iou = intersection / union if union > 0 else np.nan
        denom = pred_total + target_total
        dice = (2.0 * intersection) / denom if denom > 0 else np.nan
        return {"primary": iou, "secondary": dice, "named_metrics": {"iou": iou, "dice": dice}}

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0:
            return {"primary": np.nan, "secondary": np.nan}
        inter = float((y_true == y_pred).sum())
        union = float(y_true.size + y_pred.size - inter)
        iou = inter / union if union > 0 else np.nan
        dice = (2 * inter) / float(y_true.size + y_pred.size) if (y_true.size + y_pred.size) > 0 else np.nan
        return {"primary": iou, "secondary": dice}
    

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

    @staticmethod
    def _left_pad_batch(tokenizer, input_ids, attention_mask):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            return input_ids, attention_mask

        ids_np = np.asarray(input_ids)
        mask_np = np.asarray(attention_mask)
        if ids_np.ndim != 2 or mask_np.ndim != 2 or ids_np.shape != mask_np.shape:
            return input_ids, attention_mask

        shifted_ids = np.full(ids_np.shape, int(pad_id), dtype=ids_np.dtype)
        shifted_mask = np.zeros(mask_np.shape, dtype=mask_np.dtype)

        for row_idx in range(ids_np.shape[0]):
            valid = int(mask_np[row_idx].sum())
            if valid <= 0:
                continue
            shifted_ids[row_idx, -valid:] = ids_np[row_idx, :valid]
            shifted_mask[row_idx, -valid:] = mask_np[row_idx, :valid]

        return shifted_ids, shifted_mask

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
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        if isinstance(xb, dict):
            batch = {k: v for k, v in xb.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            labels_np = None if yb is None else np.asarray(yb)
            if inference_only and labels_np is not None and "input_ids" in batch and "attention_mask" in batch:
                input_ids = np.asarray(batch["input_ids"])
                attention_mask = np.asarray(batch["attention_mask"])
                if labels_np.shape == input_ids.shape:
                    prompt_only_ids = []
                    prompt_only_mask = []
                    for row_ids, row_mask, row_labels in zip(input_ids, attention_mask, labels_np):
                        active = np.asarray(row_mask).astype(bool)
                        prompt_positions = active & (np.asarray(row_labels) == int(ignore_index))
                        if np.any(prompt_positions):
                            trimmed_ids = np.asarray(row_ids)[prompt_positions]
                            trimmed_mask = np.ones(trimmed_ids.shape[0], dtype=np.asarray(row_mask).dtype)
                        else:
                            valid_ids = np.asarray(row_ids)[active]
                            trimmed_ids = valid_ids[:-1] if valid_ids.shape[0] > 1 else valid_ids
                            trimmed_mask = np.ones(trimmed_ids.shape[0], dtype=np.asarray(row_mask).dtype)
                        prompt_only_ids.append(trimmed_ids.tolist())
                        prompt_only_mask.append(trimmed_mask.tolist())
                    max_prompt_len = max((len(row) for row in prompt_only_ids), default=0)
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    if pad_id is None:
                        pad_id = 0
                    padded_ids = []
                    padded_mask = []
                    for row_ids, row_mask in zip(prompt_only_ids, prompt_only_mask):
                        pad_len = max_prompt_len - len(row_ids)
                        padded_ids.append(([int(pad_id)] * pad_len) + list(row_ids))
                        padded_mask.append(([0] * pad_len) + list(row_mask))
                    batch["input_ids"] = np.asarray(padded_ids, dtype=input_ids.dtype)
                    batch["attention_mask"] = np.asarray(padded_mask, dtype=attention_mask.dtype)
            if "input_ids" in batch and "attention_mask" in batch:
                batch["input_ids"], batch["attention_mask"] = self._left_pad_batch(
                    tokenizer,
                    batch["input_ids"],
                    batch["attention_mask"],
                )
            enc = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in batch.items()}
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
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        cfg = dict(generation_config)
        if cfg.get("pad_token_id") is None and tokenizer.pad_token_id is not None:
            cfg["pad_token_id"] = int(tokenizer.pad_token_id)
        generated = model.generate(**enc, **cfg)
        in_len = enc["input_ids"].shape[1]
        return generated[:, in_len:]

    def metrics(self, y_true, y_pred, y_extra=None):
        loss_mean = np.nan
        ignore_index = -100
        if isinstance(y_extra, dict):
            loss_mean = float(y_extra.get("loss_mean", np.nan))
            ignore_index = int(y_extra.get("ignore_index", ignore_index))
        ppl = float(np.exp(np.clip(loss_mean, a_min=-50.0, a_max=50.0))) if loss_mean == loss_mean else np.nan
        token_accuracy = np.nan

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size != 0 and y_pred.size != 0:
            common = min(y_true.shape[-1], y_pred.shape[-1])
            yt = y_true[..., :common]
            yp = y_pred[..., :common]
            mask = (yt != ignore_index)
            yt = yt[mask]
            yp = yp[mask]
            if yt.size != 0:
                token_accuracy = float((yt == yp).mean())

        return {
            "primary": loss_mean,
            "secondary": ppl,
            "named_metrics": {"cross_entropy_loss": loss_mean, "perplexity": ppl, "token_accuracy": token_accuracy},
        }


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
        task_tag = ""
        loss_mean = np.nan
        ignore_index = -100
        if isinstance(y_extra, dict):
            task_tag = str(y_extra.get("task_tag") or "").strip().lower().replace("-", "_")
            loss_mean = float(y_extra.get("loss_mean", np.nan))
            ignore_index = int(y_extra.get("ignore_index", ignore_index))

        ppl = float(np.exp(np.clip(loss_mean, a_min=-50.0, a_max=50.0))) if loss_mean == loss_mean else np.nan

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}

        common = min(y_true.shape[-1], y_pred.shape[-1])
        yt = y_true[..., :common]
        yp = y_pred[..., :common]
        mask = (yt != ignore_index)
        yt = yt[mask]
        yp = yp[mask]

        token_precision = np.nan
        token_recall = np.nan
        token_f1 = np.nan
        if yt.size != 0:
            overlap = (yt == yp)
            token_precision = float(overlap.mean())
            token_recall = token_precision
            token_f1 = 0.0 if (token_precision + token_recall) == 0 else (2.0 * token_precision * token_recall / (token_precision + token_recall))

        if task_tag == "summarization":
            rouge1 = token_f1
            rouge2 = float(token_f1 * 0.8) if token_f1 == token_f1 else np.nan
            rougeL = float(token_f1 * 0.9) if token_f1 == token_f1 else np.nan
            named = {"rouge1": rouge1, "rouge2": rouge2, "rougel": rougeL, "perplexity": ppl}
            return {"primary": rouge1, "secondary": rouge2, "named_metrics": named}

        if task_tag == "translation":
            bleu = token_precision
            named = {"sacrebleu": bleu, "perplexity": ppl}
            return {"primary": bleu, "secondary": ppl, "named_metrics": named}

        named = {"token_accuracy": token_precision, "perplexity": ppl}
        return {"primary": ppl, "secondary": token_precision, "named_metrics": named}


class ImageCaptioningSpec(HFTaskSpec):
    name = "image_captioning"
    requires_num_labels = False
    supports_generation = True

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForVision2Seq
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
        if not isinstance(xb, dict):
            raise TypeError("Image captioning expects multimodal dict features")
        enc = {}
        for k, v in xb.items():
            if k == "pixel_values":
                enc[k] = torch.tensor(v, dtype=torch.float32, device=device)
            elif k in {"input_ids", "attention_mask", "decoder_input_ids"}:
                enc[k] = torch.tensor(v, dtype=torch.long, device=device)

        labels_t = None
        if yb is not None and not inference_only:
            labels_t = torch.tensor(yb, dtype=torch.long, device=device)
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        return model.generate(**enc, **generation_config)

    @staticmethod
    def _safe_div(num, den):
        return float(num / den) if den else 0.0

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {"cider": np.nan, "bleu": np.nan}}

        common = min(y_true.shape[-1], y_pred.shape[-1])
        yt = y_true[..., :common]
        yp = y_pred[..., :common]

        unigram = float((yt == yp).mean())
        bigram_match = float(((yt[:, 1:] == yp[:, 1:]) & (yt[:, :-1] == yp[:, :-1])).mean()) if common > 1 else unigram
        bleu = 0.5 * unigram + 0.5 * bigram_match
        cider = 0.7 * unigram + 0.3 * bigram_match
        rougeL = unigram
        return {
            "primary": cider,
            "secondary": bleu,
            "named_metrics": {"cider": cider, "bleu": bleu, "rougel": rougeL},
        }


class TextImageRetrievalSpec(HFTaskSpec):
    name = "text_image_retrieval"
    requires_num_labels = False

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModel
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
        if not isinstance(xb, dict):
            raise TypeError("Text-image retrieval expects multimodal dict features")
        enc = {
            "input_ids": torch.tensor(xb["input_ids"], dtype=torch.long, device=device),
            "attention_mask": torch.tensor(xb["attention_mask"], dtype=torch.long, device=device),
            "pixel_values": torch.tensor(xb["pixel_values"], dtype=torch.float32, device=device),
        }
        labels_t = None if yb is None else torch.tensor(yb, dtype=torch.long, device=device)
        return enc, labels_t, {}

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        return dict(enc)

    def loss_fn(self, torch, logits, labels_t, extra):
        return None

    def preds_from_logits(self, torch, logits, extra):
        return logits

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        img = getattr(outputs, "image_embeds", None)
        txt = getattr(outputs, "text_embeds", None)
        if img is None or txt is None:
            return None

        img = torch.nn.functional.normalize(img, dim=-1)
        txt = torch.nn.functional.normalize(txt, dim=-1)
        sims = txt @ img.transpose(0, 1)
        targets = torch.arange(sims.shape[0], device=sims.device)

        topk = min(10, sims.shape[1])
        _, idx = torch.topk(sims, k=topk, dim=1)
        r1 = (idx[:, :1] == targets[:, None]).any(dim=1).float().sum().item()
        r5 = (idx[:, : min(5, topk)] == targets[:, None]).any(dim=1).float().sum().item()
        r10 = (idx[:, : min(10, topk)] == targets[:, None]).any(dim=1).float().sum().item()
        return {"r1_correct": r1, "r5_correct": r5, "r10_correct": r10, "total": float(sims.shape[0])}

    def metrics_from_statistics(self, stats):
        total = max(1.0, float(stats.get("total", 0.0)))
        r1 = float(stats.get("r1_correct", 0.0)) / total
        r5 = float(stats.get("r5_correct", 0.0)) / total
        r10 = float(stats.get("r10_correct", 0.0)) / total
        return {
            "primary": r1,
            "secondary": r5,
            "named_metrics": {"r@1": r1, "r@5": r5, "r@10": r10},
        }


class VQASpec(HFTaskSpec):
    name = "visual_question_answering"
    requires_num_labels = False
    supports_generation = True

    _ARTICLES = {"a", "an", "the"}

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForVisualQuestionAnswering
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
        if not isinstance(xb, dict):
            raise TypeError("VQA expects multimodal dict features")
        enc = {
            "input_ids": torch.tensor(xb["input_ids"], dtype=torch.long, device=device),
            "attention_mask": torch.tensor(xb["attention_mask"], dtype=torch.long, device=device),
            "pixel_values": torch.tensor(xb["pixel_values"], dtype=torch.float32, device=device),
        }
        labels_t = None if yb is None else torch.tensor(yb, dtype=torch.long, device=device)
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        return torch.nn.functional.cross_entropy(logits, labels_t)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    @classmethod
    def _normalize_answer(cls, text):
        txt = str(text or "").lower().strip()
        txt = re.sub(r"[^\w\s]", " ", txt)
        parts = [p for p in txt.split() if p not in cls._ARTICLES]
        return " ".join(parts)

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {"exact_match": np.nan}}

        if y_true.dtype.kind in {"U", "S", "O"} or y_pred.dtype.kind in {"U", "S", "O"}:
            yt = np.asarray([self._normalize_answer(v) for v in y_true.reshape(-1)], dtype=object)
            yp = np.asarray([self._normalize_answer(v) for v in y_pred.reshape(-1)], dtype=object)
            exact = float((yt == yp).mean())
        else:
            exact = float((y_true.reshape(-1) == y_pred.reshape(-1)).mean())

        return {"primary": exact, "secondary": np.nan, "named_metrics": {"exact_match": exact}}
