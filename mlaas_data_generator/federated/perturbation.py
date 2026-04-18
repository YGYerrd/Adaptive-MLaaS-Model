from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PredictionProbe:
    prediction: Any
    confidence: float
    output_value: float | None = None


def run_perturbation_stage(
    model,
    x_eval,
    y_eval=None,
    *,
    task_family: str | None = None,
    hf_task: str | None = None,
    config: dict | None = None,
    meta: dict | None = None,
    client_id: str | None = None,
    round_idx: int | None = None,
) -> dict:
    """Run a small post-evaluation perturbation probe and return aggregate metrics.

    The stage is deliberately best-effort. It never mutates model weights and
    returns a compact error metric instead of failing the client round.
    """
    config = config or {}
    if not _enabled(config):
        return {"perturbation_enabled_flag": False}

    start = time.time()
    sample_count = _sample_count(x_eval, y_eval)
    sample_limit = max(0, int(config.get("perturbation_sample_count", 1) or 0))
    if sample_count <= 0 or sample_limit <= 0:
        return {
            "perturbation_enabled_flag": True,
            "perturbation_supported_flag": False,
            "perturbation_sample_count": 0,
            "perturbation_error": "no_eval_samples",
        }

    seed = _stable_seed(config.get("seed", 42), client_id, round_idx, task_family, hf_task)
    rng = np.random.default_rng(seed)
    selected = _select_indices(sample_count, min(sample_limit, sample_count), rng)

    target_units = max(1, int(config.get("perturbation_target_units", 1) or 1))
    candidate_limit = max(1, int(config.get("perturbation_candidate_units", 4) or 4))
    trust_trials = max(1, int(config.get("perturbation_trust_trials", 2) or 2))
    strength = float(config.get("perturbation_random_strength", 0.02) or 0.02)

    per_sample = []
    errors = []
    for idx in selected:
        try:
            x_sample = _get_sample(x_eval, idx)
            y_sample = _get_sample(y_eval, idx) if y_eval is not None else None
            units = _meaningful_units(x_sample, y_sample, task_family=task_family, meta=meta, limit=candidate_limit)
            if not units:
                continue

            baseline = _predict_probe(model, x_sample, y_sample, task_family=task_family, hf_task=hf_task)
            if baseline is None:
                continue

            scored_units = []
            for unit in units:
                perturbed = _apply_targeted_mask(
                    x_sample,
                    [unit],
                    task_family=task_family,
                    model=model,
                    meta=meta,
                )
                candidate = _predict_probe(model, perturbed, y_sample, task_family=task_family, hf_task=hf_task)
                if candidate is None:
                    continue
                scored_units.append(
                    (
                        _confidence_drop(baseline, candidate),
                        _prediction_changed(baseline.prediction, candidate.prediction),
                        unit,
                    )
                )

            if not scored_units:
                continue

            scored_units.sort(key=lambda item: (item[0], item[1]), reverse=True)
            chosen_units = [unit for _, _, unit in scored_units[: min(target_units, len(scored_units))]]
            targeted_x = _apply_targeted_mask(
                x_sample,
                chosen_units,
                task_family=task_family,
                model=model,
                meta=meta,
            )
            targeted = _predict_probe(model, targeted_x, y_sample, task_family=task_family, hf_task=hf_task)
            if targeted is None:
                continue

            trust_changes = []
            trust_same = []
            trust_output_deltas = []
            for _ in range(trust_trials):
                random_x = _apply_benign_perturbation(
                    x_sample,
                    rng,
                    task_family=task_family,
                    model=model,
                    meta=meta,
                    strength=strength,
                )
                random_probe = _predict_probe(model, random_x, y_sample, task_family=task_family, hf_task=hf_task)
                if random_probe is None:
                    continue
                trust_changes.append(abs(_confidence_drop(baseline, random_probe)))
                trust_same.append(0.0 if _prediction_changed(baseline.prediction, random_probe.prediction) else 1.0)
                if baseline.output_value is not None and random_probe.output_value is not None:
                    denom = max(abs(float(baseline.output_value)), 1e-9)
                    trust_output_deltas.append(abs(float(random_probe.output_value) - float(baseline.output_value)) / denom)

            if not trust_same and not trust_changes and not trust_output_deltas:
                continue

            targeted_drop = _confidence_drop(baseline, targeted)
            targeted_changed = _prediction_changed(baseline.prediction, targeted.prediction)
            unit_fraction = float(len(chosen_units) / max(1, len(units)))
            baseline_conf = _finite_or_none(baseline.confidence)
            relative_drop = (
                max(0.0, targeted_drop) / max(float(baseline_conf), 1e-9)
                if baseline_conf is not None
                else 0.0
            )
            output_delta = None
            if baseline.output_value is not None and targeted.output_value is not None:
                denom = max(abs(float(baseline.output_value)), 1e-9)
                output_delta = abs(float(targeted.output_value) - float(baseline.output_value)) / denom

            per_sample.append(
                {
                    "sample_index": int(idx),
                    "baseline_prediction": _jsonable_prediction(baseline.prediction),
                    "baseline_confidence": baseline_conf,
                    "targeted_confidence": _finite_or_none(targeted.confidence),
                    "targeted_confidence_drop": _finite_or_none(targeted_drop),
                    "targeted_prediction_changed": bool(targeted_changed),
                    "targeted_unit_fraction": unit_fraction,
                    "targeted_relative_drop": float(relative_drop),
                    "targeted_output_relative_delta": _finite_or_none(output_delta),
                    "trust_confidence_abs_delta_mean": _mean_or_nan(trust_changes),
                    "trust_prediction_same_rate": _mean_or_nan(trust_same),
                    "trust_output_relative_delta_mean": _mean_or_nan(trust_output_deltas),
                }
            )
        except Exception as exc:
            errors.append(type(exc).__name__)
            continue

    if not per_sample:
        reason = "unsupported_prediction_probe"
        if errors:
            reason = f"{reason}:{errors[0]}"
        return {
            "perturbation_enabled_flag": True,
            "perturbation_supported_flag": False,
            "perturbation_sample_count": 0,
            "perturbation_error": reason,
            "perturbation_duration_s": float(time.time() - start),
        }

    targeted_drops = _values(per_sample, "targeted_confidence_drop")
    targeted_changes = [1.0 if item["targeted_prediction_changed"] else 0.0 for item in per_sample]
    relative_drops = _values(per_sample, "targeted_relative_drop")
    unit_fractions = _values(per_sample, "targeted_unit_fraction")
    output_deltas = _values(per_sample, "targeted_output_relative_delta")
    trust_conf_delta = _values(per_sample, "trust_confidence_abs_delta_mean")
    trust_same = _values(per_sample, "trust_prediction_same_rate")
    trust_output_delta = _values(per_sample, "trust_output_relative_delta_mean")
    per_sample_explainability_scores = [_sample_explainability_score(item) for item in per_sample]
    per_sample_trust_scores = [_sample_trust_score(item) for item in per_sample]

    compactness = 1.0 - min(1.0, _mean_or_nan(unit_fractions) if unit_fractions else 1.0)
    explainability_signal = _mean_or_nan(relative_drops)
    if math.isnan(explainability_signal) or explainability_signal == 0.0:
        explainability_signal = _mean_or_nan(output_deltas)
    if math.isnan(explainability_signal):
        explainability_signal = 0.0
    explainability_score = float(np.clip(explainability_signal * max(0.0, compactness), 0.0, 1.0))
    per_sample_explainability_score_mean = _mean_or_nan(per_sample_explainability_scores)
    if not math.isnan(per_sample_explainability_score_mean):
        explainability_score = per_sample_explainability_score_mean

    confidence_stability = 1.0 - _mean_or_nan(trust_conf_delta)
    if math.isnan(confidence_stability):
        confidence_stability = 1.0 - _mean_or_nan(trust_output_delta)
    if math.isnan(confidence_stability):
        confidence_stability = 0.0
    confidence_stability = float(np.clip(confidence_stability, 0.0, 1.0))
    prediction_stability = _mean_or_nan(trust_same)
    if math.isnan(prediction_stability):
        prediction_stability = 0.0
    trust_score = float(np.clip((confidence_stability + prediction_stability) / 2.0, 0.0, 1.0))
    per_sample_trust_score_mean = _mean_or_nan(per_sample_trust_scores)
    if not math.isnan(per_sample_trust_score_mean):
        trust_score = per_sample_trust_score_mean

    return {
        "perturbation_enabled_flag": True,
        "perturbation_supported_flag": True,
        "perturbation_sample_count": int(len(per_sample)),
        "perturbation_baseline_confidence_mean": _mean_or_nan(_values(per_sample, "baseline_confidence")),
        "explainability_confidence_drop_mean": _mean_or_nan(targeted_drops),
        "explainability_confidence_drop_std": _std_or_nan(targeted_drops),
        "explainability_confidence_drop_p50": _percentile_or_nan(targeted_drops, 50),
        "explainability_confidence_drop_p10": _percentile_or_nan(targeted_drops, 10),
        "explainability_confidence_drop_p90": _percentile_or_nan(targeted_drops, 90),
        "explainability_prediction_change_rate": _mean_or_nan(targeted_changes),
        "explainability_unit_fraction_mean": _mean_or_nan(unit_fractions),
        "explainability_unit_fraction_p95": _percentile_or_nan(unit_fractions, 95),
        "explainability_score": explainability_score,
        "explainability_score_p10": _percentile_or_nan(per_sample_explainability_scores, 10),
        "trust_confidence_delta_mean": _mean_or_nan(trust_conf_delta),
        "trust_confidence_delta_std": _std_or_nan(trust_conf_delta),
        "trust_confidence_delta_p95": _percentile_or_nan(trust_conf_delta, 95),
        "trust_confidence_delta_max": _max_or_nan(trust_conf_delta),
        "trust_prediction_stability": float(np.clip(prediction_stability, 0.0, 1.0)),
        "trust_prediction_stability_min": _min_or_nan(trust_same),
        "trust_confidence_stability": confidence_stability,
        "trust_score": trust_score,
        "trust_score_p05": _percentile_or_nan(per_sample_trust_scores, 5),
        "trust_score_min": _min_or_nan(per_sample_trust_scores),
        "perturbation_duration_s": float(time.time() - start),
        "perturbation_samples": per_sample,
    }


def _enabled(config):
    value = config.get("enable_perturbation_metrics", config.get("perturbation_enabled", True))
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _stable_seed(*parts):
    text = ":".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _select_indices(count, limit, rng):
    if limit >= count:
        return list(range(count))
    return [int(i) for i in rng.choice(count, size=limit, replace=False)]


def _sample_count(x, y=None):
    source = y if y is not None else x
    if isinstance(source, dict):
        for value in source.values():
            try:
                return int(len(value))
            except Exception:
                continue
        return 0
    try:
        return int(len(source))
    except Exception:
        return 0


def _get_sample(x, idx):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _get_sample(v, idx) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[int(idx)]
    if isinstance(x, (list, tuple)):
        return x[int(idx)]
    return x


def _batchify_x(sample):
    if isinstance(sample, dict):
        return {k: _batchify_value(v) for k, v in sample.items()}
    if isinstance(sample, str):
        return [sample]
    if isinstance(sample, (list, tuple)) and sample and isinstance(sample[0], str):
        return [list(sample)]
    return _batchify_value(sample)


def _batchify_y(sample):
    if sample is None:
        return None
    return _batchify_value(sample)


def _batchify_value(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    return np.expand_dims(arr, axis=0)


def _predict_probe(model, x_sample, y_sample=None, *, task_family=None, hf_task=None):
    if hasattr(model, "core"):
        return _predict_hf_probe(model, x_sample, y_sample, task_family=task_family, hf_task=hf_task)
    return _predict_generic_probe(model, x_sample, task_family=task_family)


def _predict_generic_probe(model, x_sample, *, task_family=None):
    if not hasattr(model, "predict"):
        return None
    x_batch = _batchify_x(x_sample)
    try:
        try:
            raw = model.predict(x_batch, verbose=0)
        except TypeError:
            raw = model.predict(x_batch)
    except Exception:
        return None
    return _probe_from_array(raw, task_family=task_family)


def _predict_hf_probe(adapter, x_sample, y_sample=None, *, task_family=None, hf_task=None):
    core = getattr(adapter, "core", None)
    if core is None or getattr(core, "model", None) is None:
        return None
    torch = core.torch
    core.model.eval()
    xb = _batchify_x(x_sample)
    yb = _batchify_y(y_sample)
    with torch.no_grad():
        enc, labels_t, extra = core.task_spec.encode_batch(
            core.tokenizer,
            xb,
            yb,
            core.max_length,
            torch,
            core.device,
            ignore_index=core.label_pad_value,
            inference_only=True,
        )
        if bool(getattr(core.task_spec, "supports_generation", False)):
            pred_t = core.task_spec.generate_predictions(
                core.model,
                enc,
                core.tokenizer,
                torch,
                core.generation_config,
            )
            confidence = np.nan
            if labels_t is not None:
                try:
                    teacher_inputs = core.task_spec.build_forward_inputs(enc, labels_t=labels_t, inference_only=False)
                    outputs = core.model(**teacher_inputs)
                    logits = _extract_hf_logits(core, outputs)
                    loss = core.task_spec.extract_loss(torch, outputs, logits, labels_t, extra)
                    if loss is not None:
                        confidence = float(np.exp(-float(loss.detach().cpu().item())))
                except Exception:
                    confidence = np.nan
            return PredictionProbe(
                prediction=_tensor_prediction(pred_t),
                confidence=confidence,
                output_value=None,
            )

        model_inputs = core.task_spec.build_forward_inputs(enc, labels_t=None, inference_only=True)
        outputs = core.model(**model_inputs)
        logits = _extract_hf_logits(core, outputs)
        pred_t = core.task_spec.preds_from_logits(torch, logits, extra)
        return _probe_from_hf_logits(torch, logits, pred_t, enc, task_family=task_family, hf_task=hf_task)


def _extract_hf_logits(core, outputs):
    extract_fn = getattr(core, "_extract_logits", None)
    if callable(extract_fn):
        return extract_fn(outputs)
    task_spec = getattr(core, "task_spec", None)
    extract_fn = getattr(task_spec, "extract_logits", None)
    if callable(extract_fn):
        return extract_fn(outputs)
    return outputs.logits


def _probe_from_array(raw, *, task_family=None):
    arr = np.asarray(raw)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        value = float(arr)
        return PredictionProbe(prediction=value, confidence=np.nan, output_value=value)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        sample = arr[0]
    else:
        sample = arr
    if sample.ndim == 0:
        value = float(sample)
        return PredictionProbe(prediction=value, confidence=np.nan, output_value=value)
    if sample.ndim == 1 and sample.size == 1 and (task_family or "") != "regression":
        prob = float(sample.reshape(-1)[0])
        if 0.0 <= prob <= 1.0:
            pred = int(prob >= 0.5)
            conf = max(prob, 1.0 - prob)
            return PredictionProbe(prediction=pred, confidence=float(conf), output_value=None)
    if sample.ndim == 1 and sample.size > 1 and (task_family or "") != "regression":
        probs = _as_probability_vector(sample.astype("float64"))
        pred = int(np.argmax(probs))
        return PredictionProbe(prediction=pred, confidence=float(np.max(probs)), output_value=None)
    if sample.ndim >= 2 and sample.shape[-1] > 1 and (task_family or "") != "regression":
        probs = _as_probability_vector(sample.reshape(-1, sample.shape[-1]).astype("float64"))
        pred = tuple(np.argmax(probs, axis=-1).astype(int).tolist()[:128])
        conf = float(np.mean(np.max(probs, axis=-1)))
        return PredictionProbe(prediction=pred, confidence=conf, output_value=None)
    value = float(np.asarray(sample).reshape(-1)[0])
    return PredictionProbe(prediction=value, confidence=np.nan, output_value=value)


def _probe_from_hf_logits(torch, logits, pred_t, enc, *, task_family=None, hf_task=None):
    logits_cpu = logits.detach().cpu()
    arr = logits_cpu.numpy()
    pred = _tensor_prediction(pred_t)
    if arr.ndim == 2:
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return PredictionProbe(
            prediction=int(np.argmax(probs[0])),
            confidence=float(np.max(probs[0])),
            output_value=None,
        )
    if arr.ndim == 3:
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        mask = None
        try:
            if isinstance(enc, dict) and "attention_mask" in enc:
                mask = enc["attention_mask"].detach().cpu().numpy()[0].astype(bool)
        except Exception:
            mask = None
        if mask is not None and mask.shape[0] == probs.shape[0]:
            probs_used = probs[mask]
        else:
            probs_used = probs.reshape(-1, probs.shape[-1])
        if probs_used.size == 0:
            probs_used = probs.reshape(-1, probs.shape[-1])
        conf = float(np.mean(np.max(probs_used, axis=-1)))
        return PredictionProbe(prediction=pred, confidence=conf, output_value=None)
    if arr.ndim == 4:
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        max_probs = np.max(probs, axis=0)
        dominant = int(np.argmax(np.bincount(np.argmax(probs, axis=0).reshape(-1))))
        return PredictionProbe(prediction=dominant, confidence=float(np.mean(max_probs)), output_value=None)
    return PredictionProbe(prediction=pred, confidence=np.nan, output_value=None)


def _as_probability_vector(values):
    arr = np.asarray(values, dtype="float64")
    if arr.ndim == 1:
        if np.all(arr >= 0.0) and np.all(arr <= 1.0) and np.isclose(float(np.sum(arr)), 1.0, atol=1e-3):
            return arr
        shifted = arr - np.max(arr)
        exp = np.exp(np.clip(shifted, -50.0, 50.0))
        return exp / max(float(np.sum(exp)), 1e-12)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)


def _tensor_prediction(value):
    try:
        arr = value.detach().cpu().numpy()
    except Exception:
        arr = np.asarray(value)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    flat = np.asarray(arr).reshape(-1)
    if flat.size == 1:
        try:
            return int(flat[0])
        except Exception:
            return float(flat[0])
    return tuple(int(v) for v in flat[:128])


def _meaningful_units(x_sample, y_sample=None, *, task_family=None, meta=None, limit=8):
    modality = _modality(x_sample, task_family=task_family, meta=meta)
    if modality == "tokens":
        positions = _token_positions(x_sample)
        if str(task_family or "") == "token_classification":
            spans = _entity_spans(y_sample, positions)
            if spans:
                return _limited(spans, limit)
        return _limited([("token", int(pos)) for pos in positions], limit)
    if modality == "text":
        words = str(x_sample).split()
        return _limited([("word", i) for i in range(len(words))], limit)
    if modality == "image":
        return _limited(_image_patches(x_sample), limit)
    if modality == "numeric":
        arr = np.asarray(x_sample)
        if arr.ndim == 0:
            return []
        flat_size = int(arr.size)
        return _limited([("feature", i) for i in range(flat_size)], limit)
    return []


def _limited(units, limit):
    units = list(units or [])
    if len(units) <= limit:
        return units
    positions = np.linspace(0, len(units) - 1, num=int(limit), dtype=int)
    return [units[int(i)] for i in positions]


def _entity_spans(y_sample, token_positions):
    if y_sample is None:
        return []
    try:
        labels = np.asarray(y_sample).reshape(-1)
    except Exception:
        return []
    valid_positions = [int(p) for p in token_positions if int(p) < labels.size]
    spans = []
    start = None
    prev = None
    for pos in valid_positions:
        label = labels[pos]
        try:
            label_int = int(label)
        except Exception:
            label_int = 0
        is_entity = label_int not in (0, -100)
        if is_entity and start is None:
            start = pos
        if (not is_entity or (prev is not None and pos != prev + 1)) and start is not None:
            end = prev + 1 if prev is not None else pos
            spans.append(("span", int(start), int(end)))
            start = pos if is_entity else None
        prev = pos
    if start is not None and prev is not None:
        spans.append(("span", int(start), int(prev + 1)))
    return spans


def _token_positions(x_sample):
    if not isinstance(x_sample, dict) or "input_ids" not in x_sample:
        return []
    input_ids = np.asarray(x_sample.get("input_ids")).reshape(-1)
    mask = x_sample.get("attention_mask")
    if mask is not None:
        try:
            keep = np.asarray(mask).reshape(-1).astype(bool)
        except Exception:
            keep = np.ones_like(input_ids, dtype=bool)
    else:
        keep = np.ones_like(input_ids, dtype=bool)
    positions = [i for i, keep_i in enumerate(keep) if keep_i]
    if len(positions) > 2:
        return positions[1:-1]
    return positions


def _image_patches(x_sample):
    arr = _image_array_view(x_sample)
    if arr is None or arr.ndim != 3:
        return []
    channel_first = arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3)
    h_axis = 1 if channel_first else 0
    w_axis = 2 if channel_first else 1
    height = int(arr.shape[h_axis])
    width = int(arr.shape[w_axis])
    grid = 3 if min(height, width) >= 12 else 2
    patches = []
    for gy in range(grid):
        for gx in range(grid):
            y0 = int(round(gy * height / grid))
            y1 = int(round((gy + 1) * height / grid))
            x0 = int(round(gx * width / grid))
            x1 = int(round((gx + 1) * width / grid))
            if y1 > y0 and x1 > x0:
                patches.append(("patch", y0, y1, x0, x1, bool(channel_first)))
    return patches


def _modality(x_sample, *, task_family=None, meta=None):
    if isinstance(x_sample, dict):
        if "input_ids" in x_sample:
            return "tokens"
        if "pixel_values" in x_sample:
            return "image"
    if isinstance(x_sample, str):
        return "text"
    if str(task_family or "") in {"detection", "segmentation"}:
        return "image"
    arr = None
    try:
        arr = np.asarray(x_sample)
    except Exception:
        arr = None
    if arr is not None and arr.ndim >= 3:
        return "image"
    if arr is not None and arr.ndim >= 1 and arr.dtype.kind in {"b", "i", "u", "f"}:
        return "numeric"
    return "unknown"


def _apply_targeted_mask(x_sample, units, *, task_family=None, model=None, meta=None):
    out = _copy_sample(x_sample)
    for unit in units:
        kind = unit[0] if unit else None
        if kind == "token":
            out = _mask_token(out, int(unit[1]), model=model)
        elif kind == "span":
            for pos in range(int(unit[1]), int(unit[2])):
                out = _mask_token(out, pos, model=model)
        elif kind == "word":
            words = str(out).split()
            idx = int(unit[1])
            if 0 <= idx < len(words):
                mask_token = _mask_token_text(model) or ""
                if mask_token:
                    words[idx] = mask_token
                else:
                    words.pop(idx)
                out = " ".join(words)
        elif kind == "patch":
            out = _mask_patch(out, unit)
        elif kind == "feature":
            out = _mask_feature(out, int(unit[1]))
    return out


def _apply_benign_perturbation(x_sample, rng, *, task_family=None, model=None, meta=None, strength=0.02):
    modality = _modality(x_sample, task_family=task_family, meta=meta)
    out = _copy_sample(x_sample)
    if modality == "image":
        return _image_noise(out, rng, strength=strength)
    if modality == "numeric":
        arr = np.asarray(out).astype("float32", copy=True)
        scale = float(np.nanstd(arr)) if arr.size else 0.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = max(float(np.nanmax(np.abs(arr))) if arr.size else 1.0, 1.0)
        return arr + rng.normal(0.0, max(1e-8, strength * scale), size=arr.shape).astype(arr.dtype)
    if modality == "tokens":
        positions = _token_positions(out)
        if not positions:
            return out
        pos = int(rng.choice(positions))
        return _mask_token(out, pos, model=model)
    if modality == "text":
        return " ".join(str(out).split())
    return out


def _copy_sample(sample):
    if isinstance(sample, dict):
        return {k: _copy_sample(v) for k, v in sample.items()}
    if isinstance(sample, np.ndarray):
        return np.array(sample, copy=True)
    if isinstance(sample, list):
        return list(sample)
    if isinstance(sample, tuple):
        return tuple(sample)
    return sample


def _mask_token(sample, pos, *, model=None):
    if not isinstance(sample, dict) or "input_ids" not in sample:
        return sample
    out = _copy_sample(sample)
    ids = np.asarray(out["input_ids"]).copy()
    flat = ids.reshape(-1)
    if 0 <= int(pos) < flat.size:
        token_id = _mask_token_id(model)
        if token_id is None:
            token_id = _pad_token_id(model)
        flat[int(pos)] = int(token_id if token_id is not None else 0)
        out["input_ids"] = flat.reshape(ids.shape)
    return out


def _mask_token_id(model):
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    value = getattr(tokenizer, "mask_token_id", None)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _pad_token_id(model):
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    value = getattr(tokenizer, "pad_token_id", None)
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _mask_token_text(model):
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    value = getattr(tokenizer, "mask_token", None)
    return str(value) if value else None


def _image_array_view(sample):
    if isinstance(sample, dict):
        value = sample.get("pixel_values")
        if value is None:
            return None
        return np.asarray(value)
    return np.asarray(sample)


def _set_image_array(sample, arr):
    if isinstance(sample, dict):
        out = _copy_sample(sample)
        out["pixel_values"] = arr
        return out
    return arr


def _mask_patch(sample, unit):
    arr = np.asarray(_image_array_view(sample)).copy()
    if arr.ndim != 3:
        return sample
    _, y0, y1, x0, x1, channel_first = unit
    fill = float(np.nanmean(arr)) if arr.size else 0.0
    if channel_first:
        arr[:, int(y0):int(y1), int(x0):int(x1)] = fill
    else:
        arr[int(y0):int(y1), int(x0):int(x1), :] = fill
    return _set_image_array(sample, arr)


def _image_noise(sample, rng, *, strength):
    arr = np.asarray(_image_array_view(sample)).astype("float32", copy=True)
    if arr.size == 0:
        return sample
    orig_min = float(np.nanmin(arr))
    orig_max = float(np.nanmax(arr))
    scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0 if orig_max <= 1.5 else 255.0
    arr = arr + rng.normal(0.0, max(1e-8, strength * scale), size=arr.shape).astype("float32")
    if orig_min >= 0.0:
        upper = 1.0 if orig_max <= 1.5 else 255.0
        arr = np.clip(arr, 0.0, upper)
    return _set_image_array(sample, arr)


def _mask_feature(sample, idx):
    arr = np.asarray(sample).copy()
    flat = arr.reshape(-1)
    if 0 <= int(idx) < flat.size:
        flat[int(idx)] = 0
    return flat.reshape(arr.shape)


def _confidence_drop(baseline: PredictionProbe, candidate: PredictionProbe):
    b = _finite_or_none(baseline.confidence)
    c = _finite_or_none(candidate.confidence)
    if b is not None and c is not None:
        return float(b - c)
    if baseline.output_value is not None and candidate.output_value is not None:
        denom = max(abs(float(baseline.output_value)), 1e-9)
        return float(abs(float(candidate.output_value) - float(baseline.output_value)) / denom)
    return 0.0


def _prediction_changed(a, b):
    try:
        arr_a = np.asarray(a)
        arr_b = np.asarray(b)
        if arr_a.shape != arr_b.shape:
            return True
        if arr_a.dtype.kind in {"f", "c"} or arr_b.dtype.kind in {"f", "c"}:
            return not bool(np.allclose(arr_a, arr_b, rtol=1e-4, atol=1e-6))
        return not bool(np.array_equal(arr_a, arr_b))
    except Exception:
        return a != b


def _finite_or_none(value):
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if np.isfinite(parsed) else None


def _finite_values(values):
    cleaned = [_finite_or_none(v) for v in (values or [])]
    return [float(v) for v in cleaned if v is not None]


def _mean_or_nan(values):
    cleaned = _finite_values(values)
    return float(np.mean(cleaned)) if cleaned else np.nan


def _std_or_nan(values):
    cleaned = _finite_values(values)
    return float(np.std(cleaned)) if cleaned else np.nan


def _percentile_or_nan(values, percentile):
    cleaned = _finite_values(values)
    return float(np.percentile(cleaned, percentile)) if cleaned else np.nan


def _min_or_nan(values):
    cleaned = _finite_values(values)
    return float(np.min(cleaned)) if cleaned else np.nan


def _max_or_nan(values):
    cleaned = _finite_values(values)
    return float(np.max(cleaned)) if cleaned else np.nan


def _sample_explainability_score(record):
    signal = _finite_or_none(record.get("targeted_relative_drop"))
    if signal is None or signal == 0.0:
        signal = _finite_or_none(record.get("targeted_output_relative_delta"))
    if signal is None:
        signal = 0.0
    unit_fraction = _finite_or_none(record.get("targeted_unit_fraction"))
    compactness = 1.0 - min(1.0, unit_fraction if unit_fraction is not None else 1.0)
    return float(np.clip(max(0.0, signal) * max(0.0, compactness), 0.0, 1.0))


def _sample_trust_score(record):
    confidence_delta = _finite_or_none(record.get("trust_confidence_abs_delta_mean"))
    if confidence_delta is None:
        confidence_delta = _finite_or_none(record.get("trust_output_relative_delta_mean"))
    confidence_stability = 1.0 - (confidence_delta if confidence_delta is not None else 1.0)
    confidence_stability = float(np.clip(confidence_stability, 0.0, 1.0))

    prediction_stability = _finite_or_none(record.get("trust_prediction_same_rate"))
    prediction_stability = float(np.clip(prediction_stability if prediction_stability is not None else 0.0, 0.0, 1.0))
    return float(np.clip((confidence_stability + prediction_stability) / 2.0, 0.0, 1.0))


def _values(records, key):
    return [item.get(key) for item in records if item.get(key) is not None]


def _jsonable_prediction(prediction):
    if isinstance(prediction, tuple):
        return list(prediction)
    if isinstance(prediction, np.ndarray):
        return prediction.tolist()
    if isinstance(prediction, np.integer):
        return int(prediction)
    if isinstance(prediction, np.floating):
        return float(prediction)
    return prediction
