from ..accounting import append_accounting_stage, finalize_accounting
import io
import os
import inspect
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def _is_pil_image(value):
    try:
        from PIL import Image
    except Exception:
        return False
    return isinstance(value, Image.Image)


def _to_numpy_rgb(image_like):
    if image_like is None:
        raise ValueError("image is None")

    if isinstance(image_like, np.ndarray):
        arr = image_like
    elif isinstance(image_like, dict):
        if "array" in image_like and image_like["array"] is not None:
            arr = np.asarray(image_like["array"])
        elif "bytes" in image_like and image_like["bytes"] is not None:
            data = image_like["bytes"]
            arr = _decode_bytes(data)
        elif "path" in image_like and image_like["path"]:
            arr = _decode_path(image_like["path"])
        else:
            raise ValueError("unsupported HF image dict payload")
    elif isinstance(image_like, (bytes, bytearray)):
        arr = _decode_bytes(image_like)
    elif isinstance(image_like, str):
        arr = _decode_path(image_like)
    elif hasattr(image_like, "__array__") or hasattr(image_like, "__array_interface__"):
        arr = np.asarray(image_like)
    else:
        raise TypeError(f"unsupported image payload type={type(image_like)}")

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC image with ndim=3, got shape={arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"expected channel-last with 1/3/4 channels, got shape={arr.shape}")

    return arr


def _decode_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("Image decoding from path requires Pillow") from e
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _decode_bytes(data):
    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("Image decoding from bytes requires Pillow") from e
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def _normalise_detection_item(boxes, classes):
    if boxes is None:
        boxes = []
    if classes is None:
        classes = []
    out_boxes = np.asarray(boxes, dtype=np.float32)
    out_classes = np.asarray(classes, dtype=np.int64)
    if out_boxes.size == 0:
        out_boxes = np.zeros((0, 4), dtype=np.float32)
    elif out_boxes.ndim == 1:
        if out_boxes.shape[0] != 4:
            raise ValueError("detection boxes must be Nx4")
        out_boxes = out_boxes.reshape(1, 4)
    elif out_boxes.ndim != 2 or out_boxes.shape[1] != 4:
        raise ValueError(f"detection boxes must be Nx4, got shape={out_boxes.shape}")
    return {"boxes": out_boxes, "classes": out_classes}


def _extract_detection_annotations(row, *, label_column=None, boxes_column=None, classes_column=None):
    boxes = row.get(boxes_column) if boxes_column else None
    classes = row.get(classes_column) if classes_column else None

    if boxes is not None or classes is not None:
        return _normalise_detection_item(boxes, classes)

    annotation = row.get(label_column) if label_column else None
    if not isinstance(annotation, dict):
        return _normalise_detection_item([], [])

    for container_key in ("objects", "annotations", "targets"):
        nested = annotation.get(container_key)
        if isinstance(nested, dict):
            annotation = nested
            break

    candidate_boxes_keys = ("boxes", "bbox", "bboxes")
    candidate_classes_keys = ("classes", "class_labels", "labels", "category", "category_id", "category_ids")

    extracted_boxes = None
    for key in candidate_boxes_keys:
        if key in annotation:
            extracted_boxes = annotation.get(key)
            break

    extracted_classes = None
    for key in candidate_classes_keys:
        if key in annotation:
            extracted_classes = annotation.get(key)
            break

    return _normalise_detection_item(extracted_boxes, extracted_classes)


def _process_split(
    ds,
    *,
    split_name,
    image_processor,
    image_column,
    task_type,
    training,
    label_column=None,
    boxes_column=None,
    classes_column=None,
    mask_column=None,
    on_decode_error="skip",
    report_decode_errors=False,
):
    if task_type == "detection":
        LOGGER.info(
            "[detection preprocessing] entering _process_split split=%s len=%d training=%s",
            split_name,
            len(ds),
            bool(training),
        )
    images = []
    labels = []
    decode_errors = []
    processor_call = getattr(image_processor, "__call__", None)
    accepts_kwargs = False
    accepted_params = set()
    if callable(processor_call):
        try:
            sig = inspect.signature(processor_call)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            accepted_params = set(sig.parameters)
        except (TypeError, ValueError):
            accepts_kwargs = True
    preprocess_call = getattr(image_processor, "preprocess", None)
    preprocess_accepts_kwargs = False
    preprocess_params = set()
    if callable(preprocess_call):
        try:
            preprocess_sig = inspect.signature(preprocess_call)
            preprocess_accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in preprocess_sig.parameters.values()
            )
            preprocess_params = set(preprocess_sig.parameters)
        except (TypeError, ValueError):
            preprocess_accepts_kwargs = True

    def _process_image(image_array, *, training_enabled):
        base_kwargs = {
            "return_tensors": None,
            "do_resize": True,
            "do_normalize": True,
        }
        supports_do_augment = ("do_augment" in accepted_params) or ("do_augment" in preprocess_params)
        if supports_do_augment:
            base_kwargs["do_augment"] = bool(training_enabled)

        if not accepts_kwargs and accepted_params:
            candidate_kwargs = {k: v for k, v in base_kwargs.items() if k in accepted_params}
        elif accepts_kwargs and preprocess_params and not preprocess_accepts_kwargs:
            candidate_kwargs = {k: v for k, v in base_kwargs.items() if k in preprocess_params}
        elif accepts_kwargs and preprocess_params:
            candidate_kwargs = {k: v for k, v in base_kwargs.items() if k in preprocess_params}
        else:
            candidate_kwargs = dict(base_kwargs)

        call_attempts = [dict(candidate_kwargs)]
        if "do_augment" in candidate_kwargs:
            call_attempts.append({k: v for k, v in candidate_kwargs.items() if k != "do_augment"})
        call_attempts.extend(
            [
                {k: v for k, v in candidate_kwargs.items() if k in {"return_tensors"}},
                {},
            ]
        )

        last_err = None
        for kwargs in call_attempts:
            try:
                return image_processor(image_array, **kwargs)
            except TypeError as e:
                last_err = e
                continue
            except ValueError as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        return image_processor(image_array)

    for idx in range(len(ds)):
        if task_type == "detection" and (idx == 0 or idx % 25 == 0):
            LOGGER.info(
                "[detection preprocessing] split=%s progress idx=%d/%d",
                split_name,
                idx,
                len(ds),
            )
        row = ds[idx]
        try:
            if task_type == "detection":
                LOGGER.info("[detection preprocessing] split=%s idx=%d before image decode", split_name, idx)
            image = _to_numpy_rgb(row.get(image_column))
            if task_type == "detection":
                LOGGER.info(
                    "[detection preprocessing] split=%s idx=%d after image decode shape=%s",
                    split_name,
                    idx,
                    tuple(getattr(image, "shape", ())),
                )
                LOGGER.info("[detection preprocessing] split=%s idx=%d before processor call", split_name, idx)
            proc = _process_image(image, training_enabled=training)
            pix = proc.get("pixel_values", proc)
            if task_type == "detection":
                LOGGER.info("[detection preprocessing] split=%s idx=%d after processor call", split_name, idx)
                LOGGER.info(
                    "[detection preprocessing] split=%s idx=%d before np.asarray(pixel_values)",
                    split_name,
                    idx,
                )
            pix = np.asarray(pix, dtype=np.float32)
            if pix.ndim == 4:
                pix = pix[0]
            if pix.ndim != 3:
                raise ValueError(f"processor output must be CHW/HWC 3D, got {pix.shape}")
            if pix.shape[0] != 3 and pix.shape[-1] == 3:
                pix = np.transpose(pix, (2, 0, 1))
            if pix.shape[0] != 3:
                raise ValueError(f"processor output must have 3 channels, got {pix.shape}")

            if task_type == "classification":
                label = int(row.get(label_column))
            elif task_type == "detection":
                LOGGER.info(
                    "[detection preprocessing] split=%s idx=%d before np.asarray(boxes/classes)",
                    split_name,
                    idx,
                )
                label = _extract_detection_annotations(
                    row,
                    label_column=label_column,
                    boxes_column=boxes_column,
                    classes_column=classes_column,
                )
            elif task_type == "segmentation":
                mask = row.get(mask_column)
                if mask is None:
                    raise ValueError("segmentation mask is missing")
                if task_type == "detection":
                    LOGGER.info("[detection preprocessing] split=%s idx=%d before np.asarray(mask)", split_name, idx)
                label = np.asarray(mask)
            else:
                label = None
        except Exception as e:
            if task_type == "detection":
                LOGGER.exception(
                    "[detection preprocessing] split=%s idx=%d failed during preprocessing",
                    split_name,
                    idx,
                )
            decode_errors.append({"index": idx, "error": str(e)})
            if on_decode_error == "raise":
                raise
            if on_decode_error == "report":
                images.append(None)
                labels.append(None)
            continue

        images.append(pix)
        labels.append(label)

    if on_decode_error != "report":
        # Object detection datasets can include thousands of high-resolution images.
        # Stacking the full split into one contiguous NCHW tensor eagerly allocates
        # all pixel storage at once and can exhaust host RAM before batching.
        #
        # Keep detection pixel values as a per-sample list and let the training loop
        # materialize tensor batches lazily in HFCore._batch_iter/encode_batch.
        if task_type == "detection":
            x = {"pixel_values": images}
        else:
            try:
                stacked_images = np.stack(images, axis=0).astype(np.float32, copy=False)
            except ValueError:
                unique_shapes = sorted({tuple(np.asarray(img).shape) for img in images})
                channel_counts = {shape[0] for shape in unique_shapes if len(shape) == 3}
                if channel_counts != {3}:
                    raise ValueError(
                        "pixel values have inconsistent non-CHW shapes after preprocessing. "
                        f"observed shapes={unique_shapes}"
                    )

                max_h = max(shape[1] for shape in unique_shapes)
                max_w = max(shape[2] for shape in unique_shapes)
                padded_images = []
                for image in images:
                    arr = np.asarray(image, dtype=np.float32)
                    if arr.shape[1] == max_h and arr.shape[2] == max_w:
                        padded_images.append(arr)
                        continue
                    pad_h = max_h - arr.shape[1]
                    pad_w = max_w - arr.shape[2]
                    padded_images.append(np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant"))
                stacked_images = np.stack(padded_images, axis=0).astype(np.float32, copy=False)
            x = {"pixel_values": stacked_images}
    else:
        x = {"pixel_values": images}
    if task_type == "classification":
        y = np.asarray(labels, dtype=np.int64)
    else:
        y = labels

    report = {"total": len(ds), "failed": len(decode_errors), "survived": len(images)}
    if report_decode_errors:
        report["errors"] = decode_errors
    return x, y, report


def preprocess_hf_image(
    train,
    test,
    meta,
    *,
    hf_model_id,
    image_column="image",
    label_column="label",
    boxes_column=None,
    classes_column=None,
    mask_column=None,
    task_type=None,
    training_augmentations=True,
    eval_augmentations=False,
    on_decode_error="skip",
    report_decode_errors=False,
):
    if (task_type or meta.get("task_type", "classification")).strip().lower() == "detection":
        LOGGER.info("[detection preprocessing] entering detection preprocessing in preprocess_hf_image")
    try:
        from transformers import AutoImageProcessor
    except Exception as e:
        raise ImportError("HF image preprocessing requires transformers[vision]") from e

    if on_decode_error not in {"skip", "raise", "report"}:
        raise ValueError("on_decode_error must be one of ['skip', 'raise', 'report']")

    ds_train, _ = train
    ds_test, _ = test
    task_type = (task_type or meta.get("task_type", "classification")).strip().lower()
    if task_type == "detection":
        LOGGER.info(
            "[detection preprocessing] split lengths train=%d test=%d",
            len(ds_train),
            len(ds_test),
        )

    processor = AutoImageProcessor.from_pretrained(hf_model_id)

    x_train, y_train, train_report = _process_split(
        ds_train,
        split_name="train",
        image_processor=processor,
        image_column=image_column,
        task_type=task_type,
        training=bool(training_augmentations),
        label_column=label_column,
        boxes_column=boxes_column,
        classes_column=classes_column,
        mask_column=mask_column,
        on_decode_error=on_decode_error,
        report_decode_errors=report_decode_errors,
    )
    x_test, y_test, test_report = _process_split(
        ds_test,
        split_name="test",
        image_processor=processor,
        image_column=image_column,
        task_type=task_type,
        training=bool(eval_augmentations),
        label_column=label_column,
        boxes_column=boxes_column,
        classes_column=classes_column,
        mask_column=mask_column,
        on_decode_error=on_decode_error,
        report_decode_errors=report_decode_errors,
    )

    meta = append_accounting_stage(
        meta,
        stage="hf_image",
        split="train",
        input_record_count=len(ds_train),
        post_filter_record_count=int(train_report["survived"]),
        emitted_record_count=len(x_train["pixel_values"]),
        sequence_count=len(x_train["pixel_values"]),
        metric_instance_count=len(y_train),
    )
    meta = append_accounting_stage(
        meta,
        stage="hf_image",
        split="test",
        input_record_count=len(ds_test),
        post_filter_record_count=int(test_report["survived"]),
        emitted_record_count=len(x_test["pixel_values"]),
        sequence_count=len(x_test["pixel_values"]),
        metric_instance_count=len(y_test),
    )
    meta = finalize_accounting(meta)
    inferred_input_shape = ()
    pixel_values = x_train.get("pixel_values") if isinstance(x_train, dict) else None
    if pixel_values is not None and len(pixel_values) > 0:
        inferred_input_shape = tuple(getattr(pixel_values[0], "shape", ()))
    inferred_num_classes = None
    if task_type == "classification" and len(y_train) > 0:
        inferred_num_classes = int(np.unique(np.asarray(y_train)).size)
    feature_num_classes = None
    if task_type == "classification":
        try:
            label_feature = (getattr(ds_train, "features", None) or {}).get(label_column)
            if label_feature is not None:
                class_names = getattr(label_feature, "names", None)
                if class_names:
                    feature_num_classes = int(len(class_names))
                else:
                    num_classes_attr = getattr(label_feature, "num_classes", None)
                    if num_classes_attr is not None:
                        feature_num_classes = int(num_classes_attr)
        except Exception:
            feature_num_classes = None

    resolved_num_classes = meta.get("num_classes")
    if task_type == "classification":
        candidates = []
        for candidate in (resolved_num_classes, feature_num_classes, inferred_num_classes):
            if candidate is None:
                continue
            try:
                value = int(candidate)
            except Exception:
                continue
            if value > 0:
                candidates.append(value)
        resolved_num_classes = int(max(candidates)) if candidates else None
    meta.update(
        {
            "input_shape": inferred_input_shape,
            "image_column": image_column,
            "label_column": label_column,
            "boxes_column": boxes_column,
            "classes_column": classes_column,
            "mask_column": mask_column,
            "task_type": task_type,
            "modality": "image",
            "hf_processor": hf_model_id,
            "channel_order": "CHW",
            "num_classes": resolved_num_classes,
            "training_augmentations": bool(training_augmentations),
            "eval_augmentations": bool(eval_augmentations),
            "decode_error_policy": on_decode_error,
            "decode_report": {"train": train_report, "test": test_report},
            "schema": {
                "image_column": image_column,
                "label_column": label_column if task_type == "classification" else None,
                "detection": {"boxes_column": boxes_column, "classes_column": classes_column},
                "segmentation": {"mask_column": mask_column},
            },
        }
    )

    return (x_train, y_train), (x_test, y_test), meta
