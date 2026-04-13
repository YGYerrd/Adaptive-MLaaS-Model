from ..accounting import append_accounting_stage, finalize_accounting
import io
import os
import inspect

import numpy as np


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
    elif _is_pil_image(image_like):
        arr = np.asarray(image_like.convert("RGB"))
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


def _process_split(
    ds,
    *,
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

    for idx in range(len(ds)):
        row = ds[idx]
        try:
            image = _to_numpy_rgb(row.get(image_column))
            processor_kwargs = {
                "return_tensors": None,
                "do_resize": True,
                "do_normalize": True,
                "do_augment": bool(training),
            }
            if not accepts_kwargs and accepted_params:
                processor_kwargs = {k: v for k, v in processor_kwargs.items() if k in accepted_params}
            elif accepts_kwargs and preprocess_params and not preprocess_accepts_kwargs:
                # Many HF processors expose permissive __call__(**kwargs) wrappers but validate
                # against preprocess(...) kwargs. Filter here to avoid warnings about ignored args.
                processor_kwargs = {k: v for k, v in processor_kwargs.items() if k in preprocess_params}
            proc = image_processor(image, **processor_kwargs)
            pix = proc.get("pixel_values", proc)
            pix = np.asarray(pix, dtype=np.float32)
            if pix.ndim == 4:
                pix = pix[0]
            if pix.ndim != 3:
                raise ValueError(f"processor output must be CHW/HWC 3D, got {pix.shape}")
            if pix.shape[0] != 3 and pix.shape[-1] == 3:
                pix = np.transpose(pix, (2, 0, 1))
            if pix.shape[0] != 3:
                raise ValueError(f"processor output must have 3 channels, got {pix.shape}")
        except Exception as e:
            decode_errors.append({"index": idx, "error": str(e)})
            if on_decode_error == "raise":
                raise
            if on_decode_error == "report":
                images.append(None)
                labels.append(None)
            continue

        images.append(pix)

        if task_type == "classification":
            labels.append(int(row.get(label_column)))
        elif task_type == "detection":
            labels.append(_normalise_detection_item(row.get(boxes_column), row.get(classes_column)))
        elif task_type == "segmentation":
            mask = row.get(mask_column)
            if mask is None:
                raise ValueError("segmentation mask is missing")
            labels.append(np.asarray(mask))
        else:
            labels.append(None)

    x = {"pixel_values": np.asarray(images, dtype=np.float32)} if on_decode_error != "report" else {"pixel_values": images}
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
    try:
        from transformers import AutoImageProcessor
    except Exception as e:
        raise ImportError("HF image preprocessing requires transformers[vision]") from e

    if on_decode_error not in {"skip", "raise", "report"}:
        raise ValueError("on_decode_error must be one of ['skip', 'raise', 'report']")

    ds_train, _ = train
    ds_test, _ = test
    task_type = (task_type or meta.get("task_type", "classification")).strip().lower()

    processor = AutoImageProcessor.from_pretrained(hf_model_id)

    x_train, y_train, train_report = _process_split(
        ds_train,
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
