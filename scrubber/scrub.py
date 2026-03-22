import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from mlaas_data_generator.hf_tasks import PIPELINE_TAG_TO_HF_TASK, resolve_hf_task_spec


@dataclass(frozen=True)
class TaskSpec:
    pipeline_tag: str
    hf_task: str
    modality: str = "text"
    task_tag: str | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    task_key: TaskSpec(
        pipeline_tag=pipeline_tag,
        hf_task=spec.hf_task,
        modality=spec.modality,
        task_tag=("language-modeling" if spec.hf_task == "causal_lm_generation" else spec.task_tag),
    )
    for pipeline_tag, task_key in (
        ("text-classification", "text_classification"),
        ("token-classification", "token_classification"),
        ("sentence-similarity", "sentence_similarity"),
        ("fill-mask", "fill_mask"),
        ("text-generation", "text_generation"),
        ("text2text-generation", "text2text_generation"),
        ("image-classification", "image_classification"),
        ("object-detection", "object_detection"),
        ("image-segmentation", "image_segmentation"),
        ("image-to-text", "image_captioning"),
        ("zero-shot-image-classification", "text_image_retrieval"),
        ("visual-question-answering", "visual_question_answering"),
    )
    for spec in (resolve_hf_task_spec(PIPELINE_TAG_TO_HF_TASK[pipeline_tag]),)
}


def _model_downloads(model: Any) -> int:
    value = getattr(model, "downloads", 0)
    return int(value or 0)


def _extract_dataset_tags(tags: list[str]) -> list[str]:
    return sorted(
        {
            str(tag).lower()
            for tag in (tags or [])
            if isinstance(tag, str) and tag.lower().startswith("dataset:")
        }
    )


def _has_dataset_tag(model: Any) -> bool:
    tags = getattr(model, "tags", None) or []
    return any(str(tag).lower().startswith("dataset:") for tag in tags)


def _fetch_models_for_tag(pipeline_tag: str, limit: int) -> list[Any]:
    api = HfApi()
    return list(
        api.list_models(
            pipeline_tag=pipeline_tag,
            sort="downloads",
            limit=limit,
            full=True,
        )
    )


def _fetch_filtered_model_info(model_id: str) -> dict[str, Any]:
    api = HfApi()
    info = api.model_info(model_id)
    raw = info.__dict__

    tags = raw.get("tags") or []
    resolved_model_id = raw.get("id") or raw.get("modelId") or model_id

    return {
        "model_id": raw.get("id") or raw.get("modelId") or model_id,
        "url": f"https://huggingface.co/{resolved_model_id}",
        "author": raw.get("author"),
        "downloads": raw.get("downloads"),
        "likes": raw.get("likes"),
        "pipeline_tag": raw.get("pipeline_tag"),
        "dataset_tags": _extract_dataset_tags(tags),
        "trending_score": raw.get("trending_score"),
        "library": raw.get("library_name"),
        "safetensors": raw.get("safetensors"),
    }


def build_hf_metadata_snapshot(
    *,
    models_per_task: int,
    fetch_limit_per_task: int,
    min_downloads: int,
) -> dict[str, Any]:
    task_snapshots: list[dict[str, Any]] = []

    scanned_total = 0
    kept_total = 0

    for task_key, task_spec in TASK_SPECS.items():
        print(f"[INFO] Scanning task={task_key} pipeline_tag={task_spec.pipeline_tag}")

        fetch_error = None
        try:
            all_models = _fetch_models_for_tag(task_spec.pipeline_tag, fetch_limit_per_task)
        except Exception as exc:
            all_models = []
            fetch_error = str(exc)

        scanned = 0
        kept = 0

        filtered_models = []

        for model in all_models:
            scanned += 1

            if _model_downloads(model) < min_downloads:
                continue

            if not _has_dataset_tag(model):
                continue

            filtered_models.append(model)
            kept += 1

        selected_models = filtered_models[:models_per_task]

        print("\n=== TASK SUMMARY ===")
        print(f"Task: {task_key}")
        print(f"Scanned: {scanned}")
        print(f"Kept: {kept}")
        print(f"Kept %: {kept / scanned:.2%}" if scanned else "0")

        model_rows: list[dict[str, Any]] = []
        for model in selected_models:
            model_id = getattr(model, "id", None)
            if not model_id:
                continue

            try:
                filtered_meta = _fetch_filtered_model_info(model_id)
                raw_fetch_error = None
            except Exception as exc:
                filtered_meta = {
                    "model_id": model_id,
                    "error": str(exc),
                }
                raw_fetch_error = str(exc)

            model_rows.append(filtered_meta)

        task_snapshots.append(
            {
                "task_key": task_key,
                "pipeline_tag": task_spec.pipeline_tag,
                "hf_task": task_spec.hf_task,
                "modality": task_spec.modality,
                "task_tag": task_spec.task_tag,
                "fetched_count": len(all_models),
                "dataset_tagged_count": len(filtered_models),
                "selected_count": len(model_rows),
                "fetch_error": fetch_error,
                "models": model_rows,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "huggingface_hub",
        "parameters": {
            "models_per_task": models_per_task,
            "fetch_limit_per_task": fetch_limit_per_task,
            "min_downloads": min_downloads,
            "require_dataset_tag": True,
        },
        "tasks": task_snapshots,
    }


def save_hf_metadata_snapshot(snapshot: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrub Hugging Face model metadata and only keep models with dataset tags"
    )
    parser.add_argument("--output", default="outputs/hf_model_metadata.json", help="Output JSON path")
    parser.add_argument("--models-per-task", type=int, default=100)
    parser.add_argument("--fetch-limit-per-task", type=int, default=5000)
    parser.add_argument("--min-downloads", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    snapshot = build_hf_metadata_snapshot(
        models_per_task=args.models_per_task,
        fetch_limit_per_task=args.fetch_limit_per_task,
        min_downloads=args.min_downloads,
    )

    output_path = Path(args.output)
    save_hf_metadata_snapshot(snapshot, output_path)

    print(f"Wrote metadata snapshot with {len(snapshot['tasks'])} task groups to {output_path}")


if __name__ == "__main__":
    main()