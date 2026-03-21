import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi



@dataclass(frozen=True)
class TaskSpec:
    pipeline_tag: str
    hf_task: str
    modality: str = "text"
    task_tag: str | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    "text_classification": TaskSpec(
        pipeline_tag="text-classification",
        hf_task="sequence_classification",
    ),
    "token_classification": TaskSpec(
        pipeline_tag="token-classification",
        hf_task="token_classification",
    ),
    "sentence_similarity": TaskSpec(
        pipeline_tag="sentence-similarity",
        hf_task="sentence_similarity",
    ),
    "fill_mask": TaskSpec(
        pipeline_tag="fill-mask",
        hf_task="fill_mask",
    ),
    "text_generation": TaskSpec(
        pipeline_tag="text-generation",
        hf_task="causal_lm_generation",
        task_tag="language-modeling",
    ),
    "text2text_generation": TaskSpec(
        pipeline_tag="text2text-generation",
        hf_task="seq2seq_generation",
    ),
    "image_classification": TaskSpec(
        pipeline_tag="image-classification",
        hf_task="image_classification",
        modality="image",
    ),
    "object_detection": TaskSpec(
        pipeline_tag="object-detection",
        hf_task="object_detection",
        modality="image",
    ),
    "image_segmentation": TaskSpec(
        pipeline_tag="image-segmentation",
        hf_task="image_segmentation",
        modality="image",
    ),
    "image_captioning": TaskSpec(
        pipeline_tag="image-to-text",
        hf_task="image_captioning",
        modality="multimodal",
        task_tag="captioning",
    ),
    "text_image_retrieval": TaskSpec(
        pipeline_tag="zero-shot-image-classification",
        hf_task="text_image_retrieval",
        modality="multimodal",
        task_tag="retrieval",
    ),
    "visual_question_answering": TaskSpec(
        pipeline_tag="visual-question-answering",
        hf_task="visual_question_answering",
        modality="multimodal",
        task_tag="vqa",
    ),
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


def _extract_family_hints(model_id: str, library_name: str | None, tags: list[str]) -> list[str]:
    hints: set[str] = set()

    normalized_model_id = (model_id or "").lower()
    normalized_library = (library_name or "").lower()
    normalized_tags = [str(tag).lower() for tag in (tags or []) if isinstance(tag, str)]

    segments = [segment for segment in normalized_model_id.replace("_", "-").split("/") if segment]
    if segments:
        hints.add(segments[0])
        leaf = segments[-1]
        hints.add(leaf.split("-")[0])

    if normalized_library:
        hints.add(normalized_library)

    known_families = (
        "bert",
        "roberta",
        "distilbert",
        "deberta",
        "gpt",
        "llama",
        "mistral",
        "mixtral",
        "t5",
        "flan",
        "bart",
        "vit",
        "clip",
        "whisper",
        "yolo",
    )
    for family in known_families:
        if family in normalized_model_id or any(family in tag for tag in normalized_tags):
            hints.add(family)

    return sorted(hint for hint in hints if hint)


def _fetch_models_for_tag(pipeline_tag: str, limit: int) -> list[Any]:
    api = HfApi(token="")
    models = list(
        api.list_models(
            pipeline_tag=pipeline_tag,
            sort="downloads",
            limit=min(limit, 20),
            full=False,
        )
    )    
    print(f"[DEBUG] pipeline_tag={pipeline_tag} returned {len(models)} models")
    return models


def _fetch_filtered_model_info(model_id: str) -> dict[str, Any]:
    api = HfApi()
    info = api.model_info(model_id)
    raw = info.__dict__

    tags = raw.get("tags") or []
    resolved_model_id = raw.get("id") or raw.get("modelId") or model_id
    library_name = raw.get("library_name")


    return {
        "model_id": resolved_model_id,
        "url": f"https://huggingface.co/{resolved_model_id}",
        "author": raw.get("author"),
        "downloads": raw.get("downloads"),
        "likes": raw.get("likes"),
        "pipeline_tag": raw.get("pipeline_tag"),
        "library": library_name,
        "safetensors": raw.get("safetensors"),
        "family_hints": _extract_family_hints(resolved_model_id, library_name, tags),
        "audit_dataset_tags": _extract_dataset_tags(tags),
        "audit_raw_tags": sorted(str(tag) for tag in tags if isinstance(tag, str)),
        "trending_score": raw.get("trending_score"),
    }


def build_hf_metadata_snapshot(
    *,
    models_per_task: int,
    fetch_limit_per_task: int,
    min_downloads: int,
) -> dict[str, Any]:
    task_snapshots: list[dict[str, Any]] = []

    for task_key, task_spec in TASK_SPECS.items():
        print(f"[INFO] Scanning task={task_key} pipeline_tag={task_spec.pipeline_tag}")

        fetch_error = None
        try:
            all_models = _fetch_models_for_tag(task_spec.pipeline_tag, fetch_limit_per_task)
        except Exception as exc:
            all_models = []
            fetch_error = str(exc)

        if fetch_error:
            print(f"[WARN] Fetch failed for task={task_key}: {fetch_error}")

        scanned = 0

        filtered_models = []

        for model in all_models:
            scanned += 1

            if _model_downloads(model) < min_downloads:
                continue

            filtered_models.append(model)

        selected_models = filtered_models[:models_per_task]

        print("\n=== TASK SUMMARY ===")
        print(f"Task: {task_key}")
        print(f"Scanned: {scanned}")
        print(f"Kept: {len(filtered_models)}")
        print(f"Kept %: {len(filtered_models) / scanned:.2%}" if scanned else "0")

        model_rows: list[dict[str, Any]] = []
        for model in selected_models:
            model_id = getattr(model, "id", None)
            if not model_id:
                continue
            try:
                filtered_meta = _fetch_filtered_model_info(model_id)
            except Exception as exc:
                filtered_meta = {
                    "model_id": model_id,
                    "error": str(exc),
                }


            model_rows.append(filtered_meta)

        task_snapshots.append(
            {
                "task_key": task_key,
                "pipeline_tag": task_spec.pipeline_tag,
                "hf_task": task_spec.hf_task,
                "modality": task_spec.modality,
                "task_tag": task_spec.task_tag,
                "fetched_count": len(all_models),
                "eligible_count": len(filtered_models),
                "selected_count": len(model_rows),
                "selection_policy": {
                    "min_downloads": min_downloads,
                    "dataset_tags_used_for_selection": False,
                },
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
        description="Scrub Hugging Face model metadata for downstream curation and audit"
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