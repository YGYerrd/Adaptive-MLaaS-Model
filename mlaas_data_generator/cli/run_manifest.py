import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import CONFIG

BASE_DEFAULTS: dict[str, Any] = {
    "num_rounds": 3,
    "num_clients": 3,
    "client_participation_rate": 1.0,
    "local_epochs": 1,
    "batch_size": 16,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "seed": 42,
    "distribution": "iid",
}

BOOL_COLUMNS = {"enabled", "measure_system_metrics", "mixed_precision"}
INT_COLUMNS = {
    "seed",
    "num_rounds",
    "num_clients",
    "local_epochs",
    "batch_size",
    "num_shards",
    "max_samples",
    "max_length",
    "num_workers",
    "timeout_s",
}
FLOAT_COLUMNS = {
    "client_participation_rate",
    "learning_rate",
    "weight_decay",
    "momentum",
    "dirichlet_alpha",
}
ENUM_COLUMNS = {"aggregation", "distribution", "optimizer", "device", "model_type", "hf_task", "task_type", "modality"}

DATASET_ARG_COLUMNS = {
    "dataset_name",  
    "dataset_config",  
    "hf_model_id",
    "hf_task",
    "max_length",
    "train_split",
    "test_split",
    "label_column",
    "text_column",
    "image_column",
    "modality",
    "missing_pair_handling",
    "max_samples",
    "task_tag",
}

REQUIRED_COLUMNS = {"model_type"}

BLANK_STRINGS = {"", "na", "nan", "null", "none"}

COLUMN_ALIASES = {
    "external run id": "external_run_id",
    "run group id": "run_group_id",
    "run_group": "run_group_id",
    "num rounds": "num_rounds",
    "num clients": "num_clients",
    "client participation rate": "client_participation_rate",
    "local epoch": "local_epochs",
    "local epochs": "local_epochs",
    "batch size": "batch_size",
    "learing_rate": "learning_rate",
    "earning_rate": "learning_rate",
    "learning rate": "learning_rate",
    "model type": "model_type",
    "task type": "task_type",
    "dataset name": "dataset_name",
    "dataset config": "dataset_config",
    "hf model id": "hf_model_id",
    "train split": "train_split",
    "test split": "test_split",
    "label column": "label_column",
    "text column": "text_column",
    "image column": "image_column",
    "task tag": "task_tag",
}

@dataclass
class RowValidation:
    ok: bool
    error: str = ""


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in BLANK_STRINGS
    return False


def _normalize_value(value: Any) -> Any:
    if _is_blank(value):
        return None
    if isinstance(value, str):
        return value.strip()
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value '{value}'")


def _coerce_by_column(column: str, value: Any) -> Any:
    value = _normalize_value(value)
    if value is None:
        return None

    if column in BOOL_COLUMNS:
        return _to_bool(value)
    if column in INT_COLUMNS:
        return int(value)
    if column in FLOAT_COLUMNS:
        return float(value)
    if column in ENUM_COLUMNS and isinstance(value, str):
        return value.lower()
    return value

def _normalize_column_name(column: Any) -> Any:
    if not isinstance(column, str):
        return column

    trimmed = column.strip()
    snake = trimmed.lower().replace("-", "_").replace(" ", "_")
    snake = "_".join(part for part in snake.split("_") if part)
    return COLUMN_ALIASES.get(trimmed.lower(), COLUMN_ALIASES.get(snake, snake))


def _normalize_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {_col: _normalize_column_name(_col) for _col in df.columns}
    out = df.rename(columns=normalized)

    if "learning_rate" not in out.columns:
        for alias in ("earning_rate", "learing_rate"):
            if alias in out.columns:
                out["learning_rate"] = out[alias]
                break

    return out

def _extract_defaults_row(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if "external_run_id" not in df.columns:
        return {}, df

    marker = df["external_run_id"].astype(str).str.strip().str.lower() == "defaults"
    if not marker.any():
        return {}, df

    defaults_row = df[marker].iloc[0].to_dict()
    cleaned_defaults = {
        key: _coerce_by_column(key, value)
        for key, value in defaults_row.items()
        if not _is_blank(value)
    }
    return cleaned_defaults, df[~marker].copy()


def load_manifest(file_path: Path, sheet: str = "runs") -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        runs_df = _normalize_manifest_columns(pd.read_csv(file_path))
        csv_defaults, runs_df = _extract_defaults_row(runs_df)
        return runs_df, csv_defaults

    if suffix == ".xlsx":
        runs_df = _normalize_manifest_columns(pd.read_excel(file_path, sheet_name=sheet))
        defaults_df = None
        try:
            defaults_df = _normalize_manifest_columns(pd.read_excel(file_path, sheet_name="defaults"))
        except ValueError:
            pass

        xlsx_defaults = {}
        if defaults_df is not None and not defaults_df.empty:
            xlsx_defaults = {
                key: _coerce_by_column(key, value)
                for key, value in defaults_df.iloc[0].to_dict().items()
                if not _is_blank(value)
            }

        csv_style_defaults, runs_df = _extract_defaults_row(runs_df)
        xlsx_defaults.update(csv_style_defaults)
        return runs_df, xlsx_defaults

    raise ValueError(f"Unsupported file extension '{suffix}'. Use .csv or .xlsx")


def _resolve_row(row: pd.Series, manifest_defaults: dict[str, Any]) -> dict[str, Any]:
    resolved = CONFIG.copy()
    resolved.update(BASE_DEFAULTS)
    resolved.update(manifest_defaults)

    for column, raw in row.items():
        value = _coerce_by_column(column, raw)
        if value is None:
            continue
        resolved[column] = value

    if "distribution" in resolved:
        resolved["distribution_type"] = resolved["distribution"]

    if "client_participation_rate" in resolved:
        resolved["sample_frac"] = resolved["client_participation_rate"]

    dataset = str(resolved.get("dataset") or "").strip().lower()
    model_type = str(resolved.get("model_type") or "").strip().lower()
    hf_task = str(resolved.get("hf_task") or "").strip().lower().replace("-", "_")

    if dataset == "hf" and model_type in {"hf", "hf_text", "transformers"} and hf_task in {
        "token_classification",
        "fill_mask",
        "sentence_similarity",
    }:
        resolved["model_type"] = "hf_finetune"

    return resolved


def _build_dataset_args(resolved: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for key in DATASET_ARG_COLUMNS:
        value = resolved.get(key)
        if value is not None:
            args[key] = value
    return args


def _validate_row(resolved: dict[str, Any]) -> RowValidation:
    for col in REQUIRED_COLUMNS:
        if _is_blank(resolved.get(col)):
            return RowValidation(False, f"Missing required column '{col}'")

    dataset = resolved.get("dataset")
    if _is_blank(dataset):
        return RowValidation(False, "Missing required dataset")

    if int(resolved.get("num_rounds", 0)) <= 0:
        return RowValidation(False, "num_rounds must be > 0")

    if int(resolved.get("num_clients", 0)) <= 0:
        return RowValidation(False, "num_clients must be > 0")

    # Added: HF runs must specify hf_model_id or case_name
    if str(dataset).strip().lower() == "hf":
        if _is_blank(resolved.get("hf_model_id")) and _is_blank(resolved.get("case_name")):
            return RowValidation(False, "HF runs require 'hf_model_id' or 'case_name'")


    modality = str(resolved.get("modality") or "").strip().lower()
    if modality == "multimodal":
        if _is_blank(resolved.get("image_column")):
            return RowValidation(False, "Multimodal runs require 'image_column'")
        if _is_blank(resolved.get("text_column")):
            return RowValidation(False, "Multimodal runs require 'text_column'")

    return RowValidation(True)


def _is_enabled(row: pd.Series) -> bool:
    if "enabled" not in row.index:
        return True
    raw = _normalize_value(row.get("enabled"))
    if raw is None:
        return True
    return _to_bool(raw)


def run_manifest(file: str, sheet: str = "runs", dry_run: bool = False) -> Path:
    manifest_path = Path(file)
    print(f"Loading manifest file: {manifest_path}")
    runs_df, manifest_defaults = load_manifest(manifest_path, sheet=sheet)

    enabled_df = runs_df[runs_df.apply(_is_enabled, axis=1)].copy()
    print(f"Loaded manifest: {manifest_path}")
    print(f"Total rows: {len(runs_df)}")
    print(f"Enabled runs: {len(enabled_df)}")

    run_group_id = str(uuid.uuid4())
    results: list[dict[str, Any]] = []

    for i, (idx, row) in enumerate(enabled_df.iterrows(), start=1):
        resolved = _resolve_row(row, manifest_defaults)
        if _is_blank(resolved.get("run_group_id")):
            resolved["run_group_id"] = run_group_id

        external_run_id = resolved.get("external_run_id") or f"row_{idx}"

        print(f"\nRunning {i}/{len(enabled_df)}: {external_run_id}")
        print(
            f"dataset={resolved.get('dataset')} "
            f"task={resolved.get('task_type', 'classification')} "
            f"model={resolved.get('model_type')} "
            f"rounds={resolved.get('num_rounds')} "
            f"clients={resolved.get('num_clients')}"
        )

        validation = _validate_row(resolved)
        if not validation.ok:
            results.append(
                {
                    "external_run_id": external_run_id,
                    "row_index": int(idx),
                    "run_group_id": resolved.get("run_group_id"),
                    "run_id": None,
                    "case_name": resolved.get("case_name"),
                    "status": "failed",
                    "error_message": validation.error,
                    "resolved_config_json": json.dumps(resolved, default=str),
                }
            )
            print(f"Skipping row {idx}: {validation.error}")
            continue

        if dry_run:
            print(json.dumps(resolved, indent=2, default=str))
            results.append(
                {
                    "external_run_id": external_run_id,
                    "row_index": int(idx),
                    "run_group_id": resolved.get("run_group_id"),
                    "run_id": None,
                    "case_name": resolved.get("case_name"),
                    "status": "success",
                    "error_message": "",
                    "resolved_config_json": json.dumps(resolved, default=str),
                }
            )
            continue

        try:
            dataset_args = _build_dataset_args(resolved)
            from ..federated.orchestrator import FederatedDataGenerator

            gen = FederatedDataGenerator(
                config=resolved,
                dataset=resolved.get("dataset"),
                task_type=resolved.get("task_type", "classification"),
                model_type=resolved.get("model_type"),
                dataset_args=dataset_args,
            )
            summary = gen.run()
            results.append(
                {
                    "external_run_id": external_run_id,
                    "row_index": int(idx),
                    "run_group_id": resolved.get("run_group_id"),
                    "run_id": summary.get("run_id"),
                    "case_name": resolved.get("case_name"),
                    "status": "success",
                    "error_message": "",
                    "resolved_config_json": json.dumps(resolved, default=str),
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "external_run_id": external_run_id,
                    "row_index": int(idx),
                    "run_group_id": resolved.get("run_group_id"),
                    "run_id": None,
                    "case_name": resolved.get("case_name"),
                    "status": "failed",
                    "error_message": str(exc),
                    "resolved_config_json": json.dumps(resolved, default=str),
                }
            )
            print(f"Run failed for row {idx}: {exc}")

    output_path = Path("outputs") / "run_manifest_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Wrote results: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run federated experiments from a CSV/XLSX manifest")
    parser.add_argument("--file", required=True, help="Path to manifest file (.csv or .xlsx)")
    parser.add_argument("--sheet", default="runs", help="Sheet name for runs (xlsx only)")
    parser.add_argument("--dry_run", action="store_true", help="Resolve configs and validate rows without executing runs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_manifest(
        file=args.file,
        sheet=args.sheet,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()