from __future__ import annotations

import argparse
import json
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import CONFIG, FAILURE_LOG_PATH, MANIFEST_RESULTS_PATH
from ..services.runner import execute_service, resolve_service_id
from ..storage.writer import make_writer

BASE_DEFAULTS: dict[str, Any] = {
    "training_regime": "finetune_transfer",
    "training_epochs": 1,
    "batch_size": 16,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "seed": 42,
    "dataset_variant": 0,
    "split_variant": 0,
    "knob_variant": 0,
}

BOOL_COLUMNS = {
    "enabled",
    "measure_system_metrics",
    "mixed_precision",
    "explainability_enabled",
    "enable_perturbation_metrics",
    "perturbation_stage_logging",
    "perturbation_progress_logging",
    "save_weights",
    "dynamic_padding",
    "report_decode_errors",
}
INT_COLUMNS = {
    "seed",
    "training_epochs",
    "batch_size",
    "sample_size",
    "max_samples",
    "max_length",
    "num_workers",
    "timeout_s",
    "source_max_length",
    "target_max_length",
    "vqa_answer_vocab_size",
    "dataset_variant",
    "split_variant",
    "knob_variant",
    "perturbation_sample_count",
    "perturbation_candidate_units",
    "perturbation_target_units",
    "perturbation_trust_trials",
    "perturbation_progress_sample_interval",
    "clustering_k",
    "clustering_n_init",
    "clustering_max_iter",
}
FLOAT_COLUMNS = {
    "learning_rate",
    "weight_decay",
    "momentum",
    "distribution_param",
    "clustering_tol",
    "realism_score",
    "perturbation_random_strength",
}
ENUM_COLUMNS = {"training_regime", "optimizer", "device", "model_type", "hf_task", "task_type", "modality", "split_strategy", "distribution_type"}
JSON_COLUMNS = {"column_mapping", "service_config", "custom_distributions"}

DATASET_ARG_COLUMNS = {
    "dataset_name",
    "dataset_config",
    "hf_model_id",
    "hf_task",
    "max_length",
    "train_split",
    "test_split",
    "benchmark_split",
    "label_column",
    "mask_column",
    "text_column",
    "image_column",
    "question_column",
    "answer_column",
    "ranking_label_column",
    "modality",
    "missing_pair_handling",
    "on_decode_error",
    "report_decode_errors",
    "vqa_label_mode",
    "vqa_answer_vocab_size",
    "vqa_unseen_answer_policy",
    "retrieval_positive_policy",
    "max_samples",
    "source_max_length",
    "target_max_length",
    "dynamic_padding",
    "column_mapping",
    "task_tag",
    "task",
    "training_regime",
    "service_source",
    "model_role",
    "input_schema",
    "fit_decision",
    "fit_reason",
    "dataset_hint",
    "hf_pipeline_tag",
    "hf_downloads",
    "hf_likes",
    "hf_dataset_id",
    "downloads",
    "likes",
    "model_size",
    "params_count",
    "pipeline_tag",
    "library_name",
    "license",
    "tags",
    "last_modified",
    "hf_author",
    "hf_url",
    "hf_service_meta_json",
    "explainability_enabled",
    "explainability_method",
    "explainability_target",
}

REQUIRED_COLUMNS = {"dataset", "model_type", "task_type"}
BLANK_STRINGS = {"", "na", "n/a", "nan", "null", "none", "not applicable", "not_applicable"}

COLUMN_ALIASES = {
    "manifest group id": "manifest_group_id",
    "training regime": "training_regime",
    "dataset variant": "dataset_variant",
    "split variant": "split_variant",
    "knob variant": "knob_variant",
    "service config": "service_config",
    "batch size": "batch_size",
    "learning rate": "learning_rate",
    "learing_rate": "learning_rate",
    "earning_rate": "learning_rate",
    "training epochs": "training_epochs",
    "epochs": "training_epochs",
    "model type": "model_type",
    "task type": "task_type",
    "dataset name": "dataset_name",
    "dataset config": "dataset_config",
    "hf model id": "hf_model_id",
    "train split": "train_split",
    "test split": "test_split",
    "benchmark split": "benchmark_split",
    "label column": "label_column",
    "mask column": "mask_column",
    "text column": "text_column",
    "image column": "image_column",
    "task tag": "task_tag",
    "dataset task": "task",
    "db": "db_path",
    "database": "db_path",
    "database path": "db_path",
    "sql db": "db_path",
    "sql db path": "db_path",
    "sqlite db": "db_path",
    "sqlite db path": "db_path",
    "sample count": "sample_size",
    "sample_count": "sample_size",
    "samples": "sample_size",
    "split strategy": "split_strategy",
    "distribution type": "distribution_type",
    "distribution param": "distribution_param",
    "custom distributions": "custom_distributions",
    "save weights": "save_weights",
}

FEDERATED_COLUMNS = {
    "external_run_id",
    "run_group_id",
    "run_group",
    "num_" + "".join(["c", "lients"]),
    "num_" + "rounds",
    "rounds",
    "".join(["c", "lients"]),
    "".join(["c", "lient"]) + "_" + "participation_rate",
    "".join(["c", "lient"]) + "_" + "dropout_rate",
    "aggregation",
    "aggregator",
    "aggregation_" + "weight",
    "aggregation_" + "weight_unit",
    "aggregation_" + "weight_value",
    "global_" + "model",
    "local_epochs",
}


@dataclass
class RowValidation:
    ok: bool
    error: str = ""


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return False
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in BLANK_STRINGS
    return False


def _normalize_value(value: Any) -> Any:
    if _is_blank(value):
        return None
    return value.strip() if isinstance(value, str) else value


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _coerce_by_column(column: str, value: Any) -> Any:
    value = _normalize_value(value)
    if value is None:
        return None
    if column in BOOL_COLUMNS:
        return _to_bool(value)
    if column in INT_COLUMNS:
        return int(float(value))
    if column in FLOAT_COLUMNS:
        return float(value)
    if column in JSON_COLUMNS and isinstance(value, str):
        return json.loads(value)
    if column in ENUM_COLUMNS and isinstance(value, str):
        return value.strip().lower()
    return value


def _normalize_column_name(column: Any) -> Any:
    if not isinstance(column, str):
        return column
    normalized = column.strip()
    lower = normalized.lower()
    return COLUMN_ALIASES.get(lower, lower.replace(" ", "_"))


def _normalize_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]
    return df


def _extract_defaults_row(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if "service_id" not in df.columns:
        return {}, df
    marker = df["service_id"].astype(str).str.strip().str.lower() == "defaults"
    if not marker.any():
        return {}, df
    defaults: dict[str, Any] = {}
    defaults_row = df.loc[marker].iloc[0]
    for column, raw in defaults_row.items():
        if column == "service_id":
            continue
        value = _coerce_by_column(column, raw)
        if value is not None:
            defaults[column] = value
    return defaults, df.loc[~marker].reset_index(drop=True)


def load_manifest(file_path: Path, sheet: str = "services") -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = _normalize_manifest_columns(pd.read_csv(file_path))
        defaults, rows = _extract_defaults_row(df)
        return rows, defaults
    if suffix in {".xlsx", ".xls"}:
        workbook = pd.read_excel(file_path, sheet_name=None)
        service_rows_df = _normalize_manifest_columns(workbook.get(sheet) if sheet in workbook else next(iter(workbook.values())))
        defaults: dict[str, Any] = {}
        if "defaults" in workbook:
            defaults_df = _normalize_manifest_columns(workbook["defaults"])
            if not defaults_df.empty:
                defaults = {
                    key: _coerce_by_column(key, value)
                    for key, value in defaults_df.iloc[0].to_dict().items()
                    if _coerce_by_column(key, value) is not None
                }
        csv_defaults, service_rows_df = _extract_defaults_row(service_rows_df)
        defaults.update(csv_defaults)
        return service_rows_df, defaults
    raise ValueError(f"Unsupported file extension '{suffix}'. Use .csv or .xlsx")


def _resolve_row(row: pd.Series, manifest_defaults: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(CONFIG)
    resolved.update(BASE_DEFAULTS)
    resolved.update(manifest_defaults)
    for column, raw in row.items():
        value = _coerce_by_column(column, raw)
        if value is not None:
            resolved[column] = value

    if resolved.get("benchmark_split") and not resolved.get("test_split"):
        resolved["test_split"] = resolved["benchmark_split"]
    if resolved.get("test_split") and not resolved.get("benchmark_split"):
        resolved["benchmark_split"] = resolved["test_split"]
    if _is_blank(resolved.get("service_id")):
        resolved["service_id"] = resolve_service_id(resolved)
    resolved["dataset_args"] = _build_dataset_args(resolved)
    return resolved


def _build_dataset_args(resolved: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for key in DATASET_ARG_COLUMNS:
        value = resolved.get(key)
        if value is not None:
            args[key] = value
    if args.get("benchmark_split") and not args.get("test_split"):
        args["test_split"] = args["benchmark_split"]
    return args


def _validate_row(resolved: dict[str, Any]) -> RowValidation:
    federated_columns = [
        key
        for key, value in resolved.items()
        if (key in FEDERATED_COLUMNS or str(key).startswith("global_")) and not _is_blank(value)
    ]
    if federated_columns:
        return RowValidation(
            False,
            "Federated columns are not accepted in service manifests: "
            + ", ".join(sorted(federated_columns)),
        )

    for col in REQUIRED_COLUMNS:
        if _is_blank(resolved.get(col)):
            return RowValidation(False, f"Missing required column '{col}'")
    if _is_blank(resolved.get("service_id")):
        return RowValidation(False, "Missing or unresolved service_id")
    training_regime = str(resolved.get("training_regime") or "").strip().lower()
    if training_regime not in {"finetune_transfer", "inference_only", "generic"}:
        return RowValidation(False, "training_regime must be one of finetune_transfer, inference_only, or generic")
    if int(resolved.get("training_epochs", 1) or 1) <= 0 and training_regime != "inference_only":
        return RowValidation(False, "training_epochs must be > 0 for trainable services")
    if int(resolved.get("batch_size", 0) or 0) <= 0:
        return RowValidation(False, "batch_size must be > 0")

    if str(resolved.get("dataset")).strip().lower() == "hf":
        if _is_blank(resolved.get("hf_model_id")):
            return RowValidation(False, "HF service rows require hf_model_id")
        hf_task = str(resolved.get("hf_task") or "").strip().lower().replace("-", "_")
        if _is_blank(hf_task):
            return RowValidation(False, "HF service rows require hf_task")
        if hf_task in {"seq2seq_generation", "text2text_generation"}:
            column_mapping = resolved.get("column_mapping") if isinstance(resolved.get("column_mapping"), dict) else {}
            source_col = column_mapping.get("source") or resolved.get("source_column") or resolved.get("text_column")
            target_col = column_mapping.get("target") or resolved.get("target_column") or resolved.get("label_column")
            if _is_blank(source_col) or _is_blank(target_col) or str(source_col) == str(target_col):
                return RowValidation(False, "seq2seq_generation requires distinct source and target columns")

    if str(resolved.get("modality") or "").strip().lower() == "multimodal":
        if _is_blank(resolved.get("image_column")):
            return RowValidation(False, "Multimodal service rows require image_column")
        if _is_blank(resolved.get("text_column")):
            return RowValidation(False, "Multimodal service rows require text_column")
    return RowValidation(True)


def _is_enabled(row: pd.Series) -> bool:
    if "enabled" not in row.index:
        return True
    raw = _normalize_value(row.get("enabled"))
    return True if raw is None else _to_bool(raw)


def _format_traceback(exc) -> str | None:
    if exc is None:
        return None
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _write_failure_log(log_path: Path, *, row_index, service_id, case_name, manifest_group_id, failure_stage, error_message, resolved, exc=None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 100,
        f"timestamp: {datetime.now().isoformat()}",
        f"row_index: {row_index}",
        f"service_id: {service_id}",
        f"case_name: {case_name}",
        f"manifest_group_id: {manifest_group_id}",
        f"failure_stage: {failure_stage}",
        f"error_message: {error_message}",
        "resolved_config:",
        json.dumps(resolved, indent=2, default=str),
    ]
    if exc is not None:
        lines.extend(["traceback:", _format_traceback(exc) or ""])
    lines.append("")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_failure_db(resolved: dict[str, Any], *, row_index, service_id, case_name, manifest_group_id, failure_stage, error_message, exc=None) -> None:
    db_path = resolved.get("db_path") or CONFIG.get("db_path")
    try:
        writer = make_writer("sqlite", db_path=db_path)
        writer.start()
        writer.write_service_failure(
            service_id=str(service_id) if service_id is not None else None,
            row_index=int(row_index) if row_index is not None else None,
            case_name=str(case_name) if case_name is not None else None,
            manifest_group_id=str(manifest_group_id) if manifest_group_id is not None else None,
            failure_stage=str(failure_stage),
            error_message=str(error_message) if error_message is not None else None,
            resolved_config_json=json.dumps(resolved, default=str),
            traceback_text=_format_traceback(exc),
        )
        writer.finish()
    except Exception as db_exc:  # noqa: BLE001
        print(f"Warning: failed to persist service failure to SQLite: {db_exc}")


def run_manifest(file: str, sheet: str = "services", dry_run: bool = False, db_path: str | None = None) -> Path:
    manifest_path = Path(file)
    print(f"Loading manifest file: {manifest_path}")
    rows_df, manifest_defaults = load_manifest(manifest_path, sheet=sheet)

    enabled_df = rows_df[rows_df.apply(_is_enabled, axis=1)].copy()
    print(f"Loaded manifest: {manifest_path}")
    print(f"Total rows: {len(rows_df)}")
    print(f"Enabled services: {len(enabled_df)}")

    manifest_group_id = str(uuid.uuid4())
    results: list[dict[str, Any]] = []

    for i, (idx, row) in enumerate(enabled_df.iterrows(), start=1):
        resolved = _resolve_row(row, manifest_defaults)
        resolved["row_index"] = int(idx)
        if db_path is not None:
            resolved["db_path"] = db_path
        if _is_blank(resolved.get("manifest_group_id")):
            resolved["manifest_group_id"] = manifest_group_id

        service_id = resolved["service_id"]
        print(f"\nService {i}/{len(enabled_df)}: {service_id}")
        print(
            f"dataset={resolved.get('dataset')} "
            f"task={resolved.get('task_type')} "
            f"model={resolved.get('model_type')} "
            f"training_regime={resolved.get('training_regime')} "
            f"db={resolved.get('db_path')}"
        )

        validation = _validate_row(resolved)
        if not validation.ok:
            results.append(_result_row(resolved, idx, "failed", validation.error, service_id=service_id))
            _write_failure_log(
                FAILURE_LOG_PATH,
                row_index=int(idx),
                service_id=service_id,
                case_name=resolved.get("case_name"),
                manifest_group_id=resolved.get("manifest_group_id"),
                failure_stage="validation_failed",
                error_message=validation.error,
                resolved=resolved,
            )
            if not dry_run:
                _write_failure_db(
                    resolved,
                    row_index=int(idx),
                    service_id=service_id,
                    case_name=resolved.get("case_name"),
                    manifest_group_id=resolved.get("manifest_group_id"),
                    failure_stage="validation_failed",
                    error_message=validation.error,
                )
            print(f"Skipping row {idx}: {validation.error}")
            continue

        if dry_run:
            print(json.dumps(resolved, indent=2, default=str))
            results.append(_result_row(resolved, idx, "success", "", service_id=service_id))
            continue

        try:
            summary = execute_service(resolved)
            status = "success" if summary.status == "success" else "failed"
            error = summary.error or ""
            results.append(_result_row(resolved, idx, status, error, service_id=summary.service_id))
            if status != "success":
                print(f"Service failed for row {idx}: {error}")
        except Exception as exc:  # noqa: BLE001
            results.append(_result_row(resolved, idx, "failed", str(exc), service_id=service_id))
            _write_failure_log(
                FAILURE_LOG_PATH,
                row_index=int(idx),
                service_id=service_id,
                case_name=resolved.get("case_name"),
                manifest_group_id=resolved.get("manifest_group_id"),
                failure_stage="runtime_exception",
                error_message=str(exc),
                resolved=resolved,
                exc=exc,
            )
            _write_failure_db(
                resolved,
                row_index=int(idx),
                service_id=service_id,
                case_name=resolved.get("case_name"),
                manifest_group_id=resolved.get("manifest_group_id"),
                failure_stage="runtime_exception",
                error_message=str(exc),
                exc=exc,
            )
            print(f"Service failed for row {idx}: {exc}")

    output_path = MANIFEST_RESULTS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Wrote results: {output_path}")
    return output_path


def _result_row(resolved: dict[str, Any], idx: int, status: str, error_message: str, *, service_id: str) -> dict[str, Any]:
    return {
        "service_id": service_id,
        "row_index": int(idx),
        "manifest_group_id": resolved.get("manifest_group_id"),
        "case_name": resolved.get("case_name"),
        "status": status,
        "error_message": error_message,
        "resolved_config_json": json.dumps(resolved, default=str),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute manifest service rows")
    parser.add_argument("--file", required=True, help="Path to manifest file (.csv or .xlsx)")
    parser.add_argument("--sheet", default="services", help="Sheet name for service rows (xlsx only)")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Resolve and validate rows without executing services")
    parser.add_argument("--db", "--db-path", dest="db_path", default=None, help="SQLite database path for service records")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_manifest(file=args.file, sheet=args.sheet, dry_run=args.dry_run, db_path=args.db_path)


if __name__ == "__main__":
    main()
