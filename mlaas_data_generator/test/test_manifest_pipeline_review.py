import json
import sqlite3

import pandas as pd

from mlaas_data_generator.cli.cmd_review_manifest_pipeline import (
    audit_registries,
    build_manifest_matrix_report,
    inspect_run_outputs,
    run_review,
)
from mlaas_data_generator.storage.writer import make_writer


def test_registry_and_manifest_matrix_review_passes_static_checks(tmp_path):
    registry_report = audit_registries()
    registry_errors = [issue for issue in registry_report["issues"] if issue["severity"] == "error"]

    assert registry_report["model_count"] == 120
    assert registry_report["dataset_count"] == 93
    assert registry_errors == []

    matrix_report = build_manifest_matrix_report(
        manifest_profile="test",
        avg_sample_size=32,
        seed=42,
        write_rehearsal_manifest=str(tmp_path / "rehearsal.xlsx"),
    )
    matrix_errors = [issue for issue in matrix_report["issues"] if issue["severity"] == "error"]

    assert matrix_errors == []
    assert matrix_report["matrices"]["combined"]["row_count"] == 1289
    assert matrix_report["matrices"]["combined"]["written_manifest"].endswith("rehearsal.xlsx")


def test_post_run_inspection_detects_complete_successful_run(tmp_path):
    db_path = tmp_path / "federated.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.seed_metrics()
    writer.write_run(
        {
            "run_id": "run-ok",
            "dataset": "synthetic",
            "task_type": "regression",
            "model_type": "mlp",
            "num_clients": 1,
            "num_rounds": 1,
        }
    )
    writer.write_round({"run_id": "run-ok", "round": 1, "scheduled_clients": 1, "attempted_clients": 1, "participating_clients": 1, "dropped_clients": 0})
    writer.write_client({"run_id": "run-ok", "client_id": "client_1", "samples_count": 8})
    writer.write_run_param("run-ok", "runner", "seed", 42)
    writer.write_run_param("run-ok", "adapter", "learning_rate", 0.001)
    writer.write_measurements("run-ok", round=1, client_id="client_1", values={"rmse": 0.2, "compute_time_s": 0.1})
    writer.write_measurements("run-ok", round=1, client_id=None, values={"global_rmse": 0.2, "global_aux_metric": 0.1})
    writer.finish()

    results_path = tmp_path / "run_manifest_results.csv"
    pd.DataFrame(
        [
            {
                "external_run_id": "run-ok",
                "run_id": "run-ok",
                "case_name": "regression_case",
                "status": "success",
                "error_message": "",
                "resolved_config_json": json.dumps({"task_type": "regression", "hf_task": "", "task_tag": ""}),
            }
        ]
    ).to_csv(results_path, index=False)

    report = inspect_run_outputs(results_csv=str(results_path), db_path=str(db_path))

    assert report["available"] is True
    assert report["status_counts"] == {"success": 1}
    assert report["success_rows_inspected"] == 1
    assert report["issues"] == []


def test_post_run_inspection_reports_failed_rows_and_missing_metrics(tmp_path):
    db_path = tmp_path / "federated.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.seed_metrics()
    writer.write_run(
        {
            "run_id": "run-missing-metric",
            "dataset": "synthetic",
            "task_type": "regression",
            "model_type": "mlp",
            "num_clients": 1,
            "num_rounds": 1,
        }
    )
    writer.write_round({"run_id": "run-missing-metric", "round": 1, "scheduled_clients": 1})
    writer.write_client({"run_id": "run-missing-metric", "client_id": "client_1"})
    writer.write_run_param("run-missing-metric", "runner", "seed", 42)
    writer.write_measurements("run-missing-metric", round=1, client_id="client_1", values={"compute_time_s": 0.1})
    writer.finish()

    results_path = tmp_path / "run_manifest_results.csv"
    pd.DataFrame(
        [
            {
                "external_run_id": "run-missing-metric",
                "run_id": "run-missing-metric",
                "case_name": "regression_case",
                "status": "success",
                "error_message": "",
                "resolved_config_json": json.dumps({"task_type": "regression", "hf_task": "", "task_tag": ""}),
            },
            {
                "external_run_id": "run-failed",
                "run_id": "",
                "case_name": "failed_case",
                "status": "failed",
                "error_message": "boom",
                "resolved_config_json": "{}",
            },
        ]
    ).to_csv(results_path, index=False)

    report = inspect_run_outputs(results_csv=str(results_path), db_path=str(db_path))
    codes = {issue["code"] for issue in report["issues"]}

    assert "failed_manifest_rows" in codes
    assert "expected_metric_missing" in codes


def test_post_run_inspection_treats_blank_success_run_ids_as_dry_run(tmp_path):
    db_path = tmp_path / "federated.db"
    writer = make_writer("sqlite", db_path=str(db_path))
    writer.start()
    writer.seed_metrics()
    writer.finish()

    results_path = tmp_path / "run_manifest_results.csv"
    pd.DataFrame(
        [
            {
                "external_run_id": "dry-run-row",
                "run_id": "",
                "case_name": "dry_case",
                "status": "success",
                "error_message": "",
                "resolved_config_json": "{}",
            }
        ]
    ).to_csv(results_path, index=False)

    report = inspect_run_outputs(results_csv=str(results_path), db_path=str(db_path))

    assert report["dry_run_results_detected"] is True
    assert report["success_rows_inspected"] == 0
    assert [issue["code"] for issue in report["issues"]] == ["dry_run_results_detected"]


def test_run_review_writes_report_without_post_run_inspection(tmp_path):
    output_json = tmp_path / "review.json"
    issue_csv = tmp_path / "issues.csv"
    report = run_review(
        output_json=str(output_json),
        issue_csv=str(issue_csv),
        rehearsal_manifest=None,
        inspect_post_run=False,
    )

    assert output_json.exists()
    assert issue_csv.exists()
    assert report["summary"]["error_count"] == 0
    with sqlite3.connect(":memory:"):
        loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["registry"]["model_count"] == 120
