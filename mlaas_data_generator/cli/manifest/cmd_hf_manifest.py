from __future__ import annotations

import argparse
from pathlib import Path

from .hf_manifest_builder import MANIFEST_PROFILES, build_hf_manifest, save_manifest


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def register_hf_manifest(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("hf-manifest", help="Generate HF model+dataset run manifest")
    p.add_argument("--input-json", help="Optional HF audit JSON used only for metadata enrichment")
    p.add_argument("--output", default="outputs/run_manifest.xlsx", help="Output .csv or .xlsx path")
    p.add_argument("--sheet", default="runs", help="Sheet name for xlsx output")
    p.add_argument("--task-keys", help="Comma-separated registry task keys")
    p.add_argument("--models-per-task", type=int, default=10)
    p.add_argument("--datasets-per-model", type=int, default=1)
    p.add_argument("--run-regimes", help="Comma-separated run regimes")
    p.add_argument("--variants-per-pair", type=int, default=1)
    p.add_argument("--total-runs", type=int, help="Total rows to emit, split as evenly as possible across requested task keys")
    p.add_argument("--manifest-profile", choices=sorted(MANIFEST_PROFILES), default="balanced")
    p.add_argument("--avg-sample-size", type=int, help="Target average max_samples across emitted manifest rows")
    p.add_argument("--seed", type=int, default=42)

    def _run(args: argparse.Namespace) -> None:
        df = build_hf_manifest(
            json_path=args.input_json,
            task_keys=_parse_csv_arg(args.task_keys),
            models_per_task=args.models_per_task,
            datasets_per_model=args.datasets_per_model,
            run_regimes=_parse_csv_arg(args.run_regimes),
            variants_per_pair=args.variants_per_pair,
            total_runs=args.total_runs,
            seed=args.seed,
            manifest_profile=args.manifest_profile,
            avg_sample_size=args.avg_sample_size,
        )
        output_path = Path(args.output)
        save_manifest(df, output_path, sheet_name=args.sheet)
        avg_samples = float(df["max_samples"].mean()) if not df.empty else 0.0
        print(f"Wrote {len(df)} rows to {output_path} (avg max_samples={avg_samples:.1f})")

    p.set_defaults(_handler=_run)
