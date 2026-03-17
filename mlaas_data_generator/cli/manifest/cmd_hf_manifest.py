from __future__ import annotations

import argparse
from pathlib import Path

from .hf_manifest_builder import build_hf_manifest, save_manifest


def register_hf_manifest(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("hf-manifest", help="Generate HF model+dataset run manifest")
    p.add_argument("--output", default="outputs/run_manifest.xlsx", help="Output .csv or .xlsx path")
    p.add_argument("--sheet", default="runs", help="Sheet name for xlsx output")
    p.add_argument("--models-per-task", type=int, default=10)
    p.add_argument("--datasets-per-model", type=int, default=1)
    p.add_argument("--fetch-limit-per-task", type=int, default=5000)
    p.add_argument("--max-per-author", type=int, default=20)
    p.add_argument("--max-per-family", type=int, default=35)
    p.add_argument("--min-downloads", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    def _run(args: argparse.Namespace) -> None:
        df = build_hf_manifest(
            models_per_task=args.models_per_task,
            datasets_per_model=args.datasets_per_model,
            fetch_limit_per_task=args.fetch_limit_per_task,
            max_per_author=args.max_per_author,
            max_per_family=args.max_per_family,
            min_downloads=args.min_downloads,
            seed=args.seed,
        )
        output_path = Path(args.output)
        save_manifest(df, output_path, sheet_name=args.sheet)
        print(f"Wrote {len(df)} rows to {output_path}")

    p.set_defaults(_handler=_run)