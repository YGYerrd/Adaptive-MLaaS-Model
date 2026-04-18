from __future__ import annotations

import argparse
import json

from ..federated.dynamics import DEFAULT_TOLERANCE, evaluate_run_dynamics


def _handle(args: argparse.Namespace) -> None:
    summaries = evaluate_run_dynamics(
        args.db,
        run_id=args.run_id,
        tolerance=args.tolerance,
    )
    if args.json:
        print(json.dumps(summaries, indent=2, default=str))
        return

    for summary in summaries:
        print(f"run_id: {summary['run_id']}")
        print(f"  task/model: {summary.get('task_type')} / {summary.get('model_type')}")
        print(f"  rounds: {summary.get('num_rounds')}")
        print(f"  update_expected_rounds: {summary.get('update_expected_rounds')}")
        print(f"  redundant_rounds: {summary.get('redundant_rounds')}")
        print(f"  expected_repeated_rounds: {summary.get('expected_repeated_rounds')}")
        issues = summary.get("issues") or []
        if issues:
            print("  issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  issues: none")


def register_dynamics(subparsers):
    p = subparsers.add_parser("evaluate-dynamics", help="Evaluate federated round learning dynamics in a run DB")
    p.add_argument("--db", required=True, help="Path to SQLite run database")
    p.add_argument("--run-id", default=None, help="Optional run_id to inspect")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Numeric equality tolerance")
    p.add_argument("--json", action="store_true", help="Emit JSON summary")
    p.set_defaults(_handler=_handle)
