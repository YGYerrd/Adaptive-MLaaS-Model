# cli/main.py
from __future__ import annotations
import argparse
from .cmd_generate import register_generate
from .cmd_merge import register_merge
from .cmd_wizard import register_wizard
from .cmd_autogen import register_autogen
from .cmd_dynamics import register_dynamics
from .cmd_export_finetune_dataset import register_export_finetune_dataset

from .run_manifest import run_manifest
from .manifest.cmd_hf_manifest import register_hf_manifest




def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate MLaaS client data")
    sub = p.add_subparsers(dest="command")
    register_generate(sub)
    register_merge(sub)
    register_wizard(sub)
    register_autogen(sub)
    register_hf_manifest(sub)
    register_dynamics(sub)
    register_export_finetune_dataset(sub)

    return p

def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "run-manifest":
        parser = argparse.ArgumentParser(description="Run manifest rows")
        parser.add_argument("--file", required=True)
        parser.add_argument("--sheet", default="runs")
        parser.add_argument("--dry_run", action="store_true")
        args = parser.parse_args(sys.argv[2:])
        run_manifest(file=args.file, sheet=args.sheet, dry_run=args.dry_run)
        return
    parser = build_parser()
    if len(sys.argv) > 1 and sys.argv[1] not in {"generate", "merge", "wizard", "autogen", "hf-manifest", "evaluate-dynamics", "export-finetune-dataset"}:
        sys.argv.insert(1, "generate")
    args = parser.parse_args()
    if not hasattr(args, "_handler"):
        parser.print_help(); return
    args._handler(args)

if __name__ == "__main__":
    main()
