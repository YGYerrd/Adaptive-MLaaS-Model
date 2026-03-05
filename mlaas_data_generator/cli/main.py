# cli/main.py
from __future__ import annotations
import argparse
from .cmd_generate import register_generate
from .cmd_merge import register_merge
from .cmd_wizard import register_wizard
from .cmd_autogen import register_autogen
from .interactive_run import register_interactive_run



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate MLaaS client data")
    sub = p.add_subparsers(dest="command")
    register_generate(sub)
    register_merge(sub)
    register_wizard(sub)
    register_autogen(sub)
    register_interactive_run(sub)
    return p

def main() -> None:
    import sys
    parser = build_parser()
    if len(sys.argv) > 1 and sys.argv[1] not in {"generate", "merge", "wizard", "autogen", "interactive-run"}:
        sys.argv.insert(1, "generate")
    args = parser.parse_args()
    if not hasattr(args, "_handler"):
        parser.print_help(); return
    args._handler(args)

if __name__ == "__main__":
    main()
