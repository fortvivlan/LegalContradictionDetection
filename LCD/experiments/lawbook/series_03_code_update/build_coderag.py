"""Local CLI for extracting КоАП CSVs and building/searching a whole-unit index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from LCD.shared.coderag import CodeRagStore, build_index, export_corpus
from LCD.shared.coderag.index import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "local/data/codes/Кодекс Российской Федерации об административных правонарушениях_редакция 01.10.2026.docx"
DEFAULT_OUTPUT = ROOT / "local/data/coderag"


def main(argv: list[str] | None = None) -> None:
    """Run one explicit extraction, index, combined build, or local search."""
    # Native Windows pipes otherwise inherit cp1252 and cannot print citations.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "index", "build", "search"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        if command in {"extract", "build"}:
            sub.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
        if command in {"index", "build", "search"}:
            sub.add_argument("--device", default="cuda")
            sub.add_argument("--cache-dir", type=Path, default=ROOT / "local/cache/huggingface")
            sub.add_argument("--local-files-only", action="store_true")
        if command in {"index", "build"}:
            sub.add_argument("--model", default=DEFAULT_MODEL)
            sub.add_argument("--revision")
            sub.add_argument("--precision", default="auto",
                             choices=("auto", "float32", "float16", "bfloat16"))
            sub.add_argument("--batch-size", type=int, default=8)
            sub.add_argument("--seed", type=int, default=42)
        if command == "search":
            sub.add_argument("--query", required=True)
            sub.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    result = None
    if args.command in {"extract", "build"}:
        result = export_corpus(args.source, args.output_dir)
        print(json.dumps(result["counts"], ensure_ascii=False), flush=True)
    if args.command in {"index", "build"}:
        result = build_index(args.output_dir, model=args.model, revision=args.revision,
                             device=args.device, precision=args.precision, batch_size=args.batch_size,
                             seed=args.seed, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    elif args.command == "search":
        store = CodeRagStore(args.output_dir, device=args.device, cache_dir=args.cache_dir,
                             local_files_only=args.local_files_only)
        result = store.search(args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
