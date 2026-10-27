"""Local CLI for portable dataset construction and subagent/human review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .review import export_split, import_annotations, import_review_csv
from .workflow import (SPLITS, DatasetEncoder, IndexedRetriever, apply_variant,
                       build_tasks, initialize_bundle, prepare_bundle, read_json,
                       validate_bundle)


def main(argv=None):
    """Execute one explicit workflow stage; importing this module performs no I/O."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("initialize", "prepare", "tasks", "apply-variant", "annotate", "verify", "review", "review-plan", "export", "validate", "refresh-filter", "refresh-rules", "audit-targets"):
        sub = commands.add_parser(command)
        sub.add_argument("--bundle", type=Path, required=True)
        if command in {"prepare", "tasks", "annotate", "verify", "review", "review-plan", "export", "refresh-filter", "audit-targets"}:
            sub.add_argument("--split", choices=SPLITS)
        if command == "initialize":
            sub.add_argument("--decisions-root", type=Path,
                             default=Path("local/data/classification/dataset0929/decisions"))
            sub.add_argument("--expert-table", type=Path,
                             default=Path("local/data/classification/dataset0929/error_types.xlsx"))
            sub.add_argument("--sizes", type=int, nargs=3, default=(600, 200, 200))
            sub.add_argument("--seed", type=int, default=42)
        if command == "prepare":
            sub.add_argument("--coderag", type=Path, default=Path("local/data/coderag"))
            sub.add_argument("--device", default="cuda")
            sub.add_argument("--batch-size", type=int, default=8)
            sub.add_argument("--max-batch-tokens", type=int, default=8192)
            sub.add_argument("--cache-dir", type=Path, default=Path("local/cache/huggingface"))
            sub.add_argument("--allow-download", action="store_true")
            sub.add_argument("--limit", type=int)
            sub.add_argument("--top-k", type=int, default=10)
            sub.add_argument("--hard-limit", type=int, default=20)
        if command == "tasks":
            sub.add_argument("--kind", choices=("annotate", "generate", "verify"), default="annotate")
            sub.add_argument("--batch-size", type=int, default=20)
        if command in {"apply-variant", "annotate", "verify", "review"}:
            sub.add_argument("--input", type=Path, required=True)
        if command == "annotate":
            sub.add_argument("--human", action="store_true")
            sub.add_argument("--skip-excluded", action="store_true")
        if command == "export":
            sub.add_argument("--released", action="store_true")
        if command == "review":
            sub.add_argument("--sampled", action="store_true")
        if command == "review-plan":
            sub.add_argument("--fraction", type=float, default=0.1)
            sub.add_argument("--seed", type=int, default=42)
        if command == "refresh-rules":
            sub.add_argument("--expert-table", type=Path,
                             default=Path("local/data/classification/dataset0929/error_types.xlsx"))
    args = parser.parse_args(argv)
    if args.command == "initialize":
        result = initialize_bundle(args.decisions_root, args.bundle, args.expert_table,
                                   sizes=tuple(args.sizes), seed=args.seed)
    elif args.command == "prepare":
        from LCD.shared.coderag import CodeRagStore
        from LCD.shared.coderag.index import QwenEncoder
        config = read_json(args.coderag / "manifest.json")["index"]
        encoder = QwenEncoder(model=config["model"], revision=config["revision"],
                              device=args.device, precision=config["precision"],
                              batch_size=args.batch_size, seed=config["seed"],
                              cache_dir=args.cache_dir, instruction=config["query_instruction"],
                              local_files_only=not args.allow_download)
        store = CodeRagStore(args.coderag, encoder=encoder)
        wrapped = DatasetEncoder(encoder, max_batch_tokens=args.max_batch_tokens)
        prepared = prepare_bundle(args.bundle, IndexedRetriever(store, wrapped), wrapped,
                                split=args.split, limit=args.limit,
                                top_k=args.top_k, hard_limit=args.hard_limit)
        from .target_audit import audit_variant_targets
        audit = audit_variant_targets(args.bundle, split=args.split)
        for name in ((args.split,) if args.split else SPLITS):
            export_split(args.bundle / name)
        result = {"prepared_variants": prepared["prepared_variants"],
                  "preparation_errors": len(prepared["errors"]),
                  "target_audit": {k: v for k, v in audit.items() if k != "splits"}}
    elif args.command == "tasks":
        result = build_tasks(args.bundle, split=args.split, kind=args.kind, batch_size=args.batch_size)
    elif args.command == "apply-variant":
        contents = args.input.read_text(encoding="utf-8-sig")
        try:
            parsed = json.loads(contents)
            proposals = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            proposals = [json.loads(line) for line in contents.splitlines() if line.strip()]
        applied = [apply_variant(args.bundle, value) for value in proposals]
        result = [{key: value[key] for key in
                   ("source_id", "target_hypothesis_id", "rule_id", "status", "variant_id", "feedback_id")
                   if key in value} for value in applied]
    elif args.command in {"annotate", "verify", "review"}:
        if not args.split:
            parser.error("annotate/review require --split to avoid ambiguous row ownership")
        split_dir = args.bundle / args.split
        if args.command == "annotate":
            result = import_annotations(split_dir, args.input, human=args.human,
                                        skip_excluded=args.skip_excluded)
        elif args.command == "verify":
            from .verification import import_verifications
            result = import_verifications(split_dir, args.input)
        else:
            result = import_review_csv(split_dir, args.input, sampled=args.sampled)
        export_split(split_dir)
    elif args.command == "export":
        validate_bundle(args.bundle)
        result = {s: export_split(args.bundle / s, released=args.released)
                  for s in ((args.split,) if args.split else SPLITS)}
    elif args.command == "refresh-filter":
        from .curation import refresh_filter
        result = refresh_filter(args.bundle, split=args.split)
    elif args.command == "review-plan":
        from .review_policy import plan_human_review
        result = {s: plan_human_review(args.bundle / s, fraction=args.fraction, seed=args.seed)
                  for s in ((args.split,) if args.split else SPLITS)}
    elif args.command == "refresh-rules":
        from .curation import refresh_rule_metadata
        result = refresh_rule_metadata(args.bundle, args.expert_table)
    elif args.command == "audit-targets":
        from .target_audit import audit_variant_targets
        audit = audit_variant_targets(args.bundle, split=args.split)
        for name in ((args.split,) if args.split else SPLITS):
            export_split(args.bundle / name)
        result = {k: v for k, v in audit.items() if k != "splits"}
    else:
        result = validate_bundle(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
