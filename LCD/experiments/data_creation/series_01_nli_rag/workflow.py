"""Mechanical dataset preparation, resumable retrieval, and agent task packaging.

This module never invents edits or labels. Those are supplied by subagents and
reviewers. All paths saved in a split bundle are relative to that split.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile

from LCD.shared.documents import read_docx_text
from LCD.shared.retrieval import extract_citations
from .preparation import extract_hypotheses, read_expert_rules, sentence_records
from ._io import replace_checkpoint
from .review import (ACTIVE_REVIEW_STATES, CSV_COLUMNS, LABELS, RELEVANCE_LABELS,
                     export_split, register_rows)

SPLITS = ("train", "val", "test")
CONTEXT_INSTRUCTION = (
    "Given a legal claim and a provision of Russian law, retrieve sentences from "
    "the same court decision that provide the factual and procedural context needed "
    "to interpret the claim and assess its consistency with the provision."
)
VERSION = 1


def digest(value) -> str:
    """Hash a JSON-compatible identity with deterministic Unicode serialization."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Return a local file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_identity(row: dict) -> tuple:
    """Identify an input pair without collapsing distinct citations with equal text."""
    metadata = row.get("metadata", {})
    unit_id = (row.get("unit_id") or metadata.get("unit_id")
               or metadata.get("retrieval", {}).get("unit_id", ""))
    return (row["source_id"], row["hypothesis"], row["premise"],
            row["article_number"], unit_id, row["context"])


def read_json(path: Path):
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """Atomically replace one JSON checkpoint on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        if path.name == "registry.json":
            # The registry grows to tens of thousands of long legal pairs.
            # Use the C encoder and one buffered write instead of millions of
            # small writes for each resumable document checkpoint.
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    replace_checkpoint(temporary, path)


def write_jsonl(path: Path, records) -> None:
    """Write inspectable agent records without pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def portable_path(split_dir: Path, relative: str) -> Path:
    """Resolve an existing plain file strictly within a portable split."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or not rel.parts:
        raise ValueError(f"Unsafe relative artifact path: {relative}")
    current = split_dir
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink is not a portable copied artifact: {relative}")
    if not current.resolve().is_relative_to(split_dir.resolve()) or not current.is_file():
        raise ValueError(f"Missing or escaping artifact: {relative}")
    return current


def initialize_bundle(decisions_root: str | Path, bundle: str | Path,
                      expert_table: str | Path, *, sizes=(600, 200, 200), seed=42) -> dict:
    """Freeze grouped article-stratified assignments and copy all originals as TXT.

    Exact counts are required; incompatible duplicate/case groups fail rather
    than leaking between splits. A fresh output directory is mandatory.
    """
    root, bundle, table = Path(decisions_root), Path(bundle), Path(expert_table)
    if bundle.exists():
        raise FileExistsError(f"Select a fresh bundle directory: {bundle}")
    if bundle.resolve().is_relative_to(root.resolve()):
        raise ValueError("Bundle must be outside the active decision corpus")
    if len(sizes) != 3 or any(n < 0 for n in sizes):
        raise ValueError("Provide three nonnegative split sizes")
    with (root / "provenance.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["group"] == "main"]
    actual = {p.relative_to(root).as_posix() for p in (root / "main").rglob("*.docx")}
    if actual != {r["filename"] for r in rows} or len(rows) != len(actual):
        raise ValueError("Active main DOCX files and provenance do not match")
    if len(rows) != sum(sizes):
        raise ValueError(f"Expected {sum(sizes)} documents; found {len(rows)}")
    rules = read_expert_rules(table)
    if not rules:
        raise ValueError("No exactly-Да expert rules")
    documents, texts = [], {}
    for row in sorted(rows, key=lambda r: r["filename"]):
        path = portable_path(root, row["filename"])
        text = read_docx_text(path)
        source_id = "doc_" + digest(row["filename"])[:16]
        texts[source_id] = text
        documents.append({"source_id": source_id, "source_filename": row["filename"],
                          "article": row["article"], "source_sha256": file_hash(path),
                          "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "provenance": row})
    # Union duplicate sources and identities of the same court case.
    parents = list(range(len(documents)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, document in enumerate(documents):
        row = document["provenance"]
        normalized = re.sub(r"\s+", " ", texts[document["source_id"]]).strip()
        keys = [("text", digest(normalized))]
        for field in ("source_url", "document_id"):
            if row.get(field):
                keys.append((field, row[field]))
        if row.get("court") and row.get("case_number"):
            keys.append(("case", row["court"].strip(), row["case_number"].strip()))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, doc in enumerate(documents):
        groups[find(i)].append(doc)
    strata = defaultdict(list)
    for group in groups.values():
        strata[group[0]["article"]].append(group)
    rng = random.Random(seed)
    assigned = Counter()
    article_totals = {article: sum(len(group) for group in article_groups)
                      for article, article_groups in strata.items()}
    article_assigned = defaultdict(Counter)
    # Place indivisible multi-document groups before singletons. Otherwise a
    # rare singleton can consume the only bin capable of holding a duplicate
    # pair even when an exact assignment exists.
    ordered_groups = [(group[0]["article"], group) for group in groups.values()]
    rng.shuffle(ordered_groups)
    ordered_groups.sort(key=lambda item: (-len(item[1]), article_totals[item[0]], item[0]))
    for article, group in ordered_groups:
        eligible = [s for s, n in zip(SPLITS, sizes) if assigned[s] + len(group) <= n]
        if not eligible:
            raise ValueError("Case groups cannot fit requested exact counts")
        chosen = max(eligible, key=lambda s: (
            article_totals[article] * sizes[SPLITS.index(s)] / len(documents)
            - article_assigned[article][s],
            sizes[SPLITS.index(s)] - assigned[s], -SPLITS.index(s)))
        assigned[chosen] += len(group)
        article_assigned[article][chosen] += len(group)
        for doc in group:
            doc["split"] = chosen
    manifest = {"schema_version": VERSION, "seed": seed, "sizes": dict(zip(SPLITS, sizes)),
                "expert_table_sha256": file_hash(table), "documents": documents,
                "context_instruction": CONTEXT_INSTRUCTION,
                "policy": {"max_train_variants_per_hypothesis": 2,
                           "val_contradiction_documents": 120, "test_contradiction_documents": 40}}
    bundle.mkdir(parents=True)
    write_json(bundle / "rules.json", rules)
    write_json(bundle / "manifest.json", manifest)
    for split in SPLITS:
        split_dir = bundle / split
        members = [d for d in documents if d["split"] == split]
        variants = []
        for doc in members:
            relative = f"decisions/originals/{doc['source_id']}.txt"
            target = split_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(texts[doc["source_id"]], encoding="utf-8", newline="")
            variants.append({"source_id": doc["source_id"], "variant_id": "original",
                             "decision_text_path": relative, "text_sha256": doc["text_sha256"],
                             "source_filename": doc["source_filename"], "active": True})
        write_json(split_dir / "manifest.json", {"split": split, "documents": members,
                                                  "variants": variants, "prepared": {}})
        write_json(split_dir / "registry.json", {"next_id": 1, "rows": {}})
        write_json(split_dir / "rules.json", rules)
    return {"documents": len(documents), "sizes": dict(assigned), "rules": len(rules)}


class DatasetEncoder:
    """Reuse one pinned Qwen model with explicit independent query instructions."""

    def __init__(self, encoder, *, max_batch_tokens=8192):
        if max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be positive")
        self.encoder = encoder
        self.max_batch_tokens = max_batch_tokens
        self.configuration = {**encoder.configuration, "max_batch_tokens": max_batch_tokens}

    def _encode(self, texts):
        import numpy as np
        from LCD.shared.coderag.index import _check_lengths
        lengths = _check_lengths(self.encoder, texts, [str(i) for i in range(len(texts))])
        # Bound padded tokens, rather than merely counting inputs. Long legal
        # units otherwise cause Windows CUDA shared-memory paging without OOM.
        order = sorted(range(len(texts)), key=lambda i: (-lengths[i], i))
        batches, batch = [], []
        for index in order:
            longest = max([lengths[index], *[lengths[i] for i in batch]])
            if batch and (len(batch) >= self.encoder.batch_size
                          or longest * (len(batch) + 1) > self.max_batch_tokens):
                batches.append(batch)
                batch = []
            batch.append(index)
        if batch:
            batches.append(batch)
        vectors = [None] * len(texts)
        for indices in batches:
            encoded = self.encoder.model.encode(
                [texts[i] for i in indices], prompt="", batch_size=len(indices),
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            for index, vector in zip(indices, encoded):
                vectors[index] = vector
        return normalize(vectors)

    def encode_documents(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts, instruction):
        from LCD.shared.coderag.index import query_text
        return self._encode([query_text(t, instruction) for t in texts])


def normalize(vectors):
    """Validate and L2-normalize a finite nonzero vector matrix."""
    import numpy as np
    values = np.asarray(vectors, dtype="float32")
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("Embeddings must be a nonempty matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms == 0).any():
        raise ValueError("Embeddings must be finite and nonzero")
    return values / norms


class IndexedRetriever:
    """Batch dense queries while preserving CodeRagStore metadata and exact lookup."""

    def __init__(self, store, encoder):
        self.store, self.encoder = store, encoder
        self.manifest = store.manifest

    def exact_lookup(self, *args, **kwargs):
        return self.store.exact_lookup(*args, **kwargs)

    def search_many(self, hypotheses, *, top_k=20):
        import numpy as np
        vectors = self.encoder.encode_queries(hypotheses, self.manifest["index"]["query_instruction"])
        scores, ids = self.store.index.search(np.ascontiguousarray(vectors),
                                             min(top_k, len(self.store.records)))
        return [[{**self.store.records[int(i)], "method": "faiss", "rank": rank,
                  "score": float(score)} for rank, (score, i) in enumerate(zip(ss, ii), 1)
                 if i >= 0] for ss, ii in zip(scores, ids)]


def retrieve_premises(hypothesis: str, store, *, top_k=10, hard_limit=20,
                      semantic=None) -> tuple[list[dict], list[dict]]:
    """Resolve explicit citations first, then fill a bounded unique semantic pool."""
    if not 0 < top_k <= hard_limit <= 20:
        raise ValueError("Require 0 < top_k <= hard_limit <= 20")
    selected, unresolved = {}, []
    for citation in extract_citations(hypothesis):
        matches = (store.exact_lookup(citation.article, part=citation.part, point=citation.point)
                   if citation.code == "КоАП РФ" else [])
        if not matches:
            unresolved.append(vars(citation))
        for result in matches:
            selected.setdefault(result["unit_id"], {**result, "methods": ["exact"]})
    if len(selected) > hard_limit:
        raise ValueError(f"Exact citations yield {len(selected)} premises, exceeding {hard_limit}")
    if semantic is None:
        semantic = store.search(hypothesis, top_k=hard_limit)
    target = max(top_k, len(selected))
    for result in semantic:
        key = result["unit_id"]
        if key in selected:
            selected[key].update(semantic_rank=result.get("rank"), semantic_score=result.get("score"))
            if "faiss" not in selected[key]["methods"]:
                selected[key]["methods"].append("faiss")
        elif len(selected) < target:
            selected[key] = {**result, "methods": ["faiss"]}
    return list(selected.values()), unresolved


def select_context(hypothesis, sentences, vectors, query_vector, *, count=5):
    """Return highest-cosine distinct sentences in source order plus rank audit."""
    import numpy as np
    if count < 1 or count > 5:
        raise ValueError("Context count must be 1–5")
    matrix, query = normalize(vectors), normalize([query_vector])[0]
    if len(matrix) != len(sentences) or matrix.shape[1] != len(query):
        raise ValueError("Context embedding dimensions differ")
    scores = matrix @ query
    canonical = lambda t: " ".join(t.split()).casefold()
    seen, chosen = {canonical(hypothesis)}, []
    for i in sorted(range(len(sentences)), key=lambda i: (-float(scores[i]), sentences[i]["start"])):
        key = canonical(sentences[i]["text"])
        if key in seen:
            continue
        seen.add(key)
        chosen.append({**sentences[i], "score": float(scores[i]), "rank": len(chosen) + 1})
        if len(chosen) == count:
            break
    chosen.sort(key=lambda r: r["start"])
    return "\n".join(r["text"] for r in chosen), chosen


def prepare_bundle(bundle: str | Path, store, encoder, *, split=None, limit=None,
                   top_k=10, hard_limit=20) -> dict:
    """Resume pair/context preparation, checkpointing each immutable document variant."""
    import numpy as np
    bundle = Path(bundle)
    if split is not None and split not in SPLITS:
        raise ValueError("Unknown split")
    configuration = {"schema_version": VERSION, "index": store.manifest,
                     "encoder": encoder.configuration, "top_k": top_k,
                     "hard_limit": hard_limit, "context_instruction": CONTEXT_INSTRUCTION}
    # Device/batch do not change identities, but model/index/filter configuration does.
    identity_config = {**configuration, "encoder": {k: v for k, v in encoder.configuration.items()
                       if k not in {"device", "batch_size", "max_batch_tokens"}}}
    fingerprint = digest(identity_config)
    config_path = bundle / "preparation.json"
    if config_path.exists() and read_json(config_path)["fingerprint"] != fingerprint:
        raise ValueError("Preparation configuration changed; select a new bundle")
    write_json(config_path, {"fingerprint": fingerprint, **configuration})
    errors_path = bundle / "preparation_errors.json"
    errors = read_json(errors_path) if errors_path.is_file() else []
    if not isinstance(errors, list):
        raise ValueError("Preparation errors checkpoint is invalid")
    error_keys = {digest(error) for error in errors}

    def record_error(value):
        key = digest(value)
        if key not in error_keys:
            error_keys.add(key)
            errors.append(value)

    processed = 0
    for name in ((split,) if split else SPLITS):
        split_dir = bundle / name
        manifest = read_json(split_dir / "manifest.json")
        manifest.setdefault("prepared_failed", {})
        registry = read_json(split_dir / "registry.json")
        # Rebuild newly supplied edits before continuing the original queue,
        # so review can validate an end-to-end synthetic sample promptly.
        for variant in sorted(manifest["variants"], key=lambda v: v["variant_id"] == "original"):
            variant_key = variant["source_id"] + ":" + variant["variant_id"]
            if variant_key in manifest["prepared"]:
                continue
            failure = manifest["prepared_failed"].get(variant_key)
            if failure and failure.get("fingerprint") == fingerprint:
                continue
            if limit is not None and processed >= limit:
                break
            path = portable_path(split_dir, variant["decision_text_path"])
            if file_hash(path) != variant["text_sha256"]:
                raise ValueError(f"Decision text changed: {path}")
            text = path.read_text(encoding="utf-8")
            try:
                hypotheses, audit = extract_hypotheses(text)
            except ValueError as error:
                failure = {"split": name, "source_id": variant["source_id"],
                           "variant_id": variant["variant_id"], "variant_key": variant_key,
                           "reason": str(error)}
                record_error(failure)
                manifest["prepared_failed"][variant_key] = {
                    "fingerprint": fingerprint, "errors": [failure]}
                write_json(split_dir / "manifest.json", manifest)
                write_json(errors_path, errors)
                continue
            if not hypotheses:
                failure = {"split": name, "source_id": variant["source_id"],
                           "variant_id": variant["variant_id"],
                           "variant_key": variant_key, "reason": "empty_extraction"}
                record_error(failure)
                manifest["prepared_failed"][variant_key] = {
                    "fingerprint": fingerprint, "errors": [failure]
                }
                write_json(split_dir / "manifest.json", manifest)
                continue
            write_json(split_dir / "extraction" / f"{digest(variant_key)[:24]}.json",
                       {"variant": variant, "hypotheses": hypotheses, "audit": audit})
            # Context can come from any section; hypothesis duplicates are excluded per query.
            sentences = sentence_records(text)
            cache_key = digest([variant["text_sha256"], fingerprint,
                                [s["text"] for s in sentences]])
            cache_path = bundle / "cache" / (cache_key + ".npy")
            if cache_path.exists():
                vectors = normalize(np.load(cache_path, allow_pickle=False))
                if len(vectors) != len(sentences):
                    raise ValueError("Cached sentence vector count mismatch")
            else:
                vectors = normalize(encoder.encode_documents([s["text"] for s in sentences]))
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("wb", dir=cache_path.parent,
                                                 delete=False) as cache_stream:
                    np.save(cache_stream, vectors, allow_pickle=False)
                    temporary_cache = Path(cache_stream.name)
                replace_checkpoint(temporary_cache, cache_path)
            semantic = (store.search_many([h["text"] for h in hypotheses], top_k=hard_limit)
                        if hasattr(store, "search_many") else [None] * len(hypotheses))
            pairs, query_texts = [], []
            for hypothesis, candidates in zip(hypotheses, semantic):
                hypothesis_id = "hyp_" + digest([variant["source_id"], hypothesis["start"],
                                                  hypothesis["text"]])[:20]
                try:
                    premises, unresolved = retrieve_premises(hypothesis["text"], store,
                        top_k=top_k, hard_limit=hard_limit, semantic=candidates)
                except ValueError as error:
                    record_error({"split": name, "source_id": variant["source_id"],
                                  "variant_id": variant["variant_id"],
                                  "variant_key": variant_key,
                                  "hypothesis_id": hypothesis_id, "hypothesis": hypothesis["text"],
                                  "source_filename": variant["source_filename"],
                                  "decision_text_path": variant["decision_text_path"],
                                  "citations": [vars(c) for c in extract_citations(hypothesis["text"])],
                                  "reason": str(error)})
                    continue
                for premise in premises:
                    pairs.append((hypothesis, hypothesis_id, premise, unresolved))
                    query_texts.append(f"Hypothesis: {hypothesis['text']}\nProvision: {premise['text']}")
            if not pairs:
                failure = {"split": name, "source_id": variant["source_id"],
                           "variant_id": variant["variant_id"],
                           "variant_key": variant_key, "reason": "no_pairs"}
                record_error(failure)
                manifest["prepared_failed"][variant_key] = {
                    "fingerprint": fingerprint, "errors": [failure]
                }
                write_json(split_dir / "manifest.json", manifest)
                continue
            queries = normalize(encoder.encode_queries(query_texts, CONTEXT_INSTRUCTION))
            prepared_rows = []
            for (hypothesis, hyp_id, premise, unresolved), query in zip(pairs, queries):
                context, context_audit = select_context(hypothesis["text"], sentences, vectors, query)
                stable_key = digest([fingerprint, variant_key, hyp_id, premise["unit_id"], context])
                prepared_rows.append({"stable_key": stable_key,
                    "hypothesis": hypothesis["text"], "premise": premise["text"],
                    "article_number": premise["article"], "context": context,
                    "label": "", "error_type": "", "source_filename": variant["source_filename"],
                    "decision_text_path": variant["decision_text_path"],
                    "source_id": variant["source_id"], "variant_id": variant["variant_id"],
                    "hypothesis_id": hyp_id, "unit_id": premise["unit_id"],
                    "active": variant.get("active", True), "status": "pending", "relevance": "",
                    "metadata": {"hypothesis_span": [hypothesis["start"], hypothesis["end"]],
                                 "retrieval": premise, "unresolved_citations": unresolved,
                                 "context_sentences": context_audit,
                                 "scenario_rule_id": variant.get("rule_id", ""),
                                 "text_sha256": variant["text_sha256"]}})
            reused_ids = {}
            if prepared_rows:
                existing = {}
                for stable_key, registered in registry["rows"].items():
                    if not registered.get("active", True):
                        continue
                    identity = pair_identity(registered)
                    existing.setdefault(identity, (stable_key, registered))
                unique_rows = []
                batch_identities, repeated_indices = {}, {}
                for index, prepared in enumerate(prepared_rows):
                    identity = pair_identity(prepared)
                    duplicate = existing.get(identity)
                    if duplicate is None:
                        if identity in batch_identities:
                            repeated_indices[index] = batch_identities[identity]
                            continue
                        batch_identities[identity] = index
                        unique_rows.append((index, prepared))
                        continue
                    _, registered = duplicate
                    metadata = registered.setdefault("metadata", {})
                    variant_ids = set(metadata.get("variant_ids", []))
                    variant_ids.update((registered.get("variant_id", "original"),
                                        prepared["variant_id"]))
                    metadata["variant_ids"] = sorted(variant_ids)
                    scenario_ids = set(metadata.get("scenario_rule_ids", []))
                    for rule_id in (metadata.get("scenario_rule_id"),
                                    prepared["metadata"].get("scenario_rule_id")):
                        if rule_id:
                            scenario_ids.add(rule_id)
                    metadata["scenario_rule_ids"] = sorted(scenario_ids)
                    reused_ids[index] = registered["row_id"]
                if reused_ids:
                    write_json(split_dir / "registry.json", registry)
                new_ids = register_rows(split_dir, [row for _, row in unique_rows], registry=registry)
                allocated = dict(reused_ids)
                allocated.update({index: row_id for (index, _), row_id
                                  in zip(unique_rows, new_ids)})
                for index, original_index in repeated_indices.items():
                    allocated[index] = allocated[original_index]
                ids = [allocated[index] for index in range(len(prepared_rows))]
            else:
                ids = register_rows(split_dir, prepared_rows, registry=registry)
            manifest["prepared"][variant_key] = {"fingerprint": fingerprint, "row_ids": ids,
                                                   "identity_version": 2}
            manifest["prepared_failed"].pop(variant_key, None)
            write_json(split_dir / "manifest.json", manifest)
            processed += 1
            print(json.dumps({"split": name, "prepared_variants": processed,
                              "source_id": variant["source_id"], "pairs": len(ids)}), flush=True)
        export_split(split_dir)
    write_json(errors_path, errors)
    return {"prepared_variants": processed, "errors": errors}


def apply_variant(bundle: str | Path, proposal: dict) -> dict:
    """Validate agent offset edits and save an immutable derived TXT in its source split."""
    bundle = Path(bundle)
    source_id = proposal["source_id"]
    root_manifest = read_json(bundle / "manifest.json")
    documents = [d for d in root_manifest["documents"] if d["source_id"] == source_id]
    if len(documents) != 1:
        raise ValueError("Unknown source ID")
    source = documents[0]
    split_dir = bundle / source["split"]
    manifest = read_json(split_dir / "manifest.json")
    rules = {r["rule_id"]: r for r in read_json(bundle / "rules.json")}
    if proposal.get("rule_id") not in rules or proposal.get("status", "proposed") not in {"proposed", "needs_review"}:
        raise ValueError("Proposal must use an eligible rule and a valid status")
    original = next(v for v in manifest["variants"] if v["source_id"] == source_id
                    and v["variant_id"] == "original")
    original_path = portable_path(split_dir, original["decision_text_path"])
    if file_hash(original_path) != original["text_sha256"]:
        raise ValueError("Original text was changed")
    text = original_path.read_text(encoding="utf-8")
    hypotheses, _ = extract_hypotheses(text)
    hyp_ids = {"hyp_" + digest([source_id, h["start"], h["text"]])[:20] for h in hypotheses}
    target = proposal.get("target_hypothesis_id")
    if target not in hyp_ids:
        raise ValueError("Target hypothesis is not from this source's original operative section")
    if proposal.get("status") == "needs_review":
        if proposal.get("edits") or not proposal.get("rationale", "").strip():
            raise ValueError("An infeasible assignment needs a rationale and no edits")
        feedback_id = digest([source_id, target, proposal["rule_id"]])
        write_json(split_dir / "generation_feedback" / (feedback_id + ".json"), proposal)
        return {"source_id": source_id, "target_hypothesis_id": target,
                "rule_id": proposal["rule_id"], "status": "needs_review", "feedback_id": feedback_id}
    target_unit_ids = proposal.get("target_unit_ids", [])
    if not isinstance(target_unit_ids, list) or any(
        not isinstance(unit_id, str) or not unit_id for unit_id in target_unit_ids
    ):
        raise ValueError("target_unit_ids must be a list of nonempty strings")
    if target_unit_ids:
        registry = read_json(split_dir / "registry.json")
        available = {
            row["unit_id"] for row in registry["rows"].values()
            if row.get("source_id") == source_id
            and row.get("variant_id") == "original"
            and row.get("hypothesis_id") == target
        }
        unknown = sorted(set(target_unit_ids) - available)
        if unknown:
            raise ValueError(
                f"target_unit_ids are not retrieved candidates for the target hypothesis: {unknown}"
            )
    edits = sorted(proposal.get("edits", []), key=lambda e: e["start"])
    if not edits or not proposal.get("rationale", "").strip():
        raise ValueError("A proposal needs edits and a rationale")
    previous_end = -1
    for edit in edits:
        start, end = edit["start"], edit["end"]
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text)
                or start < previous_end or text[start:end] != edit["before"]
                or edit["before"] == edit["after"]):
            raise ValueError("Invalid, overlapping, stale or empty edit operation")
        previous_end = end
    for edit in reversed(edits):
        text = text[:edit["start"]] + edit["after"] + text[edit["end"]:]
    extract_hypotheses(text)  # Require a usable derived operative section.
    variant_id = "syn_" + digest([source_id, edits])[:20]
    if any(v["variant_id"] == variant_id for v in manifest["variants"]):
        return next(v for v in manifest["variants"] if v["variant_id"] == variant_id)
    existing = [v for v in manifest["variants"] if v["source_id"] == source_id
                and v["variant_id"] != "original"]
    if source["split"] == "train":
        if sum(v["target_hypothesis_id"] == target for v in existing) >= 2:
            raise ValueError("Two distinct variants already exist for this train hypothesis")
    elif existing:
        raise ValueError("Val/test permit one active variant per source")
    relative = f"decisions/variants/{variant_id}.txt"
    target_path = split_dir / relative
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text, encoding="utf-8", newline="")
    variant = {**proposal, "source_id": source_id, "variant_id": variant_id,
               "decision_text_path": relative, "source_filename": source["source_filename"],
               "text_sha256": file_hash(target_path), "active": True}
    manifest["variants"].append(variant)
    if source["split"] != "train":
        original["active"] = False
        registry = read_json(split_dir / "registry.json")
        for row in registry["rows"].values():
            if row["source_id"] == source_id:
                row["active"] = False
        write_json(split_dir / "registry.json", registry)
    write_json(split_dir / "manifest.json", manifest)
    return variant


def build_tasks(bundle: str | Path, *, split=None, batch_size=20, kind="annotate") -> dict:
    """Package self-contained tasks; no model API or synthetic generation is invoked."""
    bundle = Path(bundle)
    if batch_size < 1 or kind not in {"annotate", "generate", "verify"}:
        raise ValueError("Invalid task kind or batch size")
    if split is not None and split not in SPLITS:
        raise ValueError("Unknown split")
    if kind == "generate":
        from .generation import build_generation_tasks
        return build_generation_tasks(bundle, split=split)
    rules = read_json(bundle / "rules.json")
    rule_lookup = {r["rule_id"]: r for r in rules}
    result = {}
    for name in ((split,) if split else SPLITS):
        split_dir = bundle / name
        registry = read_json(split_dir / "registry.json")
        manifest = read_json(split_dir / "manifest.json")
        variants = {(v["source_id"], v["variant_id"]): v for v in manifest["variants"]}
        grouped = defaultdict(list)
        for row in registry["rows"].values():
            if row.get("active", True) and row.get("status") != "excluded":
                if ((kind == "annotate" and row.get("status") in {"pending", "needs_review"})
                        or (kind == "verify" and row.get("status") == "proposed")):
                    grouped[(row["source_id"], row["variant_id"])].append(row)
        tasks = []
        for (source_id, variant_id), rows in sorted(grouped.items()):
            full_text = portable_path(split_dir, rows[0]["decision_text_path"]).read_text(encoding="utf-8")
            variant = variants[(source_id, variant_id)]
            for start in range(0, len(rows), batch_size):
                batch = []
                for original_row in rows[start:start + batch_size]:
                    row = {key: value for key, value in original_row.items()
                           if key not in {"metadata", "annotation_history"}}
                    metadata = {key: value for key, value in original_row.get("metadata", {}).items()
                                if key != "context_sentences"}
                    if "retrieval" in metadata:
                        metadata["retrieval"] = {key: value for key, value in metadata["retrieval"].items()
                                                 if key != "text"}
                    row["metadata"] = metadata
                    batch.append(row)
                for row in batch:
                    row["expert_rule"] = rule_lookup.get(row.get("metadata", {}).get("scenario_rule_id"))
                tasks.append({"task_id": digest([name, kind, [r["row_id"] for r in batch]])[:24],
                              "kind": kind, "split": name, "source_id": source_id,
                              "variant_id": variant_id, "source_text": full_text,
                              "pairs": batch, "expert_rules": [{key: rule.get(key, "") for key in
                                  ("rule_id", "name", "kind", "comment", "verification_prompt")}
                                  for rule in rules],
                              "variant": variant,
                              "objects": ([{"object_id": r["row_id"], "object_type": "annotation",
                                            "annotation": {k: r.get(k, "") for k in
                                                ("label", "relevance", "error_type", "annotation_rationale")}}
                                           for r in batch] if kind == "verify" else []),
                              "context_instruction": CONTEXT_INSTRUCTION,
                              "policy": {"max_train_variants_per_hypothesis": 2,
                                         "val_contradiction_documents": 120,
                                         "test_contradiction_documents": 40}})
        destination = split_dir / "tasks" / (kind + ".jsonl")
        write_jsonl(destination, tasks)
        result[name] = {"tasks": len(tasks), "path": str(destination)}
    return result


def validate_bundle(bundle: str | Path) -> dict:
    """Check split containment, copied-text hashes, IDs and exported row references."""
    bundle = Path(bundle)
    root = read_json(bundle / "manifest.json")
    rules = read_json(bundle / "rules.json")
    eligible_error_types = {rule["name"] for rule in rules}
    source_splits, all_ids = {}, set()
    report = {}
    for split in SPLITS:
        split_dir = bundle / split
        manifest = read_json(split_dir / "manifest.json")
        members = {d["source_id"] for d in manifest["documents"]}
        for member in members:
            if member in source_splits:
                raise ValueError("Source appears in multiple splits")
            source_splits[member] = split
        for variant in manifest["variants"]:
            if variant["source_id"] not in members:
                raise ValueError("Variant crosses source split")
            path = portable_path(split_dir, variant["decision_text_path"])
            if file_hash(path) != variant["text_sha256"]:
                raise ValueError(f"Copied text hash mismatch: {path}")
        registry = read_json(split_dir / "registry.json")
        by_id = {}
        path_hashes = {}
        exact_active_pairs = set()
        maximum_id = 0
        variant_keys = {(variant["source_id"], variant["variant_id"]): variant
                        for variant in manifest["variants"]}
        for stable_key, row in registry["rows"].items():
            match = re.fullmatch(re.escape(split) + r"_(\d{6,})", row.get("row_id", ""))
            if (not stable_key or match is None or row["row_id"] in all_ids
                    or row["row_id"] in by_id):
                raise ValueError("Duplicate or wrong-split row ID")
            maximum_id = max(maximum_id, int(match.group(1)))
            all_ids.add(row["row_id"])
            by_id[row["row_id"]] = row
            if row["source_id"] not in members:
                raise ValueError("Row crosses source split")
            variant = variant_keys.get((row["source_id"], row.get("variant_id")))
            if variant is None or row["decision_text_path"] != variant["decision_text_path"]:
                raise ValueError("Row does not match its recorded document variant")
            relative = row["decision_text_path"]
            if relative not in path_hashes:
                path_hashes[relative] = file_hash(portable_path(split_dir, relative))
            expected_hash = row.get("metadata", {}).get("text_sha256")
            if expected_hash and expected_hash != path_hashes[relative]:
                raise ValueError("Prepared row decision-text hash mismatch")
            status = row.get("status", "pending")
            if status not in {"pending", "proposed", "approved", "excluded"}:
                raise ValueError(f"Invalid registry status: {status}")
            if status in ACTIVE_REVIEW_STATES:
                if row.get("label") not in LABELS or row.get("relevance") not in RELEVANCE_LABELS:
                    raise ValueError("Labeled row has an invalid NLI or relevance label")
                if (row["label"] in {"contradiction", "entailment"}
                        and row["relevance"] != "related"):
                    raise ValueError("Contradiction and entailment rows must be related")
                if row["label"] == "contradiction":
                    if row.get("error_type") not in eligible_error_types:
                        raise ValueError("Contradiction row has an ineligible error type")
                elif row.get("error_type"):
                    raise ValueError("Non-contradiction row has an error type")
            if row.get("active", True) and status != "excluded":
                identity = pair_identity(row)
                if identity in exact_active_pairs:
                    raise ValueError("Active registry contains an exact duplicate pair")
                exact_active_pairs.add(identity)
        if registry.get("next_id", 0) <= maximum_id:
            raise ValueError("Registry next_id would reuse an allocated row ID")
        for checkpoint_name in ("prepared", "prepared_failed"):
            for variant_key, checkpoint in manifest.get(checkpoint_name, {}).items():
                if variant_key not in {source + ":" + variant for source, variant in variant_keys}:
                    raise ValueError(f"Unknown variant in {checkpoint_name} checkpoint")
                for row_id in checkpoint.get("row_ids", []):
                    if row_id not in by_id:
                        raise ValueError("Prepared checkpoint refers to an unknown row ID")
        csv_path = split_dir / f"{split}.csv"
        if csv_path.exists():
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != list(CSV_COLUMNS):
                    raise ValueError("CSV columns or their order differ from the portable schema")
                csv_rows = list(reader)
            ids = [r["row_id"] for r in csv_rows]
            if len(ids) != len(set(ids)) or set(ids) - by_id.keys():
                raise ValueError("CSV has duplicated or unknown IDs")
            for row in csv_rows:
                registered = by_id[row["row_id"]]
                if (not registered.get("active", True)
                        or registered.get("status") not in ACTIVE_REVIEW_STATES):
                    raise ValueError("CSV contains an inactive, excluded or pending row")
                if any(row[column] != str(registered.get(column, ""))
                       for column in CSV_COLUMNS):
                    raise ValueError("CSV contents differ from the canonical registry")
                portable_path(split_dir, row["decision_text_path"])
            expected_ids = {
                row["row_id"] for row in registry["rows"].values()
                if row.get("active", True) and row.get("status") in ACTIVE_REVIEW_STATES
            }
            if set(ids) != expected_ids:
                raise ValueError("CSV omits active labeled rows or contains stale rows")
        report[split] = {"documents": len(members), "variants": len(manifest["variants"]),
                         "registered_pairs": len(by_id),
                         "active_pairs": len(exact_active_pairs)}
    if source_splits != {d["source_id"]: d["split"] for d in root["documents"]}:
        raise ValueError("Root and split assignments differ")
    if {s: report[s]["documents"] for s in SPLITS} != root["sizes"]:
        raise ValueError("Split sizes changed")
    return report
