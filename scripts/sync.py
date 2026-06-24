"""
Synchronize BOFIP corpus with latest data from data.economie.gouv.fr.

Downloads, diffs, parses, chunks, embeds, and atomically replaces the corpus.

Usage:
    python scripts/sync.py              # Full sync
    python scripts/sync.py --check      # Show what changed, don't apply
    python scripts/sync.py --force      # Skip confirmation

Safety:
    - Builds to data/interim_tmp/, never touches live corpus until validated
    - Backs up old corpus before swap
    - Validates doc counts, chunk counts, NPY shapes before activating
    - Rollback: restore from data/backup_YYYYMMDD_HHMMSS/

Design:
    - Always does full rebuild. ~5 min on GPU. No surgical NPY edits.
    - Atomic swap: build in temp dir, validate, rename. Zero partial state.
    - Reusable: same pipeline works for BOFIP, CGI, LPF (add parsers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


# ── Config ────────────────────────────────────────────────────────────
BOFIP_API = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/bofip-vigueur/records"
API_LIMIT = 100  # records per page
SERIES_FILTER: list[str] = []
PGP_ID_RE = re.compile(r"/bofip/([^/?#]+)-PGP")
PERMALINK_IDENTIFIANT_RE = re.compile(r"identifiant=([^&#]+)")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM = DATA_DIR / "interim"
INTERIM_TMP = DATA_DIR / "interim_tmp"
SNAPSHOT_FILE = RAW_DIR / "latest_snapshot.jsonl"
SYNC_META = INTERIM / "sync_meta.json"
BACKUP_PREFIX = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
PRESERVED_INTERIM_FILES = ("eval_queries_v1.jsonl", "passage_gold_v3.jsonl")


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def source_record_key(doc: dict) -> str:
    """Return a stable row identity for one BOFiP API record."""
    existing = text_value(doc.get("source_record_id"))
    if existing:
        return existing

    permalink = text_value(doc.get("permalien") or doc.get("source_url"))
    base_ref = text_value(doc.get("identifiant_juridique") or doc.get("boi_reference") or doc.get("document_id"))
    date = text_value(doc.get("debut_de_validite") or doc.get("publication_date")).replace("-", "")

    identifiant_match = PERMALINK_IDENTIFIANT_RE.search(permalink)
    versioned_ref = identifiant_match.group(1) if identifiant_match else ""
    if not versioned_ref:
        versioned_ref = f"{base_ref}-{date}" if base_ref and date else base_ref

    pgp_match = PGP_ID_RE.search(permalink)
    if pgp_match and versioned_ref:
        return f"{versioned_ref}__PGP-{pgp_match.group(1)}"
    if permalink and versioned_ref:
        permalink_hash = hashlib.sha1(permalink.encode("utf-8")).hexdigest()[:12]
        return f"{versioned_ref}__URL-{permalink_hash}"
    return versioned_ref


def with_source_record_id(doc: dict) -> dict:
    enriched = dict(doc)
    enriched["source_record_id"] = source_record_key(enriched)
    return enriched


def build_download_params(*, offset: int, series_filter: list[str] | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "limit": API_LIMIT,
        "offset": offset,
        "select": "identifiant_juridique,titre,serie,division,contenu,contenu_html,permalien,debut_de_validite",
    }
    requested_series = SERIES_FILTER if series_filter is None else series_filter
    if requested_series:
        params["where"] = " OR ".join(f"serie='{serie}'" for serie in requested_series)
    return params


# ── Download ──────────────────────────────────────────────────────────
def download_bofip() -> list[dict]:
    """Download all BOFIP records, unless SERIES_FILTER is explicitly set."""
    try:
        import requests
    except ImportError:
        sys.exit("ERROR: pip install requests")

    print("Downloading BOFIP from data.economie.gouv.fr...")
    docs = []
    offset = 0
    page = 1
    errors = 0

    while True:
        if page % 20 == 1:
            print(f"  page {page} ({len(docs)} docs)...", flush=True)

        try:
            resp = requests.get(BOFIP_API, params=build_download_params(offset=offset), timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            errors += 1
            if errors > 5:
                print(f"  Too many errors ({errors}), stopping at {len(docs)} docs")
                break
            print(f"  Error page {page}: {e} -- retrying...", flush=True)
            time.sleep(3)
            continue

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            doc_id = text_value(r.get("identifiant_juridique"))
            if not doc_id:
                continue
            docs.append(with_source_record_id({
                "identifiant_juridique": doc_id,
                "titre": text_value(r.get("titre")),
                "serie": text_value(r.get("serie")),
                "division": text_value(r.get("division")),
                "contenu_html": text_value(r.get("contenu_html")),
                "contenu": text_value(r.get("contenu")),
                "permalien": text_value(r.get("permalien")),
                "debut_de_validite": text_value(r.get("debut_de_validite")),
            }))

        offset += API_LIMIT
        page += 1

        if len(results) < API_LIMIT:
            break

    print(f"  Downloaded {len(docs)} documents")
    return docs


# ── Diff ──────────────────────────────────────────────────────────────
def compute_hash(doc: dict) -> str:
    """Stable hash of document content fields (ignoring volatile metadata)."""
    content = text_value(doc.get("contenu_html")) + text_value(doc.get("contenu"))
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def diff_documents(new_docs: list[dict], snapshot_path: Path) -> dict:
    """Compare new download against previous snapshot.

    Returns: {
        "new": [...], "updated": [...], "removed": [...], "unchanged": [...],
        "snapshot_count": int, "new_count": int,
    }
    """
    new_by_id = {source_record_key(d): with_source_record_id(d) for d in new_docs}
    source_collision_count = len(new_docs) - len(new_by_id)

    if not snapshot_path.exists():
        print("  No previous snapshot -- all documents are new")
        return {
            "new": list(new_by_id.values()),
            "updated": [], "removed": [], "unchanged": [],
            "snapshot_count": 0, "new_count": len(new_docs),
            "source_record_count": len(new_docs),
            "source_unique_count": len(new_by_id),
            "source_collision_count": source_collision_count,
        }

    old_docs = []
    with open(snapshot_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    old_docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    old_by_id = {source_record_key(d): with_source_record_id(d) for d in old_docs}

    new_ids = set(new_by_id.keys())
    old_ids = set(old_by_id.keys())

    added_ids = new_ids - old_ids
    removed_ids = old_ids - new_ids
    common_ids = new_ids & old_ids

    added = [new_by_id[i] for i in added_ids]
    removed = [old_by_id[i] for i in removed_ids]
    updated = []
    unchanged = []

    for cid in common_ids:
        if compute_hash(new_by_id[cid]) != compute_hash(old_by_id[cid]):
            updated.append(new_by_id[cid])
        else:
            unchanged.append(new_by_id[cid])

    return {
        "new": added, "updated": updated, "removed": removed, "unchanged": unchanged,
        "snapshot_count": len(old_docs), "new_count": len(new_docs),
        "source_record_count": len(new_docs),
        "source_unique_count": len(new_by_id),
        "source_collision_count": source_collision_count,
    }


# ── Parse ─────────────────────────────────────────────────────────────
def count_document_changes(diff: dict) -> int:
    return len(diff.get("new", [])) + len(diff.get("updated", [])) + len(diff.get("removed", []))


def should_build_corpus(diff: dict, *, rebuild: bool = False) -> bool:
    return rebuild or count_document_changes(diff) > 0


def preserve_interim_files(source_dir: Path, target_dir: Path) -> list[Path]:
    copied = []
    for name in PRESERVED_INTERIM_FILES:
        source = source_dir / name
        target = target_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def write_sync_progress(
    progress_path: Path,
    *,
    phase: str,
    done: int,
    total: int,
    started_at: float,
    now: float | None = None,
) -> None:
    current = time.time() if now is None else now
    elapsed = max(0.0, current - started_at)
    percent = round((done / total) * 100, 2) if total else 100.0
    eta = round((elapsed / done) * (total - done), 1) if done > 0 and total > done else 0.0
    payload = {
        "phase": phase,
        "done": done,
        "total": total,
        "percent": percent,
        "elapsed_s": round(elapsed, 1),
        "eta_s": eta,
        "updated_at": datetime.now().isoformat(),
    }
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_documents(raw_docs: list[dict]) -> list[dict]:
    """Parse downloaded BOFIP JSON into the runtime RawDocument format."""
    from bofip_agentic.html_parser import parse_html_content
    from bofip_agentic.text_utils import extract_legal_refs, normalize_whitespace

    parsed = []
    for d in raw_docs:
        boi_reference = text_value(d.get("identifiant_juridique"))
        doc_id = source_record_key(d)
        title = text_value(d.get("titre"))
        serie = text_value(d.get("serie"))
        raw_html = text_value(d.get("contenu_html"))
        raw_text = text_value(d.get("contenu"))
        permalink = text_value(d.get("permalien"))
        publication_date = text_value(d.get("debut_de_validite"))
        html_payload = {
            "html_title": None,
            "sections": [],
            "paragraphs": [],
            "tables": [],
            "internal_links": [],
            "legal_refs": [],
        }
        if raw_html:
            try:
                html_payload = parse_html_content(raw_html, document_id=doc_id)
            except Exception:
                pass

        paragraphs = list(html_payload.get("paragraphs", []))
        legal_refs = list(html_payload.get("legal_refs", []))

        if not paragraphs and not html_payload.get("tables") and raw_text:
            raw_text = normalize_whitespace(raw_text)
            refs = extract_legal_refs(raw_text)
            paragraphs = [{
                "paragraph_id": f"{doc_id}__para_00000",
                "section_id": None, "order_index": 0, "html_tag": "p",
                "anchor": None, "paragraph_number": None,
                "text": raw_text, "legal_refs": refs, "links": [],
            }]
            legal_refs = refs

        seen = set()
        deduped_refs = []
        for ref in legal_refs:
            key = ref.lower()
            if key not in seen:
                seen.add(key)
                deduped_refs.append(ref)

        parsed.append({
            "document_id": doc_id,
            "boi_reference": boi_reference,
            "title": title,
            "document_type": "Contenu",
            "content_type": "Commentaire",
            "publication_date": publication_date,
            "source_url": permalink,
            "language": "fr-FR",
            "subjects": [serie],
            "identifiers": [],
            "relations": [],
            "category_path": ["Commentaire", serie],
            "raw_xml_path": "", "raw_html_path": "",
            "version_status": None,
            "sections": html_payload.get("sections", []),
            "paragraphs": paragraphs,
            "tables": html_payload.get("tables", []),
            "internal_links": html_payload.get("internal_links", []),
            "legal_refs": deduped_refs,
            "html_title": html_payload.get("html_title"),
            "raw_text_length": len(raw_text),
        })

    return parsed


# ── Chunk ─────────────────────────────────────────────────────────────
def chunk_documents(parsed_docs: list[dict]) -> list[dict]:
    """Build section_window chunks for the runtime corpus."""
    from bofip_agentic.chunking import build_chunks_for_documents
    from bofip_agentic.models import raw_document_from_dict

    docs = [raw_document_from_dict(d) for d in parsed_docs]
    chunks = build_chunks_for_documents(docs, strategy="section_window", max_tokens=350, min_tokens=40)

    chunk_dicts = []
    for c in chunks:
        chunk_dicts.append({
            "chunk_id": c.chunk_id, "source_type": "BOFIP",
            "document_id": c.document_id, "boi_reference": c.boi_reference,
            "doc_version": c.doc_version, "strategy": c.strategy,
            "section_id": c.section_id, "section_path": c.section_path,
            "paragraph_range": c.paragraph_range, "text": c.text,
            "token_count": c.token_count, "chunk_kind": c.chunk_kind,
            "legal_refs": c.legal_refs, "parent_chunk_id": c.parent_chunk_id,
        })
    return chunk_dicts


# ── Embed ─────────────────────────────────────────────────────────────
def embed_corpus(
    docs: list[dict],
    chunks: list[dict],
    device: str,
    *,
    batch_size: int = 32,
    progress_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode documents and chunks with E5-large."""
    from bofip_agentic.dense_retrieval import DenseEncoder
    from bofip_agentic.models import raw_document_from_dict, chunk_node_from_dict

    model_name = "intfloat/multilingual-e5-large"
    print(f"  Loading {model_name}...")
    encoder = DenseEncoder(model_name, device=device)

    doc_objs = [raw_document_from_dict(d) for d in docs]
    chunk_objs = [chunk_node_from_dict(c) for c in chunks]

    started_at = time.time()

    print(f"  Encoding {len(doc_objs)} documents...")
    t0 = time.time()
    progress_batch_size = max(batch_size * 64, 512)
    if progress_path:
        write_sync_progress(progress_path, phase="embedding_documents", done=0, total=len(doc_objs), started_at=started_at)
    doc_emb = encoder.encode_documents(
        doc_objs,
        mode="sections_firstpara",
        batch_size=batch_size,
        progress_batch_size=progress_batch_size,
        progress_callback=(
            (lambda done, total: write_sync_progress(
                progress_path,
                phase="embedding_documents",
                done=done,
                total=total,
                started_at=started_at,
            ))
            if progress_path
            else None
        ),
    )
    print(f"    done in {time.time() - t0:.1f}s, shape {doc_emb.shape}")

    print(f"  Encoding {len(chunk_objs)} chunks...")
    t0 = time.time()
    if progress_path:
        write_sync_progress(progress_path, phase="embedding_chunks", done=0, total=len(chunk_objs), started_at=started_at)
    chunk_emb = encoder.encode_chunks(
        chunk_objs,
        mode="full",
        batch_size=batch_size,
        progress_batch_size=progress_batch_size,
        progress_callback=(
            (lambda done, total: write_sync_progress(
                progress_path,
                phase="embedding_chunks",
                done=done,
                total=total,
                started_at=started_at,
            ))
            if progress_path
            else None
        ),
    )
    print(f"    done in {time.time() - t0:.1f}s, shape {chunk_emb.shape}")

    return doc_emb, chunk_emb


# ── Validate ──────────────────────────────────────────────────────────
def validate_corpus(docs_path: Path, chunks_path: Path,
                    doc_npy: Path, chunk_npy: Path,
                    min_docs: int = 9048) -> list[str]:
    """Validate corpus integrity. Returns list of error messages (empty = OK)."""
    errors = []

    if not docs_path.exists():
        errors.append(f"{docs_path} missing")
    if not chunks_path.exists():
        errors.append(f"{chunks_path} missing")
    if not doc_npy.exists():
        errors.append(f"{doc_npy} missing")
    if not chunk_npy.exists():
        errors.append(f"{chunk_npy} missing")
    if errors:
        return errors

    with open(docs_path, encoding="utf-8") as handle:
        doc_count = sum(1 for _ in handle)
    with open(chunks_path, encoding="utf-8") as handle:
        chunk_count = sum(1 for _ in handle)
    doc_shape = np.load(str(doc_npy)).shape
    chunk_shape = np.load(str(chunk_npy)).shape

    if doc_count < min_docs:
        errors.append(f"Only {doc_count} docs (min {min_docs})")
    if doc_count != doc_shape[0]:
        errors.append(f"Doc count {doc_count} != NPY rows {doc_shape[0]}")
    if chunk_count != chunk_shape[0]:
        errors.append(f"Chunk count {chunk_count} != NPY rows {chunk_shape[0]}")
    if doc_shape[1] != 1024:
        errors.append(f"Doc embedding dim {doc_shape[1]} != 1024")
    if chunk_shape[1] != 1024:
        errors.append(f"Chunk embedding dim {chunk_shape[1]} != 1024")

    return errors


# ── Main ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Sync BOFIP corpus with latest data")
    p.add_argument("--check", action="store_true", help="Show diff only, don't apply")
    p.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--rebuild", action="store_true", help="Rebuild corpus even when downloaded records are unchanged")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    args = p.parse_args()
    if args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1")
        return 1

    print("=" * 60)
    print("BOFIP Corpus Sync")
    print("=" * 60)

    # 1. Download
    print("\n[1/5] Downloading latest BOFIP data...")
    new_docs = download_bofip()
    if not new_docs:
        print("ERROR: download returned 0 documents")
        return 1

    # 2. Diff
    print("\n[2/5] Comparing with previous snapshot...")
    diff = diff_documents(new_docs, SNAPSHOT_FILE)
    print(f"  Snapshot: {diff['snapshot_count']} docs")
    print(f"  Latest:   {diff['new_count']} docs total")
    print(f"  Source unique rows: {diff['source_unique_count']} / {diff['source_record_count']}")
    print(f"  Source key collisions: {diff['source_collision_count']}")
    print(f"    New:       {len(diff['new'])}")
    print(f"    Updated:   {len(diff['updated'])}")
    print(f"    Removed:   {len(diff['removed'])}")
    print(f"    Unchanged: {len(diff['unchanged'])}")

    if args.check:
        return 0

    total_changes = count_document_changes(diff)
    if not should_build_corpus(diff, rebuild=args.rebuild):
        print("\nNo changes. Corpus is up to date.")
        return 0
    if total_changes == 0 and args.rebuild:
        print("\nNo source changes, but --rebuild requested. Rebuilding corpus artifacts.")

    if not args.force:
        change_label = f"{total_changes} documents changed" if total_changes else "rebuild requested"
        print(f"\n{change_label}. Proceed? [y/N] ", end="", flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer != "y":
            print("Aborted.")
            return 0

    # 3. Backup
    print("\n[3/5] Backing up current corpus...")
    backup_dir = DATA_DIR / BACKUP_PREFIX
    backup_dir.mkdir(parents=True, exist_ok=True)
    if INTERIM.exists():
        for f in INTERIM.glob("*"):
            shutil.copy2(f, backup_dir / f.name)
        print(f"  Backup: {backup_dir} ({len(list(backup_dir.glob('*')))} files)")

    # 4. Build
    print("\n[4/5] Building new corpus...")
    INTERIM_TMP.mkdir(parents=True, exist_ok=True)

    # Remove old temp if exists
    for f in INTERIM_TMP.glob("*"):
        f.unlink()

    all_docs = diff['unchanged'] + diff['new'] + diff['updated']
    print(f"  Parsing {len(all_docs)} documents...")
    parsed = parse_documents(all_docs)
    print(f"  Parsed {len(parsed)} documents")

    print(f"  Chunking...")
    chunk_dicts = chunk_documents(parsed)
    print(f"  {len(chunk_dicts)} chunks")

    # Write JSONL
    docs_out = INTERIM_TMP / "raw_docs.jsonl"
    chunks_out = INTERIM_TMP / "chunks.jsonl"
    with open(docs_out, "w", encoding="utf-8") as f:
        for d in parsed:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(chunks_out, "w", encoding="utf-8") as f:
        for c in chunk_dicts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Embed
    print(f"\n  Building embeddings (device: {args.device})...")
    progress_path = INTERIM_TMP / "sync_progress.json"
    doc_emb, chunk_emb = embed_corpus(
        parsed,
        chunk_dicts,
        args.device,
        batch_size=args.batch_size,
        progress_path=progress_path,
    )

    doc_npy_out = INTERIM_TMP / "doc_dense_cache.npy"
    chunk_npy_out = INTERIM_TMP / "chunk_dense_cache.npy"
    np.save(str(doc_npy_out), doc_emb)
    np.save(str(chunk_npy_out), chunk_emb)
    print(f"  Saved: {doc_npy_out.name} ({doc_emb.shape})")
    print(f"  Saved: {chunk_npy_out.name} ({chunk_emb.shape})")

    # 5. Validate & swap
    print("\n[5/5] Validating new corpus...")
    errors = validate_corpus(docs_out, chunks_out, doc_npy_out, chunk_npy_out)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        print(f"\nOld corpus preserved. Temp files at: {INTERIM_TMP}")
        print(f"Backup at: {backup_dir}")
        return 1

    print("  Validation OK.")

    # Save snapshot and metadata BEFORE atomic swap
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Snapshot saved: {SNAPSHOT_FILE}")

    with open(INTERIM_TMP / "sync_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "synced_at": datetime.now().isoformat(),
            "doc_count": len(parsed),
            "chunk_count": len(chunk_dicts),
            "changes": {"new": len(diff['new']), "updated": len(diff['updated']),
                        "removed": len(diff['removed']), "unchanged": len(diff['unchanged'])},
        }, f, ensure_ascii=False, indent=2)

    preserved = preserve_interim_files(INTERIM, INTERIM_TMP)
    if preserved:
        print(f"  Preserved interim files: {', '.join(path.name for path in preserved)}")

    # Atomic swap: remove old interim, rename temp
    if INTERIM.exists():
        shutil.rmtree(INTERIM)
    INTERIM_TMP.rename(INTERIM)

    # Clear BM25 cache (will rebuild on next query)
    bm25_cache = Path.home() / ".cache" / "bofip_rag"
    if bm25_cache.exists():
        shutil.rmtree(bm25_cache)
        print("  BM25 cache cleared (will rebuild on next run)")

    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print(f"  Documents: {len(parsed)}")
    print(f"  Chunks:    {len(chunk_dicts)}")
    print(f"  Changes:   +{len(diff['new'])} new, ~{len(diff['updated'])} updated, "
          f"-{len(diff['removed'])} removed")
    print(f"  Backup:    {backup_dir}")
    print(f"  Rollback:  restore from {backup_dir} to data/interim/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
