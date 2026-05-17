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
THEME_FILTER = ["IR", "IS", "TVA", "BIC", "BNC", "BA", "CF", "ENR", "IF", "PAT", "RPPM", "RFPI", "CTX", "REC", "INT"]
THEME_WHERE = " OR ".join(f"serie='{t}'" for t in THEME_FILTER)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM = DATA_DIR / "interim"
INTERIM_TMP = DATA_DIR / "interim_tmp"
SNAPSHOT_FILE = RAW_DIR / "latest_snapshot.jsonl"
SYNC_META = INTERIM / "sync_meta.json"
BACKUP_PREFIX = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ── Download ──────────────────────────────────────────────────────────
def download_bofip() -> list[dict]:
    """Download all BOFIP records matching THEME_FILTER."""
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
            resp = requests.get(BOFIP_API, params={
                "limit": API_LIMIT, "offset": offset,
                "select": "identifiant_juridique,titre,serie,division,contenu,contenu_html,permalien,debut_de_validite",
                "where": THEME_WHERE,
            }, timeout=30)
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
            doc_id = r.get("identifiant_juridique", "")
            if not doc_id:
                continue
            docs.append({
                "identifiant_juridique": doc_id,
                "titre": r.get("titre", ""),
                "serie": r.get("serie", ""),
                "division": r.get("division", ""),
                "contenu_html": r.get("contenu_html", ""),
                "contenu": r.get("contenu", ""),
                "permalien": r.get("permalien", ""),
                "debut_de_validite": r.get("debut_de_validite", ""),
            })

        offset += API_LIMIT
        page += 1

        if len(results) < API_LIMIT:
            break

    print(f"  Downloaded {len(docs)} documents")
    return docs


# ── Diff ──────────────────────────────────────────────────────────────
def compute_hash(doc: dict) -> str:
    """Stable hash of document content fields (ignoring volatile metadata)."""
    content = doc.get("contenu_html", "") + doc.get("contenu", "")
    return hashlib.md5(content.encode()).hexdigest()


def diff_documents(new_docs: list[dict], snapshot_path: Path) -> dict:
    """Compare new download against previous snapshot.

    Returns: {
        "new": [...], "updated": [...], "removed": [...], "unchanged": [...],
        "snapshot_count": int, "new_count": int,
    }
    """
    new_by_id = {d["identifiant_juridique"]: d for d in new_docs}
    new_by_id_dedup = {}
    for k, v in new_by_id.items():
        new_by_id_dedup[k] = v  # last occurrence wins if dupes

    if not snapshot_path.exists():
        print("  No previous snapshot -- all documents are new")
        return {
            "new": list(new_by_id_dedup.values()),
            "updated": [], "removed": [], "unchanged": [],
            "snapshot_count": 0, "new_count": len(new_docs),
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

    old_by_id = {d["identifiant_juridique"]: d for d in old_docs}

    new_ids = set(new_by_id_dedup.keys())
    old_ids = set(old_by_id.keys())

    added_ids = new_ids - old_ids
    removed_ids = old_ids - new_ids
    common_ids = new_ids & old_ids

    added = [new_by_id_dedup[i] for i in added_ids]
    removed = [old_by_id[i] for i in removed_ids]
    updated = []
    unchanged = []

    for cid in common_ids:
        if compute_hash(new_by_id_dedup[cid]) != compute_hash(old_by_id[cid]):
            updated.append(new_by_id_dedup[cid])
        else:
            unchanged.append(new_by_id_dedup[cid])

    return {
        "new": added, "updated": updated, "removed": removed, "unchanged": unchanged,
        "snapshot_count": len(old_docs), "new_count": len(new_docs),
    }


# ── Parse ─────────────────────────────────────────────────────────────
def parse_documents(raw_docs: list[dict]) -> list[dict]:
    """Parse downloaded BOFIP JSON into RawDocument format (same as setup.py)."""
    from bofip_agentic.text_utils import normalize_whitespace, extract_legal_refs

    parsed = []
    for d in raw_docs:
        raw_html = d.get("contenu_html", "")
        paragraphs = []
        sections = []
        legal_refs = []

        if raw_html:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw_html, "lxml")
                body = soup.body or soup

                headings = body.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                for idx, h in enumerate(headings):
                    h_text = normalize_whitespace(h.get_text())
                    if h_text:
                        section_id = f"{d['identifiant_juridique']}__section_{idx:03d}"
                        level = int(h.name[1])
                        sections.append({
                            "section_id": section_id, "parent_section_id": None,
                            "level": level, "order_index": idx,
                            "title": h_text, "anchor": None, "path": [h_text],
                        })

                for idx, p in enumerate(body.find_all(["p", "li", "blockquote"])):
                    text = normalize_whitespace(p.get_text())
                    if not text:
                        continue
                    refs = extract_legal_refs(text)
                    legal_refs.extend(refs)
                    paragraphs.append({
                        "paragraph_id": f"{d['identifiant_juridique']}__para_{idx:05d}",
                        "section_id": None, "order_index": idx,
                        "html_tag": p.name,
                        "anchor": p.get("id") or p.get("name"),
                        "paragraph_number": None,
                        "text": text, "legal_refs": refs, "links": [],
                    })
            except Exception:
                pass

        if not paragraphs and d.get("contenu"):
            raw_text = normalize_whitespace(d.get("contenu", ""))
            refs = extract_legal_refs(raw_text)
            paragraphs = [{
                "paragraph_id": f"{d['identifiant_juridique']}__para_00000",
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
            "document_id": d["identifiant_juridique"],
            "boi_reference": d["identifiant_juridique"],
            "title": d["titre"],
            "document_type": "Contenu",
            "content_type": "Commentaire",
            "publication_date": d.get("debut_de_validite", ""),
            "source_url": d.get("permalien", ""),
            "language": "fr-FR",
            "subjects": [d.get("serie", "")],
            "identifiers": [],
            "relations": [],
            "category_path": ["Commentaire", d.get("serie", "")],
            "raw_xml_path": "", "raw_html_path": "",
            "version_status": None,
            "sections": sections,
            "paragraphs": paragraphs,
            "tables": [], "internal_links": [],
            "legal_refs": deduped_refs,
            "html_title": None,
            "raw_text_length": len(d.get("contenu", "")),
        })

    return parsed


# ── Chunk ─────────────────────────────────────────────────────────────
def chunk_documents(parsed_docs: list[dict]) -> list[dict]:
    """Build section_window chunks (same as setup.py)."""
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
def embed_corpus(docs: list[dict], chunks: list[dict], device: str) -> tuple[np.ndarray, np.ndarray]:
    """Encode documents and chunks with E5-large."""
    from bofip_agentic.dense_retrieval import DenseEncoder
    from bofip_agentic.models import raw_document_from_dict, chunk_node_from_dict

    model_name = "intfloat/multilingual-e5-large"
    print(f"  Loading {model_name}...")
    encoder = DenseEncoder(model_name, device=device)

    doc_objs = [raw_document_from_dict(d) for d in docs]
    chunk_objs = [chunk_node_from_dict(c) for c in chunks]

    print(f"  Encoding {len(doc_objs)} documents...")
    t0 = time.time()
    doc_emb = encoder.encode_documents(doc_objs, mode="sections_firstpara", batch_size=32)
    print(f"    done in {time.time() - t0:.1f}s, shape {doc_emb.shape}")

    print(f"  Encoding {len(chunk_objs)} chunks...")
    t0 = time.time()
    chunk_emb = encoder.encode_chunks(chunk_objs, mode="full", batch_size=32)
    print(f"    done in {time.time() - t0:.1f}s, shape {chunk_emb.shape}")

    return doc_emb, chunk_emb


# ── Validate ──────────────────────────────────────────────────────────
def validate_corpus(docs_path: Path, chunks_path: Path,
                    doc_npy: Path, chunk_npy: Path,
                    min_docs: int = 5000) -> list[str]:
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

    doc_count = sum(1 for _ in open(docs_path, encoding="utf-8"))
    chunk_count = sum(1 for _ in open(chunks_path, encoding="utf-8"))
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
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

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
    print(f"    New:       {len(diff['new'])}")
    print(f"    Updated:   {len(diff['updated'])}")
    print(f"    Removed:   {len(diff['removed'])}")
    print(f"    Unchanged: {len(diff['unchanged'])}")

    if args.check:
        return 0

    total_changes = len(diff['new']) + len(diff['updated']) + len(diff['removed'])
    if total_changes == 0:
        print("\nNo changes. Corpus is up to date.")
        return 0

    if not args.force:
        print(f"\n{total_changes} documents changed. Proceed? [y/N] ", end="", flush=True)
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
    doc_emb, chunk_emb = embed_corpus(parsed, chunk_dicts, args.device)

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

    # Atomic swap: remove old interim, rename temp
    if INTERIM.exists():
        shutil.rmtree(INTERIM)
    INTERIM_TMP.rename(INTERIM)

    # Clear BM25 cache (will rebuild on next query)
    bm25_cache = Path.home() / ".cache" / "bofip_rag"
    if bm25_cache.exists():
        shutil.rmtree(bm25_cache)
        print("  BM25 cache cleared (rebuilt on next run)")

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
