"""
Legacy local setup helper for BOFiP Agentic RAG.

For the authoritative full-corpus rebuild, use scripts/sync.py. This helper is
kept for local parser/chunker development and one-off artifact creation only.
It never copies artifacts from sibling projects.

Usage:
    $env:PYTHONPATH="src"
    # Local pipeline (download + parse + chunk + embed)
    python scripts/setup.py

    # Download only
    python scripts/setup.py --download-only

    # Skip download, parse+chunk+embed existing data
    python scripts/setup.py --skip-download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from bofip_agentic.chunking import build_chunks_for_documents
from bofip_agentic.dense_retrieval import DenseEncoder
from bofip_agentic.jsonio import read_jsonl
from bofip_agentic.models import chunk_node_from_dict, raw_document_from_dict

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

RAW_DOCS_OUT = INTERIM_DIR / "raw_docs.jsonl"
CHUNKS_OUT = INTERIM_DIR / "chunks.jsonl"
DOC_DENSE_NPY = INTERIM_DIR / "doc_dense_cache.npy"
CHUNK_DENSE_NPY = INTERIM_DIR / "chunk_dense_cache.npy"

BOFIP_API = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/bofip-vigueur/records"
BOFIP_LIMIT = 100

SERIES_FILTER: list[str] = []
PGP_ID_RE = re.compile(r"/bofip/([^/?#]+)-PGP")
PERMALINK_IDENTIFIANT_RE = re.compile(r"identifiant=([^&#]+)")


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def source_record_key(doc: dict) -> str:
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


def remove_existing_files(paths: list[Path]) -> list[Path]:
    removed: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        path.unlink()
        removed.append(path)
    return removed


def build_download_params(*, offset: int, series_filter: list[str] | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "limit": BOFIP_LIMIT,
        "offset": offset,
        "select": "identifiant_juridique,titre,serie,division,contenu,contenu_html,permalien,debut_de_validite",
    }
    requested_series = SERIES_FILTER if series_filter is None else series_filter
    if requested_series:
        params["where"] = " OR ".join(f"serie='{serie}'" for serie in requested_series)
    return params


def download_bofip(max_docs: int = 0) -> int:
    """Download BOFIP commentary documents from data.economie.gouv.fr API."""
    try:
        import requests
    except ImportError:
        sys.exit("ERROR: requests package required. Run: pip install requests")

    print("Downloading BOFIP data from data.economie.gouv.fr...")
    print("  Full corpus: no series filter")

    docs = []
    offset = 0
    page = 1
    errors = 0

    while max_docs <= 0 or len(docs) < max_docs:
        if page % 10 == 1:
            print(f"  Page {page} ({len(docs)} docs so far)...")

        params = build_download_params(offset=offset)

        try:
            resp = requests.get(BOFIP_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            errors += 1
            if errors > 3:
                print(f"  Too many errors, stopping at {len(docs)} docs")
                break
            print(f"  Error page {page}: {e} — retrying...")
            time.sleep(2)
            continue

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            doc_id = text_value(r.get("identifiant_juridique"))
            if not doc_id:
                continue
            publication_date = text_value(r.get("debut_de_validite"))
            source_url = text_value(r.get("permalien"))
            series = text_value(r.get("serie"))
            source_record_id = source_record_key(
                {
                    "identifiant_juridique": doc_id,
                    "debut_de_validite": publication_date,
                    "permalien": source_url,
                }
            )
            docs.append({
                "source_record_id": source_record_id,
                "document_id": source_record_id,
                "boi_reference": doc_id,
                "title": text_value(r.get("titre")),
                "series": series,
                "division": text_value(r.get("division")),
                "publication_date": publication_date,
                "source_url": source_url,
                "content_type": "Commentaire",
                "document_type": "Contenu",
                "language": "fr-FR",
                "category_path": ["Commentaire", series],
                "subjects": [series],
                "raw_html": text_value(r.get("contenu_html")),
                "raw_text": text_value(r.get("contenu")),
            })
            if max_docs > 0 and len(docs) >= max_docs:
                break

        offset += BOFIP_LIMIT
        page += 1

        if len(results) < BOFIP_LIMIT:
            break

    out = RAW_DIR / "bofip_download.jsonl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"  Downloaded {len(docs)} documents -> {out}")
    return len(docs)


def parse_raw_docs(raw_input: Path, output: Path) -> int:
    """Parse downloaded BOFIP JSON into proper RawDocument format.

    Converts the flat API JSON format (with raw_html field) into structured
    RawDocument objects suitable for chunking. Uses html_parser to extract
    sections, paragraphs from the raw HTML content.
    """
    print("\nParsing raw documents into structured format...")

    if output.exists():
        print(f"  {output} already exists — skipping parse")
        return 0

    if not raw_input.exists():
        print(f"  ERROR: no raw input at {raw_input}")
        print("  Run with default (no flags) to download BOFIP data first.")
        return 0

    from bofip_agentic.html_parser import parse_html_content
    from bofip_agentic.text_utils import extract_legal_refs, normalize_whitespace

    docs = read_jsonl(raw_input)
    parsed = []

    for d in docs:
        raw_html = d.get("raw_html", "")
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
                html_payload = parse_html_content(raw_html, document_id=d["document_id"])
            except Exception:
                pass  # HTML parsing error, fall back to plain text

        paragraphs = list(html_payload.get("paragraphs", []))
        legal_refs = list(html_payload.get("legal_refs", []))

        # If no paragraphs extracted from HTML, use raw text as single paragraph
        if not paragraphs and not html_payload.get("tables") and d.get("raw_text"):
            raw_text = normalize_whitespace(d.get("raw_text", ""))
            refs = extract_legal_refs(raw_text)
            paragraphs = [{
                "paragraph_id": f"{d['document_id']}__para_00000",
                "section_id": None,
                "order_index": 0,
                "html_tag": "p",
                "anchor": None,
                "paragraph_number": None,
                "text": raw_text,
                "legal_refs": refs,
                "links": [],
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
            "document_id": d["document_id"],
            "boi_reference": d["boi_reference"],
            "title": d["title"],
            "document_type": d.get("document_type", "Contenu"),
            "content_type": d.get("content_type", "Commentaire"),
            "publication_date": d.get("publication_date", ""),
            "source_url": d.get("source_url", ""),
            "language": d.get("language", "fr-FR"),
            "subjects": d.get("subjects", []),
            "identifiers": [],
            "relations": [],
            "category_path": d.get("category_path", []),
            "raw_xml_path": "",
            "raw_html_path": "",
            "version_status": None,
            "sections": html_payload.get("sections", []),
            "paragraphs": paragraphs,
            "tables": html_payload.get("tables", []),
            "internal_links": html_payload.get("internal_links", []),
            "legal_refs": deduped_refs,
            "html_title": html_payload.get("html_title"),
            "raw_text_length": len(d.get("raw_text", "")),
        })

    with open(output, "w", encoding="utf-8") as f:
        for doc in parsed:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"  Written {len(parsed)} documents -> {output}")
    return len(parsed)


def build_chunks(docs_path: Path, output: Path, max_tokens: int = 350) -> int:
    """Build section_window chunks from parsed documents."""
    print("\nBuilding chunks (section_window strategy)...")

    if output.exists():
        print(f"  {output} already exists — skipping chunk build")
        return 0

    if not docs_path.exists():
        print(f"  ERROR: no docs at {docs_path}")
        return 0

    docs = [raw_document_from_dict(d) for d in read_jsonl(docs_path)]
    print(f"  Loaded {len(docs)} documents")

    chunks = build_chunks_for_documents(docs, strategy="section_window", max_tokens=max_tokens, min_tokens=40)
    chunk_dicts = []
    for c in chunks:
        chunk_dicts.append({
            "chunk_id": c.chunk_id,
            "source_type": "BOFIP",
            "document_id": c.document_id,
            "boi_reference": c.boi_reference,
            "doc_version": c.doc_version,
            "strategy": c.strategy,
            "section_id": c.section_id,
            "section_path": c.section_path,
            "paragraph_range": c.paragraph_range,
            "text": c.text,
            "token_count": c.token_count,
            "chunk_kind": c.chunk_kind,
            "legal_refs": c.legal_refs,
        })

    with open(output, "w", encoding="utf-8") as f:
        for c in chunk_dicts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"  Written {len(chunk_dicts)} chunks -> {output}")
    return len(chunk_dicts)


def build_embeddings(docs_path: Path, chunks_path: Path, device: str = "cuda") -> None:
    """Build dense embeddings for documents and chunks using E5-large."""
    print(f"\nBuilding dense embeddings (device: {device})...")

    if DOC_DENSE_NPY.exists() and CHUNK_DENSE_NPY.exists():
        print("  Embeddings already exist:")
        print(f"    {DOC_DENSE_NPY} ({DOC_DENSE_NPY.stat().st_size / 1e6:.1f} MB)")
        print(f"    {CHUNK_DENSE_NPY} ({CHUNK_DENSE_NPY.stat().st_size / 1e6:.1f} MB)")
        return

    if not docs_path.exists() or not chunks_path.exists():
        print("  ERROR: need docs and chunks JSONL. Run parse + chunk first.")
        return

    documents = [raw_document_from_dict(d) for d in read_jsonl(docs_path)]
    chunks = [chunk_node_from_dict(c) for c in read_jsonl(chunks_path)]
    print(f"  Documents: {len(documents)} | Chunks: {len(chunks)}")

    model_name = "intfloat/multilingual-e5-large"
    print(f"  Loading model: {model_name}...")
    encoder = DenseEncoder(model_name, device=device)

    if not DOC_DENSE_NPY.exists():
        print(f"  Encoding {len(documents)} documents...")
        t0 = time.time()
        doc_emb = encoder.encode_documents(documents, mode="sections_firstpara", batch_size=32)
        print(f"  Documents done in {time.time() - t0:.1f}s | shape: {doc_emb.shape}")
        np.save(DOC_DENSE_NPY, doc_emb)
        print(f"  Saved: {DOC_DENSE_NPY}")

    if not CHUNK_DENSE_NPY.exists():
        print(f"  Encoding {len(chunks)} chunks...")
        t0 = time.time()
        chunk_emb = encoder.encode_chunks(chunks, mode="full", batch_size=32)
        print(f"  Chunks done in {time.time() - t0:.1f}s | shape: {chunk_emb.shape}")
        np.save(CHUNK_DENSE_NPY, chunk_emb)
        print(f"  Saved: {CHUNK_DENSE_NPY}")

    print("  Embeddings ready.")


def main():
    p = argparse.ArgumentParser(description="Legacy local BOFiP data setup helper")
    p.add_argument("--download-only", action="store_true", help="Only download raw data from API")
    p.add_argument("--skip-download", action="store_true", help="Skip download, use existing raw data")
    p.add_argument("--device", type=str, default="cuda", help="Device for embeddings (cuda/cpu)")
    p.add_argument("--max-docs", type=int, default=0, help="Max docs to download from API; 0 means full corpus")
    p.add_argument("--force", action="store_true", help="Rebuild parsed docs, chunks, and embeddings")
    args = p.parse_args()

    print("=" * 60)
    print("BOFiP Agentic RAG - legacy local setup helper")
    print("=" * 60)
    print("Authoritative full-corpus refresh: python scripts/sync.py --rebuild --force")

    raw_input = RAW_DIR / "bofip_download.jsonl"

    if not args.skip_download:
        print("\n[1/4] Downloading BOFIP data...")
        if raw_input.exists():
            print(f"  Already downloaded: {raw_input}")
        else:
            n = download_bofip(max_docs=args.max_docs)
            if n == 0:
                print("  ERROR: download failed.")
                return 1

        if args.download_only:
            print(f"\nDownload complete. Raw data at: {raw_input}")
            return 0

    if args.force:
        print("\nForce rebuild: removing derived artifacts...")
        removed = remove_existing_files([RAW_DOCS_OUT, CHUNKS_OUT, DOC_DENSE_NPY, CHUNK_DENSE_NPY])
        for path in removed:
            print(f"  Removed: {path}")

    print("\n[2/4] Parsing raw documents...")
    if RAW_DOCS_OUT.exists():
        print(f"  Already parsed: {RAW_DOCS_OUT}")
    else:
        n = parse_raw_docs(raw_input, RAW_DOCS_OUT)
        if n == 0:
            return 1

    print("\n[3/4] Building chunks...")
    if CHUNKS_OUT.exists():
        print(f"  Chunks already built: {CHUNKS_OUT}")
    else:
        n = build_chunks(RAW_DOCS_OUT, CHUNKS_OUT)
        if n == 0:
            return 1

    print("\n[4/4] Building embeddings...")
    build_embeddings(RAW_DOCS_OUT, CHUNKS_OUT, device=args.device)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("Next: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
