"""
One-click reproducible setup for BOFIP Agentic RAG.

Downloads BOFIP data from data.economie.gouv.fr, parses, chunks, and builds embeddings.

Usage:
    $env:PYTHONPATH="src"
    # Full pipeline (download + parse + chunk + embed)
    python scripts/setup.py

    # Download only
    python scripts/setup.py --download-only

    # Skip download, parse+chunk+embed existing data
    python scripts/setup.py --skip-download

    # Copy data from a pre-built sibling project
    python scripts/setup.py --copy-from <path/to/bofip-rag-cleanroom>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from bofip_agentic.chunking import build_chunks_for_documents
from bofip_agentic.dense_retrieval import DenseEncoder
from bofip_agentic.jsonio import read_jsonl
from bofip_agentic.models import ChunkNode, RawDocument, chunk_node_from_dict, raw_document_from_dict
from bofip_agentic.settings import BOFIP_DATA_ROOT

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

THEME_FILTER = ["IR", "IS", "TVA", "BIC", "BNC", "BA", "CF", "ENR", "IF", "PAT", "RPPM", "RFPI", "CTX", "REC", "INT"]
THEME_WHERE = " OR ".join(f"serie='{t}'" for t in THEME_FILTER)


def download_bofip(max_docs: int = 6000) -> int:
    """Download BOFIP commentary documents from data.economie.gouv.fr API."""
    try:
        import requests
    except ImportError:
        sys.exit("ERROR: requests package required. Run: pip install requests")

    print("Downloading BOFIP data from data.economie.gouv.fr...")
    print(f"  Filtering categories: {', '.join(THEME_FILTER)}")

    docs = []
    offset = 0
    page = 1
    errors = 0

    while len(docs) < max_docs:
        if page % 10 == 1:
            print(f"  Page {page} ({len(docs)} docs so far)...")

        params = {
            "limit": BOFIP_LIMIT,
            "offset": offset,
            "select": "identifiant_juridique,titre,serie,division,contenu,contenu_html,permalien,debut_de_validite",
            "where": THEME_WHERE,
        }

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
            doc_id = r.get("identifiant_juridique", "")
            if not doc_id:
                continue
            docs.append({
                "document_id": doc_id,
                "boi_reference": doc_id,
                "title": r.get("titre", ""),
                "series": r.get("serie", ""),
                "division": r.get("division", ""),
                "publication_date": r.get("debut_de_validite", ""),
                "source_url": r.get("permalien", ""),
                "content_type": "Commentaire",
                "document_type": "Contenu",
                "language": "fr-FR",
                "category_path": ["Commentaire", r.get("serie", "")],
                "subjects": [r.get("serie", "")],
                "raw_html": r.get("contenu_html", ""),
                "raw_text": r.get("contenu", ""),
            })

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

    from bofip_agentic.text_utils import normalize_whitespace, extract_legal_refs

    docs = read_jsonl(raw_input)
    parsed = []

    for d in docs:
        raw_html = d.get("raw_html", "")
        paragraphs = []
        sections = []
        legal_refs = []

        # Basic HTML parsing: extract text blocks
        if raw_html:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw_html, "lxml")
                body = soup.body or soup

                # Extract sections from headings
                headings = body.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                for idx, h in enumerate(headings):
                    h_text = normalize_whitespace(h.get_text())
                    if h_text:
                        section_id = f"{d['document_id']}__section_{idx:03d}"
                        level = int(h.name[1])
                        sections.append({
                            "section_id": section_id,
                            "parent_section_id": None,
                            "level": level,
                            "order_index": idx,
                            "title": h_text,
                            "anchor": None,
                            "path": [h_text],
                        })

                # Extract paragraphs
                for idx, p in enumerate(body.find_all(["p", "li", "blockquote"])):
                    text = normalize_whitespace(p.get_text())
                    if not text:
                        continue
                    refs = extract_legal_refs(text)
                    legal_refs.extend(refs)
                    paragraphs.append({
                        "paragraph_id": f"{d['document_id']}__para_{idx:05d}",
                        "section_id": None,
                        "order_index": idx,
                        "html_tag": p.name,
                        "anchor": p.get("id") or p.get("name"),
                        "paragraph_number": None,
                        "text": text,
                        "legal_refs": refs,
                        "links": [],
                    })
            except ImportError:
                pass  # BeautifulSoup not installed, fall back to plain text
            except Exception:
                pass  # HTML parsing error, fall back to plain text

        # If no paragraphs extracted from HTML, use raw text as single paragraph
        if not paragraphs and d.get("raw_text"):
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

        from collections import OrderedDict
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
            "sections": sections,
            "paragraphs": paragraphs,
            "tables": [],
            "internal_links": [],
            "legal_refs": deduped_refs,
            "html_title": None,
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


def copy_from_project(source_path: Path) -> bool:
    """Copy pre-built corpus files from another project."""
    if not source_path.exists():
        print(f"Source project not found at: {source_path}")
        return False

    source_interim = source_path / "data" / "interim"
    if not source_interim.exists():
        print(f"No data/interim in source: {source_path}")
        return False

    files_to_copy = [
        "raw_docs_sample_5666.jsonl",
        "chunks_section_window_sample_5666.jsonl",
        "doc_dense_cache_5666_sections_firstpara_e5large.npy",
        "chunk_dense_cache_5666_full_e5large.npy",
    ]

    for f in files_to_copy:
        src = source_interim / f
        if src.exists():
            shutil.copy2(src, INTERIM_DIR / f)
            print(f"  Copied: {f} ({src.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  Not found: {f}")

    source_models = source_path / "data" / "models"
    local_models = DATA_DIR / "models"
    if source_models.exists():
        local_models.mkdir(parents=True, exist_ok=True)
        for item in source_models.iterdir():
            if item.is_dir() and ".cache" not in str(item):
                dst = local_models / item.name
                if not dst.exists():
                    shutil.copytree(item, dst)
                    print(f"  Copied model: {item.name}")

    return True


def main():
    p = argparse.ArgumentParser(description="One-click BOFIP data setup")
    p.add_argument("--download-only", action="store_true", help="Only download raw data from API")
    p.add_argument("--skip-download", action="store_true", help="Skip download, use existing raw data")
    p.add_argument("--copy-from", type=str, default="", help="Copy pre-built data from another project")
    p.add_argument("--device", type=str, default="cuda", help="Device for embeddings (cuda/cpu)")
    p.add_argument("--max-docs", type=int, default=6000, help="Max docs to download from API")
    args = p.parse_args()

    print("=" * 60)
    print("BOFIP Agentic RAG — Data Setup")
    print("=" * 60)

    if args.copy_from:
        source = Path(args.copy_from)
        if copy_from_project(source):
            print("\nData copied successfully. Ready to run.")
            return 0
        print("Falling back to download...")

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
