"""
One-click reproducible setup for BOFIP Agentic RAG.

Downloads BOFIP data from data.economie.gouv.fr, parses, chunks, and builds embeddings.

Usage:
    $env:PYTHONPATH="src"
    python scripts/setup.py                     # full pipeline (download + parse + chunk + embed)
    python scripts/setup.py --download-only     # only download BOFIP data
    python scripts/setup.py --skip-download     # skip download, parse+chunk+embed existing data
    python scripts/setup.py --from-sibling      # copy from sibling bofip-rag-cleanroom project
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from bofip_agentic.dense_retrieval import DenseEncoder
from bofip_agentic.jsonio import read_jsonl, write_json
from bofip_agentic.models import chunk_node_from_dict, raw_document_from_dict
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
BOFIP_LIMIT = 100  # records per page

# Theme mapping: BOFIP serie -> tax category
THEME_FILTER = ["IR", "IS", "TVA", "BIC", "BNC", "BA", "CF", "ENR", "IF", "PAT", "RPPM", "RFPI", "CTX", "REC", "INT"]
THEME_WHERE = " OR ".join("serie='{}'".format(t) for t in THEME_FILTER)


def download_bofip(max_docs: int = 6000) -> int:
    """Download BOFIP commentary documents from data.economie.gouv.fr API.
    Returns the number of documents downloaded."""
    try:
        import requests
    except ImportError:
        sys.exit("ERROR: requests package required. Run: pip install requests")

    print("Downloading BOFIP data from data.economie.gouv.fr...")
    print("  Filtering categories: {}".format(", ".join(THEME_FILTER)))
    print()

    docs = []
    offset = 0
    page = 1
    errors = 0

    while len(docs) < max_docs:
        if page == 1 or page % 10 == 0:
            print("  Page {} ({} docs so far)...".format(page, len(docs)))

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
                print("  Too many errors, stopping download at {} docs".format(len(docs)))
                break
            print("  Error on page {}: {} — retrying...".format(page, e))
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

    print("  Downloaded {} documents -> {}".format(len(docs), out))
    return len(docs)


def parse_raw_docs(raw_input: Path, output: Path) -> int:
    """Parse downloaded BOFIP JSON into RawDocument format JSONL."""
    print("\nParsing raw documents...")

    if output.exists():
        print("  {} already exists ({} — skipping parse".format(output, output.stat().st_size))
        return 0

    if not raw_input.exists():
        print("  ERROR: no raw input at {}".format(raw_input))
        print("  Run with --download to download BOFIP data first.")
        return 0

    docs = read_jsonl(raw_input)
    parsed = []
    for d in docs:
        # The API already gives us structured data; convert to RawDocument format
        parsed.append(d)

    write_json(output, parsed)
    print("  Written {} documents -> {}".format(len(parsed), output))
    return len(parsed)


def build_chunks(docs_path: Path, output: Path) -> int:
    """Build section_window chunks from raw documents."""
    print("\nBuilding chunks...")

    if output.exists():
        print("  {} already exists ({} — skipping chunk build".format(output, output.stat().st_size))
        return 0

    if not docs_path.exists():
        print("  ERROR: no docs at {}".format(docs_path))
        return 0

    from bofip_agentic.chunking import build_chunks_for_documents

    docs = [raw_document_from_dict(d) for d in read_jsonl(docs_path)]
    print("  Loaded {} documents".format(len(docs)))

    chunks = build_chunks_for_documents(docs, strategy="section_window", max_tokens=350, min_tokens=40)
    chunk_dicts = [
        {
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
        }
        for c in chunks
    ]

    write_json(output, chunk_dicts)
    print("  Written {} chunks -> {}".format(len(chunk_dicts), output))
    return len(chunk_dicts)


def build_embeddings(docs_path: Path, chunks_path: Path, device: str = "cuda") -> None:
    """Build dense embeddings for documents and chunks."""
    print("\nBuilding dense embeddings (device: {})...".format(device))

    # Check if already built
    if DOC_DENSE_NPY.exists() and CHUNK_DENSE_NPY.exists():
        print("  Embeddings already exist:")
        print("    {} ({})".format(DOC_DENSE_NPY, DOC_DENSE_NPY.stat().st_size))
        print("    {} ({})".format(CHUNK_DENSE_NPY, CHUNK_DENSE_NPY.stat().st_size))
        return

    if not docs_path.exists() or not chunks_path.exists():
        print("  ERROR: need docs and chunks JSONL. Run parse + chunk first.")
        return

    documents = [raw_document_from_dict(d) for d in read_jsonl(docs_path)]
    chunks = [chunk_node_from_dict(c) for c in read_jsonl(chunks_path)]
    print("  Documents: {} | Chunks: {}".format(len(documents), len(chunks)))

    model_name = "intfloat/multilingual-e5-large"
    print("  Loading model: {}...".format(model_name))
    encoder = DenseEncoder(model_name, device=device)

    # Document embeddings
    if DOC_DENSE_NPY.exists():
        doc_emb = np.load(DOC_DENSE_NPY)
    else:
        print("  Encoding {} documents...".format(len(documents)))
        t0 = time.time()
        doc_emb = encoder.encode_documents(documents, mode="sections_firstpara", batch_size=32)
        print("  Documents done in {:.1f}s | shape: {}".format(time.time() - t0, doc_emb.shape))
        np.save(DOC_DENSE_NPY, doc_emb)
        print("  Saved: {}".format(DOC_DENSE_NPY))

    # Chunk embeddings
    if CHUNK_DENSE_NPY.exists():
        chunk_emb = np.load(CHUNK_DENSE_NPY)
    else:
        print("  Encoding {} chunks...".format(len(chunks)))
        t0 = time.time()
        chunk_emb = encoder.encode_chunks(chunks, mode="full", batch_size=32)
        print("  Chunks done in {:.1f}s | shape: {}".format(time.time() - t0, chunk_emb.shape))
        np.save(CHUNK_DENSE_NPY, chunk_emb)
        print("  Saved: {}".format(CHUNK_DENSE_NPY))

    print("  Embeddings ready.")


def copy_from_sibling(sibling_path: Path | None = None) -> bool:
    """Copy corpus files from sibling bofip-rag-cleanroom project."""
    if sibling_path is None:
        sibling_path = PROJECT_ROOT.parent / "bofip-rag-cleanroom"

    sibling_interim = sibling_path / "data" / "interim"
    if not sibling_interim.exists():
        print("Sibling project not found at: {}".format(sibling_path))
        return False

    files_to_copy = [
        "raw_docs_sample_5666.jsonl",
        "chunks_section_window_sample_5666.jsonl",
        "doc_dense_cache_5666_sections_firstpara_e5large.npy",
        "chunk_dense_cache_5666_full_e5large.npy",
    ]

    for f in files_to_copy:
        src = sibling_interim / f
        if src.exists():
            shutil.copy2(src, INTERIM_DIR / f)
            print("  Copied: {} ({} MB)".format(f, src.stat().st_size // (1024 * 1024)))
        else:
            print("  Not found: {}".format(f))

    # Also copy models if available
    sibling_models = sibling_path / "data" / "models"
    local_models = DATA_DIR / "models"
    if sibling_models.exists():
        local_models.mkdir(parents=True, exist_ok=True)
        for item in sibling_models.iterdir():
            if item.is_dir():
                dst = local_models / item.name
                if not dst.exists():
                    shutil.copytree(item, dst)
                    print("  Copied model: {}".format(item.name))

    return True


def main():
    p = argparse.ArgumentParser(description="One-click BOFIP data setup")
    p.add_argument("--download-only", action="store_true", help="Only download raw data")
    p.add_argument("--skip-download", action="store_true", help="Skip download, use existing raw data")
    p.add_argument("--from-sibling", action="store_true", help="Copy from sibling bofip-rag-cleanroom project")
    p.add_argument("--sibling-path", type=str, default="", help="Path to sibling project")
    p.add_argument("--device", type=str, default="cuda", help="Device for embeddings (cuda/cpu)")
    p.add_argument("--max-docs", type=int, default=6000, help="Max docs to download")
    args = p.parse_args()

    print("=" * 60)
    print("BOFIP Agentic RAG — Data Setup")
    print("=" * 60)

    # Option C: Copy from sibling
    if args.from_sibling:
        sibling = Path(args.sibling_path) if args.sibling_path else None
        if copy_from_sibling(sibling):
            print("\nData copied from sibling project. Ready to run.")
            print("Next: python scripts/eval_agent.py")
            return 0
        else:
            print("Falling back to download...")

    # Option A: Full pipeline
    if not args.skip_download:
        print("\n[1/4] Downloading BOFIP data...")
        raw_input = RAW_DIR / "bofip_download.jsonl"
        if raw_input.exists():
            print("  Already downloaded: {}".format(raw_input))
        else:
            n = download_bofip(max_docs=args.max_docs)
            if n == 0:
                print("  ERROR: download failed. Check internet connection.")
                print("  Alternatively, use --from-sibling or --skip-download")
                return 1

        if args.download_only:
            print("\nDownload complete. Raw data at: {}".format(raw_input))
            return 0
    else:
        raw_input = RAW_DIR / "bofip_download.jsonl"

    # [2/4] Parse
    print("\n[2/4] Parsing raw documents...")
    if RAW_DOCS_OUT.exists():
        print("  Already parsed: {}".format(RAW_DOCS_OUT))
    else:
        n = parse_raw_docs(raw_input, RAW_DOCS_OUT)
        if n == 0:
            return 1

    # [3/4] Build chunks
    print("\n[3/4] Building chunks...")
    if CHUNKS_OUT.exists():
        print("  Chunks already built: {}".format(CHUNKS_OUT))
    else:
        n = build_chunks(RAW_DOCS_OUT, CHUNKS_OUT)
        if n == 0:
            return 1

    # [4/4] Build embeddings
    print("\n[4/4] Building embeddings...")
    build_embeddings(RAW_DOCS_OUT, CHUNKS_OUT, device=args.device)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("Next: python scripts/eval_agent.py")
    print("      python scripts/benchmark_agentic.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
