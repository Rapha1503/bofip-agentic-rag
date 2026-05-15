"""Build dense embeddings cache from chunks JSONL. Run after phase2_build_chunks.py.

Usage:
    $env:PYTHONPATH="src"
    python scripts/build_embeddings.py --corpus data/interim --model e5-large --device cuda
"""
from __future__ import annotations

import argparse, sys, time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from bofip_agentic.dense_retrieval import DenseEncoder
from bofip_agentic.jsonio import read_jsonl, write_json
from bofip_agentic.models import chunk_node_from_dict, raw_document_from_dict


def main():
    p = argparse.ArgumentParser(description="Build dense embeddings for BOFIP corpus")
    p.add_argument("--corpus", type=str, default="data/interim", help="Directory with JSONL files")
    p.add_argument("--model", type=str, default="intfloat/multilingual-e5-large", help="Embedding model")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    corpus = Path(args.corpus)
    raw_docs_path = corpus / "raw_docs_sample_5666.jsonl"
    chunks_path = corpus / "chunks_section_window_sample_5666.jsonl"

    if not raw_docs_path.exists():
        print("ERROR: {} not found. Run phase1_extract_raw_documents.py first.".format(raw_docs_path))
        print("   Or set BOFIP_DATA_ROOT to point to existing BOFIP corpus.")
        print("   See https://github.com/Rapha1503/bofip-rag-cleanroom for the data pipeline.")
        return 1

    print("Loading documents from {}...".format(raw_docs_path))
    documents = [raw_document_from_dict(d) for d in read_jsonl(raw_docs_path)]
    print("  {} documents loaded".format(len(documents)))

    print("Loading chunks from {}...".format(chunks_path))
    chunks = [chunk_node_from_dict(c) for c in read_jsonl(chunks_path)]
    print("  {} chunks loaded".format(len(chunks)))

    print("Loading model: {}...".format(args.model))
    encoder = DenseEncoder(args.model, device=args.device)

    # -- Document embeddings --
    doc_cache = corpus / "doc_dense_cache.npy"
    doc_meta = corpus / "doc_dense_cache.json"

    if doc_cache.exists():
        print("Loading cached doc embeddings from {}...".format(doc_cache))
        doc_emb = np.load(doc_cache)
    else:
        print("Encoding {} documents (this may take a few minutes)...".format(len(documents)))
        t0 = time.time()
        doc_emb = encoder.encode_documents(documents, mode="sections_firstpara", batch_size=args.batch_size)
        print("  Done in {:.1f}s | shape: {}".format(time.time() - t0, doc_emb.shape))
        np.save(doc_cache, doc_emb)
        write_json(doc_meta, {
            "generated_at": datetime.now(UTC).isoformat(),
            "model": args.model,
            "mode": "sections_firstpara",
            "shape": list(doc_emb.shape),
        })
        print("  Saved to {}".format(doc_cache))

    # -- Chunk embeddings --
    chunk_cache = corpus / "chunk_dense_cache.npy"
    chunk_meta = corpus / "chunk_dense_cache.json"

    if chunk_cache.exists():
        print("Loading cached chunk embeddings from {}...".format(chunk_cache))
        chunk_emb = np.load(chunk_cache)
    else:
        print("Encoding {} chunks (this WILL take a while)...".format(len(chunks)))
        t0 = time.time()
        chunk_emb = encoder.encode_chunks(chunks, mode="full", batch_size=args.batch_size)
        print("  Done in {:.1f}s | shape: {}".format(time.time() - t0, chunk_emb.shape))
        np.save(chunk_cache, chunk_emb)
        write_json(chunk_meta, {
            "generated_at": datetime.now(UTC).isoformat(),
            "model": args.model,
            "mode": "full",
            "shape": list(chunk_emb.shape),
        })
        print("  Saved to {}".format(chunk_cache))

    print("\nEmbeddings ready:")
    print("  Documents: {} → {}".format(doc_emb.shape, doc_cache))
    print("  Chunks:    {} → {}".format(chunk_emb.shape, chunk_cache))
    print("\nNext: run the evaluation: python scripts/eval_agent.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
