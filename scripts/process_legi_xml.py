"""
Process local LEGI XML files into BOFIPChunk-compatible JSON.

Default input directory:
    data/raw/legi_xml/

Outputs:
    data/processed/legi_chunks.json

Optional:
    --append merges LEGI chunks into data/processed/chunks.json
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_pipeline.legi_parser import LEGIArticleParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_chunks(chunks: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.to_dict() for c in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(payload)} chunks to {output_path}")


def append_to_main_chunks(legi_chunks: list, main_chunks_path: Path) -> None:
    if main_chunks_path.exists():
        with open(main_chunks_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    without_legi = [c for c in existing if c.get("source") != "LEGI"]
    combined = without_legi + [c.to_dict() for c in legi_chunks]

    with open(main_chunks_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Updated {main_chunks_path}: total={len(combined)} "
        f"(removed old LEGI={len(existing) - len(without_legi)}, added={len(legi_chunks)})"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Process LEGI XML files")
    parser.add_argument(
        "--input-dir",
        default=str(RAW_DATA_DIR / "legi_xml"),
        help="Directory containing LEGI XML files",
    )
    parser.add_argument(
        "--output-file",
        default=str(PROCESSED_DATA_DIR / "legi_chunks.json"),
        help="Output JSON file for LEGI chunks",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append LEGI chunks into data/processed/chunks.json",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Place XML files under data/raw/legi_xml then rerun.")
        sys.exit(1)

    parser_obj = LEGIArticleParser()
    chunks = parser_obj.parse_directory(input_dir)
    if not chunks:
        logger.warning("No LEGI chunks created.")
        sys.exit(1)

    save_chunks(chunks, output_file)

    if args.append:
        append_to_main_chunks(chunks, PROCESSED_DATA_DIR / "chunks.json")


if __name__ == "__main__":
    main()
