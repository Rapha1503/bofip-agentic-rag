from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl, write_jsonl


ADDITIONAL_ROWS = [
    {
        "id": "u11",
        "pattern": "unsupported",
        "query": "TVA applicable en 2030 aux cryptomonnaies minées par une intelligence artificielle autonome",
        "expected_behavior": "abstain",
    },
    {
        "id": "u12",
        "pattern": "unsupported",
        "query": "barème 2027 de la taxe sur les capsules spatiales privées",
        "expected_behavior": "abstain",
    },
    {
        "id": "u13",
        "pattern": "unsupported",
        "query": "procédure de contestation d'un impôt communal belge devant l'administration fiscale française",
        "expected_behavior": "abstain",
    },
    {
        "id": "u14",
        "pattern": "false_premise",
        "query": "le BOI dit-il que tous les organismes sans but lucratif sont exonérés d'IFI",
        "expected_behavior": "answer",
    },
    {
        "id": "u15",
        "pattern": "false_premise",
        "query": "le BOFIP prévoit-il qu'une plus-value immobilière est toujours exonérée après deux ans de détention",
        "expected_behavior": "answer",
    },
    {
        "id": "u16",
        "pattern": "unsupported",
        "query": "taxe française sur la revente de robots domestiques personnels en 2027",
        "expected_behavior": "abstain",
    },
    {
        "id": "u17",
        "pattern": "false_premise",
        "query": "le BOI indique-t-il que le PEA autorise toujours les crypto-actifs",
        "expected_behavior": "answer",
    },
    {
        "id": "u18",
        "pattern": "unsupported",
        "query": "régime fiscal français des revenus miniers tirés d'une exploitation sur la Lune",
        "expected_behavior": "abstain",
    },
    {
        "id": "u19",
        "pattern": "false_premise",
        "query": "la convention fiscale France Allemagne attribue-t-elle toujours l'imposition à la France pour tous les salaires privés",
        "expected_behavior": "answer",
    },
    {
        "id": "u20",
        "pattern": "unsupported",
        "query": "TVA applicable aux services de téléportation quantique facturés à des particuliers français",
        "expected_behavior": "abstain",
    },
    {
        "id": "u21",
        "pattern": "false_premise",
        "query": "le BOI dit-il qu'un organisme sans but lucratif n'a jamais d'obligations déclaratives de TVA",
        "expected_behavior": "answer",
    },
    {
        "id": "u22",
        "pattern": "unsupported",
        "query": "droits de succession français sur un patrimoine situé uniquement au Wakanda sans lien avec la France",
        "expected_behavior": "abstain",
    },
    {
        "id": "u23",
        "pattern": "false_premise",
        "query": "le BOFIP indique-t-il qu'une JEI bénéficie automatiquement du remboursement immédiat de tout crédit d'impôt",
        "expected_behavior": "answer",
    },
    {
        "id": "u24",
        "pattern": "unsupported",
        "query": "barème 2028 de la taxe sur les drones de loisir maritimes",
        "expected_behavior": "abstain",
    },
    {
        "id": "u25",
        "pattern": "false_premise",
        "query": "le BOI prévoit-il que l'IFI remplace aussi la taxe foncière",
        "expected_behavior": "answer",
    },
    {
        "id": "u26",
        "pattern": "unsupported",
        "query": "TVA applicable aux souvenirs vendus sur Mars par une société française",
        "expected_behavior": "abstain",
    },
    {
        "id": "u27",
        "pattern": "false_premise",
        "query": "le BOI indique-t-il qu'une société mère absorbée conserve toujours tous les déficits de groupe sans condition",
        "expected_behavior": "answer",
    },
    {
        "id": "u28",
        "pattern": "unsupported",
        "query": "règles françaises de TVA pour les voyages temporels commerciaux",
        "expected_behavior": "abstain",
    },
    {
        "id": "u29",
        "pattern": "false_premise",
        "query": "le BOI précise-t-il que toutes les publications de presse sont au taux super-réduit de TVA",
        "expected_behavior": "answer",
    },
    {
        "id": "u30",
        "pattern": "unsupported",
        "query": "procédure de recours contre un impôt fédéral brésilien devant la DGFIP",
        "expected_behavior": "abstain",
    },
]


def _decorate_base_rows(rows: list[dict]) -> list[dict]:
    decorated: list[dict] = []
    for row in rows:
        cloned = dict(row)
        if cloned["id"].startswith("q"):
            cloned.setdefault("expected_behavior", "answer")
        elif cloned.get("pattern") == "false_premise":
            cloned.setdefault("expected_behavior", "answer")
        else:
            cloned.setdefault("expected_behavior", "abstain")
        decorated.append(cloned)
    return decorated


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the broader retrieval query set v2.")
    parser.add_argument("--base", type=str, default="data/interim/retrieval_queries_sample_1000_v1.jsonl")
    parser.add_argument("--output", type=str, default="data/interim/retrieval_queries_sample_1000_v2.jsonl")
    args = parser.parse_args()

    base_rows = _decorate_base_rows(read_jsonl(PROJECT_ROOT / Path(args.base)))
    all_rows = base_rows + ADDITIONAL_ROWS

    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for row in all_rows:
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError(f"Duplicate query id: {row_id}")
        seen_ids.add(row_id)
        deduped.append(row)

    output_path = PROJECT_ROOT / Path(args.output)
    write_jsonl(output_path, deduped)
    print(f"Wrote {len(deduped)} queries to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
