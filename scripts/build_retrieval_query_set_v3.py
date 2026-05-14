from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl, write_jsonl


ANSWERABLE_ADDITIONS = [
    {
        "id": "q46",
        "pattern": "near_neighbor",
        "query": "conditions d'éligibilité des jeunes entreprises innovantes et universitaires",
        "expected_boi": "BOI-BIC-CHAMP-80-20-20-10-20250716",
        "expected_behavior": "answer",
    },
    {
        "id": "q47",
        "pattern": "explicit_reference",
        "query": "constitution et mise à jour de la liste des États et territoires non coopératifs",
        "expected_boi": "BOI-INT-DG-20-50-10-20210224",
        "expected_behavior": "answer",
    },
    {
        "id": "q48",
        "pattern": "near_neighbor",
        "query": "mesures de contrôle applicables aux produits de placement à revenu fixe prélèvement forfaitaire obligatoire",
        "expected_boi": "BOI-RPPM-RCM-30-20-70-20191220",
        "expected_behavior": "answer",
    },
    {
        "id": "q49",
        "pattern": "near_neighbor",
        "query": "modalités de fonctionnement du plan d'épargne en actions",
        "expected_boi": "BOI-RPPM-RCM-40-50-20-20240730",
        "expected_behavior": "answer",
    },
    {
        "id": "q50",
        "pattern": "near_neighbor",
        "query": "taxe d'aménagement opérations imposables",
        "expected_boi": "BOI-IF-TU-10-20-10-20251231",
        "expected_behavior": "answer",
    },
    {
        "id": "q51",
        "pattern": "near_neighbor",
        "query": "exonération de taxe d'aménagement pour constructions affectées à un service public ou d'utilité publique",
        "expected_boi": "BOI-IF-TU-10-20-30-10-20251231",
        "expected_behavior": "answer",
    },
    {
        "id": "q52",
        "pattern": "near_neighbor",
        "query": "scission de la société mère sort des déficits de l'ancien groupe intégré",
        "expected_boi": "BOI-IS-GPE-50-30-30-20210811",
        "expected_behavior": "answer",
    },
    {
        "id": "q53",
        "pattern": "near_neighbor",
        "query": "acquisition de 95 pour cent du capital de la société mère sort du déficit de l'ancien groupe",
        "expected_boi": "BOI-IS-GPE-50-20-20-20-20210811",
        "expected_behavior": "answer",
    },
    {
        "id": "q54",
        "pattern": "near_neighbor",
        "query": "dépenses de normalisation afférentes aux produits de l'entreprise crédit d'impôt recherche",
        "expected_boi": "BOI-BIC-RICI-10-10-20-50-20250813",
        "expected_behavior": "answer",
    },
    {
        "id": "q55",
        "pattern": "near_neighbor",
        "query": "dépenses éligibles au crédit d'impôt recherche relatives aux brevets et certificats d'obtention végétale",
        "expected_boi": "BOI-BIC-RICI-10-10-20-40-20250813",
        "expected_behavior": "answer",
    },
    {
        "id": "q56",
        "pattern": "near_neighbor",
        "query": "solidarité de paiement de l'opérateur de plateforme en ligne pour la TVA",
        "expected_boi": "BOI-TVA-DECLA-10-10-30-20-20200902",
        "expected_behavior": "answer",
    },
    {
        "id": "q57",
        "pattern": "near_neighbor",
        "query": "redevable de la TVA pour les livraisons de biens et prestations de services vue d'ensemble",
        "expected_boi": "BOI-TVA-DECLA-10-10-20200323",
        "expected_behavior": "answer",
    },
    {
        "id": "q58",
        "pattern": "near_neighbor",
        "query": "opérations d'apport-attribution dans les restructurations du groupe fiscal",
        "expected_boi": "BOI-IS-GPE-50-40-20210811",
        "expected_behavior": "answer",
    },
    {
        "id": "q59",
        "pattern": "near_neighbor",
        "query": "incidence sur l'appartenance au groupe des participations à d'autres opérations de restructuration",
        "expected_boi": "BOI-IS-GPE-50-50-30-20200415",
        "expected_behavior": "answer",
    },
    {
        "id": "q60",
        "pattern": "near_neighbor",
        "query": "délai de dépôt d'une demande de rescrit relatif aux jeunes entreprises innovantes",
        "expected_boi": "BOI-RES-SJ-000014-20210309",
        "expected_behavior": "answer",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the broader retrieval query set v3 with more answerable neighbor cases.")
    parser.add_argument("--base", type=str, default="data/interim/retrieval_queries_sample_1000_v2.jsonl")
    parser.add_argument("--output", type=str, default="data/interim/retrieval_queries_sample_1000_v3.jsonl")
    args = parser.parse_args()

    base_rows = read_jsonl(PROJECT_ROOT / Path(args.base))
    all_rows = base_rows + ANSWERABLE_ADDITIONS

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
