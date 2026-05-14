from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import write_jsonl

from build_passage_gold_v1 import PASSAGE_GOLD_V1


PASSAGE_GOLD_V2_ADDITIONS = [
    {
        "id": "pg023",
        "source_query_id": "c013",
        "pattern": "scenario_business",
        "query": "Une association non lucrative commence a faire du taxable: quel texte regarder pour ses obligations TVA specifiques ?",
        "expected_boi": "BOI-TVA-DECLA-20-30-20-10-20220323",
        "chunk_ids_any": [
            "920-PGP__section_window__920-pgp-paragraph-00008__920-pgp-paragraph-00009__ii-obligations-d-claratives-a-organismes-pouvant-b-n-ficier-de-la-franchise-en-base-ou-d-une-exon-ration-sur-leurs-activit-s-accessoires-1-organismes-exer-ant-titre-principal-une-activit-imposable"
        ],
        "note": "OSBL with taxable operations and TVA filing obligations.",
    },
    {
        "id": "pg024",
        "source_query_id": "c023",
        "pattern": "paraphrase",
        "query": "Je cherche la doctrine BIC qui explique quoi faire des produits et charges qui n'ont plus de lien direct avec l'activite.",
        "expected_boi": "BOI-BIC-BASE-90-20180704",
        "chunk_ids_any": [
            "7900-PGP__section_window__7900-pgp-paragraph-00092__7900-pgp-paragraph-00099__ii-produits-et-charges-extourner-c-cons-quences-pratiques-1-retraitements-extra-comptables"
        ],
        "note": "Products and charges to be extourned from professional result.",
    },
    {
        "id": "pg025",
        "source_query_id": "c024",
        "pattern": "topic_specific",
        "query": "Quel avantage fiscal s'applique a une creation d'entreprise implantee dans une zone de developpement prioritaire ?",
        "expected_boi": "BOI-BIC-CHAMP-80-10-100-30-20200422",
        "chunk_ids_any": [
            "12091-PGP__section_window__12091-pgp-paragraph-00000__12091-pgp-paragraph-00001__bic-champ-d-application-et-territorialit-exon-rations-entreprises-ou-activit-s-implant-es-dans-certaines-zones-du-territoire-entreprises-cr-es-dans-les-zones-de-d-veloppement-prioritaire-port-e-et-calcul-des-all-gements-fiscaux"
        ],
        "note": "ZDP enterprise creation tax relief scope.",
    },
    {
        "id": "pg026",
        "source_query_id": "c034",
        "pattern": "procedural_question",
        "query": "Je veux la doctrine de recouvrement sur la saisie des remunerations.",
        "expected_boi": "BOI-REC-FORCE-20-20-20-20191127",
        "chunk_ids_any": [
            "11630-PGP__section_window__11630-pgp-paragraph-00000__11630-pgp-paragraph-00002__rec-mise-en-uvre-du-recouvrement-forc-saisies-mobili-res-de-droit-commun-saisie-des-r-mun-rations-proc-dure"
        ],
        "note": "Salary garnishment procedure in tax collection.",
    },
    {
        "id": "pg027",
        "source_query_id": "c035",
        "pattern": "procedural_question",
        "query": "Quelles exigences formelles conditionnent la validite d'un cautionnement en recouvrement fiscal ?",
        "expected_boi": "BOI-REC-GAR-20-40-10-20-20120912",
        "chunk_ids_any": [
            "6914-PGP__section_window__6914-pgp-paragraph-00000__6914-pgp-paragraph-00000__rec-s-ret-s-et-garanties-du-recouvrement-cautionnement-r-gles-sp-cifiques-de-validit-de-l-acte-de-cautionnement__merge__i-le-cautionnement-doit-tre-expr-s"
        ],
        "note": "Specific validity requirements for tax guarantee acts.",
    },
    {
        "id": "pg028",
        "source_query_id": "c036",
        "pattern": "procedural_question",
        "query": "Comment se deroule l'instruction d'une demande fiscale devant le tribunal administratif ?",
        "expected_boi": "BOI-CTX-ADM-10-30-20120912",
        "chunk_ids_any": [
            "545-PGP__section_window__545-pgp-paragraph-00003__545-pgp-paragraph-00006__i-proc-dure-g-n-rale-d-instruction"
        ],
        "note": "Administrative court instruction general procedure.",
    },
    {
        "id": "pg029",
        "source_query_id": "c039",
        "pattern": "topic_specific",
        "query": "Dans quels cas la valeur locative servant a la CFE peut-elle etre minoree ?",
        "expected_boi": "BOI-IF-CFE-20-20-30-20160706",
        "chunk_ids_any": [
            "1253-PGP__section_window__1253-pgp-paragraph-00054__1253-pgp-paragraph-00056__v-r-duction-facultative-de-moiti-de-la-valeur-locative-des-b-timents-industriels-affect-s-la-recherche"
        ],
        "note": "Optional 50% reduction of research buildings rental value.",
    },
    {
        "id": "pg030",
        "source_query_id": "c045",
        "pattern": "topic_specific",
        "query": "Quel document BOFiP general traite de l'exoneration IS des reprises d'entreprises industrielles en difficulte ?",
        "expected_boi": "BOI-IS-GEO-20-10-20150603",
        "chunk_ids_any": [
            "4513-PGP__section_window__4513-pgp-paragraph-00000__4513-pgp-paragraph-00012__is-r-gimes-sectoriels-reprise-d-entreprises-industrielles-en-difficult"
        ],
        "note": "General overview for distressed industrial company takeover relief.",
    },
    {
        "id": "pg031",
        "source_query_id": "c047",
        "pattern": "topic_specific",
        "query": "Comment la liste francaise des ETNC est-elle etablie puis mise a jour ?",
        "expected_boi": "BOI-INT-DG-20-50-10-20210224",
        "chunk_ids_any": [
            "12855-PGP__section_window__12855-pgp-paragraph-00034__12855-pgp-paragraph-00040__ii-mise-jour-de-la-liste"
        ],
        "note": "ETNC list update mechanics.",
    },
    {
        "id": "pg032",
        "source_query_id": "c048",
        "pattern": "topic_specific",
        "query": "Au-dela du prelevement forfaitaire obligatoire, quelles mesures de controle visent les placements a revenu fixe ?",
        "expected_boi": "BOI-RPPM-RCM-30-20-70-20191220",
        "chunk_ids_any": [
            "7052-PGP__section_window__7052-pgp-paragraph-00000__7052-pgp-paragraph-00002__rppm-revenus-de-capitaux-mobiliers-gains-et-profits-assimil-s-modalit-s-particuli-res-d-imposition-pr-l-vement-forfaitaire-obligatoire-non-lib-ratoire-de-l-imp-t-sur-le-revenu-applicable-aux-produits-de-placement-revenu-fixe-aux-produits-et-gains-de-cession-de-bons-ou-contrats-de-capitalisation-et-placements-de-m-me-nature-attach-s-des-primes-vers-es-compter-du-27-septembre-2017-et-aux-revenus-distribu-s-mesures-de-contr-le-applicables-aux-produits-de-placement-revenu-fixe"
        ],
        "note": "Control measures on fixed-income investment products.",
    },
    {
        "id": "pg033",
        "source_query_id": "c056",
        "pattern": "scenario_business",
        "query": "A quel moment une plateforme devient-elle solidairement responsable de la TVA due par un vendeur ?",
        "expected_boi": "BOI-TVA-DECLA-10-10-30-20-20200902",
        "chunk_ids_any": [
            "12128-PGP__section_window__12128-pgp-paragraph-00015__12128-pgp-paragraph-00015__ii-conditions-d-application-de-la-proc-dure-de-solidarit-de-paiement"
        ],
        "note": "Platform solidarity conditions for VAT payment.",
    },
    {
        "id": "pg034",
        "source_query_id": "c061",
        "pattern": "topic_specific",
        "query": "Avant toute declaration DAC7, quelles diligences une plateforme doit-elle mener sur les vendeurs concernes ?",
        "expected_boi": "BOI-INT-AEA-30-20-20231213",
        "chunk_ids_any": [
            "13761-PGP__section_window__13761-pgp-paragraph-00000__13761-pgp-paragraph-00005__int-accords-et-change-automatique-de-renseignements-obligations-des-op-rateurs-de-plateforme-de-mise-en-relation-par-voie-lectronique-obligations-de-diligence-mises-la-charge-des-op-rateurs-de-plateforme-concern-s"
        ],
        "note": "DAC7 reasonable diligence duties.",
    },
    {
        "id": "pg035",
        "source_query_id": "c062",
        "pattern": "topic_specific",
        "query": "Pour DAC7, quelles informations une plateforme doit-elle declarer a l'administration fiscale ?",
        "expected_boi": "BOI-INT-AEA-30-30-20230111",
        "chunk_ids_any": [
            "13751-PGP__section_window__13751-pgp-paragraph-00000__13751-pgp-paragraph-00007__int-accords-et-change-automatique-de-renseignements-obligations-des-op-rateurs-de-plateforme-de-mise-en-relation-par-voie-lectronique-obligations-d-claratives-mises-la-charge-des-op-rateurs-de-plateforme-concern-s"
        ],
        "note": "DAC7 declarative obligations overview.",
    },
    {
        "id": "pg036",
        "source_query_id": "c065",
        "pattern": "near_neighbor",
        "query": "Pour savoir si un operateur est etabli en France au regard de la TVA, quel document de base faut-il lire ?",
        "expected_boi": "BOI-TVA-DECLA-10-10-10-20120912",
        "chunk_ids_any": [
            "3168-PGP__section_window__3168-pgp-paragraph-00008__3168-pgp-paragraph-00011__i-d-finition-b-pr-cisions-3-attractivit-du-si-ge"
        ],
        "note": "Definition and seat attractiveness for being established in France.",
    },
    {
        "id": "pg037",
        "source_query_id": "c067",
        "pattern": "topic_specific",
        "query": "Qui a la charge d'operer le prelevement a la source sur les revenus fixes relevant de ce regime forfaitaire obligatoire ?",
        "expected_boi": "BOI-RPPM-RCM-30-20-20-20191220",
        "chunk_ids_any": [
            "3740-PGP__section_window__3740-pgp-paragraph-00000__3740-pgp-paragraph-00002__rppm-revenus-de-capitaux-mobiliers-gains-et-profits-assimil-s-modalit-s-particuli-res-d-imposition-pr-l-vement-forfaitaire-obligatoire-non-lib-ratoire-de-l-imp-t-sur-le-revenu-applicable-aux-produits-de-placement-revenu-fixe-aux-produits-et-gains-de-cession-des-bons-ou-contrats-de-capitalisation-et-placements-de-m-me-nature-attach-s-des-primes-vers-es-compter-du-27-septembre-2017-et-aux-revenus-distribu-s-personnes-tenues-d-effectuer-le-pr-l-vement"
        ],
        "note": "Who must perform the withholding.",
    },
    {
        "id": "pg038",
        "source_query_id": "c068",
        "pattern": "near_neighbor",
        "query": "Comment l'exoneration IS pour reprise d'entreprise en difficulte s'articule-t-elle avec les autres regimes d'allegement ?",
        "expected_boi": "BOI-IS-GEO-20-10-40-20150603",
        "chunk_ids_any": [
            "4483-PGP__section_window__4483-pgp-paragraph-00009__4483-pgp-paragraph-00011__ii-articulation-de-l-exon-ration-avec-celle-applicable-en-zone-franche-urbaine-territoire-entrepreneur-zfu-te"
        ],
        "note": "Articulation with other relief regimes.",
    },
    {
        "id": "pg039",
        "source_query_id": "c069",
        "pattern": "topic_specific",
        "query": "Quand la R&D est sous-traitee, quelles depenses externes restent eligibles dans le CIR ?",
        "expected_boi": "BOI-BIC-RICI-10-10-20-30-20250813",
        "chunk_ids_any": [
            "6504-PGP__section_window__6504-pgp-paragraph-00062__6504-pgp-paragraph-00067__ii-d-penses-de-recherche-ligibles-a-d-penses-expos-es-pour-la-r-alisation-d-op-rations-de-recherche-externalis-es-ou-de-travaux-scientifiques-et-techniques-indispensables-la-r-alisation-d-op-rations-de-recherche-ligibles-men-es-en-interne-par-l-entreprise-donneuse-d-ordre"
        ],
        "note": "Externalized R&D expenses and indispensable technical work.",
    },
    {
        "id": "pg040",
        "source_query_id": "c070",
        "pattern": "topic_specific",
        "query": "Dans le CIR, comment prendre en compte l'amortissement des immobilisations utilisees pour la recherche ?",
        "expected_boi": "BOI-BIC-RICI-10-10-20-10-20250813",
        "chunk_ids_any": [
            "6494-PGP__section_window__6494-pgp-paragraph-00002__6494-pgp-paragraph-00004__i-amortissement-des-immeubles-affect-s-la-r-alisation-d-op-rations-de-recherche"
        ],
        "note": "Depreciation of buildings used in research.",
    },
]


def main() -> int:
    output_path = PROJECT_ROOT / "data" / "interim" / "passage_gold_v2.jsonl"
    rows = list(PASSAGE_GOLD_V1) + PASSAGE_GOLD_V2_ADDITIONS
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} passage gold rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
