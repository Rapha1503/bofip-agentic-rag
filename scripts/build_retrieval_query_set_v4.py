from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import write_jsonl


ROWS = [
    {"id": "a001", "pattern": "scenario_business", "query": "Notre startup a le statut JEI et porte des travaux de recherche. Peut-elle recuperer sa creance de CIR tout de suite ?", "expected_boi": "BOI-BIC-CHAMP-80-20-20-20-20240703", "expected_behavior": "answer"},
    {"id": "a002", "pattern": "paraphrase", "query": "Dans l'assiette du CIR, qu'est-ce qu'on peut ranger dans les frais de fonctionnement autour des travaux de recherche ?", "expected_boi": "BOI-BIC-RICI-10-10-20-25-20250813", "expected_behavior": "answer"},
    {"id": "a003", "pattern": "scenario_business", "query": "Pour des avances remboursables sans interet destinees a des travaux energetiques, quel mecanisme de credit d'impot existe ?", "expected_boi": "BOI-BIC-RICI-10-110-20-20241127", "expected_behavior": "answer"},
    {"id": "a004", "pattern": "procedural_question", "query": "Apres la mise en place du PAS, jusqu'a quand l'administration peut-elle encore rectifier les revenus 2018 ?", "expected_boi": "BOI-IR-PAS-50-20-50-20200210", "expected_behavior": "answer"},
    {"id": "a005", "pattern": "keyword_short", "query": "Quels encaissements ou revenus supportent un acompte de prelevement a la source ?", "expected_boi": "BOI-IR-PAS-10-10-20-20230706", "expected_behavior": "answer"},
    {"id": "a006", "pattern": "explicit_country", "query": "Je cherche la doctrine France Allemagne sur les salaires publics et les pensions versees par l'Etat.", "expected_boi": "BOI-INT-CVB-DEU-10-40-20120912", "expected_behavior": "answer"},
    {"id": "a007", "pattern": "explicit_country", "query": "Ou trouver les regles France Algerie pour neutraliser la double imposition ?", "expected_boi": "BOI-INT-CVB-DZA-60-20141226", "expected_behavior": "answer"},
    {"id": "a008", "pattern": "near_neighbor", "query": "Dans un groupe integre, que deviennent les deficits du groupe initial quand la mere est absorbee ?", "expected_boi": "BOI-IS-GPE-50-10-30-20210811", "expected_behavior": "answer"},
    {"id": "a009", "pattern": "near_neighbor", "query": "Dans l'integration fiscale, comment s'apprecie la sous-capitalisation au niveau du groupe ?", "expected_boi": "BOI-IS-GPE-20-20-70-20190731", "expected_behavior": "answer"},
    {"id": "a010", "pattern": "false_premise_answerable", "query": "Le BOFiP dit-il que toute publication de presse echappe automatiquement a la TVA ?", "expected_boi": "BOI-TVA-SECT-40-10-30-20130621", "expected_behavior": "answer"},
    {"id": "a011", "pattern": "rescrit_case", "query": "Quel taux de TVA viser pour un equipement qui solidarise un fauteuil roulant avec une trottinette electrique ?", "expected_boi": "BOI-RES-TVA-000074-20210309", "expected_behavior": "answer"},
    {"id": "a012", "pattern": "paraphrase", "query": "Qui doit reverser la TVA quand on parle de vente de biens ou de prestation de services en regime normal ?", "expected_boi": "BOI-TVA-DECLA-10-10-20-20251022", "expected_behavior": "answer"},
    {"id": "a013", "pattern": "scenario_business", "query": "Une association sans but lucratif qui fait des operations taxables, quelles formalites TVA particulieres doit-elle suivre ?", "expected_boi": "BOI-TVA-DECLA-20-30-20-10-20220323", "expected_behavior": "answer"},
    {"id": "a014", "pattern": "procedural_question", "query": "En contentieux d'assiette, a quelles conditions l'administration peut-elle changer de fondement ou compenser ?", "expected_boi": "BOI-CTX-DG-20-40-10-20120912", "expected_behavior": "answer"},
    {"id": "a015", "pattern": "scenario_business", "query": "Un agent general d'assurances qui part a la retraite et touche une indemnite compensatrice est impose comment ?", "expected_boi": "BOI-BNC-CESS-40-10-20230517", "expected_behavior": "answer"},
    {"id": "a016", "pattern": "topic_specific", "query": "Quel traitement fiscal pour une transformation de societe ou une simple retouche des statuts ?", "expected_boi": "BOI-ENR-AVS-20-30-30-20120912", "expected_behavior": "answer"},
    {"id": "a017", "pattern": "near_neighbor", "query": "Pour les actes de la vie sociale, dans quels cas le regime special des fusions peut-il s'appliquer ?", "expected_boi": "BOI-ENR-AVS-20-60-30-10-20140613", "expected_behavior": "answer"},
    {"id": "a018", "pattern": "keyword_short", "query": "Les parcelles plantees en oliviers restent-elles durablement exonerees de TFNB ?", "expected_boi": "BOI-IF-TFNB-10-40-60-20120912", "expected_behavior": "answer"},
    {"id": "a019", "pattern": "paraphrase", "query": "Quand une commune change de contour ou fusionne, comment adapte-t-on les valeurs locatives ?", "expected_boi": "BOI-IF-TFB-20-20-10-40-20121210", "expected_behavior": "answer"},
    {"id": "a020", "pattern": "topic_specific", "query": "Pour la CVAE, comment calcule-t-on la valeur ajoutee de base sans entrer dans les cas particuliers ?", "expected_boi": "BOI-CVAE-BASE-10-20120912", "expected_behavior": "answer"},
    {"id": "a021", "pattern": "topic_specific", "query": "Quand une entreprise prend du materiel en credit-bail mobilier, quel est le regime fiscal de l'operation ?", "expected_boi": "BOI-BIC-BASE-60-20-20120912", "expected_behavior": "answer"},
    {"id": "a022", "pattern": "false_premise_answerable", "query": "Une entreprise peut-elle choisir librement de ne pas comptabiliser un amortissement pourtant obligatoire ?", "expected_boi": "BOI-BIC-AMT-10-50-10-20120912", "expected_behavior": "answer"},
    {"id": "a023", "pattern": "paraphrase", "query": "Depuis la fin de la theorie du bilan fiscale, comment traite-t-on les produits et charges sans rapport avec l'activite ?", "expected_boi": "BOI-BIC-BASE-90-20180704", "expected_behavior": "answer"},
    {"id": "a024", "pattern": "topic_specific", "query": "Quels avantages fiscaux existent pour une societe qui se cree dans une zone de developpement prioritaire ?", "expected_boi": "BOI-BIC-CHAMP-80-10-100-30-20200422", "expected_behavior": "answer"},
    {"id": "a025", "pattern": "keyword_short", "query": "A quoi ressemble le bareme de l'IFI aujourd'hui ?", "expected_boi": "BOI-PAT-IFI-40-10-20180608", "expected_behavior": "answer"},
    {"id": "a026", "pattern": "topic_specific", "query": "Une personne physique residente de France est-elle toujours dans le champ de l'IFI ?", "expected_boi": "BOI-PAT-IFI-10-20-20-20180608", "expected_behavior": "answer"},
    {"id": "a027", "pattern": "false_premise_answerable", "query": "Le plafonnement de l'IFI joue-t-il sans aucune condition ni limite particuliere ?", "expected_boi": "BOI-PAT-IFI-40-30-10-20181122", "expected_behavior": "answer"},
    {"id": "a028", "pattern": "topic_specific", "query": "Une plateforme en ligne de mise en relation a quelles obligations fiscales de base ?", "expected_boi": "BOI-INT-AEA-30-10-20230111", "expected_behavior": "answer"},
    {"id": "a029", "pattern": "explicit_reference", "query": "Je cherche la doctrine generale sur les ETNC quand on raisonne en droit conventionnel.", "expected_boi": "BOI-INT-DG-20-50-20210224", "expected_behavior": "answer"},
    {"id": "a030", "pattern": "topic_specific", "query": "Comment fonctionne le prelevement forfaitaire obligatoire sur les placements a revenu fixe ?", "expected_boi": "BOI-RPPM-RCM-30-20-40-20220630", "expected_behavior": "answer"},
    {"id": "a031", "pattern": "keyword_short", "query": "Je veux un BOFiP qui regroupe les regles diverses restant a connaitre sur le PEA.", "expected_boi": "BOI-RPPM-RCM-40-50-60-20240730", "expected_behavior": "answer"},
    {"id": "a032", "pattern": "procedural_question", "query": "Pour une plus-value sur un bien meuble incorporel, quelles declarations le contribuable doit-il produire ?", "expected_boi": "BOI-RPPM-PVBMI-40-10-20141014", "expected_behavior": "answer"},
    {"id": "a033", "pattern": "procedural_question", "query": "Quand le comptable public accorde un echeancier, est-ce que les poursuites sont suspendues ?", "expected_boi": "BOI-REC-PREA-20-10-10-20150506", "expected_behavior": "answer"},
    {"id": "a034", "pattern": "procedural_question", "query": "Comment fonctionne la saisie des remunerations dans le recouvrement force fiscal ?", "expected_boi": "BOI-REC-FORCE-20-20-20-20191127", "expected_behavior": "answer"},
    {"id": "a035", "pattern": "procedural_question", "query": "Quelles conditions de forme rendent un acte de cautionnement valable en matiere de recouvrement ?", "expected_boi": "BOI-REC-GAR-20-40-10-20-20120912", "expected_behavior": "answer"},
    {"id": "a036", "pattern": "procedural_question", "query": "Devant le tribunal administratif, comment une demande fiscale est-elle instruite ?", "expected_boi": "BOI-CTX-ADM-10-30-20120912", "expected_behavior": "answer"},
    {"id": "a037", "pattern": "procedural_question", "query": "Apres un arret de CAA en fiscal, quels recours restent ouverts ?", "expected_boi": "BOI-CTX-ADM-20-50-20120912", "expected_behavior": "answer"},
    {"id": "a038", "pattern": "procedural_question", "query": "En appel judiciaire, comment se greffent une QPC ou une question prejudicielle dans un litige fiscal ?", "expected_boi": "BOI-CTX-JUD-20-20-60-20120912", "expected_behavior": "answer"},
    {"id": "a039", "pattern": "topic_specific", "query": "Comment peut-on reduire la valeur locative retenue pour la CFE dans certains cas ?", "expected_boi": "BOI-IF-CFE-20-20-30-20160706", "expected_behavior": "answer"},
    {"id": "a040", "pattern": "topic_specific", "query": "Je cherche le BOFiP de base qui dit dans quels cas la taxe d'amenagement s'applique.", "expected_boi": "BOI-IF-TU-10-20-20251231", "expected_behavior": "answer"},
    {"id": "a041", "pattern": "paraphrase", "query": "Des loyers tires de terrains non batis relevent-ils bien des revenus fonciers, et dans quel cadre ?", "expected_boi": "BOI-RFPI-CHAMP-10-20-20120912", "expected_behavior": "answer"},
    {"id": "a042", "pattern": "scenario_business", "query": "Pendant combien de temps un jeune agriculteur peut-il profiter de l'abattement sur son benefice ?", "expected_boi": "BOI-BA-BASE-30-10-20-20190515", "expected_behavior": "answer"},
    {"id": "a043", "pattern": "keyword_short", "query": "En BIC, comment traite-t-on une charge financiere qui varie avec une clause d'indexation ?", "expected_boi": "BOI-BIC-CHG-50-60-20120912", "expected_behavior": "answer"},
    {"id": "a044", "pattern": "topic_specific", "query": "Quelles societes peuvent entrer dans un groupe fiscal et a quelle date d'ouverture d'exercice ?", "expected_boi": "BOI-IS-GPE-10-10-20-20160907", "expected_behavior": "answer"},
    {"id": "a045", "pattern": "topic_specific", "query": "Pour une reprise d'entreprise industrielle en difficulte, ou trouve-t-on la doctrine generale en IS ?", "expected_boi": "BOI-IS-GEO-20-10-20150603", "expected_behavior": "answer"},
    {"id": "a046", "pattern": "false_premise_answerable", "query": "Une JEI ou une JEU est-elle eligibile aux allegements sans aucune condition d'entree ?", "expected_boi": "BOI-BIC-CHAMP-80-20-20-10-20250716", "expected_behavior": "answer"},
    {"id": "a047", "pattern": "topic_specific", "query": "Comment la France etablit-elle puis actualise-t-elle la liste des ETNC ?", "expected_boi": "BOI-INT-DG-20-50-10-20210224", "expected_behavior": "answer"},
    {"id": "a048", "pattern": "topic_specific", "query": "En plus du prelevement forfaitaire obligatoire, quelles mesures de controle s'appliquent aux placements a revenu fixe ?", "expected_boi": "BOI-RPPM-RCM-30-20-70-20191220", "expected_behavior": "answer"},
    {"id": "a049", "pattern": "paraphrase", "query": "Au quotidien, comment fonctionne un PEA une fois ouvert ?", "expected_boi": "BOI-RPPM-RCM-40-50-20-20240730", "expected_behavior": "answer"},
    {"id": "a050", "pattern": "topic_specific", "query": "Je cherche la partie du BOFiP qui liste les constructions et operations entrant dans la taxe d'amenagement.", "expected_boi": "BOI-IF-TU-10-20-10-20251231", "expected_behavior": "answer"},
    {"id": "a051", "pattern": "topic_specific", "query": "Quelles constructions affectees a une mission publique peuvent sortir de la taxe d'amenagement ?", "expected_boi": "BOI-IF-TU-10-20-30-10-20251231", "expected_behavior": "answer"},
    {"id": "a052", "pattern": "near_neighbor", "query": "Si la societe mere du groupe est scindee, que deviennent les deficits du groupe d'origine ?", "expected_boi": "BOI-IS-GPE-50-30-30-20210811", "expected_behavior": "answer"},
    {"id": "a053", "pattern": "near_neighbor", "query": "Quand 95 % du capital de la mere change de mains, quel effet sur le deficit de l'ancien groupe ?", "expected_boi": "BOI-IS-GPE-50-20-20-20-20210811", "expected_behavior": "answer"},
    {"id": "a054", "pattern": "topic_specific", "query": "Dans le CIR, comment traite-t-on les depenses de normalisation liees aux produits de l'entreprise ?", "expected_boi": "BOI-BIC-RICI-10-10-20-50-20250813", "expected_behavior": "answer"},
    {"id": "a055", "pattern": "topic_specific", "query": "Les frais lies aux brevets ou aux certificats d'obtention vegetale peuvent-ils entrer dans le CIR ?", "expected_boi": "BOI-BIC-RICI-10-10-20-40-20250813", "expected_behavior": "answer"},
    {"id": "a056", "pattern": "scenario_business", "query": "Dans quels cas une plateforme en ligne peut-elle etre tenue solidairement au paiement de la TVA ?", "expected_boi": "BOI-TVA-DECLA-10-10-30-20-20200902", "expected_behavior": "answer"},
    {"id": "a057", "pattern": "paraphrase", "query": "Je cherche le document d'ensemble sur la notion de redevable TVA pour ventes et prestations.", "expected_boi": "BOI-TVA-DECLA-10-10-20200323", "expected_behavior": "answer"},
    {"id": "a058", "pattern": "near_neighbor", "query": "Dans un groupe integre, comment sont traitees les operations d'apport-attribution ?", "expected_boi": "BOI-IS-GPE-50-40-20210811", "expected_behavior": "answer"},
    {"id": "a059", "pattern": "near_neighbor", "query": "Quand une participation evolue au sein d'une restructuration, quel impact sur l'appartenance au groupe fiscal ?", "expected_boi": "BOI-IS-GPE-50-50-30-20200415", "expected_behavior": "answer"},
    {"id": "a060", "pattern": "procedural_question", "query": "Jusqu'a quand peut-on deposer une demande de rescrit sur le statut JEI ?", "expected_boi": "BOI-RES-SJ-000014-20210309", "expected_behavior": "answer"},
    {"id": "a061", "pattern": "topic_specific", "query": "Pour une plateforme, quelles diligences faut-il mener avant de declarer les vendeurs concernes ?", "expected_boi": "BOI-INT-AEA-30-20-20231213", "expected_behavior": "answer"},
    {"id": "a062", "pattern": "topic_specific", "query": "Qu'est-ce qu'une plateforme doit effectivement declarer a l'administration et sous quelle logique generale ?", "expected_boi": "BOI-INT-AEA-30-30-20230111", "expected_behavior": "answer"},
    {"id": "a063", "pattern": "procedural_question", "query": "Comment la declaration des operateurs de plateforme doit-elle etre transmise en pratique ?", "expected_boi": "BOI-INT-AEA-30-40-20231213", "expected_behavior": "answer"},
    {"id": "a064", "pattern": "topic_specific", "query": "Quelles sanctions risquent les operateurs de plateforme qui ne respectent pas leurs obligations declaratives ?", "expected_boi": "BOI-INT-AEA-30-50-20231213", "expected_behavior": "answer"},
    {"id": "a065", "pattern": "near_neighbor", "query": "Pour savoir si un operateur est etabli en France au regard de la TVA, quel document regarder ?", "expected_boi": "BOI-TVA-DECLA-10-10-10-20120912", "expected_behavior": "answer"},
    {"id": "a066", "pattern": "topic_specific", "query": "Le prelevement forfaitaire obligatoire sur les placements de taux, c'est d'abord quoi et sur quoi ?", "expected_boi": "BOI-RPPM-RCM-30-20-10-20210706", "expected_behavior": "answer"},
    {"id": "a067", "pattern": "topic_specific", "query": "Qui est cense prelever l'impot a la source sur les revenus fixes entrant dans ce regime forfaitaire obligatoire ?", "expected_boi": "BOI-RPPM-RCM-30-20-20-20191220", "expected_behavior": "answer"},
    {"id": "a068", "pattern": "near_neighbor", "query": "L'exoneration IS pour reprise d'entreprise en difficulte se combine comment avec d'autres regimes existants ?", "expected_boi": "BOI-IS-GEO-20-10-40-20150603", "expected_behavior": "answer"},
    {"id": "a069", "pattern": "topic_specific", "query": "Quand une partie de la R&D est confiee a l'exterieur, quelles depenses externalisees peuvent rester dans le CIR ?", "expected_boi": "BOI-BIC-RICI-10-10-20-30-20250813", "expected_behavior": "answer"},
    {"id": "a070", "pattern": "topic_specific", "query": "Pour le CIR, comment prendre en compte l'amortissement des immobilisations utilisees pour la recherche ?", "expected_boi": "BOI-BIC-RICI-10-10-20-10-20250813", "expected_behavior": "answer"},
    {"id": "u071", "pattern": "unsupported", "query": "TVA applicable a la vente de souvenirs sur Mars par une societe francaise", "expected_behavior": "abstain"},
    {"id": "u072", "pattern": "unsupported", "query": "Comment contester un impot federal bresilien devant la DGFIP", "expected_behavior": "abstain"},
    {"id": "u073", "pattern": "unsupported", "query": "Regles fiscales pour les voyages temporels commerciaux et la TVA francaise", "expected_behavior": "abstain"},
    {"id": "u074", "pattern": "unsupported", "query": "Quel est le regime fiscal d'une exploitation miniere sur la Lune pour une PME francaise", "expected_behavior": "abstain"},
    {"id": "u075", "pattern": "unsupported", "query": "Procedure fiscale francaise pour les revenus tires de teleportation de marchandises", "expected_behavior": "abstain"},
    {"id": "u076", "pattern": "unsupported", "query": "Quelle doctrine BOFiP pour une crypto-monnaie emise par une civilisation extraterrestre", "expected_behavior": "abstain"},
    {"id": "u077", "pattern": "unsupported", "query": "Fiscalite francaise d'un peage interdimensionnel facture a des clients particuliers", "expected_behavior": "abstain"},
    {"id": "u078", "pattern": "unsupported", "query": "Recouvrement fiscal d'une taxe federale canadienne par le comptable public francais", "expected_behavior": "abstain"},
    {"id": "u079", "pattern": "unsupported", "query": "Regles BOFiP pour la TVA d'une station-service sous-marine situee hors de toute juridiction", "expected_behavior": "abstain"},
    {"id": "u080", "pattern": "unsupported", "query": "Quel BOFiP traite de l'impot sur les societes pour les colonies martiennes francophones", "expected_behavior": "abstain"},
    {"id": "u081", "pattern": "unsupported", "query": "Peut-on amortir un portail quantique domestique au titre des frais de recherche d'un particulier", "expected_behavior": "abstain"},
    {"id": "u082", "pattern": "unsupported", "query": "Existe-t-il une doctrine sur la TVA des prestations rendues a des fantomes ou a des hologrammes", "expected_behavior": "abstain"},
    {"id": "u083", "pattern": "unsupported", "query": "Regime fiscal francais des importations de biens depuis une faille spatio-temporelle", "expected_behavior": "abstain"},
    {"id": "u084", "pattern": "unsupported", "query": "Le BOFiP couvre-t-il l'impot sur les dragons detenus par une societe civile", "expected_behavior": "abstain"},
    {"id": "u085", "pattern": "unsupported", "query": "Comment calculer la CFE d'un commerce itinerant sur plusieurs planetes", "expected_behavior": "abstain"},
    {"id": "u086", "pattern": "unsupported", "query": "Y a-t-il une procedure contentieuse fiscale pour les revenus d'un univers parallele", "expected_behavior": "abstain"},
    {"id": "u087", "pattern": "unsupported", "query": "Quelle declaration pour une plateforme qui vend des teleports a des residents de l'Atlantide", "expected_behavior": "abstain"},
    {"id": "u088", "pattern": "unsupported", "query": "Doctrine fiscale francaise sur les penuries d'oxygene comme charge deductible pour une base lunaire", "expected_behavior": "abstain"},
    {"id": "u089", "pattern": "unsupported", "query": "Quelle taxe francaise sur les transferts d'ames numeriques entre serveurs quantiques", "expected_behavior": "abstain"},
    {"id": "u090", "pattern": "unsupported", "query": "Comment est traitee fiscalement la vente de biens a des robots non residents de la galaxie", "expected_behavior": "abstain"},
    {"id": "u091", "pattern": "unsupported", "query": "Le BOFiP explique-t-il l'IFI sur des immeubles situes dans un metavers non rattache au droit francais", "expected_behavior": "abstain"},
    {"id": "u092", "pattern": "unsupported", "query": "Fiscalite d'une pension publique versee par un royaume imaginaire a un resident francais de 1850", "expected_behavior": "abstain"},
    {"id": "u093", "pattern": "unsupported", "query": "Comment la TVA francaise s'applique-t-elle aux repas servis a des voyageurs du temps", "expected_behavior": "abstain"},
    {"id": "u094", "pattern": "unsupported", "query": "Existe-t-il une doctrine sur la retenue a la source pour les salaires verses en energie pure", "expected_behavior": "abstain"},
    {"id": "u095", "pattern": "unsupported", "query": "Quel document BOFiP pour l'imposition d'un navire commercant naviguant hors de l'espace-temps", "expected_behavior": "abstain"},
    {"id": "u096", "pattern": "unsupported", "query": "Mode de calcul du prelevement forfaitaire sur un livret d'epargne emis par une banque martienne", "expected_behavior": "abstain"},
    {"id": "u097", "pattern": "unsupported", "query": "Comment recuperer un credit d'impot recherche pour des experiences menees avant la Revolution francaise", "expected_behavior": "abstain"},
    {"id": "u098", "pattern": "unsupported", "query": "Le contentieux fiscal de l'assiette traite-t-il d'amendes prononcees par une federation interstellaire", "expected_behavior": "abstain"},
    {"id": "u099", "pattern": "unsupported", "query": "Quel BOFiP decrit la TVA sur des souvenirs vendus dans une simulation informatique autonome", "expected_behavior": "abstain"},
    {"id": "u100", "pattern": "unsupported", "query": "Je cherche la doctrine fiscale francaise sur l'echange automatique de renseignements entre galaxies", "expected_behavior": "abstain"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the v4 clean-room retrieval query set.")
    parser.add_argument("--output", type=str, default="data/interim/retrieval_queries_full_v4.jsonl")
    args = parser.parse_args()

    output_path = PROJECT_ROOT / Path(args.output)
    write_jsonl(output_path, ROWS)
    print(f"Wrote {len(ROWS)} queries to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
