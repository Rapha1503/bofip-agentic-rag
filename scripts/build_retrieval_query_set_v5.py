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
    {"id": "b001", "pattern": "scenario_business", "query": "On est une jeune boite innovante et on finance de la R&D. Est-ce qu'on peut encaisser la creance de CIR sans attendre plusieurs annees ?", "expected_boi": "BOI-BIC-CHAMP-80-20-20-20-20240703", "expected_behavior": "answer"},
    {"id": "b002", "pattern": "paraphrase", "query": "Dans les couts retenus pour le CIR, je cherche ce qui couvre l'environnement de travail de la recherche et pas seulement les salaires directs.", "expected_boi": "BOI-BIC-RICI-10-10-20-25-20250813", "expected_behavior": "answer"},
    {"id": "b003", "pattern": "scenario_business", "query": "Si une entreprise avance de l'argent via un pret remboursable sans interets pour des travaux d'energie, quel credit d'impot peut entrer en jeu ?", "expected_boi": "BOI-BIC-RICI-10-110-20-20241127", "expected_behavior": "answer"},
    {"id": "b004", "pattern": "procedural_question", "query": "Pour les revenus 2018 avec le PAS, jusqu'ou va encore le droit de reprise de l'administration ?", "expected_boi": "BOI-IR-PAS-50-20-50-20200210", "expected_behavior": "answer"},
    {"id": "b005", "pattern": "keyword_short", "query": "Je cherche la doctrine qui liste les revenus donnant lieu a acompte de prelevement a la source.", "expected_boi": "BOI-IR-PAS-10-10-20-20230706", "expected_behavior": "answer"},
    {"id": "b006", "pattern": "explicit_country", "query": "Ou est la doctrine franco-allemande pour les remunerations publiques et les pensions d'Etat ?", "expected_boi": "BOI-INT-CVB-DEU-10-40-20120912", "expected_behavior": "answer"},
    {"id": "b007", "pattern": "explicit_country", "query": "Je veux la partie France Algerie sur la facon d'eviter la double imposition.", "expected_boi": "BOI-INT-CVB-DZA-60-20141226", "expected_behavior": "answer"},
    {"id": "b008", "pattern": "near_neighbor", "query": "Quand la societe mere d'un groupe fiscal est absorbee, qu'est-ce qui se passe pour les deficits du groupe de depart ?", "expected_boi": "BOI-IS-GPE-50-10-30-20210811", "expected_behavior": "answer"},
    {"id": "b009", "pattern": "near_neighbor", "query": "Dans un groupe integre, je cherche le point precis sur la sous-capitalisation appreciee au niveau groupe.", "expected_boi": "BOI-IS-GPE-20-20-70-20190731", "expected_behavior": "answer"},
    {"id": "b010", "pattern": "false_premise_answerable", "query": "Est-ce que la presse sort toujours du champ de la TVA sans autre condition ?", "expected_boi": "BOI-TVA-SECT-40-10-30-20130621", "expected_behavior": "answer"},
    {"id": "b011", "pattern": "rescrit_case", "query": "Je cherche le rescrit TVA sur un dispositif reliant un fauteuil roulant a une trottinette electrique.", "expected_boi": "BOI-RES-TVA-000074-20210309", "expected_behavior": "answer"},
    {"id": "b012", "pattern": "paraphrase", "query": "En TVA, quel acteur est cense verser l'impot quand on parle d'operations normales de vente ou de prestation ?", "expected_boi": "BOI-TVA-DECLA-10-10-20-20251022", "expected_behavior": "answer"},
    {"id": "b013", "pattern": "scenario_business", "query": "Une asso a but non lucratif commence a faire quelques operations taxables. Quelles obligations TVA specifiques doit-elle regarder ?", "expected_boi": "BOI-TVA-DECLA-20-30-20-10-20220323", "expected_behavior": "answer"},
    {"id": "b014", "pattern": "procedural_question", "query": "En contentieux d'assiette, dans quels cas l'administration peut changer de base legale ou jouer la compensation ?", "expected_boi": "BOI-CTX-DG-20-40-10-20120912", "expected_behavior": "answer"},
    {"id": "b015", "pattern": "scenario_business", "query": "Pour un agent general d'assurances qui cesse son activite avec indemnite compensatrice, je cherche le traitement d'imposition.", "expected_boi": "BOI-BNC-CESS-40-10-20230517", "expected_behavior": "answer"},
    {"id": "b016", "pattern": "topic_specific", "query": "Je veux la doctrine sur les transformations de societes ou les ajustements des statuts, sans etre sur une fusion.", "expected_boi": "BOI-ENR-AVS-20-30-30-20120912", "expected_behavior": "answer"},
    {"id": "b017", "pattern": "near_neighbor", "query": "Dans les actes de vie sociale, ou est la partie qui borne vraiment le regime special des fusions ?", "expected_boi": "BOI-ENR-AVS-20-60-30-10-20140613", "expected_behavior": "answer"},
    {"id": "b018", "pattern": "keyword_short", "query": "Je cherche la regle sur l'exoneration durable de TFNB pour des terrains en oliviers.", "expected_boi": "BOI-IF-TFNB-10-40-60-20120912", "expected_behavior": "answer"},
    {"id": "b019", "pattern": "paraphrase", "query": "Quand le perimetre d'une commune change, comment la doctrine traite l'ajustement des valeurs locatives ?", "expected_boi": "BOI-IF-TFB-20-20-10-40-20121210", "expected_behavior": "answer"},
    {"id": "b020", "pattern": "topic_specific", "query": "Je veux le texte de base sur le calcul general de la valeur ajoutee pour la CVAE.", "expected_boi": "BOI-CVAE-BASE-10-20120912", "expected_behavior": "answer"},
    {"id": "b021", "pattern": "topic_specific", "query": "Pour du credit-bail mobilier, ou est la doctrine generale sur le regime fiscal de l'operation ?", "expected_boi": "BOI-BIC-BASE-60-20-20120912", "expected_behavior": "answer"},
    {"id": "b022", "pattern": "false_premise_answerable", "query": "Une entreprise peut-elle laisser tomber un amortissement obligatoire si cela l'arrange comptablement ?", "expected_boi": "BOI-BIC-AMT-10-50-10-20120912", "expected_behavior": "answer"},
    {"id": "b023", "pattern": "paraphrase", "query": "Je veux la doctrine BIC qui explique ce qu'on fait des produits ou charges sans lien avec l'activite depuis la fin de l'ancienne theorie du bilan.", "expected_boi": "BOI-BIC-BASE-90-20180704", "expected_behavior": "answer"},
    {"id": "b024", "pattern": "topic_specific", "query": "Quelles aides fiscales visees en BIC existent pour une creation dans une zone de developpement prioritaire ?", "expected_boi": "BOI-BIC-CHAMP-80-10-100-30-20200422", "expected_behavior": "answer"},
    {"id": "b025", "pattern": "keyword_short", "query": "Je veux juste le bareme IFI, la partie qui donne l'echelle d'imposition.", "expected_boi": "BOI-PAT-IFI-40-10-20180608", "expected_behavior": "answer"},
    {"id": "b026", "pattern": "topic_specific", "query": "Une personne domiciliee fiscalement en France entre-t-elle de plein droit dans l'IFI ou y a-t-il une regle precise a regarder ?", "expected_boi": "BOI-PAT-IFI-10-20-20-20180608", "expected_behavior": "answer"},
    {"id": "b027", "pattern": "false_premise_answerable", "query": "Le plafonnement de l'IFI marche-t-il automatiquement des qu'on paye beaucoup d'impot, sans autre filtre ?", "expected_boi": "BOI-PAT-IFI-40-30-10-20181122", "expected_behavior": "answer"},
    {"id": "b028", "pattern": "topic_specific", "query": "Je cherche la doctrine de base sur ce que doivent faire fiscalement les plateformes de mise en relation en ligne.", "expected_boi": "BOI-INT-AEA-30-10-20230111", "expected_behavior": "answer"},
    {"id": "b029", "pattern": "explicit_reference", "query": "Je veux la vue d'ensemble BOFiP sur les ETNC quand on raisonne avec les conventions fiscales.", "expected_boi": "BOI-INT-DG-20-50-20210224", "expected_behavior": "answer"},
    {"id": "b030", "pattern": "topic_specific", "query": "Ou est la doctrine sur le prelevement forfaitaire obligatoire applique aux produits de placement a revenu fixe ?", "expected_boi": "BOI-RPPM-RCM-30-20-40-20220630", "expected_behavior": "answer"},
    {"id": "b031", "pattern": "keyword_short", "query": "Je veux le document PEA qui sert un peu de rubrique fourre-tout ou regles diverses.", "expected_boi": "BOI-RPPM-RCM-40-50-60-20240730", "expected_behavior": "answer"},
    {"id": "b032", "pattern": "procedural_question", "query": "Pour une plus-value sur un bien meuble incorporel, quelle est la doctrine sur les formalites declaratives du contribuable ?", "expected_boi": "BOI-RPPM-PVBMI-40-10-20141014", "expected_behavior": "answer"},
    {"id": "b033", "pattern": "procedural_question", "query": "Quand le comptable public accepte un echeancier, je cherche la doctrine sur l'effet concret sur les poursuites.", "expected_boi": "BOI-REC-PREA-20-10-10-20150506", "expected_behavior": "answer"},
    {"id": "b034", "pattern": "procedural_question", "query": "Ou est la partie BOFiP sur la saisie des remunerations en recouvrement force ?", "expected_boi": "BOI-REC-FORCE-20-20-20-20191127", "expected_behavior": "answer"},
    {"id": "b035", "pattern": "procedural_question", "query": "Je cherche les regles de validite formelle d'un cautionnement en matiere de recouvrement fiscal.", "expected_boi": "BOI-REC-GAR-20-40-10-20-20120912", "expected_behavior": "answer"},
    {"id": "b036", "pattern": "procedural_question", "query": "Devant le tribunal administratif, quelle est la marche de l'instruction d'une demande fiscale ?", "expected_boi": "BOI-CTX-ADM-10-30-20120912", "expected_behavior": "answer"},
    {"id": "b037", "pattern": "procedural_question", "query": "Apres une decision de cour administrative d'appel en fiscal, quels sont encore les recours envisageables ?", "expected_boi": "BOI-CTX-ADM-20-50-20120912", "expected_behavior": "answer"},
    {"id": "b038", "pattern": "procedural_question", "query": "Je veux la doctrine sur l'articulation QPC / question prejudicielle dans un appel judiciaire fiscal.", "expected_boi": "BOI-CTX-JUD-20-20-60-20120912", "expected_behavior": "answer"},
    {"id": "b039", "pattern": "topic_specific", "query": "Dans quels cas la valeur locative retenue pour la CFE peut-elle etre reduite ?", "expected_boi": "BOI-IF-CFE-20-20-30-20160706", "expected_behavior": "answer"},
    {"id": "b040", "pattern": "topic_specific", "query": "Je cherche le texte cadre qui dit quand la taxe d'amenagement s'applique, avant les details.", "expected_boi": "BOI-IF-TU-10-20-20251231", "expected_behavior": "answer"},
    {"id": "b041", "pattern": "paraphrase", "query": "Des recettes tirees de terrains nus restent-elles dans les revenus fonciers, et ou la doctrine le pose-t-elle ?", "expected_boi": "BOI-RFPI-CHAMP-10-20-20120912", "expected_behavior": "answer"},
    {"id": "b042", "pattern": "scenario_business", "query": "Un jeune agriculteur profite combien de temps de l'avantage sur son benefice ?", "expected_boi": "BOI-BA-BASE-30-10-20-20190515", "expected_behavior": "answer"},
    {"id": "b043", "pattern": "keyword_short", "query": "Je cherche la doctrine BIC sur les charges financieres indexees.", "expected_boi": "BOI-BIC-CHG-50-60-20120912", "expected_behavior": "answer"},
    {"id": "b044", "pattern": "topic_specific", "query": "Pour former un groupe fiscal, quelles societes peuvent entrer et a quelle date on regarde l'exercice ?", "expected_boi": "BOI-IS-GPE-10-10-20-20160907", "expected_behavior": "answer"},
    {"id": "b045", "pattern": "topic_specific", "query": "Je veux la doctrine generale IS sur les reprises d'entreprises industrielles en difficulte.", "expected_boi": "BOI-IS-GEO-20-10-20150603", "expected_behavior": "answer"},
    {"id": "b046", "pattern": "false_premise_answerable", "query": "Une JEI ou une JEU obtient-elle les allegements sans aucune verification prealable ?", "expected_boi": "BOI-BIC-CHAMP-80-20-20-10-20250716", "expected_behavior": "answer"},
    {"id": "b047", "pattern": "topic_specific", "query": "Ou est la partie qui explique comment la France dresse puis met a jour la liste des ETNC ?", "expected_boi": "BOI-INT-DG-20-50-10-20210224", "expected_behavior": "answer"},
    {"id": "b048", "pattern": "topic_specific", "query": "Au-dela du prelevement forfaitaire obligatoire, quelles mesures de controle concernent les placements a revenu fixe ?", "expected_boi": "BOI-RPPM-RCM-30-20-70-20191220", "expected_behavior": "answer"},
    {"id": "b049", "pattern": "paraphrase", "query": "Pour un PEA deja ouvert, je cherche la doctrine qui explique son fonctionnement courant.", "expected_boi": "BOI-RPPM-RCM-40-50-20-20240730", "expected_behavior": "answer"},
    {"id": "b050", "pattern": "topic_specific", "query": "Je veux la liste doctrinale des constructions et operations qui entrent dans la taxe d'amenagement.", "expected_boi": "BOI-IF-TU-10-20-10-20251231", "expected_behavior": "answer"},
    {"id": "b051", "pattern": "topic_specific", "query": "Quelles constructions liees a une mission publique peuvent sortir de la taxe d'amenagement ?", "expected_boi": "BOI-IF-TU-10-20-30-10-20251231", "expected_behavior": "answer"},
    {"id": "b052", "pattern": "near_neighbor", "query": "Si la mere du groupe est scindee, comment la doctrine traite les deficits de l'ancien ensemble integre ?", "expected_boi": "BOI-IS-GPE-50-30-30-20210811", "expected_behavior": "answer"},
    {"id": "b053", "pattern": "near_neighbor", "query": "Quand 95 % du capital de la mere change de proprietaire, quel effet sur le deficit de l'ancien groupe fiscal ?", "expected_boi": "BOI-IS-GPE-50-20-20-20-20210811", "expected_behavior": "answer"},
    {"id": "b054", "pattern": "topic_specific", "query": "Dans le CIR, ou est la partie sur les depenses de normalisation rattachees aux produits de l'entreprise ?", "expected_boi": "BOI-BIC-RICI-10-10-20-50-20250813", "expected_behavior": "answer"},
    {"id": "b055", "pattern": "topic_specific", "query": "Je veux le point CIR sur les depenses liees aux brevets et aux certificats d'obtention vegetale.", "expected_boi": "BOI-BIC-RICI-10-10-20-40-20250813", "expected_behavior": "answer"},
    {"id": "b056", "pattern": "scenario_business", "query": "Dans quels cas une plateforme devient-elle solidairement tenue de payer la TVA due par les vendeurs ?", "expected_boi": "BOI-TVA-DECLA-10-10-30-20-20200902", "expected_behavior": "answer"},
    {"id": "b057", "pattern": "paraphrase", "query": "Je veux le document d'ensemble sur la notion de redevable TVA, pas un sous-cas precis.", "expected_boi": "BOI-TVA-DECLA-10-10-20200323", "expected_behavior": "answer"},
    {"id": "b058", "pattern": "near_neighbor", "query": "Dans un groupe fiscal, ou est la doctrine sur les operations d'apport-attribution ?", "expected_boi": "BOI-IS-GPE-50-40-20210811", "expected_behavior": "answer"},
    {"id": "b059", "pattern": "near_neighbor", "query": "Je cherche le point sur l'appartenance au groupe quand les participations bougent pendant d'autres restructurations.", "expected_boi": "BOI-IS-GPE-50-50-30-20200415", "expected_behavior": "answer"},
    {"id": "b060", "pattern": "procedural_question", "query": "Jusqu'a quelle date peut-on encore deposer une demande de rescrit concernant le statut JEI ?", "expected_boi": "BOI-RES-SJ-000014-20210309", "expected_behavior": "answer"},
    {"id": "b061", "pattern": "topic_specific", "query": "Avant de faire les declarations, quelles diligences une plateforme doit-elle mener sur les vendeurs concernes ?", "expected_boi": "BOI-INT-AEA-30-20-20231213", "expected_behavior": "answer"},
    {"id": "b062", "pattern": "topic_specific", "query": "Pour les plateformes, qu'est-ce qui doit etre declare a l'administration dans la vue d'ensemble ?", "expected_boi": "BOI-INT-AEA-30-30-20230111", "expected_behavior": "answer"},
    {"id": "b063", "pattern": "procedural_question", "query": "Comment une plateforme doit-elle transmettre sa declaration en pratique ?", "expected_boi": "BOI-INT-AEA-30-40-20231213", "expected_behavior": "answer"},
    {"id": "b064", "pattern": "topic_specific", "query": "Quelles sanctions sont prevues pour une plateforme qui ne respecte pas ses obligations declaratives ?", "expected_boi": "BOI-INT-AEA-30-50-20231213", "expected_behavior": "answer"},
    {"id": "b065", "pattern": "near_neighbor", "query": "Pour savoir si un operateur est etabli en France au regard de la TVA, quelle doctrine de base consulter ?", "expected_boi": "BOI-TVA-DECLA-10-10-10-20120912", "expected_behavior": "answer"},
    {"id": "b066", "pattern": "topic_specific", "query": "Je cherche la doctrine de base sur le prelevement forfaitaire obligatoire applicable aux placements de taux.", "expected_boi": "BOI-RPPM-RCM-30-20-10-20210706", "expected_behavior": "answer"},
    {"id": "b067", "pattern": "topic_specific", "query": "Qui est charge du prelevement a la source sur les revenus fixes entrant dans ce regime forfaitaire obligatoire ?", "expected_boi": "BOI-RPPM-RCM-30-20-20-20191220", "expected_behavior": "answer"},
    {"id": "b068", "pattern": "near_neighbor", "query": "Comment l'exoneration IS pour reprise d'entreprise en difficulte s'articule avec d'autres regimes ?", "expected_boi": "BOI-IS-GEO-20-10-40-20150603", "expected_behavior": "answer"},
    {"id": "b069", "pattern": "topic_specific", "query": "Quand la recherche est sous-traitee, quelles depenses externalisees peuvent encore entrer dans le CIR ?", "expected_boi": "BOI-BIC-RICI-10-10-20-30-20250813", "expected_behavior": "answer"},
    {"id": "b070", "pattern": "topic_specific", "query": "Pour le CIR, comment la doctrine traite l'amortissement des immobilisations employees pour la recherche ?", "expected_boi": "BOI-BIC-RICI-10-10-20-10-20250813", "expected_behavior": "answer"},
    {"id": "v071", "pattern": "unsupported", "query": "Je cherche la doctrine BOFiP sur la fiscalite d'une activite commerciale installee sur Mars.", "expected_behavior": "abstain"},
    {"id": "v072", "pattern": "unsupported", "query": "Comment la DGFIP recouvre-t-elle un impot federal canadien sans aucun ancrage francais ?", "expected_behavior": "abstain"},
    {"id": "v073", "pattern": "unsupported", "query": "Existe-t-il une doctrine fiscale sur les voyages temporels proposes a des particuliers ?", "expected_behavior": "abstain"},
    {"id": "v074", "pattern": "unsupported", "query": "Je veux le texte BOFiP sur l'imposition d'une mine lunaire detenue par une startup francaise.", "expected_behavior": "abstain"},
    {"id": "v075", "pattern": "unsupported", "query": "Quelles regles de TVA s'appliquent a un service de teleportation de marchandises ?", "expected_behavior": "abstain"},
    {"id": "v076", "pattern": "unsupported", "query": "Le BOFiP couvre-t-il une crypto-monnaie emise par une entite extraterrestre ?", "expected_behavior": "abstain"},
    {"id": "v077", "pattern": "unsupported", "query": "Je cherche la fiscalite d'un peage interdimensionnel facture a des clients particuliers.", "expected_behavior": "abstain"},
    {"id": "v078", "pattern": "unsupported", "query": "Quelle procedure contentieuse francaise pour un impot communal belge sans lien avec la France ?", "expected_behavior": "abstain"},
    {"id": "v079", "pattern": "unsupported", "query": "Y a-t-il une doctrine sur la TVA d'une station-service situee hors de toute juridiction terrestre ?", "expected_behavior": "abstain"},
    {"id": "v080", "pattern": "unsupported", "query": "Quel document BOFiP traite de l'impot sur les societes dans une colonie martienne ?", "expected_behavior": "abstain"},
    {"id": "v081", "pattern": "unsupported", "query": "Peut-on faire entrer un portail quantique prive dans le CIR d'un particulier ?", "expected_behavior": "abstain"},
    {"id": "v082", "pattern": "unsupported", "query": "Je veux la doctrine sur la TVA de services rendus a des hologrammes ou des fantomes.", "expected_behavior": "abstain"},
    {"id": "v083", "pattern": "unsupported", "query": "La doctrine BOFiP couvre-t-elle les importations depuis une faille spatio-temporelle ?", "expected_behavior": "abstain"},
    {"id": "v084", "pattern": "unsupported", "query": "Existe-t-il une doctrine sur l'imposition de dragons detenus par une societe civile ?", "expected_behavior": "abstain"},
    {"id": "v085", "pattern": "unsupported", "query": "Comment calculer la CFE d'un commerce ambulant exploite sur plusieurs planetes ?", "expected_behavior": "abstain"},
    {"id": "v086", "pattern": "unsupported", "query": "Y a-t-il une procedure fiscale pour des revenus tires d'un univers parallele ?", "expected_behavior": "abstain"},
    {"id": "v087", "pattern": "unsupported", "query": "Quelle declaration BOFiP pour une plateforme qui vend des teleports a des residents de l'Atlantide ?", "expected_behavior": "abstain"},
    {"id": "v088", "pattern": "unsupported", "query": "Je veux la doctrine sur la deduction fiscale d'un manque d'oxygene dans une base lunaire.", "expected_behavior": "abstain"},
    {"id": "v089", "pattern": "unsupported", "query": "Quelle taxe francaise sur les transferts d'ames numeriques entre serveurs quantiques ?", "expected_behavior": "abstain"},
    {"id": "v090", "pattern": "unsupported", "query": "Comment la France taxe-t-elle la vente de biens a des robots non residents d'une autre galaxie ?", "expected_behavior": "abstain"},
    {"id": "v091", "pattern": "unsupported", "query": "Le BOFiP explique-t-il l'IFI sur des immeubles purement virtuels d'un metavers autonome ?", "expected_behavior": "abstain"},
    {"id": "v092", "pattern": "unsupported", "query": "Je cherche la doctrine sur une pension publique versee par un royaume imaginaire a un resident francais.", "expected_behavior": "abstain"},
    {"id": "v093", "pattern": "unsupported", "query": "Quelle TVA pour des repas servis a des voyageurs du temps en France ?", "expected_behavior": "abstain"},
    {"id": "v094", "pattern": "unsupported", "query": "Existe-t-il une retenue a la source pour des salaires verses en energie pure ?", "expected_behavior": "abstain"},
    {"id": "v095", "pattern": "unsupported", "query": "Je cherche le BOFiP sur l'imposition d'un navire naviguant hors de l'espace-temps.", "expected_behavior": "abstain"},
    {"id": "v096", "pattern": "unsupported", "query": "Quel prelevement forfaitaire sur un livret d'epargne emis par une banque martienne ?", "expected_behavior": "abstain"},
    {"id": "v097", "pattern": "unsupported", "query": "Comment recuperer un credit d'impot recherche pour des experiences menees avant 1789 ?", "expected_behavior": "abstain"},
    {"id": "v098", "pattern": "unsupported", "query": "Le contentieux fiscal de l'assiette traite-t-il d'amendes d'une federation interstellaire ?", "expected_behavior": "abstain"},
    {"id": "v099", "pattern": "unsupported", "query": "Quel document BOFiP sur la TVA de souvenirs vendus dans une simulation autonome ?", "expected_behavior": "abstain"},
    {"id": "v100", "pattern": "unsupported", "query": "Je veux la doctrine francaise sur l'echange automatique de renseignements entre galaxies.", "expected_behavior": "abstain"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the v5 clean-room retrieval query set with new user-like phrasings.")
    parser.add_argument("--output", type=str, default="data/interim/retrieval_queries_full_v5.jsonl")
    args = parser.parse_args()

    output_path = PROJECT_ROOT / Path(args.output)
    write_jsonl(output_path, ROWS)
    print(f"Wrote {len(ROWS)} queries to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
