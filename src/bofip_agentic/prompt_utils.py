"""Shared prompt building for BOFIP RAG — used by app.py + preview_answer.py."""
from __future__ import annotations


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build a coverage-aware citation prompt from query + retrieved chunks."""
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c['rank']}] BOI: {c['boi_reference']}\n"
            f"Titre: {c['title']}\n"
            f"Date: {c['publication_date'] or 'inconnue'}\n"
            f"Section: {c['section_path'] or '(sans section)'}\n"
            f"Texte: {c['text']}"
        )
    return (
        "Question utilisateur:\n" + query + "\n\n"
        "Extraits BOFiP fournis:\n" + "\n\n".join(blocks) + "\n\n"
        "Instructions:\n"
        "- Tu es un assistant fiscal. Reponds UNIQUEMENT a partir des extraits fournis.\n"
        "- N'invente ni source, ni article, ni taux, ni reponse.\n"
        "- Les montants et taux donnes par l'utilisateur dans la question peuvent etre utilises sans citation.\n"
        "- Si la question contient un chiffre ou un taux qui est contredit par les extraits, corrige-le explicitement en citant l'extrait.\n"
        "- Si la question demande un calcul, detaille les etapes: formule, valeurs, resultat.\n"
        "- Renvoie un objet JSON valide et rien d'autre. Pas de markdown autour.\n\n"
        'Schema JSON: {"answer_status":"supported|partial|insufficient_evidence","axes_requis":["..."],"axes_couverts":["..."],"axes_manquants":["..."],"conclusion":"...","justification_bullets":["..."],"limits":"..."}\n\n'
        "Etape 1 - Identifier les axes fiscaux requis (1 a 5 axes).\n"
        "Etape 2 - Verifier couverture: supported (tous couverts) | partial (mixte) | insufficient_evidence (aucun).\n"
        "Etape 3 - Redige une reponse complete comme un comptable ou fiscaliste:\n"
        "- conclusion: resume la reponse en une phrase (inclure le montant si calcul).\n"
        "- justification_bullets: 2 a 4 puces detaillees.\n"
        "  Chaque puce cite ses sources [n] et explique le raisonnement juridique.\n"
        "  Si c'est un calcul, detaille: formule, valeurs, resultat intermediaire, total.\n"
        "  Si c'est une procedure, explique les etapes chronologiquement.\n"
        "  Si c'est une condition, enumere les criteres un par un.\n"
        "- limits obligatoire <= 50 mots. Lister axes manquants si partial.\n"
        "- Citations [n] referencent UNIQUEMENT les extraits fournis.\n"
    )
