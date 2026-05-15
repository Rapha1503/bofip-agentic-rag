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
        "- Si la question contient une premisse fausse et que les extraits la contredisent, corrige-la.\n"
        "- Renvoie un objet JSON valide et rien d'autre. Pas de markdown autour.\n\n"
        'Schema JSON: {"answer_status":"supported|partial|insufficient_evidence","axes_requis":["..."],"axes_couverts":["..."],"axes_manquants":["..."],"conclusion":"...","justification_bullets":["..."],"limits":"..."}\n\n'
        "Etape 1 - Identifier les axes fiscaux requis (1 a 5 axes).\n"
        "Etape 2 - Verifier couverture: supported (tous couverts) | partial (mixte) | insufficient_evidence (aucun).\n"
        "Etape 3 - 2-4 puces avec citations [n] pour axes couverts. Puce explicative pour chaque axe manquant.\n"
        "- limits obligatoire <= 40 mots. Lister axes manquants si partial.\n"
        "- Citations [n] referencent UNIQUEMENT les extraits fournis.\n"
    )
