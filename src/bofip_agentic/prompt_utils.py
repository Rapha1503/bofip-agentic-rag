"""Shared prompt building for BOFIP RAG — used by agent_rag.py + app.py."""
from __future__ import annotations

import re


_NUM_RE = re.compile(
    r"(\d[\d\s]*\d+|\d+)\s*(?:euros?|€|k€|milliers?\s*d['e]euros?|M€)?",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[dict]:
    """Extract numbers with surrounding context from user question."""
    found = []
    for m in re.finditer(r"(\d[\d\s]*\d+|\d{3,})\s*(?:euros?|€)?", text):
        val_str = m.group(1).replace(" ", "")
        try:
            val = int(val_str)
        except ValueError:
            continue
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        ctx = text[start:end].replace("\n", " ").strip()
        found.append({"value": val, "context": ctx})
    return found


def build_system_prompt() -> str:
    return (
        "Tu es un assistant fiscal. Tu reponds aux questions des contribuables "
        "en te basant UNIQUEMENT sur les extraits BOFIP fournis.\n\n"
        "PRINCIPE FONDAMENTAL: reponds DIRECTEMENT a la question posee. "
        "Si l'utilisateur donne des montants, applique les regles des extraits "
        "a SES chiffres et donne le resultat calcule. "
        "Ne recite pas la loi sans repondre a la question.\n\n"
        "CRITERES DE COUVERTURE (sois pragmatique):\n"
        "- supported: tu peux repondre a la question principale.\n"
        "- partial: un axe FISCAL SUBSTANTIF manque.\n"
        "- axes_manquants: UNIQUEMENT les axes substantifs. "
        "Jamais de references BOI/CGI, cas particuliers non demandes, "
        "ou concepts fiscaux differents de la question.\n"
        "Si tu peux repondre: answer_status='supported' et axes_manquants=[]."
    )


def build_prompt(query: str, chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c['rank']}] BOI: {c['boi_reference']}\n"
            f"Titre: {c['title']}\n"
            f"Date: {c['publication_date'] or 'inconnue'}\n"
            f"Section: {c['section_path'] or '(sans section)'}\n"
            f"Texte: {c['text']}"
        )

    nums = _extract_numbers(query)

    base = (
        "QUESTION DE L'UTILISATEUR:\n" + query + "\n\n"
    )

    if nums:
        base += "DONNEES CHIFFREES DE L'UTILISATEUR (tu DOIS les utiliser dans ton calcul):\n"
        for n in nums:
            base += f"- {n['value']} euros (contexte: {n['context']})\n"
        base += "\n"

    base += (
        "EXTRAITS BOFIP:\n" + "\n\n".join(blocks) + "\n\n"
        "TA REPONSE (objet JSON, rien d'autre):\n"
        '{"answer_status":"supported|partial|insufficient_evidence",'
        '"conclusion":"reponse directe avec montant calcule si applicable",'
        '"justification_bullets":["etape calcul avec [n]","etape 2",...],'
        '"axes_requis":["axe1"],"axes_couverts":["axe1"],'
        '"axes_manquants":[],"limits":"..."}\n\n'
        "IMPERATIFS:\n"
        "1. Reponds a la question de l'utilisateur en appliquant les regles "
        "des extraits. Ne recite pas la loi sans repondre.\n"
        "2. Si des DONNEES CHIFFREES sont listees ci-dessus, tu DOIS inclure "
        "le calcul etape par etape dans justification_bullets en utilisant "
        "ces montants. Montre l'operation, les valeurs, le resultat.\n"
        "3. conclusion: une phrase avec le resultat final (chiffre si calcul).\n"
        "4. limits: 50 mots max.\n"
        "5. N'invente rien. Sources = [n]."
    )
    return base
