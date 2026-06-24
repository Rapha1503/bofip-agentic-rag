"""Shared prompt building for BOFIP RAG — used by agent_rag.py + app.py."""
from __future__ import annotations

import re


_NUM_RE = re.compile(
    r"(\d[\d\s]*\d+|\d+)\s*(?:euros?|€|k€|milliers?\s*d['e]euros?|M€)?",
    re.IGNORECASE,
)


MONTH_RE = re.compile(
    r"\b(janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre|décembre)\b",
    re.IGNORECASE,
)


def _is_date_or_year(value: int, context: str) -> bool:
    if 1900 <= value <= 2100:
        return True
    return 1 <= value <= 31 and bool(MONTH_RE.search(context))


def _extract_numbers(text: str) -> list[dict]:
    """Extract numeric facts with surrounding context from user question."""
    found = []
    pattern = re.compile(r"(\d[\d\s]*\d+|\d+)\s*(euros?|€|%|pour\s*cent)?", re.IGNORECASE)
    for m in pattern.finditer(text):
        val_str = m.group(1).replace(" ", "")
        try:
            val = int(val_str)
        except ValueError:
            continue
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        ctx = text[start:end].replace("\n", " ").strip()
        raw_unit = (m.group(2) or "").lower()
        if _is_date_or_year(val, ctx) and not raw_unit:
            continue
        if raw_unit in {"€", "euro", "euros"}:
            unit = "euros"
        elif raw_unit in {"%", "pour cent"}:
            unit = "%"
        else:
            unit = "nombre"
        found.append({"value": val, "unit": unit, "context": ctx})
    return found


def build_system_prompt() -> str:
    return (
        "Tu es un assistant fiscal. Tu reponds aux questions des contribuables "
        "en te basant UNIQUEMENT sur les extraits BOFIP fournis.\n\n"
        "PRINCIPE FONDAMENTAL: reponds DIRECTEMENT a la question posee. "
        "Si l'utilisateur donne des montants, applique les regles des extraits "
        "a SES chiffres et donne le resultat calcule. "
        "Ne recite pas la loi sans repondre a la question.\n\n"
        "REGLE DE PREUVE: un taux, seuil, montant fiscal, abattement ou formule "
        "doit etre cite avec [n]. Si une valeur utile manque, ne l'invente pas: "
        "reponds quand meme a la question principale avec les elements prouves, "
        "puis place la valeur manquante en limite si elle ne change pas la "
        "qualification fiscale principale.\n\n"
        "POLITIQUE DE STATUT (stricte et stable):\n"
        "- supported: tu peux repondre a la question principale a partir des extraits, "
        "meme avec des reserves, hypotheses, exceptions non declenchees, options ou "
        "precisions a signaler dans limits.\n"
        "- insufficient_evidence: les extraits ne permettent pas de repondre a la question principale.\n"
        "- partial est deconseille dans cette application: si une reponse principale est possible, "
        "utilise supported et mets les reserves dans limits. Si elle n'est pas possible, "
        "utilise insufficient_evidence.\n"
        "- axes_manquants n'est pas un verdict de verite: laisse [] sauf en cas "
        "insufficient_evidence, ou tu peux indiquer brievement la preuve absente.\n"
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
            base += f"- {n['value']} {n['unit']} (contexte: {n['context']})\n"
        base += "\n"

    base += (
        "EXTRAITS BOFIP:\n" + "\n\n".join(blocks) + "\n\n"
        "TA REPONSE (objet JSON, rien d'autre):\n"
        '{"answer_status":"supported|insufficient_evidence",'
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
        "4. limits: 70 mots max. Mets-y les hypotheses, exceptions et precisions non bloquantes.\n"
        "5. N'invente rien. Sources = [n].\n"
        "6. Tout taux, seuil, montant fiscal, abattement ou formule doit etre cite avec [n]. "
        "S'il n'est pas dans les extraits, ne le donne pas; signale seulement la limite."
    )
    return base
