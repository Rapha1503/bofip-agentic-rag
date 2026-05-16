"""Agentic RAG — self-evaluating retrieval loop for BOFIP fiscal questions.

Architecture (inspired by Azure Agentic Retrieval + Corrective RAG):
    1. Retrieve — first pass via RagRuntime
    2. Answer + Self-Evaluate — LLM returns {answer_status, axes_requis, axes_couverts, axes_manquants, ...}
    3. IF partial → Reformulate missing axes → Retrieve 2nd pass with branch hint → Merge → Final answer
    4. IF still partial → 3rd recovery iteration with forced branch routing
    5. Trace — full audit log

LLM calls per query: 2 (first pass successful) or 3-4 (with reformulation).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .prompt_utils import build_prompt
from .rag_runtime import RagRuntime


# ---------------------------------------------------------------------------
# BOFIP domain routing (soft boost, never filter)
# ---------------------------------------------------------------------------

_BRANCH_KEYWORDS: dict[str, dict[str, str]] = {
    "TVA": {"déduction": "BOI-TVA-DED", "déduire": "BOI-TVA-DED", "récupérer": "BOI-TVA-DED",
            "taux": "BOI-TVA-LIQ", "champ": "BOI-TVA-CHAMP",
            "déclaration": "BOI-TVA-DECLA", "déclarer": "BOI-TVA-DECLA",
            "franchise": "BOI-TVA-DECLA", "seuil": "BOI-TVA-DECLA",
            "immobilier": "BOI-TVA-IMM", "immeuble": "BOI-TVA-IMM",
            "base": "BOI-TVA-BASE", "sectoriel": "BOI-TVA-SECT",
            "remboursement": "BOI-TVA-DED", "crédit": "BOI-TVA-DED"},
    "BIC": {"charges": "BOI-BIC-CHG", "amortissement": "BOI-BIC-AMT", "amortir": "BOI-BIC-AMT",
            "base": "BOI-BIC-BASE", "déduction": "BOI-BIC-CHG", "frais": "BOI-BIC-CHG",
            "déductible": "BOI-BIC-CHG", "justificatif": "BOI-BIC-CHG",
            "provision": "BOI-BIC-PROV"},
    "IS": {"taux": "BOI-IS-LIQ", "groupe": "BOI-IS-GPE", "cessation": "BOI-IS-CESS",
           "déficit": "BOI-IS-GPE", "intégration": "BOI-IS-GPE",
           "acompte": "BOI-IS-GPE", "dividende": "BOI-IS-BASE"},
    "CF": {"procédure": "BOI-CF-IOR", "contrôle": "BOI-CF-IOR", "rectification": "BOI-CF-IOR",
           "sanction": "BOI-CF-INF", "pénalité": "BOI-CF-INF", "intérêt": "BOI-CF-INF",
           "abus": "BOI-CF-IOR", "vérification": "BOI-CF-IOR",
           "droit": "BOI-CF-COM", "communication": "BOI-CF-COM"},
    "IR": {"revenus": "BOI-IR-BASE", "BNC": "BOI-BNC-BASE", "foncier": "BOI-RFPI-BASE",
           "plus-value": "BOI-RFPI-PVI", "télétravail": "BOI-RSA-BASE",
           "salarié": "BOI-RSA-BASE", "loyer": "BOI-RFPI-BASE"},
    "IF": {"taxe foncière": "BOI-IF-TFB", "CFE": "BOI-IF-CFE"},
    "ENR": {"donation": "BOI-ENR-DMTG", "succession": "BOI-ENR-DMTG",
            "enregistrement": "BOI-ENR-DG", "notaire": "BOI-ENR-DG"},
    "PAT": {"IFI": "BOI-PAT-IFI"},
}

# Minimum confidence to apply branch boost (lowered for pre-inference)
_MIN_BRANCH_CONFIDENCE = 0.50
# Score multiplier for matching docs
_BRANCH_BOOST = 1.15


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgenticRAG:
    def __init__(
        self,
        runtime: RagRuntime,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_iterations: int = 2,
    ):
        self.rt = runtime
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict:
        """Main entry point. Returns structured result with trace."""
        trace: list[dict] = []
        all_chunks: list[dict] = []
        seen_ids: set[str] = set()
        branch_hint: str | None = None
        branch_confidence: float = 0.0

        t_start = time.time()
        original_question = question

        # Pre-infer branch from original question (used on iter1 if confidence high enough)
        pre_branch = _infer_bofip_branch(
            original_question, "",
            {"axes_couverts": [], "axes_manquants": []},
            []
        )
        branch_hint: str | None = None
        branch_confidence: float = 0.0
        if pre_branch.get("confidence", 0) >= _MIN_BRANCH_CONFIDENCE:
            branch_hint = pre_branch["branch"]
            branch_confidence = pre_branch["confidence"]

        for iteration in range(1, self.max_iterations + 1):
            step_log = {"iteration": iteration}

            # --- Retrieve ---
            t0 = time.time()
            hint = branch_hint if branch_confidence >= _MIN_BRANCH_CONFIDENCE else None
            result = self.rt.retrieve(question, top_docs=8, max_chunks=8, branch_hint=hint)
            chunks = _chunks_from_result(result)
            step_log["retrieve_s"] = round(time.time() - t0, 2)
            step_log["docs_found"] = len(result.stage1_hits)
            step_log["chunks_found"] = len(chunks)
            if hint and iteration == 1:
                step_log["branch_hint_initial"] = hint

            # Merge & deduplicate
            new_chunks = [c for c in chunks if c["chunk_id"] not in seen_ids]
            for c in new_chunks:
                seen_ids.add(c["chunk_id"])
            all_chunks.extend(new_chunks)
            step_log["chunks_new"] = len(new_chunks)
            step_log["chunks_total"] = len(all_chunks)

            # --- Answer + Self-Evaluate ---
            t0 = time.time()
            answer = self._answer(question, all_chunks)
            step_log["answer_s"] = round(time.time() - t0, 2)
            step_log["answer_status"] = answer.get("answer_status", "?")
            step_log["axes_requis"] = answer.get("axes_requis", [])
            step_log["axes_couverts"] = answer.get("axes_couverts", [])
            step_log["axes_manquants"] = answer.get("axes_manquants", [])

            trace.append(step_log)

            # Check if sufficient
            status = answer.get("answer_status", "partial")
            missing = answer.get("axes_manquants", [])

            # Filter nitpicky missing axes (references, edge cases, trivia)
            raw_missing = list(missing)
            missing = _filter_trivial_missing(raw_missing)
            answer["axes_manquants"] = missing

            # Upgrade status if all missing axes were trivial
            if not missing and raw_missing:
                answer["answer_status"] = "supported"
                status = "supported"

            if status == "supported" and not missing:
                break

            if iteration >= self.max_iterations:
                break

            # --- Reformulate ---
            t0 = time.time()
            branch_info = _infer_bofip_branch(
                original_question, question, answer,
                [h.boi_reference for h in result.stage1_hits]
            )
            reformulated = self._reformulate(question, answer, branch_info)
            question = reformulated["query"]
            branch_hint = reformulated.get("branch_hint")
            branch_confidence = reformulated.get("branch_confidence", branch_info.get("confidence", 0))
            step_log["reformulated_query"] = question
            step_log["reformulate_s"] = round(time.time() - t0, 2)
            if branch_hint:
                step_log["inferred_branch"] = branch_hint
                step_log["branch_confidence"] = branch_confidence

            # Re-rank all chunks before next answer
            all_chunks = _sort_chunks(all_chunks)

        total_s = round(time.time() - t_start, 2)
        coverage = (
            len(answer.get("axes_couverts", [])) / max(len(answer.get("axes_requis", [])), 1)
        )

        return {
            "question": question,
            "answer_status": answer.get("answer_status", "?"),
            "axes_requis": answer.get("axes_requis", []),
            "axes_couverts": answer.get("axes_couverts", []),
            "axes_manquants": answer.get("axes_manquants", []),
            "conclusion": answer.get("conclusion", ""),
            "justification_bullets": answer.get("justification_bullets", []),
            "limits": answer.get("limits", ""),
            "coverage": round(coverage, 3),
            "iterations": len(trace),
            "total_s": total_s,
            "chunks_used": len(all_chunks),
            "sources": _sort_chunks(all_chunks)[:8],
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _answer(self, question: str, chunks: list[dict]) -> dict:
        """LLM generates answer + self-evaluates coverage (existing build_prompt format)."""
        prompt = build_prompt(question, chunks)
        system = (
            "Tu es un assistant fiscal pragmatique. Reponds UNIQUEMENT a partir des extraits fournis. "
            "Schema JSON strict. Pas de citation inventee.\n\n"
            "CRITERES DE COUVERTURE (sois PRAGMATIQUE, pas perfectionniste):\n"
            "- supported: les axes principaux de la question sont couverts. "
            "Des details mineurs, des cas particuliers non demandes, ou l'absence de reference "
            "BOFIP exacte NE JUSTIFIENT PAS un statut partial.\n"
            "- partial: un axe FISCAL SUBSTANTIF manque et l'utilisateur aurait une reponse incomplate.\n"
            "- axes_manquants: liste UNIQUEMENT les axes substantifs reellement non couverts. "
            "N'inclus JAMAIS: references numeriques BOI/CGI/LPF, cas particuliers non demandes, "
            "details administratifs mineurs, taux ou seuils que l'utilisateur n'a pas demandes. "
            "Si tu peux repondre a la question de l'utilisateur avec les extraits, "
            "alors answer_status DOIT etre 'supported' et axes_manquants DOIT etre vide [].\n\n"
            "Si les extraits proviennent de la mauvaise branche BOFIP (ex: BIC au lieu de TVA-DED), "
            "signale-le dans axes_manquants avec le nom de la branche correcte (ex: 'manque BOI-TVA-DED')."
        )
        return self._call_llm(prompt, system, json_mode=True)

    def _reformulate(self, original_question: str, answer: dict, branch_info: dict) -> dict:
        """Generate a targeted BOFIP search query from missing axes. Returns {query, branch_hint, branch_confidence}."""
        missing = answer.get("axes_manquants", [])
        if not missing:
            return {"query": original_question, "branch_hint": None, "branch_confidence": 0}

        prompt = (
            "Question originale: " + original_question + "\n\n"
            "Les axes suivants ne sont PAS couverts par la recherche initiale. "
            "Genere UNE SEULE requete de recherche technique (20 mots max) "
            "pour trouver les documents BOFIP pertinents.\n\n"
            "Axes manquants:\n" + "\n".join("- " + m for m in missing) + "\n\n"
            "Reponds UNIQUEMENT avec la requete de recherche, sans guillemets ni commentaire."
        )
        system = (
            "Tu es un expert en recherche documentaire BOFIP. "
            "Genere une requete technique courte et precise. "
            "Utilise le vocabulaire exact du BOFIP (ex: 'deduction TVA', 'charges deductibles BIC')."
        )
        resp = self._call_llm(prompt, system, json_mode=False)
        query = resp.get("_raw", "").strip()[:200] or original_question

        # Use inferred branch as soft hint (never prepended into query)
        branch = branch_info.get("branch")
        confidence = branch_info.get("confidence", 0)
        return {"query": query, "branch_hint": branch, "branch_confidence": confidence}

    def _call_llm(self, prompt: str, system: str, json_mode: bool = True) -> dict:
        """Call DeepSeek API, parse JSON response. Returns dict with _raw key for text responses."""
        try:
            from openai import OpenAI
        except ImportError:
            return {"answer_status": "insufficient_evidence", "axes_requis": [], "error": "openai not installed"}

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 2800 if json_mode else 200,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(1, 4):
            try:
                resp = client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                if not json_mode:
                    return {"_raw": content}
                parsed = _parse_json(content)
                if parsed:
                    return parsed
                if attempt >= 3:
                    return {"answer_status": "insufficient_evidence", "axes_requis": [], "raw": content[:200]}
            except Exception:
                if attempt >= 3:
                    return {"answer_status": "insufficient_evidence", "axes_requis": [], "error": "llm call failed"}
                time.sleep(3 * attempt)
        return {"answer_status": "insufficient_evidence", "axes_requis": [], "error": "max attempts"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_bofip_branch(original_question: str, reformulated_query: str,
                         answer: dict, retrieved_refs: list[str]) -> dict:
    """Infer BOFIP branch from multiple signals. Returns {branch, confidence, reason}."""
    import unicodedata

    def _norm(text: str) -> str:
        """Remove accents for accent-insensitive matching."""
        nfkd = unicodedata.normalize("NFKD", text.lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    signals = _norm(" ".join([
        original_question,
        reformulated_query,
        " ".join(answer.get("axes_couverts", [])),
        " ".join(answer.get("axes_manquants", [])),
    ]))

    scores: dict[str, int] = {}
    reasons: dict[str, list[str]] = {}
    for domain, sub in _BRANCH_KEYWORDS.items():
        if _norm(domain) in signals:
            for kw, branch in sub.items():
                if _norm(kw) in signals:
                    scores[branch] = scores.get(branch, 0) + 1
                    reasons.setdefault(branch, []).append(kw)

    if not scores:
        return {"branch": None, "confidence": 0, "reason": ""}

    best = max(scores, key=scores.get)
    confidence = min(scores[best] / 2.0, 1.0)  # lowered from /3.0 to /2.0 for better sensitivity
    reason = " + ".join(reasons[best][:3])
    return {"branch": best, "confidence": round(confidence, 2), "reason": reason}


_TRIVIAL_PATTERN = re.compile(
    r"(boi[\s-]|cgic?\s|article\s+\d|r[eé]f[eé]rence\s+(boi|pr[eé]cise|l[eé]gale|exacte)|"
    r"lpf|sp[eé]cifique.*boi|num[eé]ro.*boi|"
    r"cas particulier|pick[\s-]?up|compensation$|"
    r"radiation.*rcs|modalit[eé]s d.option|cr[eé]dit d.imp[oô]t.*formation)",
    re.IGNORECASE,
)


def _filter_trivial_missing(missing: list[str]) -> list[str]:
    """Remove nitpicky missing axes (BOFIP references, edge cases, trivia)."""
    substantive = [m for m in missing if not _TRIVIAL_PATTERN.search(m)]
    return substantive


def _parse_json(raw: str) -> dict | None:
    import re
    candidates = [raw]
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL):
        c = m.group(1).strip()
        if c:
            candidates.append(c)
    for c in candidates:
        try:
            p = json.loads(c)
            if isinstance(p, dict):
                return p
        except json.JSONDecodeError:
            continue
    return None


def _chunks_from_result(result) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "boi_reference": c.boi_reference,
            "title": c.title,
            "publication_date": c.publication_date,
            "section_path": c.section_path,
            "text": c.text,
            "chunk_id": c.chunk_id,
        }
        for i, c in enumerate(result.stage2_chunks)
    ]


def _sort_chunks(chunks: list[dict]) -> list[dict]:
    """Sort by rank (preserves original retrieval order)."""
    return sorted(chunks, key=lambda c: c.get("rank", 999))
