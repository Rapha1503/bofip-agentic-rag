"""Agentic RAG — self-evaluating retrieval loop for BOFIP fiscal questions.

Architecture (inspired by Azure Agentic Retrieval + Corrective RAG):
    1. Retrieve — first pass via RagRuntime
    2. Answer + Self-Evaluate — LLM returns {answer_status, axes_requis, axes_couverts, axes_manquants, ...}
    3. IF partial → Reformulate missing axes → Retrieve 2nd pass → Merge chunks → Final answer
    4. Trace — full audit log

LLM calls per query: 2 (first pass successful) or 3 (with reformulation).
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field

from .prompt_utils import build_prompt, build_system_prompt
from .rag_runtime import RagRuntime


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgenticRAG:
    def __init__(
        self,
        runtime: RagRuntime,
        *,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_iterations: int = 2,
        client=None,
        use_reranker: bool = True,
        progress_callback=None,
    ):
        self.rt = runtime
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_iterations = max_iterations
        self._client = client
        self.use_reranker = use_reranker
        self.progress_callback = progress_callback

    def _progress(self, label: str, **payload) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(label, payload)
        except Exception:
            pass
    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict:
        """Main entry point. Returns structured result with trace."""
        original_question = question
        trace: list[dict] = []
        all_chunks: list[dict] = []
        seen_ids: set[str] = set()

        t_start = time.time()

        # Before first retrieval: ask the LLM to classify the BOFIP domain.
        # One cheap call (max_tokens=150) extracts the right family + sub-family prefix.
        self._progress("Classification BOFiP", detail="Détection de la famille documentaire.")
        domain = _classify_domain(question, self._call_llm)
        boost_prefix = domain
        if domain:
            question = f"{domain} {question}"
        self._progress("Famille BOFiP", detail=boost_prefix or "Non déterminée")

        for iteration in range(1, self.max_iterations + 1):
            step_log = {"iteration": iteration, "domain_prefix": boost_prefix, "query_used": question}

            # --- Retrieve ---
            self._progress("Recherche documentaire", detail=f"Itération {iteration}: corpus complet, BM25 et embeddings si disponibles.")
            t0 = time.time()
            result = self.rt.retrieve(question, top_docs=8, max_chunks=8, use_reranker=self.use_reranker, boost_prefix=boost_prefix)

            # Mismatch detection: compare LLM-predicted domain vs actual retrieved families.
            # If retrieval pulled mostly wrong documents, retry with the predicted prefix.
            if iteration == 1 and domain and len(result.stage1_hits) >= 4:
                expected_ref = domain.upper().removeprefix("BOI-")
                expected = expected_ref.split("-")[0] if "-" in expected_ref else expected_ref
                fams = []
                for h in result.stage1_hits[:8]:
                    ref = h.boi_reference.upper().removeprefix("BOI-")
                    fams.append(ref.split("-")[0] if "-" in ref else "")
                from collections import Counter
                fam_dist = Counter(fams)
                total = sum(fam_dist.values())
                expected_count = fam_dist.get(expected, 0)
                if expected_count / total < 0.5 and total > 0:
                    question = f"{domain} {question}"
                    step_log["mismatch_fix"] = f"retry with {domain} ({expected_count}/{total} {expected})"
                    result = self.rt.retrieve(question, top_docs=8, max_chunks=8, use_reranker=self.use_reranker, boost_prefix=boost_prefix)

            chunks = _chunks_from_result(result)
            step_log["retrieve_s"] = round(time.time() - t0, 2)
            step_log["docs_found"] = len(result.stage1_hits)
            step_log["chunks_found"] = len(chunks)

            # Merge & deduplicate
            new_chunks = [c for c in chunks if c["chunk_id"] not in seen_ids]
            for c in new_chunks:
                seen_ids.add(c["chunk_id"])
            all_chunks.extend(new_chunks)
            step_log["chunks_new"] = len(new_chunks)
            step_log["chunks_total"] = len(all_chunks)
            self._progress("Passages retenus", detail=f"{len(result.stage1_hits)} documents candidats, {len(chunks)} passages analysés.")

            # --- Answer + Self-Evaluate ---
            self._progress("Auto-évaluation", detail=f"Itération {iteration}: réponse structurée et contrôle des axes.")
            t0 = time.time()
            answer = self._answer(question, all_chunks)
            step_log["answer_s"] = round(time.time() - t0, 2)

            # Check if sufficient
            status = answer.get("answer_status", "partial")
            missing = answer.get("axes_manquants", [])

            # Filter nitpicky missing axes (references, edge cases, trivia)
            _trivial = re.compile(
                r"(boi[\s-]|cgic?\s|article\s+\d|r[eé]f[eé]rence\s+(boi|pr[eé]cise|l[eé]gale|exacte)|"
                r"lpf|sp[eé]cifique.*boi|num[eé]ro.*boi|"
                r"cas particulier|pick[\s-]?up|compensation$|"
                r"radiation.*rcs|modalit[eé]s d.option|cr[eé]dit d.imp[oô]t.*formation)",
                re.IGNORECASE,
            )
            substantive_missing = [m for m in missing if not _trivial.search(m)]
            if not substantive_missing and missing:
                answer["answer_status"] = "supported"
                answer["axes_manquants"] = []
                status = "supported"
                missing = []
            elif substantive_missing:
                missing = substantive_missing
                answer["axes_manquants"] = substantive_missing
            if status == "supported" and not missing:
                step_log["answer_status"] = answer.get("answer_status", "?")
                step_log["axes_requis"] = answer.get("axes_requis", [])
                step_log["axes_couverts"] = answer.get("axes_couverts", [])
                step_log["axes_manquants"] = answer.get("axes_manquants", [])
                trace.append(step_log)
                self._progress("Couverture suffisante", detail="Les axes essentiels sont couverts par les passages retenus.")
                break

            step_log["answer_status"] = answer.get("answer_status", "?")
            step_log["axes_requis"] = answer.get("axes_requis", [])
            step_log["axes_couverts"] = answer.get("axes_couverts", [])
            step_log["axes_manquants"] = answer.get("axes_manquants", [])
            trace.append(step_log)

            if iteration >= self.max_iterations:
                break

            # --- Reformulate ---
            self._progress("Relance ciblée", detail="Axes manquants détectés, génération de requête BOFiP plus précise.")
            t0 = time.time()
            question = self._reformulate(question, answer)
            step_log["reformulated_query"] = question
            step_log["reformulate_s"] = round(time.time() - t0, 2)

            # Re-rank all chunks before next answer
            all_chunks = _sort_chunks(all_chunks)

        total_s = round(time.time() - t_start, 2)
        coverage = (
            min(1.0, len(answer.get("axes_couverts", [])) / len(answer.get("axes_requis", [])))
            if answer.get("axes_requis")
            else 1.0
        )

        return {
            "question": original_question,
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
        system = build_system_prompt()
        return self._call_llm(prompt, system, json_mode=True)

    def _reformulate(self, original_question: str, answer: dict) -> str:
        """Generate a targeted BOFIP search query from missing axes.

        Uses structured JSON output to force the LLM to produce a clean search query
        with proper BOFIP vocabulary (RPPM for particuliers, BIC for entreprises, etc.).
        """
        missing = answer.get("axes_manquants", [])
        if not missing:
            return original_question

        prompt = (
            "Question originale: " + original_question + "\n\n"
            "Axes fiscaux NON couverts par la recherche precedente:\n"
            + "\n".join("- " + m for m in missing) + "\n\n"
            "Genere une requete de recherche BOFIP optimisee. Retourne UN JSON:\n"
            '{"bofip_family": "RPPM|BIC|IS|TVA|IR|CF|ENR|IF|PAT",'
            '"search_query": "requete de 8-15 mots en vocabulaire technique BOFIP"}\n\n'
            "REGLES:\n"
            "- bofip_family: RPPM pour particuliers/revenus mobiliers, BIC pour benefices"
            " industriels, IS pour societes, TVA pour taxe valeur ajoutee, IR pour impot revenu\n"
            "- search_query: UNIQUEMENT des termes techniques BOFIP, pas de phrases.\n"
            "  Exemple bon: 'particuliers plus-values mobilieres imputation moins-values RPPM PVBMI'\n"
            "  Exemple mauvais: 'je voudrais savoir comment les particuliers sont imposes sur...'\n"
            "- MAX 15 mots. Pas de guillemets, pas de phrases, pas de politesse."
        )
        system = (
            "Tu es un expert en recherche documentaire BOFIP. "
            "Tu connais les familles: RPPM (particuliers/revenus), BIC (entreprises/benefices), "
            "IS (societes), TVA, IR, CF (controle fiscal), ENR (enregistrement), IF (impots fonciers), PAT (patrimoine). "
            "Tu generes des requetes de recherche purement techniques."
        )
        resp = self._call_llm(prompt, system, json_mode=True)

        query = resp.get("search_query", "").strip()
        family = resp.get("bofip_family", "").strip().upper()

        if query:
            if family:
                query = f"{family} {query}"
            return query[:200]

        return original_question

    def _call_llm(self, prompt: str, system: str, json_mode: bool = True) -> dict:
        """Call LLM API, parse JSON response. Uses shared client if provided."""
        if self._client is not None:
            client = self._client
        else:
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


def _ascii_lower(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _fallback_domain_from_question(question: str) -> str:
    q = _ascii_lower(question)
    rules = [
        ("TVA", ("tva", "taxe sur la valeur ajoutee")),
        ("IR", ("impot sur le revenu", "prelevement a la source", "foyer fiscal", "quotient familial")),
        ("BIC", ("bic", "micro-bic", "benefices industriels", "chiffre d'affaires", "auto-entrepreneur", "entrepreneur")),
        ("IS", ("impot sur les societes", "societe soumise a l'is", " is ")),
        ("CF", ("controle fiscal", "redressement", "interet de retard", "majoration", "penalite")),
        ("ENR", ("succession", "donation", "droits d'enregistrement")),
        ("IF", ("taxe fonciere", "cfe", "cvae", "cotisation fonciere")),
        ("RPPM", ("plus-value", "dividende", "valeurs mobilieres", "per ", "plan epargne retraite")),
    ]
    for family, markers in rules:
        if any(marker in q for marker in markers):
            return family
    return ""

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


def _classify_domain(question: str, call_llm) -> str:
    """Ask the LLM to classify the question into a BOFiP document prefix."""
    prompt = (
        "Identifie le prefixe documentaire BOFIP le PLUS PRECIS pour cette question. "
        "Donne le prefixe complet jusqu'au niveau section si identifiable "
        "(ex: RPPM-PVBMI-20-10-40 pas juste RPPM-PVBMI). "
        "Si tu n'es pas sur du sous-chapitre, donne juste la famille (ex: TVA). "
        "Retourne UNIQUEMENT le prefixe, rien d'autre.\n\n"
        "Question: " + question
    )
    resp = call_llm(prompt, "Tu es un classifieur de taxonomie BOFIP.", json_mode=False)
    raw = resp.get("_raw", "").strip()
    first_line = raw.split("\n")[0].strip().strip('"').strip("'").strip("`")
    match = re.search(
        r"\b(?:BOI-)?(RPPM|BIC|IS|TVA|IR|CF|ENR|IF|PAT)(?:-[A-Z0-9]+){0,8}\b",
        first_line.upper(),
    )
    if match:
        return match.group(0).removeprefix("BOI-")[:80]
    fallback = _fallback_domain_from_question(question)
    if fallback:
        return fallback
    if 2 <= len(first_line) <= 80:
        return first_line.upper().removeprefix("BOI-")
    return ""


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
