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
import time
from dataclasses import dataclass, field

from .prompt_utils import build_prompt
from .rag_runtime import RagRuntime


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

        t_start = time.time()

        for iteration in range(1, self.max_iterations + 1):
            step_log = {"iteration": iteration}

            # --- Retrieve ---
            t0 = time.time()
            result = self.rt.retrieve(question, top_docs=8, max_chunks=8)
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
            if status == "supported" and not missing:
                break

            if iteration >= self.max_iterations:
                break

            # --- Reformulate ---
            t0 = time.time()
            question = self._reformulate(question, answer)
            step_log["reformulated_query"] = question
            step_log["reformulate_s"] = round(time.time() - t0, 2)

            # Re-rank all chunks before next answer
            all_chunks = _sort_chunks(all_chunks)

        total_s = round(time.time() - t_start, 2)
        coverage = (
            len(answer.get("axes_couverts", [])) / len(answer.get("axes_requis", []))
            if answer.get("axes_requis")
            else 1.0
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
            "trace": trace,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _answer(self, question: str, chunks: list[dict]) -> dict:
        """LLM generates answer + self-evaluates coverage (existing build_prompt format)."""
        prompt = build_prompt(question, chunks)
        system = (
            "Tu es un assistant fiscal. Reponds UNIQUEMENT a partir des extraits fournis. "
            "Schema JSON strict. Pas de citation inventee. "
            "Utilise axes_manquants pour indiquer quels aspects fiscaux ne sont pas couverts "
            "par les extraits — cela declenchera une recherche supplementaire."
        )
        return self._call_llm(prompt, system)

    def _reformulate(self, original_question: str, answer: dict) -> str:
        """Generate a targeted BOFIP search query from missing axes."""
        missing = answer.get("axes_manquants", [])
        if not missing:
            return original_question

        prompt = (
            "Question originale: " + original_question + "\n\n"
            "Les axes suivants ne sont PAS couverts par la recherche initiale. "
            "Genere UNE SEULE requete de recherche (20 mots max) en vocabulaire technique "
            "BOFIP/fiscal pour trouver les documents pertinents.\n\n"
            "Axes manquants:\n" + "\n".join("- " + m for m in missing) + "\n\n"
            "Reponds UNIQUEMENT avec la requete de recherche, sans guillemets ni commentaire."
        )
        system = "Tu es un expert en recherche documentaire BOFIP. Genere une requete de recherche technique."
        resp = self._call_llm(prompt, system, response_format=None)
        return resp.get("_raw", "").strip()[:200] or original_question

    def _call_llm(self, prompt: str, system: str, response_format: dict | None = None) -> dict:
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
            "max_tokens": 2800,
        }
        if response_format is None:
            # Text response (reformulation)
            kwargs["max_tokens"] = 200
        else:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(1, 4):
            try:
                resp = client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                if response_format is None:
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
