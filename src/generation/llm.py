"""
BOFIP LLM Client

Handles LLM interaction with Groq API (and Ollama fallback).
"""

import logging
from typing import Optional, List, Dict, Any
import hashlib
import json
import re
import time
from pathlib import Path

from groq import Groq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MODELS,
    LLM_MAX_CONTEXT_CHUNKS,
    LLM_MAX_CONTEXT_TOKENS,
    CACHE_DIR,
    CACHE_TTL_LLM,
    FAITHFULNESS_GUARDRAIL_ENABLED,
    FAITHFULNESS_VERIFIER_MODEL,
    FAITHFULNESS_MIN_CONFIDENCE,
    FAITHFULNESS_MAX_CONTEXT_CHUNKS,
)
from src.generation.prompts import (
    SYSTEM_PROMPT,
    FAITHFULNESS_VERIFIER_SYSTEM_PROMPT,
    create_user_prompt,
    create_faithfulness_prompt,
    DISCLAIMER,
)

logger = logging.getLogger(__name__)

# Response cache
RESPONSE_CACHE_DIR = CACHE_DIR / 'llm_responses'
RESPONSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Track which model is currently being used (for display in UI)
_current_model_info = {"id": GROQ_MODEL, "name": GROQ_MODELS[0]["name"] if GROQ_MODELS else "Unknown"}


def get_current_model_name() -> str:
    """Get the display name of the currently active model."""
    return _current_model_info.get("name", "Unknown")


class BOFIPLLMClient:
    """
    LLM client for BOFIP question answering.
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize LLM client.

        Args:
            api_key: Groq API key (uses env if not provided)
            model: Model name (default from config)
        """
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self.cache_ttl = CACHE_TTL_LLM

        if not self.api_key:
            logger.warning("No Groq API key provided. Set GROQ_API_KEY in .env")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.model}")

    def _is_cache_expired(self, cache_path: Path) -> bool:
        """Return True if cache entry is older than configured TTL."""
        if self.cache_ttl is None:
            return False

        age_seconds = time.time() - cache_path.stat().st_mtime
        return age_seconds > self.cache_ttl

    def _get_cache_key(self, question: str, context_hash: str) -> str:
        """Generate cache key for a query"""
        combined = f"{question}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        cache_path = RESPONSE_CACHE_DIR / f"{cache_key}.json"
        if cache_path.exists():
            if self._is_cache_expired(cache_path):
                logger.info(f"Cache expired, removing entry: {cache_path.name}")
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(f"Could not delete expired cache file {cache_path.name}: {e}")
                return None

            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('response')
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Invalid cache file {cache_path.name}, ignoring it: {e}")
        return None

    def _cache_response(self, cache_key: str, response: str, question: str):
        """Cache a response"""
        cache_path = RESPONSE_CACHE_DIR / f"{cache_key}.json"
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'question': question,
                'response': response,
                'model': self.model
            }, f, ensure_ascii=False, indent=2)

    def _extract_sources_from_chunks(self, chunks: List[Dict[str, Any]], max_sources: int = 4) -> List[Dict[str, str]]:
        """
        Extract and deduplicate source references from retrieved chunks.

        Dedup key is BOI base reference (without trailing date segment) to avoid
        repeating near-identical dated entries.
        """
        seen_refs = set()
        sources = []

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            boi_ref = metadata.get('boi_reference', 'N/A')

            # Remove trailing date suffix in BOI refs when present.
            base_ref = '-'.join(boi_ref.split('-')[:-1]) if boi_ref.count('-') > 4 else boi_ref
            if base_ref in seen_refs:
                continue

            seen_refs.add(base_ref)
            sources.append({
                'boi_reference': boi_ref,
                'section_title': metadata.get('section_title', ''),
                'source_url': metadata.get('source_url', ''),
                'series': metadata.get('series', '')
            })

        return sources[:max_sources]

    @staticmethod
    def _estimate_chunk_tokens(chunk: Dict[str, Any]) -> int:
        """
        Estimate chunk token size from metadata with text fallback.
        """
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        token_count = metadata.get("token_count")
        try:
            if token_count is not None:
                return max(1, int(token_count))
        except (TypeError, ValueError):
            pass

        text = chunk.get("text", "") if isinstance(chunk, dict) else ""
        # Rough fallback for missing metadata.
        return max(1, int(len((text or "").split()) * 1.1))

    def _select_context_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: int = LLM_MAX_CONTEXT_CHUNKS,
        max_tokens: int = LLM_MAX_CONTEXT_TOKENS,
    ) -> List[Dict[str, Any]]:
        """
        Keep highest-ranked chunks while respecting LLM context budget.
        """
        if not chunks:
            return []

        max_chunks = max(1, int(max_chunks))
        max_tokens = max(500, int(max_tokens))

        selected = []
        total_tokens = 0

        for chunk in chunks:
            if len(selected) >= max_chunks:
                break

            chunk_tokens = self._estimate_chunk_tokens(chunk)
            # Always keep first chunk, then respect budget.
            if selected and (total_tokens + chunk_tokens > max_tokens):
                continue

            selected.append(chunk)
            total_tokens += chunk_tokens

        if not selected:
            selected = [chunks[0]]
            total_tokens = self._estimate_chunk_tokens(chunks[0])

        logger.info(
            f"Context selection: kept {len(selected)}/{len(chunks)} chunks "
            f"(~{total_tokens} tokens, limits chunks={max_chunks}, tokens={max_tokens})"
        )
        return selected

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON object from raw model text with light cleanup."""
        if not text:
            return None
        text = text.strip()

        # Fast path: content is already a JSON object.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first {...} block.
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = text[start:end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
        return None

    @staticmethod
    def _normalize_number(num_text: str) -> str:
        """Normalize number text for robust containment checks."""
        cleaned = (num_text or "").replace("\xa0", " ").replace(" ", "")
        cleaned = cleaned.replace(",", ".")
        return cleaned

    def _verify_faithfulness_heuristic(self, question: str, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deterministic fallback when verifier call fails.

        Conservative logic: block only clearly unsupported outputs.
        """
        if not chunks:
            return {
                "pass": False,
                "mode": "heuristic",
                "reason": "Aucun extrait recuperé.",
                "unsupported_claims": ["Aucun contexte disponible pour justifier la reponse."],
                "confidence": 0.0,
            }

        question_text = question or ""
        context_text = " ".join(chunk.get("text", "") for chunk in chunks)
        context_refs = {
            (chunk.get("metadata", {}) or {}).get("boi_reference", "")
            for chunk in chunks
        }
        context_refs_text = " ".join(r for r in context_refs if r)

        # Check legal/BOI references mentioned by the answer.
        ref_pattern = re.compile(
            r"(BOI-[A-Z0-9-]+|(?:CGI|LPF)\s+Art\.\s*[A-Z]?\*?\.?\s*[\d][\dA-Z\-\s\.]*)",
            re.IGNORECASE,
        )
        answer_refs = {m.group(1).strip() for m in ref_pattern.finditer(answer or "")}
        unsupported_refs = []
        for ref in answer_refs:
            ref_upper = ref.upper()
            if ref_upper not in context_text.upper() and ref_upper not in context_refs_text.upper():
                unsupported_refs.append(ref)

        # Check numeric claims that do not appear in question/context.
        number_pattern = re.compile(r"\d[\d\s,.]*")
        corpus_numbers = {
            self._normalize_number(m.group(0))
            for m in number_pattern.finditer((question_text + " " + context_text))
            if self._normalize_number(m.group(0))
        }
        answer_numbers = {
            self._normalize_number(m.group(0))
            for m in number_pattern.finditer(answer or "")
            if self._normalize_number(m.group(0))
        }
        unsupported_numbers = sorted(n for n in answer_numbers if n not in corpus_numbers)

        # Fail only on clear evidence of unsupported claims.
        too_many_unsupported_numbers = len(answer_numbers) >= 3 and len(unsupported_numbers) > (len(answer_numbers) * 0.6)
        if unsupported_refs or too_many_unsupported_numbers:
            reason = "Affirmations potentiellement non supportees detectees."
            unsupported = []
            if unsupported_refs:
                unsupported.extend([f"Reference non supportee: {r}" for r in unsupported_refs])
            if too_many_unsupported_numbers:
                unsupported.append(
                    f"Valeurs numeriques possiblement non supportees: {', '.join(unsupported_numbers[:5])}"
                )
            return {
                "pass": False,
                "mode": "heuristic",
                "reason": reason,
                "unsupported_claims": unsupported,
                "confidence": 0.2,
            }

        return {
            "pass": True,
            "mode": "heuristic",
            "reason": "Aucune non-conformite evidente detectee (fallback heuristique).",
            "unsupported_claims": [],
            "confidence": 0.6,
        }

    def _verify_faithfulness_with_llm(self, question: str, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        LLM-based faithfulness verification.

        Returns pass/fail + reason. If verifier cannot run, returns checked=False.
        """
        if not self.client:
            return {
                "checked": False,
                "error": "no_client",
            }

        verifier_prompt = create_faithfulness_prompt(
            question=question,
            answer=answer,
            chunks=chunks,
            max_chunks=FAITHFULNESS_MAX_CONTEXT_CHUNKS,
        )

        try:
            response = self.client.chat.completions.create(
                model=FAITHFULNESS_VERIFIER_MODEL,
                messages=[
                    {"role": "system", "content": FAITHFULNESS_VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": verifier_prompt},
                ],
                temperature=0.0,
                max_tokens=350,
            )
            raw = response.choices[0].message.content or ""
            verdict = self._parse_json_object(raw)
            if not verdict:
                return {
                    "checked": False,
                    "error": "invalid_verifier_json",
                    "raw": raw[:300],
                }

            grounded = bool(verdict.get("grounded", False))
            confidence = self._safe_float(verdict.get("confidence"), default=0.0)
            unsupported = verdict.get("unsupported_claims", [])
            if not isinstance(unsupported, list):
                unsupported = [str(unsupported)]
            reason = str(verdict.get("reason", "")).strip() or str(verdict.get("verdict", "")).strip()

            passed = grounded and confidence >= FAITHFULNESS_MIN_CONFIDENCE and len(unsupported) == 0

            return {
                "checked": True,
                "pass": passed,
                "mode": "llm_verifier",
                "reason": reason or ("Verifier a confirme la reponse." if passed else "Verifier a detecte un manque de support."),
                "unsupported_claims": unsupported,
                "confidence": confidence,
                "verdict": str(verdict.get("verdict", "")),
            }
        except Exception as e:
            logger.warning(f"Faithfulness verifier failed, fallback to heuristic: {e}")
            return {
                "checked": False,
                "error": f"verifier_exception:{e}",
            }

    def _run_faithfulness_guardrail(self, question: str, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run verifier with deterministic heuristic fallback."""
        llm_check = self._verify_faithfulness_with_llm(question, answer, chunks)
        if llm_check.get("checked"):
            return llm_check

        heuristic_check = self._verify_faithfulness_heuristic(question, answer, chunks)
        heuristic_check["verifier_fallback_reason"] = llm_check.get("error", "unknown")
        return heuristic_check

    @staticmethod
    def _build_abstention_answer(reason: str, unsupported_claims: Optional[List[str]] = None) -> str:
        """
        Standard abstention answer when faithfulness is insufficient.
        """
        details = ""
        if unsupported_claims:
            details = "\n".join(f"- {c}" for c in unsupported_claims[:3] if c)
            if details:
                details = f"\nPoints non confirmes:\n{details}"

        return (
            "Je ne peux pas confirmer une reponse suffisamment fiable a partir des extraits recuperes.\n\n"
            f"Motif: {reason or 'Evidence insuffisante.'}"
            f"{details}\n\n"
            "Merci de reformuler la question ou d'ajouter des precisions factuelles pour obtenir une reponse "
            "strictement justifiee par les sources."
        )

    def generate(self, question: str,
                 chunks: List[Dict[str, Any]],
                 use_cache: bool = True,
                 temperature: float = 0.1,
                 max_tokens: int = 1500) -> str:
        """
        Generate an answer using the LLM.

        Args:
            question: User's question
            chunks: Retrieved context chunks
            use_cache: Whether to use response cache
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum response tokens

        Returns:
            Generated answer
        """
        if not self.client:
            return "Erreur: Client LLM non configure. Verifiez votre cle API Groq."

        # Create context hash for caching
        context_hash = hashlib.md5(
            json.dumps([c.get('chunk_id', '') for c in chunks]).encode()
        ).hexdigest()[:16]

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(question, context_hash)
            cached = self._get_cached_response(cache_key)
            if cached:
                logger.info("Using cached response")
                return cached

        # Create prompt
        user_prompt = create_user_prompt(question, chunks)

        # Try models in order until one works
        global _current_model_info
        models_to_try = [m for m in GROQ_MODELS if m["id"] != self.model]
        models_to_try.insert(0, {"id": self.model, "name": _current_model_info.get("name", "Unknown")})

        last_error = None
        for model_info in models_to_try:
            model_id = model_info["id"]
            model_name = model_info["name"]

            try:
                logger.info(f"Calling Groq API ({model_name})...")
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                answer = response.choices[0].message.content

                # Update current model tracking
                _current_model_info["id"] = model_id
                _current_model_info["name"] = model_name
                self.model = model_id

                # Cache response
                if use_cache:
                    self._cache_response(cache_key, answer, question)

                return answer

            except Exception as e:
                error_str = str(e)
                last_error = e

                # Check if it's a rate limit error
                if "429" in error_str or "rate_limit" in error_str.lower():
                    logger.warning(f"Rate limit hit for {model_name}, trying next model...")
                    continue
                else:
                    # Non-rate-limit error, don't try other models
                    logger.error(f"LLM generation failed: {e}")
                    return f"Erreur lors de la generation de la reponse: {error_str}"

        # All models failed (rate limited)
        logger.error(f"All models rate limited. Last error: {last_error}")
        return f"Erreur: Tous les modeles sont temporairement indisponibles (limite de taux atteinte). Reessayez dans quelques minutes."

    def generate_with_sources(self, question: str,
                              chunks: List[Dict[str, Any]],
                              **kwargs) -> Dict[str, Any]:
        """
        Generate answer with source information.

        Args:
            question: User's question
            chunks: Retrieved context chunks
            **kwargs: Additional arguments for generate()

        Returns:
            Dict with 'answer', 'sources', and 'chunks_used'
        """
        context_chunks = self._select_context_chunks(chunks)
        answer = self.generate(question, context_chunks, **kwargs)
        sources = self._extract_sources_from_chunks(context_chunks, max_sources=4)

        faithfulness = {
            "enabled": FAITHFULNESS_GUARDRAIL_ENABLED,
            "pass": True,
            "mode": "disabled",
            "reason": "Guardrail desactive.",
            "unsupported_claims": [],
            "confidence": 1.0,
        }

        if FAITHFULNESS_GUARDRAIL_ENABLED:
            if answer.startswith("Erreur"):
                faithfulness = {
                    "enabled": True,
                    "pass": False,
                    "mode": "skipped_error",
                    "reason": "Generation en erreur, verification sautee.",
                    "unsupported_claims": [],
                    "confidence": 0.0,
                }
            else:
                faithfulness = self._run_faithfulness_guardrail(question, answer, context_chunks)
                if not faithfulness.get("pass", False):
                    logger.warning(
                        "Faithfulness guardrail triggered abstention "
                        f"(mode={faithfulness.get('mode')}, reason={faithfulness.get('reason')})"
                    )
                    answer = self._build_abstention_answer(
                        reason=faithfulness.get("reason", "Evidence insuffisante."),
                        unsupported_claims=faithfulness.get("unsupported_claims", []),
                    )

        return {
            'answer': answer,
            'sources': sources,
            'chunks_used': len(context_chunks),
            'disclaimer': DISCLAIMER,
            'faithfulness': faithfulness,
        }


# Singleton instance
_llm_client: Optional[BOFIPLLMClient] = None


def get_llm_client() -> BOFIPLLMClient:
    """Get or create singleton LLM client"""
    global _llm_client
    if _llm_client is None:
        _llm_client = BOFIPLLMClient()
    return _llm_client


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Test with mock chunks
    mock_chunks = [
        {
            'chunk_id': 'test_1',
            'text': 'Le taux de TVA applicable a la restauration sur place est de 10%.',
            'metadata': {
                'boi_reference': 'BOI-TVA-LIQ-30-20-10',
                'section_title': 'Taux reduit de 10%',
                'source_url': 'https://bofip.impots.gouv.fr/...'
            }
        }
    ]

    client = get_llm_client()

    if client.client:
        print("Testing LLM generation...")
        result = client.generate_with_sources(
            "Quel est le taux de TVA pour la restauration?",
            mock_chunks,
            use_cache=False
        )
        print("\nAnswer:")
        print(result['answer'])
        print("\nSources:")
        for s in result['sources']:
            print(f"  - {s['boi_reference']}")
    else:
        print("No API key configured. Set GROQ_API_KEY in .env")
