"""Agentic RAG - plan-and-route retrieval loop for BOFiP fiscal questions.

The agent separates fiscal understanding from retrieval:
1. a planner turns the user question into generic fiscal axes;
2. retrieval runs independently for each axis;
3. a source critic rejects off-topic passages and asks for targeted relaunches;
4. the final answer is written only from the retained BOFiP passages.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .prompt_utils import build_prompt, build_system_prompt

if TYPE_CHECKING:
    from .rag_runtime import RagRuntime


BOFIP_PREFIX_FAMILIES = "RPPM|RFPI|BIC|BNC|BA|IS|TVA|IR|RSA|CF|ENR|IF|PAT"
BOI_REFERENCE_RE = re.compile(r"\bBOI-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", re.IGNORECASE)
RETRIEVAL_RESCUE_STAGES = {"evidence_rescue", "intra_document", "global_relaunch"}


@dataclass(frozen=True)
class SearchFacet:
    name: str
    goal: str
    query: str
    prefix: str = ""
    priority: int = 1
    expected_evidence: list[str] = field(default_factory=list)
    role: str = "core"
    blocking: bool = True


@dataclass(frozen=True)
class SearchPlan:
    reformulated_question: str
    facts: list[str]
    ambiguities: list[str]
    facets: list[SearchFacet]
    excluded_axes: list[dict[str, str]] = field(default_factory=list)


class AgenticRAG:
    def __init__(
        self,
        runtime: "RagRuntime",
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
        self._run_started_at: float | None = None
        self._last_progress_at: float | None = None
        self._step_timings: list[dict] = []

    def _progress(self, label: str, **payload) -> None:
        payload = dict(payload)
        now = time.time()
        if self._run_started_at is not None:
            previous = self._last_progress_at or self._run_started_at
            step_s = round(now - previous, 2)
            elapsed_s = round(now - self._run_started_at, 2)
            self._last_progress_at = now
            payload.setdefault("step_s", step_s)
            payload.setdefault("elapsed_s", elapsed_s)
            self._step_timings.append(
                {
                    "label": label,
                    "detail": payload.get("detail", ""),
                    "step_s": step_s,
                    "elapsed_s": elapsed_s,
                    "fields": payload.get("fields", []) or [],
                }
            )
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(label, payload)
        except Exception:
            pass

    def run(self, question: str) -> dict:
        original_question = question.strip()
        t_start = time.time()
        self._run_started_at = t_start
        self._last_progress_at = t_start
        self._step_timings = []
        seen_ids: set[str] = set()
        all_chunks: list[dict] = []

        self._progress(
            "Question posée au planneur fiscal",
            detail="Extraction des faits, ambiguïtés et axes fiscaux avant toute recherche.",
            fields=[{"label": "Question", "value": original_question}],
        )
        plan = self._plan_question(original_question)
        self._progress(
            "Plan fiscal produit",
            detail="La recherche est découpée en axes indépendants pour limiter les mots-pièges.",
            fields=[
                {"label": "Reformulation fiscale", "value": plan.reformulated_question},
                {"label": "Faits compris", "value": "\n".join(plan.facts) or "Non renseigné"},
                {"label": "Axes retenus", "value": _format_facets(plan.facets)},
                {"label": "Axes écartés", "value": _format_excluded_axes(plan.excluded_axes) or "Aucun"},
                {"label": "Ambiguïtés à signaler", "value": "\n".join(plan.ambiguities) or "Aucune"},
            ],
        )

        route_log: list[dict] = []
        for facet in plan.facets:
            result, new_chunks = self._retrieve_for_facet(facet, original_question, seen_ids)
            pipeline_log = getattr(result, "pipeline_log", {}) or {}
            all_chunks.extend(new_chunks)
            route_log.append(
                {
                    "facet": _facet_to_dict(facet),
                    "query_used": result.query,
                    "docs_found": len(result.stage1_hits),
                    "chunks_new": len(new_chunks),
                    "stage1_refs": [h.boi_reference for h in result.stage1_hits[:8]],
                    "stage2_refs": _unique_strings(pipeline_log.get("stage2_candidate_doc_refs", []))[:12],
                    "selected_refs": _unique_strings(pipeline_log.get("final_doc_refs", []))[:8],
                    "stage2_chunk_ids": _unique_strings(pipeline_log.get("stage2_candidate_chunk_ids", []))[:12],
                    "selected_chunk_ids": _unique_strings(pipeline_log.get("final_chunk_ids", []))[:8],
                    "evidence_rescue_queries": _unique_strings(pipeline_log.get("evidence_rescue_queries", []))[:6],
                    "evidence_rescue_chunk_ids": _unique_strings(pipeline_log.get("evidence_rescue_chunk_ids", []))[:8],
                }
            )
            self._progress(
                "Recherche par axe",
                detail=f"Axe: {facet.name}",
                fields=[
                    {"label": "Objectif", "value": facet.goal},
                    {"label": "Préfixe BOFiP", "value": facet.prefix or "Aucun"},
                    {"label": "Requête", "value": result.query},
                    {
                        "label": "Sources candidates",
                        "value": "\n".join(
                            f"{h.boi_reference} - {_short(h.title, 120)}" for h in result.stage1_hits[:5]
                        )
                        or "Aucune",
                    },
                ],
            )

        self._progress(
            "Critique des sources demandée",
            detail="Contrôle documentaire avant rédaction: utile, hors sujet, ou axe à relancer.",
            fields=[{"label": "Passages à juger", "value": str(len(all_chunks))}],
        )
        source_review = self._review_sources(original_question, plan, all_chunks)
        source_review = _complete_source_review_with_plan_gaps(
            _ensure_nonempty_source_review(source_review, plan),
            plan,
            all_chunks,
        )
        selected_chunks = _select_reviewed_chunks(all_chunks, source_review)
        self._progress(
            "Critique des sources",
            detail="Les passages hors sujet sont séparés des sources utiles avant la réponse.",
            fields=[
                {"label": "Axes couverts par les sources", "value": "\n".join(source_review.get("covered_axes", [])) or "Non renseigné"},
                {"label": "Axes à relancer", "value": _format_missing_axes(source_review.get("missing_axes", [])) or "Aucun"},
                {"label": "Sources utiles", "value": "\n".join(_source_labels(selected_chunks[:8])) or "Aucune"},
            ],
        )

        relaunch_log: list[dict] = []
        missing_axes = source_review.get("missing_axes", []) or []
        if self.max_iterations > 1 and missing_axes:
            for missing in missing_axes[:3]:
                facet = _facet_from_missing_axis(missing)
                if not facet.query:
                    continue
                candidate_refs = _candidate_refs_for_missing_axis(missing, all_chunks, route_log)
                rescue_result, rescue_chunks = self._retrieve_within_candidate_docs(
                    facet, original_question, candidate_refs, seen_ids
                )
                if rescue_chunks:
                    rescue_log = getattr(rescue_result, "pipeline_log", {}) or {}
                    all_chunks.extend(rescue_chunks)
                    relaunch_log.append(
                        {
                            "stage": "intra_document",
                            "facet": _facet_to_dict(facet),
                            "query_used": rescue_result.query,
                            "docs_searched": candidate_refs[:5],
                            "chunks_new": len(rescue_chunks),
                            "stage2_refs": _unique_strings(rescue_log.get("stage2_candidate_doc_refs", []))[:12],
                            "selected_refs": _unique_strings(rescue_log.get("final_doc_refs", []))[:8],
                            "stage2_chunk_ids": _unique_strings(rescue_log.get("stage2_candidate_chunk_ids", []))[:12],
                            "selected_chunk_ids": _unique_strings(rescue_log.get("final_chunk_ids", []))[:8],
                        }
                    )
                    self._progress(
                        "Recherche intra-document",
                        detail="Le bon BOI semble deja candidat; l'agent descend dans ses sections avant de relancer tout le corpus.",
                        fields=[
                            {"label": "Axe manquant", "value": facet.name},
                            {"label": "BOI inspectes", "value": "\n".join(candidate_refs[:5]) or "Aucun"},
                            {
                                "label": "Passages remontes",
                                "value": "\n".join(_source_labels(rescue_chunks[:4])) or "Aucun nouveau passage",
                            },
                        ],
                    )
                    continue
                result, new_chunks = self._retrieve_for_facet(facet, original_question, seen_ids)
                pipeline_log = getattr(result, "pipeline_log", {}) or {}
                for chunk in new_chunks:
                    chunk["retrieval_stage"] = "global_relaunch"
                all_chunks.extend(new_chunks)
                relaunch_log.append(
                    {
                        "stage": "global_relaunch",
                        "facet": _facet_to_dict(facet),
                        "query_used": result.query,
                        "docs_found": len(result.stage1_hits),
                        "chunks_new": len(new_chunks),
                        "stage1_refs": [h.boi_reference for h in result.stage1_hits[:8]],
                        "stage2_refs": _unique_strings(pipeline_log.get("stage2_candidate_doc_refs", []))[:12],
                        "selected_refs": _unique_strings(pipeline_log.get("final_doc_refs", []))[:8],
                        "stage2_chunk_ids": _unique_strings(pipeline_log.get("stage2_candidate_chunk_ids", []))[:12],
                        "selected_chunk_ids": _unique_strings(pipeline_log.get("final_chunk_ids", []))[:8],
                    }
                )
                self._progress(
                    "Relance documentaire ciblée",
                    detail="Relance déclenchée par un axe manquant identifié, pas par une reformulation globale.",
                    fields=[
                        {"label": "Axe manquant", "value": facet.name},
                        {"label": "Requête relancée", "value": result.query},
                        {
                            "label": "Nouvelles sources",
                            "value": "\n".join(
                                f"{h.boi_reference} - {_short(h.title, 120)}" for h in result.stage1_hits[:4]
                            )
                            or "Aucune",
                        },
                    ],
                )

            if relaunch_log:
                source_review = _complete_source_review_with_plan_gaps(
                    _merge_source_reviews(
                        source_review,
                        _ensure_nonempty_source_review(self._review_sources(original_question, plan, all_chunks), plan),
                        all_chunks,
                    ),
                    plan,
                    all_chunks,
                )
                selected_chunks = _select_reviewed_chunks(all_chunks, source_review)
                self._progress(
                    "Critique après relance",
                    detail="Les nouvelles sources sont réévaluées avant la réponse finale.",
                    fields=[
                        {"label": "Sources utiles", "value": "\n".join(_source_labels(selected_chunks[:10])) or "Aucune"},
                        {"label": "Axes encore manquants", "value": _format_missing_axes(source_review.get("missing_axes", [])) or "Aucun"},
                    ],
                )

        final_chunks = selected_chunks[:12] if selected_chunks else _sort_chunks(all_chunks)[:12]
        evidence_matrix = _build_evidence_matrix(plan, all_chunks, source_review, final_chunks)
        self._progress(
            "Question posée au modèle de réponse",
            detail="Rédaction sourcée avec le plan fiscal et les passages retenus.",
            fields=[
                {"label": "Question originale", "value": original_question},
                {"label": "Sources transmises", "value": "\n".join(_source_labels(final_chunks[:10])) or "Aucune"},
            ],
        )
        answer = _clean_answer_status(self._answer(original_question, final_chunks, plan, source_review), plan, source_review)
        self._progress(
            "Auto-évaluation de couverture",
            detail=f"Statut {answer.get('answer_status', '?')}",
            fields=[
                {"label": "Axes requis", "value": "\n".join(answer.get("axes_requis", []) or []) or "Non renseigné"},
                {"label": "Axes couverts", "value": "\n".join(answer.get("axes_couverts", []) or []) or "Non renseigné"},
                {"label": "Axes manquants", "value": "\n".join(answer.get("axes_manquants", []) or []) or "Aucun"},
            ],
        )

        trace = [
            {
                "iteration": 1,
                "stage": "plan_and_route",
                "plan": _plan_to_dict(plan),
                "routes": route_log,
                "source_review": source_review,
                "relaunches": relaunch_log,
                "evidence_matrix": evidence_matrix,
                "answer_status": answer.get("answer_status", "?"),
                "axes_requis": answer.get("axes_requis", []),
                "axes_couverts": answer.get("axes_couverts", []),
                "axes_manquants": answer.get("axes_manquants", []),
                "step_timings": list(self._step_timings),
            }
        ]

        total_s = round(time.time() - t_start, 2)
        coverage = _compute_coverage(answer)

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
            "iterations": 1 + (1 if relaunch_log else 0),
            "total_s": total_s,
            "chunks_used": len(final_chunks),
            "sources": final_chunks[:8],
            "trace": trace,
            "plan": _plan_to_dict(plan),
            "source_review": source_review,
            "evidence_matrix": evidence_matrix,
            "step_timings": list(self._step_timings),
        }

    def _retrieve_for_facet(self, facet: SearchFacet, original_question: str, seen_ids: set[str]):
        query = _build_facet_retrieval_query(original_question, facet)
        chunk_query = _build_facet_chunk_query(facet, original_question)
        result = self.rt.retrieve(
            query,
            top_docs=6,
            max_chunks=4,
            use_reranker=self.use_reranker,
            boost_prefix=facet.prefix,
            chunk_query=chunk_query,
        )
        chunks = _chunks_from_result(result, facet=facet.name)
        new_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in seen_ids]
        for chunk in new_chunks:
            seen_ids.add(chunk["chunk_id"])
        evidence_chunks = self._retrieve_expected_evidence_chunks(facet, result, seen_ids)
        if evidence_chunks:
            new_chunks.extend(evidence_chunks)
            pipeline_log = getattr(result, "pipeline_log", None)
            if isinstance(pipeline_log, dict):
                pipeline_log["evidence_rescue_chunk_ids"] = [chunk["chunk_id"] for chunk in evidence_chunks]
                pipeline_log["evidence_rescue_queries"] = _unique_strings(
                    chunk.get("evidence_query", "") for chunk in evidence_chunks
                )
            self._progress(
                "Recherche de sous-preuves",
                detail="L'agent complete une facette multi-preuves dans les BOI deja candidats.",
                fields=[
                    {"label": "Axe", "value": facet.name},
                    {"label": "Passages ajoutes", "value": "\n".join(_source_labels(evidence_chunks[:4])) or "Aucun"},
                ],
            )
        return result, new_chunks

    def _retrieve_expected_evidence_chunks(self, facet: SearchFacet, result, seen_ids: set[str]) -> list[dict]:
        evidence_values = _specific_expected_evidence_values(facet.expected_evidence)
        if not _should_rescue_expected_evidence(facet, evidence_values) or not hasattr(self.rt, "retrieve_within_documents"):
            return []
        candidate_refs = _candidate_refs_from_retrieval_result(result)
        if not candidate_refs:
            return []

        rescued: list[dict] = []
        for evidence in evidence_values[:3]:
            evidence_query = _build_expected_evidence_chunk_query(facet, evidence)
            if not evidence_query:
                continue
            evidence_result = self.rt.retrieve_within_documents(
                evidence_query,
                candidate_refs[:6],
                chunks_per_doc=2,
                max_chunks=4,
            )
            for chunk in _chunks_from_result(evidence_result, facet=facet.name):
                chunk_id = chunk.get("chunk_id", "")
                if not chunk_id or chunk_id in seen_ids:
                    continue
                chunk["retrieval_stage"] = "evidence_rescue"
                chunk["evidence_query"] = evidence_query
                rescued.append(chunk)
                seen_ids.add(chunk_id)
        return rescued

    def _retrieve_within_candidate_docs(
        self,
        facet: SearchFacet,
        original_question: str,
        candidate_refs: list[str],
        seen_ids: set[str],
    ):
        if not candidate_refs or not hasattr(self.rt, "retrieve_within_documents"):
            return None, []
        query = _build_facet_chunk_query(facet, original_question)
        result = self.rt.retrieve_within_documents(
            query,
            candidate_refs[:6],
            chunks_per_doc=6,
            max_chunks=8,
        )
        chunks = _chunks_from_result(result, facet=facet.name)
        new_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in seen_ids]
        for chunk in new_chunks:
            chunk["retrieval_stage"] = "intra_document"
            seen_ids.add(chunk["chunk_id"])
        return result, new_chunks

    def _plan_question(self, question: str) -> SearchPlan:
        prompt = (
            "Analyse cette question fiscale avant toute recherche documentaire BOFiP.\n"
            "Retourne uniquement un JSON avec ce schema exact:\n"
            "{"
            '"reformulated_question":"question fiscale clarifiee en une phrase",'
            '"facts":["fait utilisateur neutre"],'
            '"ambiguities":["hypothese ou information manquante a signaler"],'
            '"facets":[{"name":"axe fiscal court","goal":"preuve a trouver",'
            '"bofip_prefix":"famille BOFiP probable","search_query":"8 a 18 mots techniques BOFiP",'
            '"priority":1,"role":"core|calculation|reserve|alternative","blocking":true,'
            '"expected_evidence":["type de preuve attendue"]}],'
            '"excluded_axes":[{"axis":"axe ecarte","reason":"raison courte"}]'
            "}\n\n"
            "Contraintes:\n"
            "- Ne conclus pas sur le droit applicable; tu planifies seulement la recherche.\n"
            "- Ne calcule aucun impot dans ce plan.\n"
            "- N'infere jamais un regime fiscal depuis le montant seul.\n"
            "- Si le statut juridique est ambigu, cree des axes alternatifs au lieu de choisir un seul regime.\n"
            "- Ne remplace pas entreprise, PME ou societe par micro-entrepreneur si ce n'est pas explicite.\n"
            "- Distingue toujours chiffre d'affaires, benefice/resultat, TVA collectee et revenu imposable.\n"
            "- Decoupe les questions mixtes en plusieurs axes, meme si les mots se ressemblent.\n"
            "- Marque role='core' ou 'calculation' pour les axes necessaires a la reponse principale.\n"
            "- Marque role='reserve' ou 'alternative' et blocking=false pour les reserves, exceptions ou hypotheses non declenchees par les faits.\n"
            "- Un axe ajoute seulement pour eviter une confusion, cadrer le hors-perimetre ou signaler ce qui n'est pas calcule doit etre role='reserve' et blocking=false.\n"
            "- Les exonerations facultatives, temporaires, zonage, dispositifs sectoriels ou cas particuliers restent reserve/alternative sauf si les faits utilisateur les declenchent explicitement.\n"
            "- Utilise un vocabulaire BOFiP technique pour search_query.\n"
            "- bofip_prefix doit etre vide ou commencer par une famille connue: "
            + BOFIP_PREFIX_FAMILIES.replace("|", ", ")
            + ".\n"
            "- Exclue explicitement les axes qui attireraient des sources hors sujet.\n\n"
            "Question: "
            + question
        )
        system = (
            "Tu es un planneur de recherche fiscale BOFiP. "
            "Tu transformes une question utilisateur en axes de recherche generiques, "
            "sans repondre et sans inventer de seuils."
        )
        return _normalize_plan(question, self._call_llm(prompt, system, json_mode=True))

    def _review_sources(self, question: str, plan: SearchPlan, chunks: list[dict]) -> dict:
        if not chunks:
            return {
                "coverage_status": "needs_more_sources",
                "covered_axes": [],
                "missing_axes": [
                    {
                        "axis": facet.name,
                        "bofip_prefix": facet.prefix,
                        "search_query": facet.query,
                        "why_needed": facet.goal,
                    }
                    for facet in plan.facets[:3]
                ],
                "useful_chunk_ids": [],
                "rejected_chunks": [],
            }

        chunk_blocks = []
        for chunk in _chunks_for_source_review(chunks, limit=16):
            chunk_blocks.append(
                f"ID: {chunk['chunk_id']}\n"
                f"Axe de recherche: {chunk.get('facet', '')}\n"
                f"BOI: {chunk['boi_reference']}\n"
                f"Titre: {chunk['title']}\n"
                f"Section: {chunk['section_path']}\n"
                f"Texte: {_short(chunk['text'], 900)}"
            )

        prompt = (
            "Question utilisateur:\n"
            + question
            + "\n\nPlan fiscal:\n"
            + json.dumps(_plan_to_dict(plan), ensure_ascii=False)
            + "\n\nPassages candidats:\n"
            + "\n\n---\n\n".join(chunk_blocks)
            + "\n\nRetourne uniquement un JSON:\n"
            "{"
            '"coverage_status":"ready|needs_more_sources",'
            '"useful_chunk_ids":["id"],'
            '"rejected_chunks":[{"chunk_id":"id","reason":"raison courte"}],'
            '"covered_axes":["axe couvert"],'
            '"missing_axes":[{"axis":"axe manquant","bofip_prefix":"prefixe",'
            '"search_query":"requete BOFiP ciblee","why_needed":"preuve manquante",'
            '"role":"core|calculation|reserve|alternative","blocking":true}]'
            "}\n\n"
            "Regles:\n"
            "- Garde seulement les passages qui aident a repondre a un axe du plan.\n"
            "- Rejette les documents attires par un mot ambigu mais fiscalement hors sujet.\n"
            "- Si un passage ne donne qu'un renvoi vers un BOI de fond sans contenir la regle de fond, traite-le comme source passerelle: ne le considere pas suffisant et ajoute le BOI cite dans missing_axes.search_query.\n"
            "- coverage_status='ready' si les passages permettent de repondre a la question principale, meme avec des limites.\n"
            "- coverage_status='needs_more_sources' seulement si une preuve manquante peut changer la conclusion, le taux, le regime, le calcul ou l'obligation principale.\n"
            "- useful_chunk_ids est obligatoire pour chaque axe couvert; si aucun passage n'est utile pour la question principale, coverage_status vaut needs_more_sources.\n"
            "- missing_axes doit contenir uniquement des axes bloquants pour la conclusion principale, pas des references formelles, precisions annexes, exceptions non demandees ou modalites secondaires.\n"
            "- Si un axe sert seulement a eviter une confusion, cadrer le hors-perimetre ou signaler une imposition non demandee, classe-le non bloquant.\n"
            "- Pour une reserve ou alternative non bloquante, mets role='reserve' ou 'alternative' et blocking=false; elle ne doit pas declencher de relance documentaire.\n"
            "- Une exoneration facultative, temporaire, locale, zonage, dispositif sectoriel ou cas particulier ne bloque pas si les faits utilisateur ne la declenchent pas explicitement.\n"
            "- search_query doit rester technique et courte, pas une phrase utilisateur."
        )
        system = (
            "Tu es un critique de sources BOFiP. "
            "Tu ne reponds pas a la question; tu qualifies l'utilite documentaire des passages."
        )
        return _normalize_source_review(self._call_llm(prompt, system, json_mode=True), chunks)

    def _answer(self, question: str, chunks: list[dict], plan: SearchPlan, source_review: dict) -> dict:
        prompt = build_prompt(_build_answer_question(question, plan, source_review), chunks)
        system = build_system_prompt() + (
            "\n\nTu disposes aussi d'un plan fiscal et d'une critique de sources. "
            "Ils servent a structurer la reponse, mais seules les sources BOFiP citees prouvent le droit. "
            "Distingue clairement base taxable, taux, seuil, hypothese et information manquante."
        )
        return self._call_llm(prompt, system, json_mode=True)

    def _call_llm(self, prompt: str, system: str, json_mode: bool = True) -> dict:
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


TAXONOMY_HINTS: tuple[dict[str, object], ...] = (
    {
        "prefix": "TVA",
        "label": "taxe sur la valeur ajoutée",
        "markers": (
            "tva",
            "taxe sur la valeur ajoutee",
            "territorialite tva",
            "lieu des prestations",
            "prestation b2b",
            "b2b",
            "preneur assujetti",
            "intracommunautaire",
        ),
        "terms": "TVA taxe valeur ajoutée territorialité facturation redevable déclaration",
    },
    {
        "prefix": "IR",
        "label": "impôt sur le revenu des particuliers",
        "markers": ("impot sur le revenu", "prelevement a la source", "foyer fiscal", "quotient familial"),
        "terms": "IR impôt revenu foyer fiscal quotient familial prélèvement source",
    },
    {
        "prefix": "RSA",
        "label": "traitements, salaires et revenus assimilés",
        "markers": (
            "salaire",
            "salaires",
            "traitements salaires",
            "traitement salaire",
            "rupture conventionnelle",
            "indemnite de rupture",
            "indemnites de rupture",
            "rupture du contrat de travail",
            "fin de contrat",
            "licenciement",
            "mandataire social",
            "mandataires sociaux",
            "dirigeant",
            "dirigeants",
            "cessation de fonctions",
            "cessation des fonctions",
        ),
        "terms": (
            "RSA traitements salaires revenus assimilés indemnités imposables exonération limites"
        ),
    },
    {
        "prefix": "BIC",
        "label": "bénéfices industriels et commerciaux",
        "markers": (
            "bic",
            "micro-bic",
            "benefices industriels",
            "chiffre d'affaires",
            "auto-entrepreneur",
            "entrepreneur",
            "location meublee",
            "loueur en meuble",
            "loueurs en meuble",
            "lmnp",
            "meuble non professionnel",
            "location meublee non professionnelle",
        ),
        "terms": (
            "BIC bénéfices industriels commerciaux micro entreprise chiffre affaires bénéfice imposable régime"
        ),
    },
    {
        "prefix": "BNC",
        "label": "bénéfices non commerciaux",
        "markers": ("bnc", "profession liberale", "honoraires", "micro-bnc"),
        "terms": "BNC profession libérale honoraires micro BNC recettes imposables",
    },
    {
        "prefix": "IS",
        "label": "impôt sur les sociétés",
        "markers": ("impot sur les societes", "societe soumise a l'is", " is "),
        "terms": "IS impôt sociétés résultat fiscal bénéfice imposable",
    },
    {
        "prefix": "RFPI",
        "label": "revenus fonciers et plus-values immobilières",
        "markers": (
            "revenus fonciers",
            "location nue",
            "micro-foncier",
            "micro foncier",
            "loyers",
            "charges de copropriete",
            "copropriete",
            "plus-value immobiliere",
            "plus value immobiliere",
            "vente appartement",
            "vente maison",
            "residence principale",
        ),
        "terms": "RFPI revenus fonciers location nue micro-foncier régime réel charges déductibles copropriété plus-values immobilières",
    },
    {
        "prefix": "RPPM",
        "label": "revenus et produits du patrimoine mobilier",
        "markers": (
            "cto",
            "compte titre",
            "compte-titre",
            "compte titres",
            "compte-titres",
            "valeurs mobilieres",
            "titre financier",
            "titres financiers",
            "cession de titres",
            "plus-value",
            "plus value",
            "moins-value",
            "moins value",
            "actions",
            "pea",
            "pea-pme",
            "plan epargne en actions",
            "plan d epargne en actions",
            "plan d'epargne en actions",
            "dividende",
            "revenus mobiliers",
            "per ",
            "plan epargne retraite",
        ),
        "terms": (
            "RPPM revenus capitaux mobiliers dividendes intérêts produits placements valeurs mobilières"
        ),
    },
    {
        "prefix": "CF",
        "label": "contrôle fiscal et pénalités",
        "markers": ("controle fiscal", "redressement", "interet de retard", "interets de retard", "majoration", "penalite"),
        "terms": "CF contrôle fiscal redressement intérêts retard majoration pénalité procédure",
    },
    {
        "prefix": "ENR",
        "label": "enregistrement, succession, donation",
        "markers": ("succession", "donation", "droits d'enregistrement"),
        "terms": "ENR succession donation droits enregistrement mutation abattement",
    },
    {
        "prefix": "IF",
        "label": "impôts fonciers et contribution économique territoriale",
        "markers": ("taxe fonciere", "cfe", "cvae", "cotisation fonciere"),
        "terms": "IF taxe foncière CFE CVAE cotisation foncière base exonération",
    },
)
def _short(value: object, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _ascii_lower(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _taxonomy_facets_for_question(question: str) -> list[dict[str, object]]:
    q = _ascii_lower(question)
    scored: list[tuple[int, dict[str, object]]] = []
    for hint in TAXONOMY_HINTS:
        markers = hint["markers"]
        score = sum(1 for marker in markers if str(marker) in q)
        if score:
            scored.append((score, hint))
    scored.sort(key=lambda item: (-item[0], str(item[1]["prefix"])))
    return [hint for _, hint in scored]


def _prefix_once(prefix: str, query: str) -> str:
    prefix = (prefix or "").strip()
    query = (query or "").strip()
    if not prefix:
        return query
    return query if query.upper().startswith(prefix.upper() + " ") else f"{prefix} {query}"


def _normalize_prefix(value: object) -> str:
    raw = str(value or "").strip().upper().removeprefix("BOI-")
    match = re.search(rf"\b({BOFIP_PREFIX_FAMILIES})(?:-[A-Z0-9]+){{0,8}}\b", raw)
    return match.group(0)[:80] if match else ""


def _normalize_facet_prefix(value: object, context: str = "") -> str:
    return _normalize_prefix(value)


def _prefix_family(prefix: str) -> str:
    normalized = _normalize_prefix(prefix)
    return normalized.split("-", 1)[0] if normalized else ""


def _hint_matches_domain(hint_prefix: str, domain: str) -> bool:
    domain_family = _prefix_family(domain)
    if not domain_family:
        return True
    return _prefix_family(hint_prefix) == domain_family


def _build_retrieval_header(question: str, domain: str = "") -> str:
    parts: list[str] = []
    normalized_domain = _normalize_prefix(domain)
    if normalized_domain:
        parts.append(normalized_domain)
    for facet in _taxonomy_facets_for_question(question)[:4]:
        prefix = str(facet["prefix"])
        if not _hint_matches_domain(prefix, normalized_domain):
            continue
        terms = str(facet["terms"])
        if prefix and prefix != normalized_domain:
            parts.append(prefix)
        if terms:
            parts.append(terms)
    return " ".join(part for part in parts if part).strip()


def _build_retrieval_query(question: str, domain: str = "") -> str:
    return " ".join(part for part in (_build_retrieval_header(question, domain), question) if part).strip()


def _expected_evidence_query_text(values: list[str]) -> str:
    parts: list[str] = []
    for value in values[:3]:
        cleaned = BOI_REFERENCE_RE.sub(" ", str(value or ""))
        cleaned = " ".join(cleaned.split())
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


def _specific_expected_evidence_values(values: list[str]) -> list[str]:
    generic_terms = {
        "application",
        "applicable",
        "champ",
        "conditions",
        "doctrine",
        "limite",
        "limites",
        "modalite",
        "modalites",
        "pertinente",
        "preuve",
        "regle",
        "source",
        "sources",
    }
    specific: list[str] = []
    for value in values or []:
        text = " ".join(str(value or "").split())
        if not text:
            continue
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", _ascii_lower(text))
            if len(token) >= 4
        }
        if not tokens:
            continue
        if tokens - generic_terms or re.search(r"\d|%|€", text):
            specific.append(text)
    return _unique_strings(specific)


def _should_rescue_expected_evidence(facet: SearchFacet, evidence_values: list[str]) -> bool:
    if len(evidence_values) >= 2:
        return True
    if not evidence_values:
        return False
    context = _ascii_lower(" ".join([facet.name, facet.goal, facet.query, " ".join(evidence_values)]))
    return any(
        marker in context
        for marker in (
            "abattement",
            "bareme",
            "barème",
            "calcul",
            "forfait",
            "montant",
            "plafond",
            "seuil",
        )
    ) or bool(re.search(r"\d|%", context))


def _build_facet_retrieval_query(original_question: str, facet: SearchFacet) -> str:
    focus_text = " ".join(
        part
        for part in (
            facet.name,
            facet.query,
        )
        if part
    )
    taxonomy_header = _normalize_facet_prefix(facet.prefix, focus_text)
    parts = [
        taxonomy_header,
        facet.name,
        facet.query,
        original_question,
    ]
    return " ".join(part for part in parts if part).strip()[:500]


def _build_facet_chunk_query(facet: SearchFacet, original_question: str = "") -> str:
    expected_evidence = _expected_evidence_query_text(facet.expected_evidence)
    focus_text = " ".join(
        part
        for part in (
            facet.query,
            facet.goal,
            expected_evidence,
        )
        if part
    )
    taxonomy_header = _normalize_facet_prefix(facet.prefix, focus_text)
    parts = [
        taxonomy_header,
        facet.query,
        _numeric_threshold_terms(f"{original_question} {focus_text}"),
        facet.goal,
        expected_evidence,
        _question_terms_without_numbers(original_question),
    ]
    return " ".join(part for part in parts if part).strip()[:500]


def _build_expected_evidence_chunk_query(facet: SearchFacet, evidence: str) -> str:
    evidence_text = _expected_evidence_query_text([evidence])
    if not evidence_text:
        return ""
    taxonomy_header = _normalize_facet_prefix(facet.prefix, " ".join((facet.name, evidence_text)))
    expansion = _expected_evidence_expansion(evidence_text)
    parts = [
        taxonomy_header,
        facet.prefix,
        facet.name,
        facet.query,
        evidence_text,
        expansion,
    ]
    return " ".join(part for part in parts if part).strip()[:300]


def _expected_evidence_expansion(evidence: str) -> str:
    normalized = _ascii_lower(evidence)
    terms: list[str] = []
    if "abattement" in normalized and any(marker in normalized for marker in ("forfaitaire", "charges", "charge")):
        terms.extend(["application abattement", "taux abattement", "conditions abattement"])
    if "taux" in normalized and "abattement" in normalized:
        terms.extend(["application abattement", "pourcentage"])
    return " ".join(_unique_strings(terms))


def _numeric_threshold_terms(value: str) -> str:
    if not re.search(r"\d", value or ""):
        return ""
    normalized = _ascii_lower(value)
    terms: list[str] = []
    if any(marker in normalized for marker in ("chiffre d'affaires", "chiffre affaires", "recettes", "ca ")):
        terms.extend(["seuil", "plafond", "inferieur egal", "faible chiffre affaires", "recettes"])
    if any(marker in normalized for marker in ("loyers", "revenus", "recettes")):
        terms.extend(["seuil", "plafond", "abattement", "regime"])
    if any(marker in normalized for marker in ("taux", "pourcentage", "combien", "calcul")):
        terms.extend(["taux", "bareme", "calcul"])
    deduped: list[str] = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return " ".join(deduped)


def _question_terms_without_numbers(question: str) -> str:
    if not question:
        return ""
    without_numbers = re.sub(r"\b\d[\d\s\u00a0.,]*\d?\b", " ", question)
    without_amount_words = re.sub(r"\b(euros?|€|ht|ttc)\b", " ", without_numbers, flags=re.IGNORECASE)
    stopwords = {
        "avec",
        "cela",
        "dans",
        "depuis",
        "dois",
        "donc",
        "dont",
        "elle",
        "est",
        "mes",
        "mon",
        "pour",
        "puis",
        "quel",
        "quelle",
        "sans",
        "suis",
        "une",
        "vous",
    }
    tokens = []
    for token in re.findall(r"[A-Za-zÀ-ÿ']+", without_amount_words):
        normalized = _ascii_lower(token).strip("'")
        if len(normalized) < 4 or normalized in stopwords:
            continue
        if normalized not in tokens:
            tokens.append(normalized)
    return " ".join(tokens[:36])


def _fallback_domain_from_question(question: str) -> str:
    facets = _taxonomy_facets_for_question(question)
    return str(facets[0]["prefix"]) if facets else ""


def _fallback_plan(question: str) -> SearchPlan:
    facets: list[SearchFacet] = []
    for hint in _taxonomy_facets_for_question(question)[:4]:
        facets.append(
            SearchFacet(
                name=str(hint["label"]),
                goal=f"Identifier la doctrine BOFiP pertinente pour {hint['label']}",
                query=str(hint["terms"]),
                prefix=str(hint["prefix"]),
                priority=len(facets) + 1,
                expected_evidence=["règle applicable", "champ d'application", "limites"],
                role="core",
                blocking=True,
            )
        )
    if not facets:
        facets.append(
            SearchFacet(
                name="Doctrine fiscale applicable",
                goal="Identifier les règles BOFiP répondant à la question",
                query=question,
                prefix="",
                priority=1,
                expected_evidence=["règle applicable", "conditions", "limites"],
                role="core",
                blocking=True,
            )
        )
    return SearchPlan(
        reformulated_question=question,
        facts=[question],
        ambiguities=[],
        facets=facets,
        excluded_axes=[],
    )


def _normalize_axis_role(value: object) -> str:
    role = str(value or "").strip().lower()
    if role in {"core", "calculation", "reserve", "alternative"}:
        return role
    return "core"


def _normalize_blocking_flag(value: object, role: object = "") -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"false", "0", "no", "non"}:
        return False
    if text in {"true", "1", "yes", "oui"}:
        return True
    return _normalize_axis_role(role) in {"core", "calculation"}


def _looks_nonblocking_axis(item: dict) -> bool:
    text = _ascii_lower(
        " ".join(
            str(item.get(key, "") or "")
            for key in ("axis", "search_query", "why_needed")
        )
    )
    return any(
        marker in text
        for marker in (
            "non bloquant",
            "ne change pas la conclusion",
            "ne modifie pas la conclusion",
            "precision annexe",
            "modalite secondaire",
            "utile en limite",
            "reserve",
            "hors perimetre",
            "hors sujet",
            "pas demande",
            "non demande",
            "eviter une confusion",
            "utile seulement",
            "seulement si necessaire",
            "seulement si",
            "cadrer la reponse",
            "eventuel",
            "eventuelle",
            "conditions particulieres",
            "non precise",
            "non precisees",
            "pourrait modifier",
            "pourraient modifier",
            "faute de faits",
            "fait declencheur",
            "faits declencheurs",
            "non declenche",
            "non declenchee",
            )
    )


def _question_requires_numeric_or_threshold_axis(question: str, axis_context: str) -> bool:
    q = _ascii_lower(question)
    axis = _ascii_lower(axis_context)
    asks_for_liability_or_amount = any(
        marker in q
        for marker in (
            "combien",
            "taux",
            "montant",
            "calcul",
            "calcule",
            "imposable",
            "soumis",
            "exonere",
            "exoneration",
            "taxe",
            "impot",
            "cotisation",
        )
    )
    axis_is_threshold_or_amount = any(
        marker in axis
        for marker in (
            "seuil",
            "plafond",
            "bareme",
            "taux",
            "montant",
            "chiffre d'affaires",
            "chiffre affaires",
            "recettes",
            "cotisation minimum",
            "base minimum",
        )
    )
    return asks_for_liability_or_amount and axis_is_threshold_or_amount


def _question_assumes_regime_for_calculation(question: str) -> bool:
    q = _ascii_lower(question)
    assumes_regime = any(
        marker in q
        for marker in (
            "si je releve",
            "si nous relevons",
            "si je suis au regime",
            "si je suis sous le regime",
            "si j applique le regime",
            "relevant du regime",
            "sous le regime",
            "au regime applicable",
        )
    )
    asks_calculation = any(
        marker in q
        for marker in (
            "benefice imposable",
            "base imposable",
            "resultat imposable",
            "dois je retenir",
            "combien",
            "calcul",
            "calcule",
            "montant",
            "taux",
        )
    )
    asks_eligibility = any(
        marker in q
        for marker in (
            "puis je",
            "ai je droit",
            "suis je eligible",
            "est ce que je releve",
            "conditions pour relever",
            "puis je beneficier",
            "ai je droit au regime",
        )
    )
    return assumes_regime and asks_calculation and not asks_eligibility


def _facet_is_regime_reserve(context: str) -> bool:
    text = _ascii_lower(context)
    return any(
        marker in text
        for marker in (
            "changement",
            "sortie",
            "exclusion",
            "exclusions",
            "exclu",
            "exclus",
            "option",
            "cessation",
            "creation",
            "cessent",
            "cesse de s appliquer",
        )
    )


def _facet_is_scope_clarification_reserve(context: str) -> bool:
    text = _ascii_lower(context)
    strong_markers = (
        "eviter une confusion",
        "hors perimetre",
        "hors sujet",
        "utile seulement",
        "seulement si necessaire",
        "cadrer la reponse",
        "pas demande",
        "non demande",
        "ne pas calculer",
        "ne calcule pas",
        "sans calculer",
    )
    if any(marker in text for marker in strong_markers):
        return True
    has_limited_scope = any(marker in text for marker in ("uniquement", "seulement", "limite", "reserve"))
    has_clarification = any(marker in text for marker in ("distinction", "distinguer", "clarification", "cadrer"))
    return has_limited_scope and has_clarification


def _facet_is_untriggered_optional_reserve(question: str, context: str) -> bool:
    text = _ascii_lower(context)
    q = _ascii_lower(question)
    optional_markers = (
        "eventuel",
        "eventuelle",
        "cas particulier",
        "conditions particulieres",
        "condition particuliere",
        "non precise",
        "non precisees",
        "non indique",
        "non indiquees",
        "non declenche",
        "non declenchee",
        "fait declencheur",
        "faits declencheurs",
        "faute de faits",
        "pourrait modifier",
        "pourraient modifier",
        "sous reserve d'informations",
        "hypothese non",
        "dispositif special",
        "regime special",
    )
    if not any(marker in text for marker in optional_markers):
        return False
    explicit_question_markers = (
        "ai je droit",
        "a t on droit",
        "suis je eligible",
        "sommes nous eligibles",
        "puis je beneficier",
        "peut on beneficier",
        "conditions pour beneficier",
        "conditions d application",
        "exception",
        "cas particulier",
        "dispositif special",
        "regime special",
    )
    if any(marker in q for marker in explicit_question_markers):
        return False
    return True


def _normalize_plan(question: str, raw: dict) -> SearchPlan:
    if not isinstance(raw, dict):
        return _fallback_plan(question)

    facets: list[SearchFacet] = []
    for idx, item in enumerate(raw.get("facets", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        query = _short(item.get("search_query", ""), 220)
        goal = _short(item.get("goal", ""), 220)
        name = _short(item.get("name", ""), 90)
        if not query and not goal:
            continue
        expected = [
            _short(value, 120)
            for value in (item.get("expected_evidence", []) or [])
            if str(value or "").strip()
        ][:4]
        prefix_context = " ".join(
            str(part or "")
            for part in (
                question,
                raw.get("reformulated_question", ""),
                name,
                goal,
                query,
                " ".join(expected),
            )
        )
        facet_context = " ".join(str(part or "") for part in (name, goal, query, " ".join(expected)))
        try:
            priority = int(item.get("priority", idx))
        except (TypeError, ValueError):
            priority = idx
        role = _normalize_axis_role(item.get("role", ""))
        blocking = _normalize_blocking_flag(item.get("blocking", None), item.get("role", ""))
        if _question_requires_numeric_or_threshold_axis(question, prefix_context):
            role = "calculation"
            blocking = True
        if _question_assumes_regime_for_calculation(question) and _facet_is_regime_reserve(prefix_context):
            role = "reserve"
            blocking = False
        if _facet_is_scope_clarification_reserve(facet_context):
            role = "reserve"
            blocking = False
        if _facet_is_untriggered_optional_reserve(question, facet_context):
            role = "reserve"
            blocking = False
        facets.append(
            SearchFacet(
                name=name or f"Axe fiscal {idx}",
                goal=goal or "Identifier la doctrine BOFiP applicable",
                query=query or goal,
                prefix=_normalize_facet_prefix(item.get("bofip_prefix", ""), prefix_context),
                priority=max(1, min(priority, 9)),
                expected_evidence=expected,
                role=role,
                blocking=blocking,
            )
        )

    if not facets:
        return _fallback_plan(question)

    facets = sorted(facets, key=lambda facet: facet.priority)[:5]
    facts = [_short(value, 180) for value in (raw.get("facts", []) or []) if str(value or "").strip()][:8]
    ambiguities = [_short(value, 180) for value in (raw.get("ambiguities", []) or []) if str(value or "").strip()][:8]
    excluded_axes = []
    for item in raw.get("excluded_axes", []) or []:
        if isinstance(item, dict):
            axis = _short(item.get("axis", ""), 90)
            reason = _short(item.get("reason", ""), 160)
            if axis or reason:
                excluded_axes.append({"axis": axis, "reason": reason})
    return SearchPlan(
        reformulated_question=_short(raw.get("reformulated_question", ""), 260) or question,
        facts=facts,
        ambiguities=ambiguities,
        facets=facets,
        excluded_axes=excluded_axes[:8],
    )

def _normalize_source_review(review: dict, chunks: list[dict]) -> dict:
    known_ids = {chunk["chunk_id"] for chunk in chunks}
    useful_ids = []
    if isinstance(review, dict):
        for chunk_id in review.get("useful_chunk_ids", []) or []:
            if chunk_id in known_ids and chunk_id not in useful_ids:
                useful_ids.append(chunk_id)

    covered_axes = []
    missing_axes = []
    non_blocking_axes = []
    rejected = []
    if isinstance(review, dict):
        covered_axes = [_short(value, 160) for value in review.get("covered_axes", []) if str(value or "").strip()][:8]
        for item in review.get("missing_axes", []) or []:
            if not isinstance(item, dict):
                continue
            axis = _short(item.get("axis", ""), 120)
            query = _short(item.get("search_query", ""), 220)
            if axis or query:
                role = _normalize_axis_role(item.get("role", ""))
                blocking = _normalize_blocking_flag(item.get("blocking", None), item.get("role", ""))
                if "blocking" not in item and not item.get("role") and _looks_nonblocking_axis(item):
                    role = "reserve"
                    blocking = False
                normalized_item = {
                    "axis": axis or query,
                    "bofip_prefix": _normalize_facet_prefix(
                        item.get("bofip_prefix", ""),
                        " ".join(str(part or "") for part in (axis, query, item.get("why_needed", ""))),
                    ),
                    "search_query": query,
                    "why_needed": _short(item.get("why_needed", ""), 180),
                    "role": role,
                    "blocking": blocking,
                }
                if normalized_item["blocking"]:
                    missing_axes.append(normalized_item)
                else:
                    non_blocking_axes.append(normalized_item)
        for item in review.get("rejected_chunks", []) or []:
            if isinstance(item, dict) and item.get("chunk_id") in known_ids:
                rejected.append(
                    {
                        "chunk_id": item.get("chunk_id"),
                        "reason": _short(item.get("reason", ""), 160),
                    }
                )

    coverage_status = review.get("coverage_status", "ready") if isinstance(review, dict) else "ready"
    if covered_axes and not useful_ids:
        useful_ids = _infer_useful_ids_for_covered_axes(chunks, covered_axes, rejected)
    if covered_axes and not useful_ids:
        coverage_status = "needs_more_sources"
        known_missing = {item["axis"].lower() for item in missing_axes if item.get("axis")}
        for axis in covered_axes:
            if axis.lower() in known_missing:
                continue
            missing_axes.append(
                {
                    "axis": axis,
                    "bofip_prefix": "",
                    "search_query": axis,
                    "why_needed": "Aucun passage utile explicitement retenu pour cet axe.",
                    "role": "core",
                    "blocking": True,
                }
            )
        covered_axes = []

    return {
        "coverage_status": coverage_status,
        "useful_chunk_ids": useful_ids,
        "rejected_chunks": rejected[:12],
        "covered_axes": covered_axes,
        "missing_axes": missing_axes[:4],
        "non_blocking_axes": non_blocking_axes[:6],
    }


def _infer_useful_ids_for_covered_axes(chunks: list[dict], covered_axes: list[str], rejected: list[dict]) -> list[str]:
    rejected_ids = {item.get("chunk_id") for item in rejected if isinstance(item, dict)}
    useful: list[str] = []
    for axis in covered_axes:
        for chunk in _sort_chunks(chunks):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id or chunk_id in rejected_ids or chunk_id in useful:
                continue
            facet = str(chunk.get("facet", ""))
            if _axes_match(axis, facet) or _chunk_axis_overlap_score(axis, chunk) >= 2:
                useful.append(chunk_id)
                break
    return useful[:12]


def _merge_source_reviews(previous: dict, current: dict, chunks: list[dict]) -> dict:
    known_ids = {chunk["chunk_id"] for chunk in chunks}
    rejected_ids = {
        item.get("chunk_id")
        for item in (current.get("rejected_chunks", []) or [])
        if isinstance(item, dict) and item.get("chunk_id")
    }

    useful_ids: list[str] = []
    for review in (previous, current):
        for chunk_id in review.get("useful_chunk_ids", []) or []:
            if chunk_id in known_ids and chunk_id not in rejected_ids and chunk_id not in useful_ids:
                useful_ids.append(chunk_id)

    covered_axes: list[str] = []
    for review in (previous, current):
        for axis in review.get("covered_axes", []) or []:
            if axis and axis not in covered_axes:
                covered_axes.append(axis)

    non_blocking_axes: list[dict] = []
    for review in (previous, current):
        for item in review.get("non_blocking_axes", []) or []:
            if isinstance(item, dict) and item not in non_blocking_axes:
                non_blocking_axes.append(item)

    missing_axes = [
        item
        for item in current.get("missing_axes", []) or []
        if isinstance(item, dict) and item.get("blocking", True)
    ]
    merged = dict(current)
    merged["useful_chunk_ids"] = useful_ids[:16]
    merged["covered_axes"] = covered_axes[:12]
    merged["missing_axes"] = missing_axes[:4]
    merged["non_blocking_axes"] = non_blocking_axes[:8]
    if missing_axes:
        merged["coverage_status"] = "needs_more_sources"
    elif useful_ids or covered_axes:
        merged["coverage_status"] = "ready"
    return merged


def _ensure_nonempty_source_review(review: dict, plan: SearchPlan) -> dict:
    has_signal = any(
        review.get(key)
        for key in ("useful_chunk_ids", "covered_axes", "missing_axes", "non_blocking_axes", "rejected_chunks")
    )
    if has_signal:
        return review

    missing_axes = []
    for facet in plan.facets:
        if not facet.blocking:
            continue
        missing_axes.append(
            {
                "axis": facet.name,
                "bofip_prefix": facet.prefix,
                "search_query": facet.query,
                "why_needed": facet.goal,
                "role": facet.role,
                "blocking": True,
            }
        )
    if not missing_axes:
        return review
    updated = dict(review)
    updated["coverage_status"] = "needs_more_sources"
    updated["missing_axes"] = missing_axes[:4]
    return updated


def _matching_nonblocking_plan_facet(item: dict, plan: SearchPlan) -> SearchFacet | None:
    axis_context = " ".join(
        str(item.get(key, "") or "")
        for key in ("axis", "search_query", "why_needed")
    )
    for facet in plan.facets:
        if facet.blocking and facet.role not in {"reserve", "alternative"}:
            continue
        if (
            _axes_match(str(item.get("axis", "")), facet.name)
            or _axes_match(axis_context, facet.name)
            or _axes_match(str(item.get("axis", "")), facet.query)
        ):
            return facet
    return None


def _complete_source_review_with_plan_gaps(
    review: dict,
    plan: SearchPlan,
    chunks: list[dict] | None = None,
) -> dict:
    covered_axes = [str(axis) for axis in review.get("covered_axes", []) or [] if str(axis or "").strip()]
    missing_axes = [
        item
        for item in review.get("missing_axes", []) or []
        if isinstance(item, dict) and item.get("axis")
    ]
    non_blocking_axes = [
        item
        for item in review.get("non_blocking_axes", []) or []
        if isinstance(item, dict) and item.get("axis")
    ]
    useful_id_set = set(_unique_strings(review.get("useful_chunk_ids", []) or []))
    useful_chunks = [
        chunk
        for chunk in chunks or []
        if chunk.get("chunk_id") in useful_id_set
    ]
    demoted_missing: list[dict] = []
    blocking_missing: list[dict] = []
    for item in missing_axes:
        facet = _matching_nonblocking_plan_facet(item, plan)
        if facet is None:
            blocking_missing.append(item)
            continue
        demoted = dict(item)
        demoted["role"] = facet.role if facet.role in {"reserve", "alternative"} else "reserve"
        demoted["blocking"] = False
        demoted["why_needed"] = demoted.get("why_needed") or facet.goal
        demoted_missing.append(demoted)
    if demoted_missing:
        missing_axes = blocking_missing
        for item in demoted_missing:
            if not any(_axes_match(str(item.get("axis", "")), str(existing.get("axis", ""))) for existing in non_blocking_axes):
                non_blocking_axes.append(item)

    def covered_by_useful_source(facet: SearchFacet) -> bool:
        matching_covered_axis = any(_axes_match(facet.name, covered) for covered in covered_axes)
        if not matching_covered_axis:
            return False
        if not chunks:
            return True
        return any(_chunk_matches_axis(facet.name, chunk) for chunk in useful_chunks)

    def already_reviewed(facet: SearchFacet) -> bool:
        if covered_by_useful_source(facet):
            return True
        axis = facet.name
        if any(_axes_match(axis, str(item.get("axis", ""))) for item in missing_axes):
            return True
        if not facet.blocking:
            return any(_axes_match(axis, str(item.get("axis", ""))) for item in non_blocking_axes)
        return False

    added_missing: list[dict] = []
    for facet in plan.facets:
        if not facet.blocking or already_reviewed(facet):
            continue
        added_missing.append(
            {
                "axis": facet.name,
                "bofip_prefix": facet.prefix,
                "search_query": facet.query,
                "why_needed": facet.goal or "Axe bloquant planifie mais non statue par la critique des sources.",
                "role": facet.role,
                "blocking": True,
            }
        )

    if not added_missing:
        if demoted_missing:
            updated = dict(review)
            updated["missing_axes"] = missing_axes[:4]
            updated["non_blocking_axes"] = non_blocking_axes[:8]
            if not missing_axes and updated.get("coverage_status") == "needs_more_sources":
                updated["coverage_status"] = "ready"
            return updated
        return review

    updated = dict(review)
    updated["coverage_status"] = "needs_more_sources"
    updated["missing_axes"] = (missing_axes + added_missing)[:4]
    updated["non_blocking_axes"] = non_blocking_axes[:8]
    return updated


def _select_reviewed_chunks(chunks: list[dict], review: dict) -> list[dict]:
    useful_ids = _unique_strings(review.get("useful_chunk_ids", []) or [])
    rejected_ids = {
        item.get("chunk_id")
        for item in review.get("rejected_chunks", []) or []
        if isinstance(item, dict) and item.get("chunk_id")
    }
    if not useful_ids:
        if rejected_ids or review.get("covered_axes") or review.get("missing_axes"):
            filtered = [chunk for chunk in _sort_chunks(chunks) if chunk["chunk_id"] not in rejected_ids]
            if filtered:
                promoted = _promote_axis_evidence(filtered, review, [], rejected_ids)
                relaunch_chunks = [
                    chunk
                    for chunk in filtered
                    if chunk.get("retrieval_stage") in RETRIEVAL_RESCUE_STAGES
                    and _stage_chunk_matches_review_if_needed(chunk, review)
                    and _relaunch_chunk_matches_missing_review(chunk, review)
                ]
                return _dedupe_chunks(promoted + relaunch_chunks + filtered)[:12]
        return _sort_chunks(chunks)[:12]
    useful_id_set = set(useful_ids)
    selected = [chunk for chunk in _sort_chunks(chunks) if chunk["chunk_id"] in useful_id_set]
    selected = _promote_axis_evidence(chunks, review, selected, rejected_ids)
    selected_ids = {chunk["chunk_id"] for chunk in selected}
    relaunch_chunks: list[dict] = []
    for chunk in _sort_chunks(chunks):
        if chunk.get("retrieval_stage") not in RETRIEVAL_RESCUE_STAGES:
            continue
        if not _stage_chunk_matches_review_if_needed(chunk, review):
            continue
        if not _relaunch_chunk_matches_missing_review(chunk, review):
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id in selected_ids or chunk_id in rejected_ids:
            continue
        relaunch_chunks.append(chunk)
        selected_ids.add(chunk_id)
    return _dedupe_chunks(selected[:8] + relaunch_chunks + selected[8:])[:12]


def _promote_axis_evidence(
    chunks: list[dict],
    review: dict,
    selected: list[dict],
    rejected_ids: set[str],
) -> list[dict]:
    selected_ids = {chunk.get("chunk_id") for chunk in selected}
    promoted: list[dict] = []
    covered_axes = [
        str(axis)
        for axis in review.get("covered_axes", []) or []
        if str(axis or "").strip()
    ]
    missing_axes = _blocking_missing_axis_specs(review)
    for axis in covered_axes:
        if any(_chunk_matches_axis(axis, chunk) for chunk in selected):
            continue
        best = _best_chunk_for_axis(axis, chunks, rejected_ids, selected_ids)
        if not best:
            continue
        promoted.append(best)
        selected_ids.add(best["chunk_id"])
    for axis, preferred_prefix in missing_axes:
        best = _best_chunk_for_axis(
            axis,
            chunks,
            rejected_ids,
            selected_ids,
            preferred_prefix=preferred_prefix,
        )
        if not best:
            continue
        promoted.append(best)
        selected_ids.add(best["chunk_id"])
    return _dedupe_chunks(promoted + selected)


def _blocking_missing_axis_specs(review: dict) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in review.get("missing_axes", []) or []:
        if not (isinstance(item, dict) and item.get("blocking", True)):
            continue
        axis_text = " ".join(
            str(item.get(key, ""))
            for key in ("axis", "search_query", "why_needed")
            if item.get(key)
        )
        if axis_text.strip():
            specs.append((axis_text, _normalize_facet_prefix(item.get("bofip_prefix", ""), axis_text)))
    return specs


def _review_axis_texts(review: dict) -> list[str]:
    axes = [
        str(axis)
        for axis in review.get("covered_axes", []) or []
        if str(axis or "").strip()
    ]
    axes.extend(axis for axis, _prefix in _blocking_missing_axis_specs(review))
    return axes


def _stage_chunk_matches_review(chunk: dict, review: dict) -> bool:
    axes = _review_axis_texts(review)
    if not axes:
        return True
    facet_text = str(chunk.get("facet", ""))
    for axis in axes:
        if _axes_match(axis, facet_text) or _chunk_matches_axis(axis, chunk):
            return True
    return False


def _stage_chunk_matches_review_if_needed(chunk: dict, review: dict) -> bool:
    if chunk.get("retrieval_stage") != "evidence_rescue":
        return True
    return _stage_chunk_matches_review(chunk, review)


def _relaunch_chunk_matches_missing_review(chunk: dict, review: dict) -> bool:
    missing_axes = _blocking_missing_axis_specs(review)
    if not missing_axes:
        return True
    for axis, preferred_prefix in missing_axes:
        if _chunk_matches_axis(axis, chunk):
            return True
    return False


def _best_chunk_for_axis(
    axis: str,
    chunks: list[dict],
    rejected_ids: set[str],
    selected_ids: set[str],
    *,
    preferred_prefix: str = "",
) -> dict | None:
    candidates = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_id")
        and chunk.get("chunk_id") not in rejected_ids
        and chunk.get("chunk_id") not in selected_ids
        and _chunk_matches_axis(axis, chunk)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda chunk: (
            _chunk_prefix_rank(chunk, preferred_prefix),
            0 if _axes_match(axis, str(chunk.get("facet", ""))) else 1,
            -_chunk_axis_overlap_score(axis, chunk),
            chunk.get("rank", 999),
            str(chunk.get("chunk_id", "")),
        ),
    )[0]


def _chunk_prefix_rank(chunk: dict, preferred_prefix: str) -> int:
    prefix = _normalize_prefix(preferred_prefix)
    if not prefix:
        return 0
    ref = str(chunk.get("boi_reference", ""))
    if _ref_matches_prefix(ref, prefix):
        return 0
    if _ref_shares_parent_prefix(ref, prefix):
        return 1
    return 2


def _chunk_matches_axis(axis: str, chunk: dict) -> bool:
    facet = str(chunk.get("facet", ""))
    return _axes_match(axis, facet) or _chunk_axis_overlap_score(axis, chunk) >= 2


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id and chunk_id in seen:
            continue
        deduped.append(chunk)
        if chunk_id:
            seen.add(chunk_id)
    return deduped


def _unique_strings(values) -> list[str]:
    unique: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def _build_evidence_matrix(
    plan: SearchPlan,
    chunks: list[dict],
    source_review: dict,
    final_chunks: list[dict],
) -> list[dict]:
    covered_axes = [str(axis) for axis in source_review.get("covered_axes", []) or [] if str(axis or "").strip()]
    missing_axes = [
        item
        for item in source_review.get("missing_axes", []) or []
        if isinstance(item, dict) and item.get("axis")
    ]
    rows: list[dict] = []
    seen_axes: set[str] = set()

    def add_axis(axis: str, *, role: str = "core", blocking: bool = True, missing_reason: str = "") -> None:
        axis_key = _ascii_lower(axis)
        if not axis_key or axis_key in seen_axes:
            return
        seen_axes.add(axis_key)
        candidate_chunks = [chunk for chunk in chunks if _chunk_matches_axis(axis, chunk)]
        final_axis_chunks = [chunk for chunk in final_chunks if _chunk_matches_axis(axis, chunk)]
        is_covered = any(_axes_match(axis, covered) for covered in covered_axes)
        is_missing = any(_axes_match(axis, item.get("axis", "")) for item in missing_axes)
        if final_axis_chunks and is_covered:
            status = "covered_final"
        elif final_axis_chunks:
            status = "final_unclaimed"
        elif candidate_chunks:
            status = "candidate_only"
        elif is_missing:
            status = "missing"
        else:
            status = "not_found"
        rows.append(
            {
                "axis": axis,
                "role": role,
                "blocking": bool(blocking),
                "status": status,
                "candidate_refs": _unique_strings(chunk.get("boi_reference", "") for chunk in candidate_chunks)[:8],
                "final_refs": _unique_strings(chunk.get("boi_reference", "") for chunk in final_axis_chunks)[:8],
                "candidate_chunk_ids": _unique_strings(chunk.get("chunk_id", "") for chunk in candidate_chunks)[:8],
                "final_chunk_ids": _unique_strings(chunk.get("chunk_id", "") for chunk in final_axis_chunks)[:8],
                "missing_reason": missing_reason,
            }
        )

    for facet in plan.facets:
        matching_missing = next(
            (item for item in missing_axes if _axes_match(facet.name, item.get("axis", ""))),
            None,
        )
        add_axis(
            facet.name,
            role=facet.role,
            blocking=facet.blocking,
            missing_reason=(matching_missing or {}).get("why_needed", ""),
        )
    for axis in covered_axes:
        add_axis(axis)
    for item in missing_axes:
        add_axis(
            str(item.get("axis", "")),
            role=_normalize_axis_role(item.get("role", "")),
            blocking=_normalize_blocking_flag(item.get("blocking", True), item.get("role", "")),
            missing_reason=str(item.get("why_needed", "")),
        )
    return rows


def _facet_from_missing_axis(item: dict) -> SearchFacet:
    prefix_context = " ".join(
        str(part or "")
        for part in (
            item.get("axis", ""),
            item.get("search_query", ""),
            item.get("why_needed", ""),
        )
    )
    return SearchFacet(
        name=_short(item.get("axis", ""), 90) or "Axe fiscal manquant",
        goal=_short(item.get("why_needed", ""), 180) or "Compléter la couverture documentaire",
        query=_short(item.get("search_query", ""), 220),
        prefix=_normalize_facet_prefix(item.get("bofip_prefix", ""), prefix_context),
        priority=1,
        expected_evidence=[_short(item.get("why_needed", ""), 120)] if item.get("why_needed") else [],
        role=_normalize_axis_role(item.get("role", "")),
        blocking=_normalize_blocking_flag(item.get("blocking", True), item.get("role", "")),
    )


def _candidate_refs_for_missing_axis(missing: dict, chunks: list[dict], route_log: list[dict]) -> list[str]:
    axis_query = " ".join(
        str(missing.get(key, ""))
        for key in ("axis", "search_query", "why_needed")
        if missing.get(key)
    )
    prefix = _normalize_facet_prefix(missing.get("bofip_prefix", ""), axis_query)
    candidates: dict[str, tuple[float, int]] = {}
    order = 0

    def normalize_ref(ref: object) -> str:
        value = str(ref or "").strip().upper()
        if not value:
            return ""
        if not value.startswith("BOI-"):
            return ""
        return value

    def add(ref: object, score: float) -> None:
        nonlocal order
        value = normalize_ref(ref)
        if not value:
            return
        if value not in candidates:
            candidates[value] = (score, order)
            order += 1
            return
        old_score, old_order = candidates[value]
        candidates[value] = (max(old_score, score), old_order)

    for ref in _exact_boi_references(axis_query):
        add(ref, 1000.0)

    for route in route_log:
        facet = route.get("facet", {}) if isinstance(route, dict) else {}
        route_axis_score = _route_axis_overlap_score(axis_query, route)
        for key, evidence_boost in (("stage2_refs", 6.0), ("stage1_refs", 3.0), ("selected_refs", 1.0)):
            for ref in route.get(key, []) or []:
                ref_score = float(_chunk_axis_overlap_score(axis_query, {"boi_reference": normalize_ref(ref)}))
                ref_matches_prefix = _ref_matches_prefix(str(ref), prefix) if prefix else False
                ref_shares_parent = _ref_shares_parent_prefix(str(ref), prefix) if prefix else False
                if not (route_axis_score > 0 or ref_score > 0 or ref_matches_prefix or ref_shares_parent):
                    continue
                route_score = (route_axis_score * 2.0) + ref_score + evidence_boost
                if ref_matches_prefix:
                    route_score += 0.5
                elif ref_shares_parent:
                    route_score += 0.75
                add(ref, route_score)

    for chunk in chunks:
        ref = str(chunk.get("boi_reference", ""))
        ref_matches_prefix = _ref_matches_prefix(ref, prefix) if prefix else False
        ref_shares_parent = _ref_shares_parent_prefix(ref, prefix) if prefix else False
        score = _chunk_axis_overlap_score(axis_query, chunk)
        if ref_matches_prefix:
            score += 2
        elif ref_shares_parent:
            score += 1
        if score > 0:
            add(chunk.get("boi_reference", ""), float(score))

    ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], item[1][1], item[0]))
    return _diversify_candidate_refs([ref for ref, _score_order in ranked], limit=6)


def _candidate_refs_from_retrieval_result(result) -> list[str]:
    pipeline_log = getattr(result, "pipeline_log", {}) or {}
    refs: list[str] = []
    if isinstance(pipeline_log, dict):
        refs.extend(pipeline_log.get("final_doc_refs", []) or [])
        refs.extend(pipeline_log.get("stage2_candidate_doc_refs", []) or [])
        refs.extend(pipeline_log.get("stage1_doc_refs", []) or [])
    refs.extend(getattr(hit, "boi_reference", "") for hit in getattr(result, "stage1_hits", []) or [])
    return _unique_strings(refs)[:8]


def _diversify_candidate_refs(refs: list[str], *, limit: int) -> list[str]:
    selected: list[str] = []
    seen_refs: set[str] = set()
    seen_branches: set[str] = set()

    for ref in refs:
        branch = _doc_branch_key(ref)
        if branch and branch in seen_branches:
            continue
        selected.append(ref)
        seen_refs.add(ref)
        if branch:
            seen_branches.add(branch)
        if len(selected) >= limit:
            return selected

    for ref in refs:
        if ref in seen_refs:
            continue
        selected.append(ref)
        seen_refs.add(ref)
        if len(selected) >= limit:
            break
    return selected


def _doc_branch_key(ref: str) -> str:
    parts = _prefix_parts_without_date(ref)
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return "-".join(parts)


def _exact_boi_references(value: str) -> list[str]:
    refs: list[str] = []
    for match in BOI_REFERENCE_RE.finditer(str(value or "")):
        ref = match.group(0).upper()
        if ref not in refs:
            refs.append(ref)
    return refs[:6]


def _chunks_for_source_review(chunks: list[dict], *, limit: int = 16) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()

    def add(chunk: dict) -> None:
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id and chunk_id in seen:
            return
        selected.append(chunk)
        if chunk_id:
            seen.add(chunk_id)

    for chunk in chunks:
        if chunk.get("retrieval_stage") in RETRIEVAL_RESCUE_STAGES:
            add(chunk)
    for chunk in chunks:
        add(chunk)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _route_axis_overlap_score(axis_query: str, route: dict) -> int:
    if not isinstance(route, dict):
        return 0
    facet = route.get("facet", {})
    if not isinstance(facet, dict):
        return 0
    parts: list[str] = []
    for key in ("name", "goal", "query", "search_query"):
        value = facet.get(key)
        if value:
            parts.append(str(value))
    evidence = facet.get("expected_evidence")
    if isinstance(evidence, list):
        parts.extend(str(item) for item in evidence if item)
    elif evidence:
        parts.append(str(evidence))
    route_text = " ".join(parts)
    return _chunk_axis_overlap_score(
        axis_query,
        {
            "title": route_text,
            "section_path": route_text,
            "text": route_text,
        },
    )


def _ref_matches_prefix(ref: str, prefix: str) -> bool:
    normalized = _normalize_prefix(prefix)
    if not normalized:
        return False
    value = str(ref or "").upper()
    return (
        value.startswith(normalized)
        or value.startswith(f"BOI-{normalized}")
        or value.startswith(f"BOI-RES-{normalized}")
    )


def _ref_shares_parent_prefix(ref: str, prefix: str) -> bool:
    ref_parts = _prefix_parts_without_date(ref)
    prefix_parts = _prefix_parts_without_date(prefix)
    if len(ref_parts) < 4 or len(prefix_parts) < 4:
        return False
    ref_parent = ref_parts[:-1]
    prefix_parent = prefix_parts[:-1]
    return len(prefix_parent) >= 3 and ref_parent == prefix_parent


def _prefix_parts_without_date(value: object) -> list[str]:
    normalized = _normalize_prefix(value)
    parts = normalized.split("-") if normalized else []
    if parts and re.fullmatch(r"\d{8}", parts[-1] or ""):
        parts = parts[:-1]
    return parts


def _prefixes_overlap(left: str, right: str) -> bool:
    l_norm = _normalize_prefix(left)
    r_norm = _normalize_prefix(right)
    return bool(l_norm and r_norm and (l_norm.startswith(r_norm) or r_norm.startswith(l_norm)))


def _chunk_axis_overlap_score(axis_query: str, chunk: dict) -> int:
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", _ascii_lower(axis_query))
        if len(token) >= 4
    }
    if not query_terms:
        return 0
    haystack = " ".join(
        str(part or "")
        for part in (
            chunk.get("boi_reference", ""),
            chunk.get("title", ""),
            chunk.get("section_path", ""),
            _short(chunk.get("text", ""), 300),
        )
    )
    text_terms = set(re.findall(r"[a-z0-9]+", _ascii_lower(haystack)))
    return len(query_terms & text_terms)


def _build_answer_question(question: str, plan: SearchPlan, source_review: dict) -> str:
    lines = [
        "QUESTION ORIGINALE:",
        question,
        "",
        "REFORMULATION FISCALE DE TRAVAIL:",
        plan.reformulated_question,
    ]
    if plan.facts:
        lines.extend(["", "FAITS UTILISATEUR A CONSERVER:"])
        lines.extend(f"- {fact}" for fact in plan.facts)
    if plan.ambiguities:
        lines.extend(["", "AMBIGUITES A SIGNALER SI ELLES EMPECHENT UN CALCUL:"])
        lines.extend(f"- {value}" for value in plan.ambiguities)
    if plan.facets:
        lines.extend(["", "AXES FISCAUX A TRAITER:"])
        lines.extend(
            f"- {facet.name} ({facet.role}, blocking={facet.blocking}): {facet.goal}"
            for facet in plan.facets
        )
    covered = source_review.get("covered_axes", []) or []
    non_blocking = source_review.get("non_blocking_axes", []) or []
    if covered:
        lines.extend(["", "AXES DOCUMENTES PAR LES SOURCES:"])
        lines.extend(f"- {axis}" for axis in covered)
    if non_blocking:
        lines.extend(["", "RESERVES NON BLOQUANTES A PLACER DANS LIMITS SI UTILE:"])
        lines.extend(f"- {item.get('axis', '')}: {item.get('why_needed', '')}" for item in non_blocking if isinstance(item, dict))
    lines.extend(
        [
            "",
            "CONSIGNE:",
            "Reponds a la question originale a partir des extraits fournis. Le reviewer documentaire a seulement servi a trouver plus de sources: il ne decide pas du statut final. Si une reponse principale est possible, classe supported et place les reserves dans limits. Utilise insufficient_evidence seulement si les extraits ne permettent pas de repondre a la question principale. N'utilise pas partial pour des reserves, hypotheses, options ou axes que tu penses incomplets. Si l'utilisateur demande un taux sur une base qui n'est pas la base taxable correcte, explique la distinction avec les sources disponibles. Ne calcule que ce que les extraits permettent de calculer.",
        ]
    )
    return "\n".join(lines)


def _clean_answer_status(answer: dict, plan: SearchPlan | None = None, source_review: dict | None = None) -> dict:
    if not isinstance(answer, dict):
        return {
            "answer_status": "insufficient_evidence",
            "conclusion": "Le modèle n'a pas retourné un JSON exploitable.",
            "axes_requis": [],
            "axes_couverts": [],
            "axes_manquants": ["réponse structurée"],
            "justification_bullets": [],
            "limits": "",
        }
    required_axes = [str(axis) for axis in answer.get("axes_requis", []) or [] if str(axis or "").strip()]
    nonblocking_plan_axes = [
        facet.name
        for facet in (plan.facets if plan else [])
        if not facet.blocking or facet.role in {"reserve", "alternative"}
    ]
    nonblocking_source_axes: list[str] = []
    if source_review:
        for item in (source_review.get("non_blocking_axes", []) or []) + (source_review.get("missing_axes", []) or []):
            if not isinstance(item, dict) or not item.get("axis"):
                continue
            role = _normalize_axis_role(item.get("role", ""))
            if not item.get("blocking", True) or role in {"reserve", "alternative"}:
                nonblocking_source_axes.append(str(item.get("axis", "")))
    nonblocking_axes = _unique_strings(nonblocking_plan_axes + nonblocking_source_axes)
    if nonblocking_axes:
        required_axes = [
            axis
            for axis in required_axes
            if not any(_axes_match(axis, nonblocking_axis) for nonblocking_axis in nonblocking_axes)
        ]
        answer["axes_requis"] = required_axes
    declared_missing = [
        str(axis)
        for axis in (answer.get("axes_manquants", []) or [])
        if str(axis or "").strip()
    ]
    if nonblocking_axes:
        declared_missing = [
            axis
            for axis in declared_missing
            if not any(_axes_match(axis, nonblocking_axis) for nonblocking_axis in nonblocking_axes)
        ]

    # A model cannot know missing fiscal axes with certainty. Its declared
    # missing axes are explanatory only; they must not veto a usable answer.
    # Source review is likewise advisory and only drives retrieval relaunches.
    if answer.get("answer_status") == "insufficient_evidence" or not _has_principal_answer_signal(answer):
        answer["axes_manquants"] = declared_missing or ["sources insuffisantes pour conclure complètement"]
        answer["answer_status"] = "insufficient_evidence"
    else:
        answer["answer_status"] = "supported"
        answer["axes_manquants"] = []
    return answer


def _has_principal_answer_signal(answer: dict) -> bool:
    conclusion = str(answer.get("conclusion", "") or "").strip()
    bullets = [item for item in answer.get("justification_bullets", []) or [] if str(item or "").strip()]
    covered_axes = [item for item in answer.get("axes_couverts", []) or [] if str(item or "").strip()]
    return bool(conclusion or bullets or covered_axes)


def _axes_match(left: str, right: str) -> bool:
    left_norm = _ascii_lower(left)
    right_norm = _ascii_lower(right)
    if not left_norm or not right_norm:
        return False
    if left_norm in right_norm or right_norm in left_norm:
        return True
    left_tokens = _axis_tokens(left_norm)
    right_tokens = _axis_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return False
    smaller, larger = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if smaller <= larger:
        return True
    overlap = left_tokens & right_tokens
    return len(overlap) >= 2 and len(overlap) >= min(len(left_tokens), len(right_tokens)) * 0.67


def _axis_tokens(value: str) -> set[str]:
    stopwords = {
        "avec",
        "dans",
        "pour",
        "sans",
        "selon",
        "cette",
        "celui",
        "celle",
        "entre",
        "ainsi",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _ascii_lower(value))
        if len(token) >= 4 and token not in stopwords
    }


def _compute_coverage(answer: dict) -> float:
    covered = answer.get("axes_couverts", []) or []
    required = answer.get("axes_requis", []) or []
    missing = answer.get("axes_manquants", []) or []
    denominator = max(len(required), len(covered) + len(missing), 1)
    coverage = min(1.0, len(covered) / denominator)
    status = answer.get("answer_status")
    if status == "partial":
        coverage = min(coverage, 0.75)
    elif status == "insufficient_evidence":
        coverage = min(coverage, 0.25)
    return round(coverage, 3)


def _format_facets(facets: list[SearchFacet]) -> str:
    return "\n".join(
        f"{facet.priority}. {facet.name} [{facet.prefix or 'sans préfixe'}; {facet.role}; blocking={facet.blocking}] - {facet.goal}"
        for facet in facets
    )


def _format_excluded_axes(excluded: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{item.get('axis', 'Axe')} - {item.get('reason', '')}".strip(" -")
        for item in excluded
        if item.get("axis") or item.get("reason")
    )


def _format_missing_axes(missing_axes: list[dict]) -> str:
    rows = []
    for item in missing_axes:
        if not isinstance(item, dict):
            continue
        rows.append(
            f"{item.get('axis', 'Axe')} [{item.get('bofip_prefix', '') or 'sans préfixe'}] - {item.get('search_query', '')}"
        )
    return "\n".join(rows)


def _source_labels(chunks: list[dict]) -> list[str]:
    return [
        f"{chunk.get('boi_reference', 'BOFiP')} - {_short(chunk.get('title', ''), 110)}"
        for chunk in chunks
    ]


def _facet_to_dict(facet: SearchFacet) -> dict:
    return {
        "name": facet.name,
        "goal": facet.goal,
        "query": facet.query,
        "prefix": facet.prefix,
        "priority": facet.priority,
        "expected_evidence": facet.expected_evidence,
        "role": facet.role,
        "blocking": facet.blocking,
    }


def _plan_to_dict(plan: SearchPlan) -> dict:
    return {
        "reformulated_question": plan.reformulated_question,
        "facts": plan.facts,
        "ambiguities": plan.ambiguities,
        "facets": [_facet_to_dict(facet) for facet in plan.facets],
        "excluded_axes": plan.excluded_axes,
    }


def _parse_json(raw: str) -> dict | None:
    candidates = [raw]
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL):
        candidate = match.group(1).strip()
        if candidate:
            candidates.append(candidate)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _chunks_from_result(result, *, facet: str = "") -> list[dict]:
    return [
        {
            "rank": i + 1,
            "boi_reference": c.boi_reference,
            "title": c.title,
            "publication_date": c.publication_date,
            "section_path": c.section_path,
            "text": c.text,
            "chunk_id": c.chunk_id,
            "facet": facet,
        }
        for i, c in enumerate(result.stage2_chunks)
    ]


def _sort_chunks(chunks: list[dict]) -> list[dict]:
    return sorted(chunks, key=lambda c: c.get("rank", 999))
