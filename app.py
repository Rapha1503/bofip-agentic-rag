"""BOFIP Agentic RAG — Streamlit application.
Multi-provider LLM support, self-evaluating retrieval loop, batch processing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# Inject project root into path so no PYTHONPATH env var is needed
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st
from openai import OpenAI

from bofip_agentic.agent_rag import AgenticRAG
from bofip_agentic.rag_runtime import RagRuntime
from bofip_agentic.providers import PROVIDERS


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _rewrite_query(query: str, client, model: str) -> tuple[str, list[str]]:
    system = (
        "Analyse cette question fiscale. Retourne UNIQUEMENT un JSON valide sans markdown:\n"
        '{"rewritten_query":"question reformulee en francais administratif formel",'
        '"facets":[{"name":"axe","query":"sous-requete"}]}\n'
        "Identifie les axes juridiques distincts (1 a 5). "
        "Noms: regle_de_fond, procedure, doctrine, garanties, sanctions, prescription."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
            temperature=0.0, max_tokens=400,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        return query, [query]

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        cleaned = re.sub(r"```(?:json)?\s*", "", content).replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return query, [query]

    rewritten = data.get("rewritten_query", query) or query
    facets = data.get("facets", [])
    facet_queries = [f.get("query", rewritten) for f in facets if f.get("query")]
    return rewritten, facet_queries if facet_queries else [rewritten]


def _parse_json_safe(content: str) -> dict | None:
    candidates = [content]
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL):
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


##### ─── Page config (must be first st call) ───
st.set_page_config(page_title="BOFIP Agentic RAG", layout="wide", initial_sidebar_state="expanded")

# ─── CSS ───
st.markdown("""
<style>
.stApp { background: #0d1117; }
section.main > div { padding-top: 1rem; }
div[data-testid="stTextArea"] textarea {
    background: #161b22 !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; color: #e6edf3 !important; font-size: 0.95rem !important;
    padding: 1rem !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(200,169,81,0.4) !important;
    box-shadow: 0 0 0 2px rgba(200,169,81,0.1) !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1a3a6b, #1e4d8c) !important;
    border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important;
    color: #e6edf3 !important; font-weight: 500 !important; padding: 0.6rem 2rem !important;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #1e4d8c, #2563a8) !important;
}
.result-card {
    background: #161b22; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;
}
.status-badge {
    display: inline-block; padding: 0.25rem 0.75rem; border-radius: 6px;
    font-size: 0.8rem; font-weight: 600; margin-bottom: 1rem;
}
.badge-supported { background: rgba(63,185,80,0.12); color: #3fb950; border: 1px solid rgba(63,185,80,0.2); }
.badge-partial  { background: rgba(210,153,34,0.12); color: #d29922; border: 1px solid rgba(210,153,34,0.2); }
.badge-insufficient { background: rgba(248,81,73,0.12); color: #f85149; border: 1px solid rgba(248,81,73,0.2); }
.conclusion-text { font-size: 1.05rem; line-height: 1.6; color: #e6edf3; margin: 1rem 0; }
.meta-row { display: flex; gap: 1.5rem; margin: 1rem 0; flex-wrap: wrap; }
.meta-item { font-size: 0.82rem; color: #8b949e; }
.meta-item strong { color: #c9d1d9; font-weight: 500; }
.source-item {
    background: #0d1117; border: 1px solid rgba(255,255,255,0.04);
    border-radius: 6px; padding: 0.75rem; margin: 0.5rem 0; font-size: 0.82rem;
}
.source-ref { color: #58a6ff; font-family: monospace; font-size: 0.78rem; }
.source-title { color: #e6edf3; font-weight: 500; margin-top: 0.25rem; }
.source-section { color: #8b949e; font-size: 0.75rem; margin-top: 0.25rem; }
.source-text { color: #c9d1d9; margin-top: 0.5rem; font-size: 0.8rem; line-height: 1.5; max-height: 120px; overflow-y: auto; }
.trace-step {
    border-left: 2px solid rgba(200,169,81,0.3); padding-left: 1rem; margin: 0.75rem 0;
}
.trace-step-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; }
.trace-step-value { color: #e6edf3; font-size: 0.85rem; }
.info-box {
    background: rgba(88,166,255,0.06); border: 1px solid rgba(88,166,255,0.1);
    border-radius: 8px; padding: 1rem; margin: 1rem 0; font-size: 0.85rem; color: #c9d1d9;
}
.footer-meta { font-size: 0.72rem; color: #484f58; text-align: center; margin-top: 2rem; }
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
[data-testid="stExpander"] {
    background: transparent !important; border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important; margin: 0.5rem 0 !important;
}
</style>
""", unsafe_allow_html=True)


def _get_runtime(device: str = "cuda"):
    return RagRuntime.from_local_corpus(corpus="commentary", device=device)


def _run_query(question, agent, llm_client, model, use_rewrite):
    cache_key = hashlib.md5((question + model + str(use_rewrite)).encode()).hexdigest()[:12]
    if "result_cache" not in st.session_state:
        st.session_state.result_cache = {}
    if cache_key in st.session_state.result_cache:
        return st.session_state.result_cache[cache_key]

    t0 = time.time()
    rewritten = question

    if use_rewrite:
        rewritten, _ = _rewrite_query(question, llm_client, model)

    query_for_agent = rewritten if use_rewrite else question
    agent_result = agent.run(query_for_agent)

    elapsed = round(time.time() - t0, 1)

    result = {
        "question": question,
        "rewritten": rewritten if rewritten != question else None,
        "parsed": {
            "answer_status": agent_result.get("answer_status", "partial"),
            "conclusion": agent_result.get("conclusion", ""),
            "axes_requis": agent_result.get("axes_requis", []),
            "axes_couverts": agent_result.get("axes_couverts", []),
            "axes_manquants": agent_result.get("axes_manquants", []),
            "justification_bullets": agent_result.get("justification_bullets", []),
            "limits": agent_result.get("limits", ""),
        },
        "chunks": agent_result.get("sources", []),
        "stage1": agent_result.get("sources", []),
        "llm_raw": json.dumps(agent_result, ensure_ascii=False, indent=2),
        "trace": agent_result.get("trace", []),
        "elapsed": elapsed,
        "iterations": agent_result.get("iterations", 1),
        "coverage": agent_result.get("coverage", 0),
        "total_s": agent_result.get("total_s", 0),
        "chunks_used": agent_result.get("chunks_used", 0),
    }

    st.session_state.result_cache[cache_key] = result
    return result


def _parse_json_safe(content: str) -> dict | None:
    candidates = [content]
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL):
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


# --- Sidebar -----------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    _load_env_file(PROJECT_ROOT / ".env.local")
    _load_env_file(PROJECT_ROOT / ".env")

    provider_id = st.selectbox("Fournisseur LLM", list(PROVIDERS.keys()), key="provider")
    provider = PROVIDERS[provider_id]

    saved_key = os.environ.get(provider["env_key"], "")
    api_key = st.text_input(
        f"Cle API ({provider['env_key']})",
        value=saved_key, type="password", key="api_key",
        help=f"Chargee depuis .env.local si disponible."
    )

    model = st.selectbox("Modele", provider["models"], key=f"model_{provider_id}")

    use_rewrite = st.checkbox("Reecriture de la question", value=True,
                              help="Reformule la question en vocabulaire fiscal + detection multi-axes.")

    st.divider()
    st.caption("Corpus: 5 666 documents BOFIP")
    st.caption("Embedding: E5-large (1024-dim)")
    st.caption("Reranker: bge-reranker-v2-m3")

    if st.button("Vider le cache"):
        st.session_state.result_cache = {}
        st.rerun()


# --- Header ------------------------------------------------------------
st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid rgba(255,255,255,0.06);">
    <div>
        <h1 style="font-size:1.75rem;font-weight:600;color:#f0f0f0;margin:0;">BOFIP Agentic RAG</h1>
        <span style="font-size:0.85rem;color:#8b949e;">Recherche fiscale augmentee · 5 666 documents · Analyse auto-evaluee</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Runtime init (cached) ---------------------------------------------
if not api_key:
    st.warning(f"Entrez une cle API **{provider['env_key']}** dans la barre laterale.")
    st.stop()


@st.cache_resource(show_spinner="Chargement du moteur de recherche (GPU)...")
def _cached_runtime(device):
    return _get_runtime(device)


try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

rt = _cached_runtime(device)
st.caption(f"Appareil: {'GPU' if device == 'cuda' else 'CPU'} | Pret")

extra_headers = {"x-api-key": api_key} if provider_id == "Anthropic" else {}
llm_client = OpenAI(api_key=api_key, base_url=provider["base_url"], **({"default_headers": extra_headers} if extra_headers else {}))

agent = AgenticRAG(rt, api_key=api_key, base_url=provider["base_url"], model=model, max_iterations=2, client=llm_client)

# --- Tabs --------------------------------------------------------------
tab1, tab2 = st.tabs(["Question unique", "Test par lot"])

with tab1:
    question = st.text_area(
        "Votre question",
        placeholder="Ex: Puis-je recuperer la TVA sur l'achat d'une voiture de tourisme pour mon entreprise ?",
        height=100,
        label_visibility="collapsed",
        key="question_input",
    )

    run_btn = st.button("Analyser", use_container_width=True, disabled=not question.strip(), key="run_btn")

    if run_btn:
        t0 = time.time()
        with st.spinner("Analyse en cours..."):
            result = _run_query(question, agent, llm_client, model, use_rewrite)

        st.markdown("<hr>", unsafe_allow_html=True)

        if result.get("rewritten"):
            st.caption(f"Reformulee: {result['rewritten'][:200]}")

        if result.get("error"):
            st.error(result["error"])
        else:
            parsed = result.get("parsed", {})
            status = parsed.get("answer_status", "partial")
            status_labels = {"supported": "Reponse complete", "partial": "Reponse partielle",
                             "insufficient_evidence": "Informations insuffisantes"}
            badge_class = {"supported": "badge-supported", "partial": "badge-partial",
                           "insufficient_evidence": "badge-insufficient"}

            coverage_val = result.get("coverage", 0)
            elapsed = result.get("total_s", result.get("elapsed", 0))
            iters = result.get("iterations", 1)
            chunks_n = result.get("chunks_used", len(result.get("chunks", [])))

            st.markdown(f"""
            <div class="result-card">
                <span class="status-badge {badge_class.get(status, 'badge-partial')}">{status_labels.get(status, status)}</span>
                <div class="conclusion-text">{parsed.get("conclusion", "")}</div>
                <div class="meta-row">
                    <div class="meta-item">Couverture <strong>{coverage_val:.0%}</strong></div>
                    <div class="meta-item">Extraits <strong>{chunks_n}</strong></div>
                    <div class="meta-item">Iterations <strong>{iters}</strong></div>
                    <div class="meta-item">Temps <strong>{elapsed}s</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detailed analysis
            with st.expander("Analyse detaillee", expanded=False):
                for a in parsed.get("axes_requis", []):
                    icon = "+" if a in parsed.get("axes_couverts", []) else "-"
                    st.markdown(f"{icon} {a}")
                for b in parsed.get("justification_bullets", []):
                    st.markdown(f"- {b}")
                if parsed.get("axes_manquants"):
                    st.markdown("**Axes non couverts:**")
                    for a in parsed["axes_manquants"]:
                        st.markdown(f"- {a}")
                if parsed.get("limits"):
                    st.caption(parsed["limits"])

            # Sources
            with st.expander("Sources consultees", expanded=False):
                for i, s in enumerate(result.get("chunks", [])[:8]):
                    ref = s.get("boi_reference", "")
                    title = s.get("title", "")
                    section = s.get("section_path", "")
                    text = s.get("text", "")
                    if len(text) > 500:
                        text = text[:500] + "..."
                    st.markdown(f"""
                    <div class="source-item">
                        <div class="source-ref">[{i+1}] {ref}</div>
                        <div class="source-title">{title}</div>
                        <div class="source-section">{section}</div>
                        <div class="source-text">{text}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Trace
            with st.expander("Trace agent (pipeline complet)", expanded=True):
                trace = result.get("trace", [])
                if trace:
                    for t in trace:
                        it = t.get("iteration", "?")
                        sts = t.get("answer_status", "?")
                        docs = t.get("docs_found", 0)
                        chks = t.get("chunks_found", 0)
                        new_c = t.get("chunks_new", 0)
                        tot_c = t.get("chunks_total", 0)
                        ret_s = t.get("retrieve_s", 0)
                        ans_s = t.get("answer_s", 0)
                        axes_r = t.get("axes_requis", [])
                        axes_c = t.get("axes_couverts", [])
                        axes_m = t.get("axes_manquants", [])

                        st.markdown(f"**Iteration {it}** -- Statut: `{sts}` | "
                                    f"Retrieval: {ret_s}s, Answer: {ans_s}s")
                        st.caption(f"Documents retrouves: {docs} | Extraits: {chks}"
                                   f" | Nouveaux: {new_c} | Total cumule: {tot_c}")

                        if axes_r:
                            cols = st.columns(3)
                            with cols[0]:
                                st.caption("Axes requis:")
                                for a in axes_r:
                                    st.markdown(f"- {a}")
                            if axes_c:
                                with cols[1]:
                                    st.caption("Axes couverts:")
                                    for a in axes_c:
                                        st.markdown(f"- {a}")
                            if axes_m:
                                with cols[2]:
                                    st.caption("Axes manquants:")
                                    for a in axes_m:
                                        st.markdown(f"- {a}")

                        if t.get("reformulated_query"):
                            st.info(f"Reformulation: {t['reformulated_query'][:300]}")
                else:
                    st.caption("Aucune trace disponible.")

            # Debug: full agent output
            with st.expander("Debug: reponse agent brute (JSON)", expanded=False):
                st.code(result.get("llm_raw", "")[:5000], language="json")


with tab2:
    st.caption("Collez plusieurs questions (une par ligne vide)")
    batch_text = st.text_area("Questions", height=120, key="batch_input",
                              placeholder="Quel taux de TVA pour une pompe a chaleur ?\n\nComment sont imposes les gains de cession de valeurs mobilieres ?")

    if st.button("Lancer le lot", type="primary", disabled=not batch_text.strip(), key="batch_btn"):
        queries = [q.strip() for q in batch_text.strip().split("\n\n") if q.strip()]
        if queries:
            progress = st.progress(0)
            status_text = st.empty()
            all_results = []
            for i, q in enumerate(queries):
                status_text.text(f"[{i+1}/{len(queries)}] {q[:80]}...")
                progress.progress((i + 1) / len(queries))
                all_results.append(_run_query(q, agent, llm_client, model, use_rewrite))
            progress.empty()
            status_text.empty()

            st.markdown("### Resume")
            rows = []
            for res in all_results:
                parsed = res.get("parsed")
                sts = parsed.get("answer_status", "error") if parsed else "error"
                conc = (parsed.get("conclusion", "")[:80] if parsed else str(res.get("error", "")))
                rows.append({"Question": res["question"][:80], "Statut": sts, "Reponse": conc})
            st.dataframe(rows, use_container_width=True, hide_index=True)

            for i, res in enumerate(all_results):
                st.markdown("---")
                with st.expander(f"Q{i+1}: {res['question'][:100]}", expanded=False):
                    if res.get("error"):
                        st.error(res["error"])
                    else:
                        parsed = res.get("parsed", {})
                        st.markdown(f"**Statut:** {parsed.get('answer_status', '?')}")
                        st.markdown(f"> {parsed.get('conclusion', '')}")
                        for b in parsed.get("justification_bullets", []):
                            st.markdown(f"- {b}")
                        st.caption(parsed.get("limits", ""))

# --- Footer ------------------------------------------------------------
st.markdown(f"""
<div class="info-box">
    Ce moteur utilise un agent qui s'auto-evalue. Il detecte les axes fiscaux manquants
    et reformule automatiquement la recherche en vocabulaire BOFIP technique.
</div>
<div class="footer-meta">
    BOFIP Agentic RAG · 5 666 documents · E5-large + bge-reranker-v2-m3 · {provider_id}
</div>
""", unsafe_allow_html=True)
