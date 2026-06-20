"""BOFiP Agentic RAG - Streamlit app."""
from __future__ import annotations
import html
import hashlib, json, logging, os, re, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import streamlit as st
from openai import OpenAI
from bofip_cleanroom.artifact_download import (
    download_missing_runtime_artifacts,
    should_auto_download_artifacts,
    validate_runtime_artifacts,
)
from bofip_cleanroom.env_utils import load_default_env_files
from bofip_cleanroom.prompt_utils import build_prompt

# Optional file logging. Disabled by default for public/demo runs because
# queries can contain sensitive facts.
LOG_DIR = PROJECT_ROOT / "data" / "logs"
ENABLE_PIPELINE_LOG = (
    not os.environ.get("SPACE_ID")
    and os.environ.get("BOFIP_PIPELINE_LOG", "").strip().lower() in {"1", "true", "yes"}
)
if ENABLE_PIPELINE_LOG:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

_pipeline_logger = logging.getLogger("bofip_pipeline")
_pipeline_logger.setLevel(logging.DEBUG)
if ENABLE_PIPELINE_LOG and not _pipeline_logger.handlers:
    _fh = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    _pipeline_logger.addHandler(_fh)
_pipeline_logger.propagate = False

def _log(step: str, data: dict):
    if not ENABLE_PIPELINE_LOG:
        return
    _pipeline_logger.info(f"[{step}] {json.dumps(data, ensure_ascii=False, default=str)[:2000]}")


PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-v4-flash",
        "env_key": "DEEPSEEK_API_KEY",
        "note": "DeepSeek v4 au 20/06/2026. deepseek-chat/reasoner restent en aliases compatibilite jusqu'au 24/07/2026.",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.4-mini", "gpt-5.5", "gpt-5.4", "gpt-5.4-nano"],
        "default_model": "gpt-5.4-mini",
        "env_key": "OPENAI_API_KEY",
        "note": "OpenAI frontier models au 20/06/2026. Mini par defaut pour limiter le cout.",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-medium-latest", "mistral-small-latest", "mistral-large-latest", "magistral-medium-latest"],
        "default_model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
        "note": "Aliases Mistral latest pour suivre Medium 3.5 / Small 4 sans figer une version retiree.",
    },
    "Google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "default_model": "gemini-3.5-flash",
        "env_key": "GEMINI_API_KEY",
        "note": "Modele OpenAI-compatible recommande par la doc Gemini au 20/06/2026.",
    },
}

REQUIRED_RUNTIME_PATHS = [
    PROJECT_ROOT / "data" / "interim" / "raw_docs_sample_5666.jsonl",
    PROJECT_ROOT / "data" / "interim" / "chunks_section_window_sample_5666.jsonl",
    PROJECT_ROOT / "data" / "interim" / "doc_dense_cache_5666_sections_firstpara_e5large.npy",
    PROJECT_ROOT / "data" / "interim" / "chunk_dense_cache_5666_full_e5large.npy",
]
E5_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "intfloat--multilingual-e5-large"
RERANKER_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "BAAI--bge-reranker-v2-m3"
RUNNING_ON_SPACE = bool(os.environ.get("SPACE_ID"))
SHOW_DEBUG_DETAILS = (
    not RUNNING_ON_SPACE
    or os.environ.get("BOFIP_SHOW_DEBUG", "").strip().lower() in {"1", "true", "yes"}
)


def _missing_runtime_paths() -> list[Path]:
    return [path for path in REQUIRED_RUNTIME_PATHS if not path.exists()]


st.set_page_config(
    page_title="BOFiP Agentic RAG",
    page_icon="§",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #18212f;
        --muted: #647184;
        --line: #d8ddd4;
        --paper: #fffef9;
        --soft: #f3f5f0;
        --navy: #14213d;
        --forest: #2f6f58;
        --saffron: #c58b22;
        --burgundy: #8c2f39;
        --blue: #245c9f;
        --red: #9f1d2f;
      }

      [data-testid="stAppViewContainer"] {
        background:
          linear-gradient(180deg, rgba(255, 254, 249, .96), rgba(243, 245, 240, .96)),
          repeating-linear-gradient(90deg, rgba(20,33,61,.035) 0, rgba(20,33,61,.035) 1px, transparent 1px, transparent 74px);
      }

      .block-container {
        max-width: 1220px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        color: var(--ink);
      }

      [data-testid="stSidebar"] {
        background: #fbfaf4;
        border-right: 1px solid var(--line);
      }

      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {
        color: var(--ink);
      }

      .bofip-hero {
        border-radius: 8px;
        background: var(--navy);
        color: #fffef9;
        padding: 26px 30px;
        border: 1px solid #20304f;
        border-left: 7px solid var(--saffron);
        box-shadow: 0 16px 44px rgba(20, 33, 61, .14);
        margin-bottom: 18px;
      }

      .bofip-hero h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.65rem;
        line-height: 1.04;
        margin: 0 0 10px 0;
        letter-spacing: 0;
        color: #fffef9;
        font-weight: 700;
      }

      .bofip-hero p {
        margin: 0;
        max-width: 780px;
        color: rgba(255,254,249,.84);
        font-size: 1rem;
      }

      .hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
      }

      .chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        border-radius: 999px;
        border: 1px solid rgba(255,254,249,.28);
        background: rgba(255,254,249,.08);
        padding: 7px 11px;
        color: #fffef9;
        font-size: .86rem;
        white-space: nowrap;
      }

      .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 10px 0 18px;
      }

      .metric-card,
      .workspace-panel,
      .answer-panel,
      .source-card,
      .notice-panel {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(20, 33, 61, .06);
      }

      .metric-card {
        padding: 15px 16px;
        min-height: 92px;
      }

      .metric-card strong {
        display: block;
        color: var(--navy);
        font-size: 1.12rem;
        line-height: 1.1;
      }

      .metric-card span {
        display: block;
        margin-top: 8px;
        color: var(--muted);
        font-size: .86rem;
      }

      .workspace-panel {
        padding: 18px;
        margin-top: 6px;
      }

      .section-kicker {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 6px;
      }

      .section-title {
        color: var(--ink);
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 12px;
      }

      .status-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
      }

      .status-supported {
        color: #174d3a;
        background: #e2f2eb;
        border: 1px solid #b9dacb;
      }

      .status-partial {
        color: #81520d;
        background: #fff4d8;
        border: 1px solid #e8c36c;
      }

      .status-insufficient,
      .status-error {
        color: #991b1b;
        background: #fee2e2;
        border: 1px solid #fecaca;
      }

      .answer-panel {
        padding: 18px;
        margin: 16px 0;
      }

      .answer-panel h3 {
        color: var(--ink);
        margin: 10px 0 8px;
        font-size: 1.08rem;
      }

      .answer-panel p,
      .answer-panel li {
        color: #243149;
        line-height: 1.55;
      }

      .answer-panel blockquote {
        border-left: 4px solid var(--teal);
        margin: 10px 0;
        padding: 7px 0 7px 14px;
        color: #243149;
        background: #f3f8f5;
      }

      .source-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }

      .source-card {
        padding: 14px;
        min-height: 170px;
      }

      .source-card .ref {
        color: var(--forest);
        font-weight: 800;
        font-size: .88rem;
      }

      .source-card h4 {
        color: var(--ink);
        margin: 7px 0;
        font-size: .98rem;
        line-height: 1.28;
      }

      .source-card .path {
        color: var(--muted);
        font-size: .78rem;
        margin-bottom: 8px;
      }

      .source-card p {
        color: #31405b;
        font-size: .88rem;
        line-height: 1.45;
      }

      .notice-panel {
        padding: 16px 18px;
        border-left: 4px solid var(--saffron);
      }

      .app-footer {
        color: var(--muted);
        font-size: .83rem;
        margin-top: 26px;
        padding-top: 16px;
        border-top: 1px solid var(--line);
      }

      div[data-testid="stButton"] > button {
        border-radius: 7px;
        font-weight: 700;
      }

      @media (max-width: 840px) {
        .bofip-hero { padding: 22px; }
        .bofip-hero h1 { font-size: 2.05rem; }
        .metric-grid,
        .source-grid { grid-template-columns: 1fr; }
        .chip { white-space: normal; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_META = {
    "SUPPORTED": ("Preuve suffisante", "status-supported"),
    "PARTIAL": ("Preuve partielle", "status-partial"),
    "INSUFFICIENT_EVIDENCE": ("Preuve insuffisante", "status-insufficient"),
    "PARSE_ERROR": ("Sortie invalide", "status-error"),
}


def _escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _truncate(value: object, limit: int = 380) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _status_meta(status: object) -> tuple[str, str, str]:
    normalized = str(status or "INSUFFICIENT_EVIDENCE").upper()
    label, css_class = STATUS_META.get(normalized, (normalized.replace("_", " ").title(), "status-error"))
    return normalized, label, css_class

@st.cache_resource(show_spinner="Chargement du runtime full corpus...")
def get_runtime(load_reranker: bool, reranker_model: str | None):
    import torch
    from bofip_cleanroom.rag_runtime import RagRuntime

    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc_model = str(E5_MODEL_PATH) if E5_MODEL_PATH.exists() else "intfloat/multilingual-e5-large"
    kwargs = {
        "corpus": "commentary",
        "device": device,
        "doc_model": doc_model,
        "chunk_model": doc_model,
        "load_reranker": load_reranker,
    }
    if reranker_model:
        kwargs["reranker_model"] = reranker_model
    return RagRuntime.from_local_corpus(**kwargs)

def rewrite_query(query, client, model):
    """Rewrite query + optionally detect legal facets. Returns (rewritten_str, facet_queries_list)."""
    system = (
        "Analyse cette question fiscale. Retourne UNIQUEMENT un JSON valide sans markdown ni commentaire:\n"
        '{"rewritten_query":"question reformulee en francais administratif formel",'
        '"facets":[{"name":"axe","query":"sous-requete pour cet axe"}]}\n'
        "Identifie les axes juridiques distincts (1 a 5). "
        "Si question simple, 1 seul facet. "
        "Noms de facets possibles: regle_de_fond, procedure, doctrine, garanties, sanctions, prescription."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":query}],
        temperature=0.0, max_tokens=400,
        response_format={"type":"json_object"},
    )
    content = (resp.choices[0].message.content or "").strip()
    # Robust JSON extraction
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Strip markdown code blocks
        cleaned = re.sub(r"```(?:json)?\s*", "", content).replace("```", "").strip()
        # Find first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end+1]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e2:
            _log("REWRITE_PARSE_ERROR", {"raw": content[:200], "error": str(e2)})
            return query, [query]
    rewritten = data.get("rewritten_query", query) or query
    facets = data.get("facets", [])
    facet_queries = [f.get("query", rewritten) for f in facets if f.get("query")]
    return rewritten, facet_queries if facet_queries else [rewritten]


def _detect_multi_axis(query: str) -> bool:
    """Heuristic: does this query need multi-facet retrieval?"""
    signals = 0
    qt = query.lower()
    # Multiple clauses
    if query.count("?") >= 2 or query.count(",") >= 2:
        signals += 1
    # Procedure/control keywords (with conjugations)
    if any(w in qt for w in ("procédure", "redressement", "redresser", "contrôle", "vérificateur",
                              "vérification", "garantie", "protéger", "sanction", "délai",
                              "prescription", "recours", "réclamation", "opposabilité",
                              "opposable", "doctrine", "personnel", "excessif")):
        signals += 1
    # Source references
    if any(w in qt for w in ("cgi", "lpf", "bofip", "article", "textes")):
        signals += 1
    # Length
    if len(query) > 80:
        signals += 1
    return signals >= 2

def call_llm(prompt, client, model):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"Tu es un assistant fiscal prudent. Schema JSON strict."},
                  {"role":"user","content":prompt}],
        temperature=0.0, max_tokens=2800,
        response_format={"type":"json_object"},
    )
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return {"raw":content,"ptokens":getattr(usage,"prompt_tokens",None) if usage else None,
            "ctokens":getattr(usage,"completion_tokens",None) if usage else None}

def render_answer(parsed):
    normalized, label, css_class = _status_meta(parsed.get("answer_status"))
    conclusion = _escape(parsed.get("conclusion", ""))
    bullets = parsed.get("justification_bullets", []) or []
    limits = _escape(parsed.get("limits", ""))

    bullet_html = "".join(f"<li>{_escape(item)}</li>" for item in bullets)
    if not bullet_html:
        bullet_html = "<li>Aucune justification detaillee n'a ete retournee.</li>"

    st.markdown(
        f"""
        <div class="answer-panel">
          <span class="status-pill {css_class}">{_escape(label)}</span>
          <h3>Conclusion</h3>
          <blockquote>{conclusion}</blockquote>
          <h3>Raisonnement</h3>
          <ul>{bullet_html}</ul>
          <p><strong>Limites:</strong> {limits or "Non precisees."}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    axes = {
        "Axes requis": parsed.get("axes_requis", []) or [],
        "Axes couverts": parsed.get("axes_couverts", []) or [],
        "Axes manquants": parsed.get("axes_manquants", []) or [],
    }
    if any(axes.values()):
        with st.expander("Couverture juridique", expanded=normalized != "SUPPORTED"):
            cols = st.columns(3)
            for col, (title, values) in zip(cols, axes.items()):
                col.markdown(f"**{title}**")
                if values:
                    for value in values:
                        col.markdown(f"- {_escape(value)}")
                else:
                    col.caption("Non renseigne")


def process_query(query, rt, client, llm_model, use_rewrite, use_reranker):
    results = {"query":query,"error":None}

    # Cache check
    cache_key = hashlib.md5((query + llm_model + str(use_rewrite)).encode()).hexdigest()[:12]
    if "result_cache" not in st.session_state:
        st.session_state.result_cache = {}
    if cache_key in st.session_state.result_cache:
        cached = st.session_state.result_cache[cache_key]
        _log("CACHE_HIT", {"key": cache_key, "query": query[:80]})
        return cached

    t0 = time.time()
    _log("QUERY_START", {"query": query[:200], "model": llm_model, "rewrite": use_rewrite})
    # Rewrite + optional facets
    if use_rewrite:
        try:
            rewritten, facet_queries = rewrite_query(query, client, llm_model)
        except Exception as e:
            return {**results,"error":f"Erreur réécriture: {e}"}
    else:
        rewritten, facet_queries = query, [query]
    # Auto-detect multi-axis if rewrite didn't produce facets
    if len(facet_queries) <= 1 and _detect_multi_axis(query):
        facet_queries = [rewritten, query]
    results["rewritten"] = rewritten
    results["facet_queries"] = facet_queries
    # Computation-aware: inject taux/rate sub-query
    _COMPUTE_WORDS = {"calculer","montant","intérêt","amende","majoration","taux","pourcentage",
                       "barème","plafond","pénalité","intérêts","somme","quel est le",
                       "combien","due","dû","dus"}
    if any(w in query.lower() for w in _COMPUTE_WORDS):
        compute_q = (rewritten + " taux pourcentage applicable").strip()
        if compute_q not in facet_queries:
            facet_queries.append(compute_q)
            _log("COMPUTE_FACET", {"added": compute_q[:120]})
    # Multi-component sub-queries (only if under 3 total facets)
    _COMPONENT_PAIRS = [("intérêt","intérêt de retard taux calcul"),
                        ("majoration","majoration pourcentage taux calcul"),
                        ("amende","amende montant calcul"),
                        ("pénalité","pénalité taux calcul")]
    qt = query.lower()
    for keyword, subq in _COMPONENT_PAIRS:
        if len(facet_queries) >= 3:
            break
        if keyword in qt and subq not in facet_queries:
            facet_queries.append(subq)
    _log("REWRITE", {"original": query[:120], "rewritten": rewritten[:120], "facets": len(facet_queries)})
    # Retrieval — per facet, merge with diversity
    all_chunks_raw = []; all_stage1 = []; seen_docs = set(); main_log = {}
    for fq in facet_queries:
        try:
            res = rt.retrieve(fq, top_docs=5, use_reranker=use_reranker)
            for h in res.stage1_hits:
                if h.boi_reference not in seen_docs:
                    all_stage1.append(h); seen_docs.add(h.boi_reference)
            for c in res.stage2_chunks: all_chunks_raw.append(c)
            main_log = getattr(res, "pipeline_log", {})
        except Exception as e:
            return {**results,"error":f"Erreur retrieval: {e}"}
    # Post-merge diversity: sort by score, then max 3 chunks per document
    all_chunks_raw.sort(key=lambda c: float(getattr(c, "score", 0)), reverse=True)
    merged = []
    doc_counts = {}
    for c in all_chunks_raw:
        d = c.boi_reference
        doc_counts[d] = doc_counts.get(d, 0) + 1
        if doc_counts[d] <= 3:
            merged.append(c)
    all_chunks_raw = merged
    # Deduplicate by chunk_id (can happen with multi-facet)
    seen_ids = set()
    deduped = []
    for c in all_chunks_raw:
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            deduped.append(c)
    all_chunks_raw = deduped
    results["stage1"] = all_stage1[:8]
    # Compute diversity log from merged result (not single facet)
    from collections import Counter
    final_docs = [c.boi_reference for c in all_chunks_raw]
    doc_dist = Counter(final_docs)
    stage1_refs = [h.boi_reference for h in all_stage1]
    merged_log = dict(main_log)
    merged_log.update({
        "unique_docs_final": len(set(final_docs)),
        "max_chunks_per_doc": max(doc_dist.values()) if doc_dist else 0,
        "doc_distribution_final": {k[:30]: v for k, v in doc_dist.items()},
        "stage2_candidates": len(all_chunks_raw),
        "stage1_docs_dropped": [r for r in stage1_refs if r not in set(final_docs)],
        "facets_used": len(facet_queries),
    })
    results["pipeline_log"] = merged_log
    _log("RETRIEVAL", {"stage1_docs": [h.boi_reference for h in all_stage1[:8]], 
          "merged_chunks": len(all_chunks_raw), "pipeline_log": main_log})
    chunks = [{"rank":idx+1,"boi_reference":c.boi_reference,"title":c.title,
               "publication_date":c.publication_date,"section_path":c.section_path,
               "text":c.text,"chunk_id":c.chunk_id,"score":float(getattr(c,"score",0))}
              for idx,c in enumerate(all_chunks_raw[:8])]
    results["chunks"] = chunks
    if not chunks:
        results["parsed"] = {"answer_status":"insufficient_evidence","conclusion":"Aucun extrait trouvé.",
                              "justification_bullets":["La recherche n'a retourné aucun résultat."],
                              "limits":"Aucune source disponible.","axes_requis":[],"axes_couverts":[],"axes_manquants":[]}
        return results
    # LLM
    prompt = build_prompt(query, chunks)
    results["prompt"] = prompt
    try:
        llm_r = call_llm(prompt, client, llm_model)
    except Exception as e:
        return {**results,"error":f"Erreur LLM: {e}"}
    results["llm_raw"] = llm_r["raw"]
    results["ptokens"] = llm_r["ptokens"]
    results["ctokens"] = llm_r["ctokens"]
    _log("LLM_DONE", {"ptokens": llm_r["ptokens"], "ctokens": llm_r["ctokens"], 
          "raw_len": len(llm_r["raw"]), "valid_json": True})
    try:
        parsed = json.loads(llm_r["raw"])
    except json.JSONDecodeError:
        parsed = None
    results["parsed"] = parsed
    elapsed = time.time() - t0
    _log("QUERY_DONE", {"cache_key": cache_key, "elapsed_s": round(elapsed, 1),
          "unique_docs": main_log.get("unique_docs_final", "?"), "status": parsed.get("answer_status", "?") if parsed else "parse_error"})
    st.session_state.result_cache[cache_key] = results
    return results

def _source_card_html(chunk: dict) -> str:
    title = _escape(chunk.get("title", "Sans titre"))
    ref = _escape(chunk.get("boi_reference", "BOFiP"))
    path = _escape(_truncate(chunk.get("section_path", ""), 120))
    publication_date = _escape(chunk.get("publication_date") or "date non renseignee")
    excerpt = _escape(_truncate(chunk.get("text", ""), 430))
    score = float(chunk.get("score", 0) or 0)
    return f"""
    <div class="source-card">
      <div class="ref">#{chunk.get("rank", "?")} - {ref} - score {score:.3f}</div>
      <h4>{title}</h4>
      <div class="path">{publication_date} - {path}</div>
      <p>{excerpt}</p>
    </div>
    """


def render_source_cards(chunks: list[dict]):
    for index in range(0, min(len(chunks), 6), 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            card_index = index + offset
            if card_index < min(len(chunks), 6):
                with col:
                    st.markdown(_source_card_html(chunks[card_index]), unsafe_allow_html=True)


def display_results(results):
    if results.get("error"):
        st.error(results["error"])
        return

    st.markdown(
        f"""
        <div class="notice-panel">
          <div class="section-kicker">Question analysee</div>
          <strong>{_escape(results.get("query", ""))}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rewritten = results.get("rewritten")
    if rewritten and rewritten != results.get("query"):
        with st.expander("Reformulation utilisee", expanded=False):
            st.write(rewritten)
            facets = results.get("facet_queries", []) or []
            if len(facets) > 1:
                st.write("Facettes de recherche:")
                for facet in facets:
                    st.markdown(f"- {facet}")

    parsed = results.get("parsed")
    if parsed:
        render_answer(parsed)
    else:
        st.warning("Le modele n'a pas retourne un JSON exploitable.")
        st.code(results.get("llm_raw", "")[:1200], language="json")

    chunks = results.get("chunks", []) or []
    st.markdown('<div class="section-kicker">Sources retenues</div>', unsafe_allow_html=True)
    if chunks:
        render_source_cards(chunks)
    else:
        st.info("Aucun passage source n'a ete retenu.")

    with st.expander(f"Documents candidats stage 1 ({len(results.get('stage1', []))})", expanded=False):
        rows = [
            {"rang": h.rank, "score": f"{h.score:.4f}", "reference": h.boi_reference, "titre": h.title[:140]}
            for h in results.get("stage1", [])
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Aucun document candidat a afficher.")

    with st.expander("Trace technique", expanded=False):
        plog = results.get("pipeline_log", {}) or {}
        if plog:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("docs uniques", plog.get("unique_docs_final", "?"))
            c2.metric("max chunks/doc", plog.get("max_chunks_per_doc", "?"))
            c3.metric("candidats S2", plog.get("stage2_candidates", "?"))
            c4.metric("facettes", plog.get("facets_used", "?"))
            dist = plog.get("doc_distribution_final", {})
            if dist:
                st.caption("Distribution finale: " + json.dumps(dist, ensure_ascii=False))
        st.caption(f"Prompt tokens: {results.get('ptokens', '?')} | Completion tokens: {results.get('ctokens', '?')}")
        if SHOW_DEBUG_DETAILS:
            trace_tabs = st.tabs(["Chunks", "Prompt", "JSON brut"])
            chunk_container = trace_tabs[0]
        else:
            st.caption("Prompt et JSON brut masques en demo publique. Activez BOFIP_SHOW_DEBUG=1 pour audit local.")
            chunk_container = st.container()

        with chunk_container:
            for chunk in chunks:
                st.markdown(f"**[{chunk['rank']}] {chunk['boi_reference']}** - {chunk['score']:.4f}")
                st.caption(chunk.get("section_path", ""))
                st.write(_truncate(chunk.get("text", ""), 700))

        if SHOW_DEBUG_DETAILS:
            with trace_tabs[1]:
                st.code(results.get("prompt", ""), language="text")
            with trace_tabs[2]:
                st.code(results.get("llm_raw", ""), language="json")


def render_hero(provider_name: str, model_name: str, use_reranker: bool):
    reranker_chip = '<span class="chip">reranking qualite active</span>' if use_reranker else ""
    st.markdown(
        f"""
        <div class="bofip-hero">
          <h1>BOFiP Agentic RAG</h1>
          <p>Assistant de recherche dans la doctrine fiscale BOFiP: retrieval hybride, citations controlees et statut de couverture.</p>
          <div class="hero-chips">
            <span class="chip">5 666 documents BOFiP commentaire</span>
            <span class="chip">Corpus observe jusqu'au 28/01/2026</span>
            <span class="chip">{_escape(provider_name)} - {_escape(model_name)}</span>
            {reranker_chip}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics():
    st.markdown(
        """
        <div class="metric-grid">
          <div class="metric-card"><strong>Corpus BOFiP</strong><span>5 666 documents commentaire, 66 289 passages indexes, fraicheur observee: 28/01/2026.</span></div>
          <div class="metric-card"><strong>Retrieval hybride</strong><span>BM25, embeddings E5 et fusion RRF pour remonter documents puis passages.</span></div>
          <div class="metric-card"><strong>Reponse citee</strong><span>Conclusion, limites et extraits sources visibles avant interpretation.</span></div>
          <div class="metric-card"><strong>Cle utilisateur</strong><span>La demo utilise la cle API saisie dans la session, sans stockage applicatif.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_missing_key(provider: dict):
    st.markdown(
        f"""
        <div class="notice-panel">
          <div class="section-kicker">Configuration requise</div>
          <strong>Ajoutez une cle {provider['env_key']} dans la barre laterale pour lancer une analyse.</strong>
          <p>La demo hebergee transmet la question et la cle au serveur Streamlit puis au fournisseur LLM choisi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# UI
load_default_env_files()

with st.sidebar:
    st.markdown("### Parametres")
    provider_id = st.selectbox("Fournisseur LLM", list(PROVIDERS.keys()), key="provider_select")
    provider = PROVIDERS[provider_id]

    api_key = st.text_input(
        f"Cle API ({provider['env_key']})",
        value="" if RUNNING_ON_SPACE else os.environ.get(provider["env_key"], ""),
        type="password",
        key="api_key_input",
        help="Sur HF, le champ reste vide pour forcer une cle utilisateur. En local, .env.local peut pre-remplir.",
    )

    model_options = provider["models"]
    default_model = provider["default_model"]
    default_index = model_options.index(default_model) if default_model in model_options else 0
    model = st.selectbox(
        "Modele",
        model_options,
        index=default_index,
        key=f"model_{provider_id}_select",
        help="Liste limitee aux modeles configures pour cette demo.",
    )
    st.caption(provider.get("note", ""))

    st.divider()
    st.markdown("### Recherche")
    use_rewrite = st.checkbox(
        "Reformuler la question",
        value=True,
        help="Reformule la question en vocabulaire fiscal avant retrieval.",
    )
    reranker_available = RERANKER_MODEL_PATH.exists()
    with st.expander("Mode qualite avance", expanded=False):
        st.caption("Le reranker reclasse les passages candidats apres retrieval. Utile pour la precision, plus lent sur CPU.")
        use_reranker = st.checkbox(
            "Activer le reranking des passages",
            value=reranker_available and not RUNNING_ON_SPACE,
            help="A garder desactive sur hebergement gratuit si la latence compte plus que le gain de precision.",
        )
        if use_reranker and not reranker_available:
            st.warning("Modele reranker absent localement: le chargement peut tenter un telechargement Hugging Face.")

    st.divider()
    st.caption("Les questions peuvent contenir des donnees sensibles. N'envoyez pas de cas reel non anonymise.")
    st.caption("Prototype de recherche, pas conseil fiscal.")
    if st.button("Vider le cache", use_container_width=True):
        st.session_state.result_cache = {}
        st.rerun()

render_hero(provider_id, model, use_reranker)
render_metrics()

if _missing_runtime_paths() and should_auto_download_artifacts():
    with st.spinner("Telechargement des artefacts full corpus..."):
        try:
            download_missing_runtime_artifacts(PROJECT_ROOT)
        except Exception as exc:
            st.error(f"Telechargement des artefacts impossible: {exc}")

missing_paths = _missing_runtime_paths()
if missing_paths:
    st.error("Artefacts full-corpus manquants. Ajoutez-les localement avant de lancer la demo.")
    st.code("\n".join(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in missing_paths))
    st.info("Commande de verification: python scripts/check_setup.py --deep")
    st.stop()

check_hashes = os.environ.get("BOFIP_VALIDATE_HASHES", "").strip().lower() in {"1", "true", "yes"}
artifact_errors = validate_runtime_artifacts(PROJECT_ROOT, check_hashes=check_hashes)
if artifact_errors:
    st.error("Artefacts full-corpus invalides.")
    st.code("\n".join(artifact_errors))
    st.stop()

if not api_key:
    render_missing_key(provider)
    st.stop()

reranker_model = str(RERANKER_MODEL_PATH) if RERANKER_MODEL_PATH.exists() else None
rt = get_runtime(use_reranker, reranker_model)
import torch
base_url = provider["base_url"]
client = OpenAI(api_key=api_key, base_url=base_url)

mode = st.radio("Mode", ["Question unique", "Lot de questions"], horizontal=True, label_visibility="collapsed")

if mode == "Question unique":
    st.markdown('<div class="workspace-panel"><div class="section-kicker">Analyse</div><div class="section-title">Question fiscale</div></div>', unsafe_allow_html=True)
    query = st.text_area(
        "Votre question",
        placeholder="Quel taux de TVA pour la pose d'une pompe a chaleur chez un particulier ?",
        height=115,
        label_visibility="collapsed",
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.caption(f"Runtime: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    c2.caption(f"Fournisseur: {provider_id}")
    submit = c3.button("Analyser", type="primary", disabled=not query.strip(), use_container_width=True)
    if submit:
        with st.spinner("Recherche, selection des sources et generation..."):
            results = process_query(query.strip(), rt, client, model, use_rewrite, use_reranker)
        display_results(results)
else:
    st.markdown('<div class="workspace-panel"><div class="section-kicker">Batch</div><div class="section-title">Plusieurs questions</div></div>', unsafe_allow_html=True)
    batch_text = st.text_area(
        "Questions",
        height=150,
        placeholder="Une question par paragraphe. Maximum 5 questions pour la demo publique.",
        label_visibility="collapsed",
    )
    submit_batch = st.button("Lancer le lot", type="primary", disabled=not batch_text.strip(), use_container_width=True)
    if submit_batch:
        queries = [q.strip() for q in re.split(r"\n\s*\n", batch_text.strip()) if q.strip()]
        if len(queries) > 5:
            st.warning("Lot limite a 5 questions pour la demo publique.")
            queries = queries[:5]
        progress = st.progress(0)
        status_text = st.empty()
        all_results = []
        for index, question in enumerate(queries, start=1):
            status_text.text(f"{index}/{len(queries)} - {question[:90]}")
            all_results.append(process_query(question, rt, client, model, use_rewrite, use_reranker))
            progress.progress(index / len(queries))
        progress.empty()
        status_text.empty()

        summary_rows = []
        for res in all_results:
            parsed = res.get("parsed") or {}
            status, label, _ = _status_meta(parsed.get("answer_status", "error"))
            summary_rows.append(
                {
                    "question": _truncate(res.get("query", ""), 90),
                    "statut": label,
                    "conclusion": _truncate(parsed.get("conclusion", res.get("error", "")), 120),
                }
            )
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
        expand_all = st.checkbox("Developper toutes les reponses", value=False, key="expand_batch")
        for index, res in enumerate(all_results, start=1):
            with st.expander(f"Question {index}: {_truncate(res.get('query', ''), 100)}", expanded=expand_all):
                display_results(res)

st.markdown('<div class="app-footer">BOFiP Agentic RAG - prototype par Raphael Ifergan - sources BOFiP a verifier avant usage professionnel.</div>', unsafe_allow_html=True)
