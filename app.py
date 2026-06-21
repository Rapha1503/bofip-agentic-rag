"""BOFiP Agentic RAG - Streamlit BYOK application."""
from __future__ import annotations

import hashlib
import html
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st
from openai import OpenAI

from bofip_agentic.artifact_download import (
    download_missing_runtime_artifacts,
    missing_runtime_artifacts,
    should_auto_download_artifacts,
    validate_runtime_artifacts,
)
from bofip_agentic.env_utils import load_default_env_files
from bofip_agentic.providers import PROVIDERS

RUNNING_ON_SPACE = bool(os.environ.get("SPACE_ID"))
SHOW_DEBUG_DETAILS = os.environ.get("BOFIP_SHOW_DEBUG", "").strip().lower() in {"1", "true", "yes"}

st.set_page_config(
    page_title="BOFiP Agentic RAG",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _truncate(value: str, limit: int) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "?"


def _status_meta(status: str | None) -> tuple[str, str, str]:
    mapping = {
        "supported": ("supported", "R?ponse sourc?e", "Les passages retenus couvrent les axes essentiels."),
        "partial": ("partial", "R?ponse partielle", "La r?ponse couvre une partie du cas et signale les limites."),
        "insufficient_evidence": ("insufficient", "Preuve insuffisante", "Le corpus retenu ne suffit pas ? conclure proprement."),
    }
    return mapping.get(status or "", ("partial", status or "Statut inconnu", "Statut retourn? par l'agent."))


st.markdown(
    """
    <style>
      :root {
        --ink: #151116;
        --text: #3c3138;
        --muted: #755f69;
        --faint: #9f8792;
        --paper: #ffffff;
        --canvas: #f6eff2;
        --soft: #fbf7f8;
        --line: #dcc8d0;
        --line-soft: #eddfe5;
        --burgundy: #86183d;
        --burgundy-dark: #4b0d22;
        --burgundy-soft: #f7e4eb;
        --gold: #b77a20;
        --green: #0f6f61;
        --green-soft: #e6f5f0;
      }

      html, body, [class*="css"] {
        font-family: "Aptos", "Segoe UI", Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      }

      [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); }
      [data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stToolbar"],
      [data-testid="stStatusWidget"], #MainMenu, footer { display: none !important; }

      .block-container { max-width: 1450px; padding-top: .75rem; padding-bottom: 3rem; }

      [data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: var(--text); }

      .app-shell {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 24px 60px rgba(75, 13, 34, .13);
        margin-bottom: 18px;
      }

      .app-shell::before {
        content: "";
        display: block;
        height: 7px;
        background: linear-gradient(90deg, var(--burgundy-dark), var(--burgundy), var(--gold));
      }

      .app-header { text-align: center; padding: 36px 28px 30px; border-bottom: 1px solid var(--line-soft); }
      .brand-line {
        display: flex; align-items: center; justify-content: center; gap: 10px;
        color: var(--burgundy-dark); font-size: .78rem; font-weight: 850;
        text-transform: uppercase; margin-bottom: 14px;
      }
      .brand-mark {
        width: 38px; height: 38px; border-radius: 8px; display: inline-flex;
        align-items: center; justify-content: center; color: #fff;
        font-family: Georgia, "Times New Roman", serif; font-size: 1.1rem; font-weight: 800;
        background: linear-gradient(135deg, var(--burgundy) 0 58%, var(--burgundy-dark) 58% 100%);
        border: 1px solid var(--burgundy-dark);
      }
      .app-header h1 {
        margin: 0 auto 16px; max-width: 980px; color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.45rem, 4.6vw, 4.9rem); line-height: .98; font-weight: 850;
        letter-spacing: 0;
      }
      .accent-word { color: var(--burgundy); }
      .app-header p { margin: 0 auto; max-width: 820px; color: var(--text); font-size: 1.08rem; line-height: 1.6; }
      .app-header strong { color: var(--burgundy); font-weight: 850; }

      .system-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); background: #fff9fb; }
      .system-item { padding: 15px 18px; border-right: 1px solid var(--line-soft); border-top: 3px solid transparent; }
      .system-item:last-child { border-right: 0; }
      .system-item:nth-child(1) { border-top-color: var(--burgundy); }
      .system-item:nth-child(2) { border-top-color: var(--burgundy-dark); }
      .system-item:nth-child(3) { border-top-color: var(--gold); }
      .system-item:nth-child(4) { border-top-color: var(--green); }
      .system-item span { display: block; color: var(--burgundy); font-size: .74rem; font-weight: 850; margin-bottom: 4px; }
      .system-item strong { display: block; color: var(--ink); font-size: 1.05rem; font-weight: 850; line-height: 1.25; }
      .system-item small { display: block; color: var(--muted); margin-top: 4px; line-height: 1.35; }

      [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--paper); border-color: var(--line) !important; border-radius: 10px !important;
        box-shadow: 0 18px 45px rgba(75, 13, 34, .09); padding: 18px !important;
      }
      [data-testid="stVerticalBlockBorderWrapper"] h2,
      [data-testid="stVerticalBlockBorderWrapper"] h3 { color: var(--ink); }

      .panel-lede { color: var(--text); margin: 0 0 14px; font-size: .96rem; line-height: 1.5; }
      .field-note { color: var(--muted); font-size: .86rem; line-height: 1.45; margin-top: 8px; }

      .stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div {
        background: #fff !important; color: var(--ink) !important; border: 1px solid var(--line) !important;
        border-radius: 7px !important; box-shadow: none !important;
      }
      .stTextArea textarea:focus, .stTextInput input:focus, [data-baseweb="select"] > div:focus-within {
        border-color: var(--burgundy) !important; box-shadow: 0 0 0 3px rgba(134, 24, 61, .12) !important;
      }
      [data-baseweb="select"] span, [data-baseweb="select"] input { color: var(--ink) !important; }

      div[data-testid="stButton"] button {
        background: var(--burgundy-dark) !important; color: #fff !important; border: 1px solid var(--burgundy-dark) !important;
        border-radius: 7px !important; min-height: 42px; font-weight: 850 !important;
      }
      div[data-testid="stButton"] button:hover { background: var(--burgundy) !important; border-color: var(--burgundy) !important; }

      .loading-button {
        width: 100%; min-height: 42px; border-radius: 7px; background: var(--burgundy-dark);
        color: #fff; display: flex; align-items: center; justify-content: center; gap: 10px;
        font-weight: 850; border: 1px solid var(--burgundy-dark);
      }
      .loading-dot {
        width: 16px; height: 16px; border-radius: 50%; border: 2px solid rgba(255,255,255,.38);
        border-top-color: #fff; animation: spin .8s linear infinite;
      }
      @keyframes spin { to { transform: rotate(360deg); } }

      .notice-panel, .answer-panel, .trace-panel, .coverage-panel, .source-card {
        background: var(--paper); border: 1px solid var(--line); border-radius: 8px;
      }
      .notice-panel { padding: 18px 20px; border-left: 5px solid var(--burgundy); margin: 14px 0; }
      .section-kicker { color: var(--burgundy); font-size: .76rem; font-weight: 850; margin-bottom: 6px; }
      .notice-panel strong { color: var(--ink); }
      .notice-panel p { color: var(--text); margin: 8px 0 0; line-height: 1.55; }

      .answer-panel { padding: 18px 20px; margin: 16px 0; }
      .status-pill { display: inline-flex; border-radius: 7px; padding: 5px 10px; font-size: .78rem; font-weight: 850; }
      .status-supported { color: #0f5f52; background: var(--green-soft); border: 1px solid #b8dfd3; }
      .status-partial { color: #8a4b05; background: #fff5df; border: 1px solid #edcf8f; }
      .status-insufficient { color: var(--burgundy-dark); background: var(--burgundy-soft); border: 1px solid #e6bfd0; }
      .answer-panel h3 { color: var(--ink); margin: 12px 0 8px; font-size: 1.08rem; }
      .answer-panel p, .answer-panel li { color: var(--text); line-height: 1.58; }
      .answer-panel blockquote { border-left: 4px solid var(--burgundy); margin: 10px 0; padding: 8px 0 8px 14px; background: #fff8fa; }

      .metric-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }
      .metric-card { background: var(--soft); border: 1px solid var(--line-soft); border-radius: 8px; padding: 12px 14px; }
      .metric-card span { display: block; color: var(--muted); font-size: .74rem; font-weight: 800; margin-bottom: 4px; }
      .metric-card strong { color: var(--ink); font-size: 1.08rem; }

      .coverage-panel { margin: 14px 0 18px; overflow: hidden; }
      .coverage-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .coverage-item { padding: 14px 16px; border-right: 1px solid var(--line-soft); }
      .coverage-item:last-child { border-right: 0; }
      .coverage-title { display: block; color: var(--burgundy); font-size: .78rem; font-weight: 850; margin-bottom: 8px; }
      .coverage-value { color: var(--ink); font-size: .92rem; line-height: 1.45; }
      .empty-value { color: var(--text); background: var(--burgundy-soft); border: 1px solid #e6bfd0; border-radius: 6px; display: inline-block; padding: 5px 8px; font-weight: 800; }

      .agent-trace { background: #fff; border: 1px solid var(--line); border-radius: 8px; margin: 14px 0 18px; overflow: hidden; }
      .agent-trace-head { padding: 13px 16px; border-bottom: 1px solid var(--line-soft); color: var(--burgundy-dark); font-weight: 850; }
      .agent-step { padding: 14px 16px; border-bottom: 1px solid var(--line-soft); }
      .agent-step:last-child { border-bottom: 0; }
      .agent-step span { color: var(--burgundy); display: block; font-size: .75rem; font-weight: 850; margin-bottom: 6px; }
      .agent-step strong { color: var(--ink); display: block; font-size: .96rem; line-height: 1.35; }
      .agent-step small { color: var(--text); display: block; margin-top: 4px; line-height: 1.38; }
      .agent-query { color: var(--muted); background: #fff8fa; border: 1px solid var(--line-soft); border-radius: 6px; padding: 8px 10px; margin-top: 8px; font-size: .84rem; }

      .source-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
      .source-card { padding: 14px; min-height: 168px; }
      .source-card .ref { color: var(--burgundy); font-weight: 850; font-size: .87rem; }
      .source-card h4 { color: var(--ink); margin: 7px 0; font-size: .98rem; line-height: 1.28; }
      .source-card .path { color: var(--muted); font-size: .78rem; margin-bottom: 8px; }
      .source-card p { color: var(--text); font-size: .88rem; line-height: 1.45; }

      .app-footer { color: var(--muted); font-size: .82rem; margin-top: 26px; padding-top: 16px; border-top: 1px solid var(--line); }
      [data-testid="stSpinner"] p, [data-testid="stSpinner"] span { color: var(--burgundy-dark) !important; font-weight: 800 !important; }

      @media (max-width: 900px) {
        .system-strip, .metric-strip, .coverage-grid, .source-grid { grid-template-columns: 1fr; }
        .system-item, .coverage-item { border-right: 0; border-bottom: 1px solid var(--line-soft); }
        .system-item:last-child, .coverage-item:last-child { border-bottom: 0; }
        .app-header { padding: 28px 18px 24px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_app_shell() -> None:
    st.markdown(
        """
        <div class="app-shell">
          <div class="app-header">
            <div class="brand-line"><span class="brand-mark">B</span><span>BOFiP Agentic RAG</span></div>
            <h1>Doctrine BOFiP.<br><span class="accent-word">R?ponse sourc?e.</span></h1>
            <p>Un poste de recherche fiscal: l'agent classe la question, interroge le corpus BOFiP, auto-?value la couverture, puis relance une recherche cibl?e si des axes restent manquants.</p>
          </div>
          <div class="system-strip">
            <div class="system-item"><span>Corpus</span><strong>5 666 documents</strong><small>Commentaires BOFiP observ?s jusqu'au 28/01/2026</small></div>
            <div class="system-item"><span>Index</span><strong>66 289 passages</strong><small>Documents puis passages sectionn?s</small></div>
            <div class="system-item"><span>Agent</span><strong>Self-eval + relance</strong><small>Axes manquants, reformulation, second passage</small></div>
            <div class="system-item"><span>Sortie</span><strong>Citations + limites</strong><small>Sources visibles avant interpr?tation</small></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_missing_key(provider: dict) -> None:
    st.markdown(
        f"""
        <div class="notice-panel">
          <div class="section-kicker">Cl? API requise</div>
          <strong>Saisissez une cl? {provider['env_key']} dans le panneau Connexion LLM pour lancer l'agent.</strong>
          <p>La cl? reste dans la session Streamlit et sert uniquement ? appeler le fournisseur choisi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_button(slot, label: str = "Analyse en cours") -> None:
    slot.markdown(
        f'<div class="loading-button"><span class="loading-dot" aria-hidden="true"></span><span>{_escape(label)}</span></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_runtime(load_reranker: bool, device: str):
    from bofip_agentic.rag_runtime import RagRuntime

    return RagRuntime.from_local_corpus(corpus="commentary", device=device, load_reranker=load_reranker)


def ensure_runtime_ready() -> bool:
    missing = missing_runtime_artifacts(PROJECT_ROOT)
    if missing and should_auto_download_artifacts():
        loader = st.empty()
        loader.markdown(
            '<div class="notice-panel"><div class="section-kicker">Pr?paration du corpus</div><strong>T?l?chargement des artefacts full-corpus.</strong><p>Cette ?tape conserve la couverture BOFiP compl?te.</p></div>',
            unsafe_allow_html=True,
        )
        try:
            download_missing_runtime_artifacts(PROJECT_ROOT)
        except Exception as exc:
            st.error(f"T?l?chargement des artefacts impossible: {exc}")
            return False
        finally:
            loader.empty()

    missing = missing_runtime_artifacts(PROJECT_ROOT)
    if missing:
        st.error("Artefacts full-corpus manquants. Ajoutez-les localement avant de lancer la d?mo.")
        st.code("\n".join(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in missing))
        st.info("Commande de v?rification: python scripts/check_setup.py --deep")
        return False

    check_hashes = os.environ.get("BOFIP_VALIDATE_HASHES", "").strip().lower() in {"1", "true", "yes"}
    artifact_errors = validate_runtime_artifacts(PROJECT_ROOT, check_hashes=check_hashes)
    if artifact_errors:
        st.error("Artefacts full-corpus invalides.")
        st.code("\n".join(artifact_errors))
        return False
    return True


def selected_device() -> str:
    if RUNNING_ON_SPACE:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_agent_query(query: str, provider: dict, api_key: str, model: str, *, use_reranker: bool) -> dict:
    cache_key = hashlib.md5((query + provider["base_url"] + model + str(use_reranker)).encode("utf-8")).hexdigest()[:16]
    st.session_state.setdefault("result_cache", {})
    if cache_key in st.session_state.result_cache:
        return st.session_state.result_cache[cache_key]

    start = time.time()
    device = selected_device()
    from bofip_agentic.agent_rag import AgenticRAG

    runtime = get_runtime(load_reranker=use_reranker, device=device)
    client = OpenAI(api_key=api_key, base_url=provider["base_url"])
    agent = AgenticRAG(
        runtime,
        api_key=api_key,
        base_url=provider["base_url"],
        model=model,
        max_iterations=2,
        client=client,
        use_reranker=use_reranker,
    )
    agent_result = agent.run(query)
    parsed = {
        "answer_status": agent_result.get("answer_status", "partial"),
        "conclusion": agent_result.get("conclusion", ""),
        "axes_requis": agent_result.get("axes_requis", []),
        "axes_couverts": agent_result.get("axes_couverts", []),
        "axes_manquants": agent_result.get("axes_manquants", []),
        "justification_bullets": agent_result.get("justification_bullets", []),
        "limits": agent_result.get("limits", ""),
    }
    result = {
        "query": query,
        "parsed": parsed,
        "chunks": agent_result.get("sources", []),
        "trace": agent_result.get("trace", []),
        "coverage": agent_result.get("coverage", 0),
        "iterations": agent_result.get("iterations", 0),
        "total_s": agent_result.get("total_s", round(time.time() - start, 1)),
        "elapsed_s": round(time.time() - start, 1),
        "chunks_used": agent_result.get("chunks_used", len(agent_result.get("sources", []))),
        "raw_agent": agent_result,
        "device": device,
        "reranker": use_reranker,
    }
    st.session_state.result_cache[cache_key] = result
    return result


def render_answer(parsed: dict, result: dict) -> None:
    status_class, status_label, status_detail = _status_meta(parsed.get("answer_status"))
    conclusion = parsed.get("conclusion") or "Aucune conclusion structur?e n'a ?t? retourn?e."
    bullets = parsed.get("justification_bullets", []) or []
    limits = parsed.get("limits") or ""
    st.markdown(
        f"""
        <div class="answer-panel">
          <span class="status-pill status-{status_class}">{_escape(status_label)}</span>
          <h3>Conclusion</h3>
          <blockquote>{_escape(conclusion)}</blockquote>
          <div class="metric-strip">
            <div class="metric-card"><span>Couverture</span><strong>{float(result.get('coverage', 0)):.0%}</strong></div>
            <div class="metric-card"><span>It?rations agent</span><strong>{int(result.get('iterations') or 0)}</strong></div>
            <div class="metric-card"><span>Passages cumul?s</span><strong>{int(result.get('chunks_used') or 0)}</strong></div>
            <div class="metric-card"><span>Temps</span><strong>{_escape(result.get('total_s', '?'))}s</strong></div>
          </div>
          <p>{_escape(status_detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if bullets:
        st.markdown('<div class="answer-panel"><h3>Raisonnement</h3><ul>' + ''.join(f'<li>{_escape(item)}</li>' for item in bullets) + '</ul></div>', unsafe_allow_html=True)
    if limits:
        st.markdown(f'<div class="notice-panel"><div class="section-kicker">Limites</div><p>{_escape(limits)}</p></div>', unsafe_allow_html=True)


def render_coverage(parsed: dict) -> None:
    axes = {
        "Axes requis": parsed.get("axes_requis", []) or [],
        "Axes couverts": parsed.get("axes_couverts", []) or [],
        "Axes manquants": parsed.get("axes_manquants", []) or [],
    }
    if not any(axes.values()):
        return
    cells = []
    for title, values in axes.items():
        if values:
            value_html = "".join(f'<div class="coverage-value">{_escape(value)}</div>' for value in values)
        else:
            value_html = '<span class="empty-value">Non renseign?</span>'
        cells.append(f'<div class="coverage-item"><span class="coverage-title">{_escape(title)}</span>{value_html}</div>')
    st.markdown('<div class="coverage-panel"><div class="coverage-grid">' + ''.join(cells) + '</div></div>', unsafe_allow_html=True)


def render_agent_trace(result: dict) -> None:
    trace = result.get("trace", []) or []
    if not trace:
        st.markdown('<div class="agent-trace"><div class="agent-trace-head">Parcours agentique</div><div class="agent-step"><strong>Aucune trace retourn?e.</strong></div></div>', unsafe_allow_html=True)
        return
    steps = []
    for step in trace:
        iteration = step.get("iteration", "?")
        status = step.get("answer_status", "?")
        prefix = step.get("domain_prefix") or "non d?termin?"
        query_used = step.get("query_used") or ""
        details = (
            f"Pr?fixe BOFiP: {prefix} ? documents: {step.get('docs_found', 0)} ? "
            f"passages: {step.get('chunks_total', step.get('chunks_found', 0))} ? "
            f"retrieval: {step.get('retrieve_s', '?')}s ? r?ponse: {step.get('answer_s', '?')}s"
        )
        missing = step.get("axes_manquants", []) or []
        covered = step.get("axes_couverts", []) or []
        axis_line = f"Axes couverts: {len(covered)} ? axes manquants: {len(missing)}"
        reformulated = step.get("reformulated_query")
        reform_html = f'<div class="agent-query"><strong>Relance cibl?e:</strong> {_escape(reformulated)}</div>' if reformulated else ""
        mismatch = step.get("mismatch_fix")
        mismatch_html = f'<div class="agent-query"><strong>Correction taxonomique:</strong> {_escape(mismatch)}</div>' if mismatch else ""
        query_html = f'<div class="agent-query"><strong>Requ?te utilis?e:</strong> {_escape(_truncate(query_used, 260))}</div>' if query_used else ""
        steps.append(
            f'<div class="agent-step"><span>It?ration {iteration}</span><strong>Auto-?valuation: {_escape(status)}</strong>'
            f'<small>{_escape(details)}</small><small>{_escape(axis_line)}</small>{query_html}{mismatch_html}{reform_html}</div>'
        )
    st.markdown('<div class="agent-trace"><div class="agent-trace-head">Parcours agentique r?el</div>' + ''.join(steps) + '</div>', unsafe_allow_html=True)


def _source_card_html(chunk: dict, index: int) -> str:
    ref = _escape(chunk.get("boi_reference", "BOFiP"))
    title = _escape(chunk.get("title", "Sans titre"))
    section = _escape(_truncate(chunk.get("section_path", ""), 120))
    publication_date = _escape(chunk.get("publication_date") or "date non renseign?e")
    excerpt = _escape(_truncate(chunk.get("text", ""), 430))
    return f'<div class="source-card"><div class="ref">#{index} ? {ref}</div><h4>{title}</h4><div class="path">{publication_date} ? {section}</div><p>{excerpt}</p></div>'


def render_sources(chunks: list[dict]) -> None:
    st.markdown('<div class="section-kicker">Sources retenues</div>', unsafe_allow_html=True)
    if not chunks:
        st.info("Aucun passage source n'a ?t? retenu.")
        return
    cards = ''.join(_source_card_html(chunk, index) for index, chunk in enumerate(chunks[:8], start=1))
    st.markdown(f'<div class="source-grid">{cards}</div>', unsafe_allow_html=True)


def display_results(result: dict) -> None:
    st.markdown(
        f'<div class="notice-panel"><div class="section-kicker">Question analys?e</div><strong>{_escape(result.get("query", ""))}</strong></div>',
        unsafe_allow_html=True,
    )
    parsed = result.get("parsed", {}) or {}
    render_answer(parsed, result)
    render_coverage(parsed)
    render_agent_trace(result)
    render_sources(result.get("chunks", []) or [])
    if SHOW_DEBUG_DETAILS:
        with st.expander("JSON agent brut", expanded=False):
            st.code(json.dumps(result.get("raw_agent", {}), ensure_ascii=False, indent=2)[:12000], language="json")


load_default_env_files()

with st.sidebar:
    st.markdown("### BOFiP Agentic RAG")
    st.caption("Prototype par Raphael Ifergan.")
    if st.button("Vider le cache", use_container_width=True):
        st.session_state.result_cache = {}
        st.session_state.latest_results = None
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.caption("Anonymisez les cas r?els avant usage.")
    st.caption("Prototype de recherche, pas conseil fiscal.")

render_app_shell()

query_col, config_col = st.columns([1.65, 0.75], gap="large")

with config_col:
    with st.container(border=True):
        st.markdown("### Connexion LLM")
        st.caption("Votre cl? reste dans la session Streamlit.")
        provider_id = st.selectbox("Fournisseur", list(PROVIDERS.keys()), key="provider_select")
        provider = PROVIDERS[provider_id]
        api_key = st.text_input(
            f"Cl? API ({provider['env_key']})",
            value="" if RUNNING_ON_SPACE else os.environ.get(provider["env_key"], ""),
            type="password",
            key="api_key_input",
        )
        model_options = provider["models"]
        default_model = provider["default_model"]
        default_index = model_options.index(default_model) if default_model in model_options else 0
        model = st.selectbox("Mod?le", model_options, index=default_index, key=f"model_{provider_id}_select")
        st.markdown(f'<p class="field-note">{_escape(provider.get("note", ""))}</p>', unsafe_allow_html=True)
        use_reranker = False
        st.markdown('<p class="field-note">Reranker d?sactiv? sur la d?mo CPU. Le moteur garde la boucle agentique compl?te.</p>', unsafe_allow_html=True)

with query_col:
    with st.container(border=True):
        st.markdown("### Question fiscale")
        st.markdown(
            '<p class="panel-lede">D?crivez le cas en fran?ais. L?agent affichera ses it?rations, les axes de couverture et les passages BOFiP utilis?s.</p>',
            unsafe_allow_html=True,
        )
        query = st.text_area(
            "Votre question",
            placeholder="Exemple : quel taux de TVA pour la pose d'une pompe ? chaleur chez un particulier ?",
            height=150,
            label_visibility="collapsed",
            key="single_question",
        )
        button_slot = st.empty()
        submit = button_slot.button("Analyser la question", type="primary", use_container_width=True)
        if submit:
            if not query.strip():
                st.warning("Saisissez une question avant de lancer l'analyse.")
            elif not api_key:
                st.warning("Saisissez une cl? API dans le panneau Connexion LLM.")
            elif ensure_runtime_ready():
                render_loading_button(button_slot)
                try:
                    st.session_state.latest_results = run_agent_query(
                        query.strip(), provider, api_key, model, use_reranker=use_reranker
                    )
                    st.rerun()
                finally:
                    button_slot.empty()

if not api_key:
    render_missing_key(provider)

latest_results = st.session_state.get("latest_results")
if latest_results:
    display_results(latest_results)

st.markdown(
    '<div class="app-footer">BOFiP Agentic RAG ? prototype par Raphael Ifergan ? sources BOFiP ? v?rifier avant usage professionnel.</div>',
    unsafe_allow_html=True,
)
