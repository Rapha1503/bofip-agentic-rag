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
from bofip_agentic.providers import PROVIDERS, coerce_model_for_provider
from bofip_agentic.retrieval_config import selected_retrieval_profile

RUNNING_ON_SPACE = bool(os.environ.get("SPACE_ID"))
SHOW_DEBUG_DETAILS = os.environ.get("BOFIP_SHOW_DEBUG", "").strip().lower() in {"1", "true", "yes"}
APP_RESULT_CACHE_VERSION = "status-logic-v3-2026-06-24"
VISIBLE_PROVIDERS = {
    name: provider
    for name, provider in PROVIDERS.items()
    if not (RUNNING_ON_SPACE and provider.get("local_only"))
}

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
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _status_meta(status: str | None) -> tuple[str, str, str]:
    mapping = {
        "supported": ("supported", "Réponse sourcée", "La réponse principale est couverte; les réserves restent en limites."),
        "partial": ("supported", "Réponse sourcée", "La réponse principale est traitée; vérifiez les limites affichées."),
        "insufficient_evidence": ("insufficient", "Preuve insuffisante", "Les passages retenus ne permettent pas de répondre à la question principale."),
    }
    return mapping.get(status or "", ("insufficient", status or "Statut inconnu", "Le statut structuré n'a pas été reconnu."))


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
      [data-testid="stAppViewContainer"] [data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
        pointer-events: auto !important;
        visibility: visible !important;
      }
      [data-testid="stAppViewContainer"] .staleElement,
      [data-testid="stAppViewContainer"] .stale-element,
      [data-testid="stAppViewContainer"] [data-testid="staleElement"],
      [data-testid="stAppViewContainer"] [class*="stale"],
      [data-testid="stAppViewContainer"] [class*="stale"] * {
        opacity: 1 !important;
        filter: none !important;
        color: inherit;
      }
      [data-testid="stAppViewContainer"] * { transition: none !important; }

      .block-container { max-width: 1450px; padding-top: .75rem; padding-bottom: 3rem; }

      [data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: var(--text) !important; }
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
      .stSelectbox label, .stTextInput label, .stTextArea label, label {
        color: var(--ink) !important;
        font-weight: 850 !important;
      }
      [data-testid="stCaptionContainer"], [data-testid="InputInstructions"] { color: var(--muted) !important; }
      [data-testid="stForm"] { border: 0 !important; padding: 0 !important; background: transparent !important; }

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
      .stTextArea textarea:disabled {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        opacity: 1 !important;
      }
      [data-testid="stAppViewContainer"] .st-key-running_question_preview [data-testid="stTextAreaRootElement"] textarea:disabled {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        opacity: 1 !important;
      }
      .stTextArea textarea:focus, .stTextInput input:focus, [data-baseweb="select"] > div:focus-within {
        border-color: var(--burgundy) !important; box-shadow: 0 0 0 3px rgba(134, 24, 61, .12) !important;
      }
      [data-baseweb="select"] span, [data-baseweb="select"] input { color: var(--ink) !important; }

      div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
        background: var(--burgundy-dark) !important; color: #fff !important; border: 1px solid var(--burgundy-dark) !important;
        border-radius: 7px !important; min-height: 42px; font-weight: 850 !important;
      }
      div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--burgundy) !important; border-color: var(--burgundy) !important;
      }

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
      .progress-panel {
        background: #fff8fa; border: 1px solid var(--line); border-radius: 8px;
        padding: 14px 16px; margin: 12px 0 2px;
      }
      .progress-title { color: var(--burgundy-dark); font-weight: 850; margin-bottom: 4px; }
      .progress-subtitle { color: var(--muted); font-size: .84rem; margin-bottom: 10px; line-height: 1.35; }
      .progress-row { display: flex; gap: 10px; align-items: flex-start; padding: 10px 0; border-top: 1px solid var(--line-soft); }
      .progress-row:first-of-type { border-top: 0; }
      .progress-index {
        flex: 0 0 24px; height: 24px; border-radius: 999px; background: var(--burgundy);
        color: #fff; display: inline-flex; align-items: center; justify-content: center;
        font-size: .76rem; font-weight: 850;
      }
      .progress-copy strong { color: var(--ink); display: block; line-height: 1.25; }
      .progress-time {
        display: inline-flex; gap: 6px; margin-left: 8px; color: var(--burgundy);
        font-size: .76rem; font-weight: 800; white-space: nowrap;
      }
      .progress-copy span { color: var(--muted); display: block; font-size: .86rem; margin-top: 2px; line-height: 1.35; }
      .progress-fields { display: grid; gap: 8px; margin-top: 9px; }
      .progress-field {
        background: #fff; border: 1px solid var(--line-soft); border-radius: 7px; padding: 8px 10px;
      }
      .progress-field b {
        display: block; color: var(--burgundy); font-size: .72rem; font-weight: 850;
        text-transform: uppercase; margin-bottom: 4px;
      }
      .progress-field code {
        white-space: pre-wrap; color: var(--ink); background: transparent; padding: 0;
        font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace; font-size: .78rem; line-height: 1.45;
      }
      .progress-details {
        margin-top: 8px;
        border: 1px solid var(--line-soft);
        border-radius: 7px;
        background: #fff;
        overflow: hidden;
      }
      .progress-details summary {
        cursor: pointer;
        color: var(--burgundy-dark);
        font-weight: 850;
        padding: 8px 10px;
        list-style: none;
      }
      .progress-details summary::-webkit-details-marker { display: none; }
      .progress-details summary::after {
        content: "+";
        float: right;
        color: var(--burgundy);
      }
      .progress-details[open] summary::after { content: "-"; }
      .progress-details ul {
        margin: 0;
        padding: 0 14px 10px 28px;
        color: var(--text);
        font-size: .84rem;
        line-height: 1.45;
      }

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
      .success-value { color: #0f5f52; background: var(--green-soft); border: 1px solid #b8dfd3; border-radius: 6px; display: inline-block; padding: 5px 8px; font-weight: 850; }
      [data-testid="stExpander"] {
        background: var(--paper) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
      }
      [data-testid="stExpander"] summary,
      [data-testid="stExpander"] summary p,
      [data-testid="stExpander"] summary span,
      [data-testid="stExpander"] summary svg {
        color: var(--burgundy-dark) !important;
        fill: var(--burgundy-dark) !important;
        font-weight: 850 !important;
      }
      [data-testid="stExpander"] summary:hover,
      [data-testid="stExpander"] summary:hover p,
      [data-testid="stExpander"] summary:hover span,
      [data-testid="stExpander"] summary:hover svg {
        color: var(--burgundy) !important;
        fill: var(--burgundy) !important;
      }
      div[data-testid="stButton"] button:disabled,
      div[data-testid="stFormSubmitButton"] button:disabled {
        opacity: 1 !important;
        background: var(--burgundy-dark) !important;
        color: #fff !important;
      }
      div[data-testid="stButton"] button:disabled::before,
      div[data-testid="stFormSubmitButton"] button:disabled::before {
        content: "";
        width: 15px;
        height: 15px;
        border-radius: 50%;
        border: 2px solid rgba(255,255,255,.42);
        border-top-color: #fff;
        animation: spin .8s linear infinite;
        margin-right: 8px;
      }

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
            <h1>Doctrine BOFiP.<br><span class="accent-word">Réponse sourcée.</span></h1>
            <p>Un poste de recherche fiscal: l'agent transforme la question en axes fiscaux, route la recherche par axe, critique les sources, puis rédige une réponse sourcée avec limites explicites.</p>
          </div>
          <div class="system-strip">
            <div class="system-item"><span>Corpus</span><strong>9 048 documents</strong><small>Snapshot API BOFiP du 23/06/2026</small></div>
            <div class="system-item"><span>Index</span><strong>79 160 passages</strong><small>Documents puis passages sectionnés</small></div>
            <div class="system-item"><span>Agent</span><strong>Plan + critique</strong><small>Axes fiscaux, routage, relance ciblée</small></div>
            <div class="system-item"><span>Sortie</span><strong>Citations + limites</strong><small>Sources visibles avant interprétation</small></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_missing_key(provider: dict) -> None:
    st.markdown(
        f"""
        <div class="notice-panel">
          <div class="section-kicker">Clé API requise</div>
          <strong>Saisissez une clé {provider['env_key']} dans le formulaire de question pour lancer l'agent.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_button(slot, label: str = "Analyse en cours") -> None:
    slot.markdown(
        f'<div class="loading-button"><span class="loading-dot" aria-hidden="true"></span><span>{_escape(label)}</span></div>',
        unsafe_allow_html=True,
    )

def render_runtime_progress(
    slot,
    events: list[dict],
    *,
    title: str = "Journal de l'agent",
    subtitle: str | None = None,
) -> None:
    if slot is None:
        return
    rows = []
    for idx, event in enumerate(events, start=1):
        label = _escape(event.get("label", "Étape"))
        detail = _escape(event.get("detail", ""))
        timing_parts = []
        if event.get("step_s") is not None:
            timing_parts.append(f"+{float(event.get('step_s') or 0):.1f}s")
        if event.get("elapsed_s") is not None:
            timing_parts.append(f"{float(event.get('elapsed_s') or 0):.1f}s total")
        timing_html = (
            f'<span class="progress-time">{" · ".join(_escape(part) for part in timing_parts)}</span>'
            if timing_parts
            else ""
        )
        detail_html = f"<span>{detail}</span>" if detail else ""
        fields = event.get("fields", []) or []
        if fields:
            field_html = []
            for field in fields[:5]:
                if not isinstance(field, dict):
                    continue
                field_html.append(
                    '<div class="progress-field">'
                    f'<b>{_escape(field.get("label", "Détail"))}</b>'
                    f'<code>{_escape(_truncate(str(field.get("value", "")), 600))}</code>'
                    '</div>'
                )
            if field_html:
                detail_html += '<div class="progress-fields">' + "".join(field_html) + '</div>'
        items = event.get("items", []) or []
        if items:
            item_html = "".join(f"<li>{_escape(_truncate(str(item), 220))}</li>" for item in items[:8])
            detail_html += f'<details class="progress-details"><summary>Détails</summary><ul>{item_html}</ul></details>'
        rows.append(
            f'<div class="progress-row"><span class="progress-index">{idx}</span>'
            f'<div class="progress-copy"><strong>{label}{timing_html}</strong>{detail_html}</div></div>'
        )
    subtitle_text = subtitle or "Etapes utiles de la recherche, sans prompts systeme ni JSON brut."
    slot.markdown(
        f'<div class="progress-panel"><div class="progress-title">{_escape(title)}</div>'
        f'<div class="progress-subtitle">{_escape(subtitle_text)}</div>'
        + ''.join(rows) + '</div>',
        unsafe_allow_html=True,
    )

@st.cache_resource(show_spinner=False)
def get_runtime(load_reranker: bool, load_dense: bool, device: str):
    from bofip_agentic.rag_runtime import RagRuntime

    return RagRuntime.from_local_corpus(
        corpus="commentary",
        device=device,
        load_reranker=load_reranker,
        load_dense=load_dense,
        allow_lexical_fallback=True,
    )


def ensure_runtime_ready() -> bool:
    missing = missing_runtime_artifacts(PROJECT_ROOT)
    if missing and should_auto_download_artifacts():
        loader = st.empty()
        loader.markdown(
            '<div class="notice-panel"><div class="section-kicker">Préparation du corpus</div><strong>Téléchargement des artefacts full-corpus.</strong><p>Cette étape conserve la couverture BOFiP complète.</p></div>',
            unsafe_allow_html=True,
        )
        try:
            download_missing_runtime_artifacts(PROJECT_ROOT)
        except Exception as exc:
            st.error(f"Téléchargement des artefacts impossible: {exc}")
            return False
        finally:
            loader.empty()

    missing = missing_runtime_artifacts(PROJECT_ROOT)
    if missing:
        st.error("Artefacts full-corpus manquants. Ajoutez-les localement avant de lancer la démo.")
        st.code("\n".join(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in missing))
        st.info("Commande de vérification: python scripts/check_setup.py --deep --skip-models")
        return False

    check_hashes = os.environ.get("BOFIP_VALIDATE_HASHES", "").strip().lower() in {"1", "true", "yes"}
    artifact_errors = validate_runtime_artifacts(PROJECT_ROOT, check_hashes=check_hashes)
    if artifact_errors:
        st.error("Artefacts full-corpus invalides.")
        st.code("\n".join(artifact_errors))
        return False
    return True


def selected_device() -> str:
    forced = os.environ.get("BOFIP_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda"}:
        return forced
    return "cpu"


def run_agent_query(query: str, provider: dict, api_key: str, model: str, *, use_reranker: bool, progress_slot=None) -> dict:
    retrieval_profile = selected_retrieval_profile()
    cache_material = "\n".join(
        [
            APP_RESULT_CACHE_VERSION,
            query,
            provider["base_url"],
            model,
            str(use_reranker),
            retrieval_profile.mode,
        ]
    )
    cache_key = hashlib.md5(cache_material.encode("utf-8")).hexdigest()[:16]
    st.session_state.setdefault("result_cache", {})
    if cache_key in st.session_state.result_cache:
        st.session_state.latest_progress_events = st.session_state.result_cache[cache_key].get("progress_events", [])
        return st.session_state.result_cache[cache_key]

    start = time.time()
    last_progress_at = start
    progress_events: list[dict] = []

    def emit_progress(
        label: str,
        detail: str = "",
        items: list[str] | None = None,
        fields: list[dict] | None = None,
        step_s: float | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        nonlocal last_progress_at
        now = time.time()
        event_step_s = round(now - last_progress_at, 2) if step_s is None else round(float(step_s), 2)
        event_elapsed_s = round(now - start, 2) if elapsed_s is None else round(float(elapsed_s), 2)
        last_progress_at = now
        progress_events.append(
            {
                "label": label,
                "detail": detail,
                "items": items or [],
                "fields": fields or [],
                "step_s": event_step_s,
                "elapsed_s": event_elapsed_s,
            }
        )
        st.session_state.latest_progress_events = list(progress_events)
        render_runtime_progress(progress_slot, progress_events, title="Analyse en cours")

    device = selected_device()
    from bofip_agentic.agent_rag import AgenticRAG

    load_dense = retrieval_profile.load_dense
    emit_progress(
        "Préparation locale",
        "Chargement des artefacts nécessaires si le cache n'est pas encore prêt.",
        fields=[{"label": "Couverture", "value": "Corpus BOFiP complet"}],
    )
    runtime = get_runtime(load_reranker=use_reranker, load_dense=load_dense, device=device)
    retrieval_mode = (
        retrieval_profile.label
        if retrieval_profile.load_dense and getattr(runtime, "doc_encoder", None) is not None
        else "BM25 full-corpus"
    )

    if provider.get("type") == "codex_cli":
        from bofip_agentic.codex_cli_client import CodexCliClient

        client = CodexCliClient(model=model, project_root=PROJECT_ROOT)
    else:
        client = OpenAI(api_key=api_key, base_url=provider["base_url"])

    def on_agent_progress(label: str, payload: dict) -> None:
        detail = payload.get("detail", "") if isinstance(payload, dict) else ""
        items = payload.get("items", []) if isinstance(payload, dict) else []
        fields = payload.get("fields", []) if isinstance(payload, dict) else []
        step_s = payload.get("step_s") if isinstance(payload, dict) else None
        _agent_elapsed_s = payload.get("elapsed_s") if isinstance(payload, dict) else None
        emit_progress(label, detail, items=items, fields=fields, step_s=step_s)

    agent = AgenticRAG(
        runtime,
        api_key="",
        base_url=provider["base_url"],
        model=model,
        max_iterations=2,
        client=client,
        use_reranker=use_reranker,
        progress_callback=on_agent_progress,
    )
    agent_result = agent.run(query)
    emit_progress(
        "Réponse prête",
        "La réponse sourcée est prête; le parcours reste disponible en détail.",
        fields=[{"label": "Statut", "value": agent_result.get("answer_status", "?")}],
    )
    parsed = {
        "answer_status": agent_result.get("answer_status", "insufficient_evidence"),
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
        "step_timings": agent_result.get("step_timings", []),
        "raw_agent": agent_result,
        "device": device,
        "reranker": use_reranker,
        "retrieval_profile": retrieval_profile.mode,
        "retrieval_mode": retrieval_mode,
        "progress_events": list(progress_events),
    }
    st.session_state.result_cache[cache_key] = result
    return result

def render_answer(parsed: dict, result: dict) -> None:
    status_class, status_label, status_detail = _status_meta(parsed.get("answer_status"))
    conclusion = parsed.get("conclusion") or "Aucune conclusion structurée n'a été retournée."
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
            <div class="metric-card"><span>Itérations agent</span><strong>{int(result.get('iterations') or 0)}</strong></div>
            <div class="metric-card"><span>Passages cumulés</span><strong>{int(result.get('chunks_used') or 0)}</strong></div>
            <div class="metric-card"><span>Temps</span><strong>{_escape(result.get('total_s', '?'))}s</strong></div>
          </div>
          <p>{_escape(status_detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if bullets:
        st.markdown('<div class="answer-panel"><h3>Justification sourcée</h3><ul>' + ''.join(f'<li>{_escape(item)}</li>' for item in bullets) + '</ul></div>', unsafe_allow_html=True)
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
        elif title == "Axes manquants":
            value_html = '<span class="success-value">Tous les axes couverts</span>'
        else:
            value_html = '<span class="empty-value">Non renseigné</span>'
        cells.append(f'<div class="coverage-item"><span class="coverage-title">{_escape(title)}</span>{value_html}</div>')
    st.markdown('<div class="coverage-panel"><div class="coverage-grid">' + ''.join(cells) + '</div></div>', unsafe_allow_html=True)


def _source_card_html(chunk: dict, index: int) -> str:
    ref = _escape(chunk.get("boi_reference", "BOFiP"))
    title = _escape(chunk.get("title", "Sans titre"))
    section = _escape(_truncate(chunk.get("section_path", ""), 120))
    publication_date = _escape(chunk.get("publication_date") or "date non renseignée")
    excerpt = _escape(_truncate(chunk.get("text", ""), 430))
    return f'<div class="source-card"><div class="ref">#{index} - {ref}</div><h4>{title}</h4><div class="path">{publication_date} - {section}</div><p>{excerpt}</p></div>'


def render_sources(chunks: list[dict]) -> None:
    st.markdown('<div class="section-kicker">Sources retenues</div>', unsafe_allow_html=True)
    if not chunks:
        st.info("Aucun passage source n'a été retenu.")
        return
    cards = ''.join(_source_card_html(chunk, index) for index, chunk in enumerate(chunks[:8], start=1))
    st.markdown(f'<div class="source-grid">{cards}</div>', unsafe_allow_html=True)


def render_agentic_details(result: dict) -> None:
    events = result.get("progress_events", []) or []
    if not events:
        return
    with st.expander("Parcours agentique détaillé", expanded=False):
        render_runtime_progress(
            st,
            events,
            title="Parcours agentique",
            subtitle="Plan fiscal, recherches ciblées, critique des sources et relances utiles.",
        )


def display_results(result: dict) -> None:
    st.markdown(
        f'<div class="notice-panel"><div class="section-kicker">Question analysée</div><strong>{_escape(result.get("query", ""))}</strong></div>',
        unsafe_allow_html=True,
    )
    parsed = result.get("parsed", {}) or {}
    render_answer(parsed, result)
    render_coverage(parsed)
    render_sources(result.get("chunks", []) or [])
    render_agentic_details(result)
    if SHOW_DEBUG_DETAILS:
        with st.expander("JSON agent brut", expanded=False):
            st.code(json.dumps(result.get("raw_agent", {}), ensure_ascii=False, indent=2)[:12000], language="json")


pending_analysis = st.session_state.get("pending_analysis")
is_running = bool(pending_analysis)
latest_results = None if is_running else st.session_state.get("latest_results")
latest_progress_events = st.session_state.get("latest_progress_events", [])
latest_error = None if is_running else st.session_state.get("latest_error")
if not is_running and not st.session_state.get("single_question") and st.session_state.get("last_submitted_query"):
    st.session_state.single_question = st.session_state.last_submitted_query

provider_names = list(VISIBLE_PROVIDERS.keys())
if "provider_select" not in st.session_state or st.session_state.provider_select not in VISIBLE_PROVIDERS:
    st.session_state.provider_select = provider_names[0]
st.session_state.selected_model = coerce_model_for_provider(
    st.session_state.provider_select,
    st.session_state.get("selected_model"),
)


def _api_key_state_key(provider_id: str) -> str:
    safe_provider = "".join(ch.lower() if ch.isalnum() else "_" for ch in provider_id)
    return f"api_key_{safe_provider}"


with st.sidebar:
    st.markdown("### BOFiP Agentic RAG")
    st.caption("Anonymisez les cas réels avant usage.")
    st.caption("Prototype de recherche, pas conseil fiscal.")

render_app_shell()

query_col, config_col = st.columns([1.65, 0.75], gap="large")

with config_col:
    with st.container(border=True):
        st.markdown("### Paramètres")
        provider_id = st.selectbox(
            "Fournisseur",
            provider_names,
            index=provider_names.index(st.session_state.provider_select),
            key="provider_select",
        )
        provider = VISIBLE_PROVIDERS[provider_id]
        st.session_state.selected_model = coerce_model_for_provider(provider_id, st.session_state.get("selected_model"))
        model_options = provider["models"]
        selected_model = st.session_state.selected_model
        model = st.selectbox(
            "Modèle",
            model_options,
            index=model_options.index(selected_model) if selected_model in model_options else 0,
            key="selected_model",
        )
        requires_api_key = provider.get("requires_api_key", True)
        if requires_api_key:
            api_key_key = _api_key_state_key(provider_id)
        else:
            api_key_key = ""
            api_key = ""
        if st.button("Vider le cache", use_container_width=True):
            st.session_state.result_cache = {}
            st.session_state.latest_results = None
            st.session_state.latest_error = None
            st.session_state.latest_progress_events = []
            st.session_state.pending_analysis = None
            st.rerun()
        use_reranker = False

with query_col:
    with st.container(border=True):
        st.markdown("### Question fiscale")
        st.markdown(
            '<p class="panel-lede">Décrivez le cas en français. Le moteur affiche ses étapes, les axes de couverture et les passages BOFiP utilisés.</p>',
            unsafe_allow_html=True,
        )
        with st.form("question_form", clear_on_submit=False, border=False):
            if is_running:
                st.session_state["running_question_preview"] = pending_analysis.get("query", "")
                query = st.text_area(
                    "Votre question",
                    height=150,
                    label_visibility="collapsed",
                    key="running_question_preview",
                    disabled=True,
                )
            else:
                query = st.text_area(
                    "Votre question",
                    placeholder="Exemple : quel taux de TVA pour la pose d'une pompe à chaleur chez un particulier ?",
                    height=150,
                    label_visibility="collapsed",
                    key="single_question",
                    disabled=False,
                )
            if requires_api_key:
                api_key = st.text_input(
                    f"Clé API ({provider['env_key']})",
                    type="password",
                    key=api_key_key,
                )
            submit = st.form_submit_button(
                "Analyse en cours" if is_running else "Analyser la question",
                type="primary",
                use_container_width=True,
                disabled=is_running,
            )
        status_slot = st.empty()
        if is_running:
            render_runtime_progress(
                status_slot,
                latest_progress_events or [{"label": "Analyse lancée", "detail": "La question entre dans le parcours agentique."}],
                title="Analyse en cours",
                subtitle="Le résultat précédent est masqué; seules les étapes de l'analyse courante sont affichées.",
            )

if submit:
    st.session_state.latest_error = None
    submitted_query = str(st.session_state.get("single_question") or query or "").strip()
    if not submitted_query:
        status_slot.warning("Saisissez une question avant de lancer l'analyse.")
    elif provider.get("requires_api_key", True) and not api_key:
        status_slot.warning("Ajoutez une clé API pour lancer l'analyse.")
    elif ensure_runtime_ready():
        st.session_state.last_submitted_query = submitted_query
        st.session_state.latest_results = None
        st.session_state.latest_progress_events = [
            {"label": "Analyse lancée", "detail": "La question entre dans le parcours agentique."}
        ]
        st.session_state.pending_analysis = {
            "query": submitted_query,
            "provider_id": provider_id,
            "model": model,
            "api_key": api_key if provider.get("requires_api_key", True) else "",
            "use_reranker": use_reranker,
        }
        st.rerun()

results_slot = st.empty()
pending_analysis = st.session_state.get("pending_analysis")
if pending_analysis:
    latest_results = None
    results_slot.empty()
    pending_provider_id = pending_analysis.get("provider_id")
    pending_provider = VISIBLE_PROVIDERS.get(pending_provider_id)
    if pending_provider is None:
        st.session_state.latest_error = "Analyse interrompue: fournisseur inconnu."
        st.session_state.pending_analysis = None
        st.rerun()
    pending_api_key = (
        pending_analysis.get("api_key", "")
        or st.session_state.get(_api_key_state_key(str(pending_provider_id)), "")
        if pending_provider.get("requires_api_key", True)
        else ""
    )
    if pending_provider.get("requires_api_key", True) and not pending_api_key:
        st.session_state.latest_error = "Analyse interrompue: clé API manquante."
        st.session_state.pending_analysis = None
        st.rerun()
    try:
        latest_results = run_agent_query(
            pending_analysis["query"],
            pending_provider,
            pending_api_key,
            pending_analysis["model"],
            use_reranker=bool(pending_analysis.get("use_reranker")),
            progress_slot=status_slot,
        )
        st.session_state.latest_results = latest_results
        st.session_state.latest_error = None
    except Exception as exc:
        st.session_state.latest_results = None
        st.session_state.latest_error = f"Analyse interrompue: {exc}"
    finally:
        st.session_state.pending_analysis = None
    st.rerun()

if latest_error:
    results_slot.error(latest_error)
elif latest_results:
    with results_slot.container():
        display_results(latest_results)
else:
    results_slot.empty()

st.markdown(
    '<div class="app-footer">BOFiP Agentic RAG - prototype par Raphael Ifergan - sources BOFiP à vérifier avant usage professionnel.</div>',
    unsafe_allow_html=True,
)
