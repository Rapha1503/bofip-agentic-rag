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
        "note": "DeepSeek v4 au 20/06/2026. deepseek-chat/reasoner restent en alias de compatibilité jusqu'au 24/07/2026.",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.4-mini", "gpt-5.5", "gpt-5.4", "gpt-5.4-nano"],
        "default_model": "gpt-5.4-mini",
        "env_key": "OPENAI_API_KEY",
        "note": "OpenAI frontier models au 20/06/2026. Mini par défaut pour limiter le coût.",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-medium-latest", "mistral-small-latest", "mistral-large-latest", "magistral-medium-latest"],
        "default_model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
        "note": "Alias Mistral latest pour suivre Medium 3.5 / Small 4 sans figer une version retirée.",
    },
    "Google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "default_model": "gemini-3.5-flash",
        "env_key": "GEMINI_API_KEY",
        "note": "Modèle OpenAI-compatible recommandé par la doc Gemini au 20/06/2026.",
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
SHOW_DEBUG_DETAILS = os.environ.get("BOFIP_SHOW_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def _missing_runtime_paths() -> list[Path]:
    return [path for path in REQUIRED_RUNTIME_PATHS if not path.exists()]


st.set_page_config(
    page_title="BOFiP Agentic RAG",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #14171f;
        --text: #2d3a46;
        --muted: #607081;
        --faint: #98a4af;
        --line: #bdc9d1;
        --line-soft: #dfe7eb;
        --paper: #ffffff;
        --canvas: #edf3f2;
        --blue: #126a63;
        --blue-dark: #0a4b47;
        --blue-soft: #e3f2ef;
        --green: #126a63;
        --amber: #b46f12;
        --red: #8f2d52;
        --plum: #8f2d52;
      }

      html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      }

      [data-testid="stAppViewContainer"] {
        background: var(--canvas);
        color: var(--ink);
      }

      [data-testid="stHeader"] {
        background: rgba(237, 243, 242, .96);
        border-bottom: 1px solid var(--line-soft);
      }

      [data-testid="stDecoration"],
      [data-testid="stToolbar"] {
        display: none;
      }

      .block-container {
        max-width: 1440px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        color: var(--ink);
      }

      [data-testid="stSidebar"] {
        background: var(--paper);
        border-right: 1px solid var(--line);
      }

      [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
      }

      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {
        color: var(--ink);
        font-size: .95rem;
        font-weight: 760;
        margin-bottom: .45rem;
      }

      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span {
        color: var(--text);
      }

      [data-testid="stSidebar"] small,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
      }

      [data-testid="stSidebar"] [data-baseweb="select"] > div,
      [data-testid="stSidebar"] .stTextInput input {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        box-shadow: none !important;
      }

      [data-testid="stSidebar"] [data-baseweb="select"] span,
      [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: var(--ink) !important;
      }

      [data-testid="stSidebar"] svg {
        color: var(--muted);
        fill: currentColor;
      }

      [data-testid="stSidebar"] hr {
        border-color: var(--line-soft);
        margin: 1.2rem 0;
      }

      .app-shell {
        border: 1px solid var(--line);
        border-top: 0;
        background: linear-gradient(120deg, #ffffff 0%, #f4fbf9 58%, #fff4f7 100%);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 20px 48px rgba(15, 61, 102, .10);
        margin-bottom: 16px;
        position: relative;
      }

      .app-shell::before {
        content: "";
        display: block;
        height: 5px;
        background: linear-gradient(90deg, var(--plum), var(--blue), var(--amber));
      }

      .app-header {
        display: flex;
        justify-content: space-between;
        gap: 24px;
        padding: 26px 30px 22px;
        border-bottom: 1px solid var(--line-soft);
      }

      .brand-line {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--blue-dark);
        font-size: .82rem;
        font-weight: 760;
        margin-bottom: 9px;
      }

      .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 7px;
        border: 1px solid var(--blue-dark);
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--plum) 0 42%, var(--blue) 42% 100%);
      }

      .app-header h1 {
        color: var(--ink);
        font-size: 2rem;
        line-height: 1.1;
        margin: 0 0 8px;
        font-weight: 760;
        letter-spacing: 0;
      }

      .app-header p {
        margin: 0;
        max-width: 760px;
        color: #435466;
        font-size: .98rem;
        line-height: 1.55;
      }

      .model-panel {
        min-width: 260px;
        align-self: flex-start;
        background: rgba(255, 255, 255, .76);
        border: 1px solid var(--line-soft);
        border-radius: 8px;
        padding: 12px 14px;
      }

      .model-panel span {
        display: block;
        color: var(--muted);
        font-size: .76rem;
        font-weight: 700;
        margin-bottom: 5px;
      }

      .model-panel strong {
        display: block;
        color: var(--ink);
        font-size: .93rem;
        line-height: 1.3;
      }

      .model-panel small {
        color: var(--muted);
        display: block;
        margin-top: 7px;
        line-height: 1.35;
      }

      .field-note {
        color: var(--muted);
        font-size: .86rem;
        line-height: 1.45;
        margin-top: 10px;
      }

      .inline-status {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 14px;
      }

      .inline-status span {
        border: 1px solid var(--line-soft);
        background: #f5faf9;
        border-radius: 6px;
        color: var(--text);
        font-size: .82rem;
        padding: 6px 9px;
      }

      [data-testid="stVerticalBlockBorderWrapper"],
      div[data-testid="stVerticalBlock"].st-emotion-cache-1ne20ew {
        background: var(--paper);
        border-color: var(--line) !important;
        border-radius: 10px !important;
        box-shadow: 0 18px 45px rgba(18, 106, 99, .08);
        padding: 18px !important;
      }

      [data-testid="stVerticalBlockBorderWrapper"] h2,
      [data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: var(--ink);
        letter-spacing: 0;
      }

      .system-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border-bottom: 1px solid var(--line-soft);
        background: rgba(255, 255, 255, .62);
      }

      .system-item {
        padding: 15px 18px;
        border-right: 1px solid var(--line-soft);
        border-top: 3px solid transparent;
      }

      .system-item:last-child {
        border-right: 0;
      }

      .system-item:nth-child(1) { border-top-color: var(--plum); }
      .system-item:nth-child(2) { border-top-color: var(--blue); }
      .system-item:nth-child(3) { border-top-color: var(--amber); }
      .system-item:nth-child(4) { border-top-color: #40566e; }

      .system-item span {
        display: block;
        color: var(--blue-dark);
        font-size: .74rem;
        font-weight: 730;
        margin-bottom: 4px;
      }

      .system-item strong {
        display: block;
        color: var(--ink);
        font-size: .98rem;
        font-weight: 750;
        line-height: 1.25;
      }

      .system-item small {
        display: block;
        color: var(--muted);
        margin-top: 4px;
        line-height: 1.35;
      }

      .workspace-panel,
      .answer-panel,
      .source-card,
      .notice-panel {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: none;
      }

      .workspace-panel {
        padding: 18px 20px;
        margin: 14px 0 10px;
      }

      .section-kicker {
        color: var(--plum);
        font-size: .75rem;
        font-weight: 760;
        letter-spacing: 0;
        text-transform: none;
        margin-bottom: 6px;
      }

      .section-title {
        color: var(--ink);
        font-size: 1.16rem;
        font-weight: 760;
        margin-bottom: 12px;
      }

      .status-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 6px;
        padding: 5px 10px;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: 0;
      }

      .status-supported {
        color: #075e45;
        background: #e7f5ef;
        border: 1px solid #b7dfcf;
      }

      .status-partial {
        color: #8a4b05;
        background: #fff7e6;
        border: 1px solid #f2d59b;
      }

      .status-insufficient,
      .status-error {
        color: #9a3412;
        background: #fff1e8;
        border: 1px solid #fed7aa;
      }

      .answer-panel {
        padding: 18px 20px;
        margin: 16px 0;
      }

      .answer-panel h3 {
        color: var(--ink);
        margin: 10px 0 8px;
        font-size: 1.08rem;
      }

      .answer-panel p,
      .answer-panel li {
        color: var(--text);
        line-height: 1.55;
      }

      .answer-panel blockquote {
        border-left: 3px solid var(--blue);
        margin: 10px 0;
        padding: 8px 0 8px 14px;
        color: var(--text);
        background: #f8fafc;
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
        color: var(--blue);
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
        color: var(--text);
        font-size: .88rem;
        line-height: 1.45;
      }

      .notice-panel {
        padding: 20px 22px;
        border-left: 4px solid var(--amber);
        margin-top: 14px;
      }

      .notice-panel p {
        color: var(--text);
        margin: 8px 0 0;
        line-height: 1.55;
      }

      .app-footer {
        color: var(--muted);
        font-size: .83rem;
        margin-top: 26px;
        padding-top: 16px;
        border-top: 1px solid var(--line);
      }

      .stTextArea textarea,
      .stTextInput input,
      [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        box-shadow: none !important;
      }

      .stTextInput [data-baseweb="input"],
      .stTextInput [data-baseweb="input"] > div,
      .stTextInput [data-baseweb="input"] button,
      button[aria-label*="password" i] {
        background: #ffffff !important;
        color: var(--muted) !important;
        border-color: var(--line) !important;
        box-shadow: none !important;
      }

      .stTextArea textarea:focus,
      .stTextInput input:focus,
      [data-baseweb="select"] > div:focus-within {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(35, 87, 137, .12) !important;
      }

      .stCheckbox [data-testid="stWidgetLabel"] p,
      .stRadio [data-testid="stWidgetLabel"] p {
        color: var(--text);
      }

      div[data-testid="stButton"] > button {
        border-radius: 6px;
        font-weight: 730;
        border: 1px solid var(--line);
      }

      div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--blue);
        border-color: var(--blue);
        color: white;
      }

      div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--blue-dark);
        border-color: var(--blue-dark);
      }

      div[data-testid="stButton"] > button:disabled,
      div[data-testid="stButton"] > button[kind="primary"]:disabled {
        background: #eef2f6 !important;
        border-color: var(--line) !important;
        color: var(--faint) !important;
        opacity: 1 !important;
      }

      @media (max-width: 840px) {
        .app-header { flex-direction: column; padding: 22px; }
        .app-header h1 { font-size: 1.72rem; }
        .model-panel { min-width: 0; width: 100%; }
        .system-strip,
        .source-grid { grid-template-columns: 1fr; }
        .system-item { border-right: 0; border-bottom: 1px solid var(--line-soft); }
        .system-item:last-child { border-bottom: 0; }
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
        '"facets":[{"name":"axe","query":"sous-requête pour cet axe"}]}\n'
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

def _extract_json_object(raw: str) -> dict | None:
    content = (raw or "").strip()
    if not content:
        return None
    candidates = [content]
    cleaned = re.sub(r"```(?:json)?\s*", "", content).replace("```", "").strip()
    if cleaned != content:
        candidates.append(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _compact_answer_prompt(query: str, chunks: list[dict], previous_raw: str = "") -> str:
    evidence = []
    for chunk in chunks[:5]:
        evidence.append(
            f"[{chunk['rank']}] {chunk['boi_reference']} | {chunk['title']} | "
            f"{_truncate(chunk.get('text', ''), 520)}"
        )
    previous_note = ""
    if previous_raw:
        previous_note = (
            "\nSortie précédente invalide ou tronquée, à corriger sans reprendre le texte long:\n"
            + _truncate(previous_raw, 900)
            + "\n"
        )
    return (
        "Question utilisateur:\n"
        + query
        + "\n\nExtraits BOFiP:\n"
        + "\n\n".join(evidence)
        + previous_note
        + "\nRéponds en JSON strict uniquement. Pas de markdown. Réponse concise.\n"
        'Schema: {"answer_status":"supported|partial|insufficient_evidence",'
        '"axes_requis":["..."],"axes_couverts":["..."],"axes_manquants":["..."],'
        '"conclusion":"80 mots maximum","justification_bullets":["2 a 3 puces courtes avec citations [n]"],'
        '"limits":"40 mots maximum"}'
    )


def call_llm(prompt, client, model, *, max_tokens: int = 1800):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"Tu es un assistant fiscal prudent. Schema JSON strict."},
                  {"role":"user","content":prompt}],
        temperature=0.0, max_tokens=max_tokens,
        response_format={"type":"json_object"},
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    return {"raw":content,"ptokens":getattr(usage,"prompt_tokens",None) if usage else None,
            "ctokens":getattr(usage,"completion_tokens",None) if usage else None,
            "finish_reason": getattr(choice, "finish_reason", None)}

def render_answer(parsed):
    normalized, label, css_class = _status_meta(parsed.get("answer_status"))
    conclusion = _escape(parsed.get("conclusion", ""))
    bullets = parsed.get("justification_bullets", []) or []
    limits = _escape(parsed.get("limits", ""))

    bullet_html = "".join(f"<li>{_escape(item)}</li>" for item in bullets)
    if not bullet_html:
        bullet_html = "<li>Aucune justification détaillée n'a été retournée.</li>"

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
    results["finish_reason"] = llm_r.get("finish_reason")
    _log("LLM_DONE", {"ptokens": llm_r["ptokens"], "ctokens": llm_r["ctokens"], 
          "raw_len": len(llm_r["raw"]), "finish_reason": llm_r.get("finish_reason")})
    parsed = _extract_json_object(llm_r["raw"])
    if parsed is None or llm_r.get("finish_reason") == "length":
        try:
            retry_prompt = _compact_answer_prompt(query, chunks, llm_r["raw"])
            retry_r = call_llm(retry_prompt, client, llm_model, max_tokens=1200)
            retry_parsed = _extract_json_object(retry_r["raw"])
            results["llm_retry_raw"] = retry_r["raw"]
            results["retry_finish_reason"] = retry_r.get("finish_reason")
            if retry_parsed is not None:
                parsed = retry_parsed
                results["ctokens"] = retry_r["ctokens"]
                results["finish_reason"] = retry_r.get("finish_reason")
        except Exception as e:
            results["llm_retry_error"] = str(e)
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
          <div class="section-kicker">Question analysée</div>
          <strong>{_escape(results.get("query", ""))}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rewritten = results.get("rewritten")
    if rewritten and rewritten != results.get("query"):
        with st.expander("Reformulation utilisée", expanded=False):
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
        st.warning("Le modèle n'a pas retourné une réponse structurée exploitable. Les sources retrouvées restent affichées ci-dessous.")
        if SHOW_DEBUG_DETAILS:
            st.code(results.get("llm_raw", "")[:1200], language="json")

    chunks = results.get("chunks", []) or []
    st.markdown('<div class="section-kicker">Sources retenues</div>', unsafe_allow_html=True)
    if chunks:
        render_source_cards(chunks)
    else:
        st.info("Aucun passage source n'a été retenu.")

    if SHOW_DEBUG_DETAILS:
        with st.expander(f"Audit retrieval - documents candidats ({len(results.get('stage1', []))})", expanded=False):
            rows = [
                {"rang": h.rank, "score": f"{h.score:.4f}", "reference": h.boi_reference, "titre": h.title[:140]}
                for h in results.get("stage1", [])
            ]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("Aucun document candidat à afficher.")

        with st.expander("Audit technique", expanded=False):
            plog = results.get("pipeline_log", {}) or {}
            if plog:
                audit_rows = [
                    {"métrique": "Documents uniques", "valeur": plog.get("unique_docs_final", "?")},
                    {"métrique": "Maximum passages par document", "valeur": plog.get("max_chunks_per_doc", "?")},
                    {"métrique": "Candidats passage", "valeur": plog.get("stage2_candidates", "?")},
                    {"métrique": "Facettes", "valeur": plog.get("facets_used", "?")},
                ]
                st.dataframe(audit_rows, use_container_width=True, hide_index=True)
                dist = plog.get("doc_distribution_final", {})
                if dist:
                    st.json(dist)
            st.caption(
                f"Prompt tokens: {results.get('ptokens', '?')} | Completion tokens: {results.get('ctokens', '?')} | "
                f"finish_reason: {results.get('finish_reason', '?')}"
            )
            trace_tabs = st.tabs(["Passages", "Prompt", "JSON brut"])
            with trace_tabs[0]:
                for chunk in chunks:
                    st.markdown(f"**[{chunk['rank']}] {chunk['boi_reference']}** - {chunk['score']:.4f}")
                    st.caption(chunk.get("section_path", ""))
                    st.write(_truncate(chunk.get("text", ""), 700))
            with trace_tabs[1]:
                st.code(results.get("prompt", ""), language="text")
            with trace_tabs[2]:
                st.code(results.get("llm_retry_raw") or results.get("llm_raw", ""), language="json")


def render_app_shell():
    st.markdown(
        """
        <div class="app-shell">
          <div class="app-header">
            <div>
              <div class="brand-line"><span class="brand-mark">B</span><span>BOFiP Agentic RAG</span></div>
              <h1>Recherche citée dans la doctrine fiscale</h1>
              <p>Un poste de recherche pour interroger le corpus BOFiP commentaires, vérifier les passages retenus et obtenir une réponse prudente avec limites explicites.</p>
            </div>
          </div>
          <div class="system-strip">
            <div class="system-item"><span>Corpus</span><strong>5 666 documents</strong><small>BOFiP commentaires observés jusqu'au 28/01/2026</small></div>
            <div class="system-item"><span>Index</span><strong>66 289 passages</strong><small>Recherche documents puis passages</small></div>
            <div class="system-item"><span>Retrieval</span><strong>BM25 + embeddings E5</strong><small>Fusion RRF, diversité documentaire</small></div>
            <div class="system-item"><span>Sortie</span><strong>Citations et limites</strong><small>Sources visibles avant interprétation</small></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_missing_key(provider: dict):
    st.markdown(
        f"""
        <div class="notice-panel">
          <div class="section-kicker">Clé API requise</div>
          <strong>Saisissez une clé {provider['env_key']} dans le panneau Connexion LLM pour lancer une recherche.</strong>
          <p>La clé reste dans la session Streamlit et sert uniquement à appeler le fournisseur choisi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# UI
load_default_env_files()


def ensure_runtime_ready() -> bool:
    if _missing_runtime_paths() and should_auto_download_artifacts():
        with st.spinner("Téléchargement des artefacts full corpus..."):
            try:
                download_missing_runtime_artifacts(PROJECT_ROOT)
            except Exception as exc:
                st.error(f"Téléchargement des artefacts impossible: {exc}")
                return False

    missing_paths = _missing_runtime_paths()
    if missing_paths:
        st.error("Artefacts full-corpus manquants. Ajoutez-les localement avant de lancer la démo.")
        st.code("\n".join(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in missing_paths))
        st.info("Commande de vérification : python scripts/check_setup.py --deep")
        return False

    check_hashes = os.environ.get("BOFIP_VALIDATE_HASHES", "").strip().lower() in {"1", "true", "yes"}
    artifact_errors = validate_runtime_artifacts(PROJECT_ROOT, check_hashes=check_hashes)
    if artifact_errors:
        st.error("Artefacts full-corpus invalides.")
        st.code("\n".join(artifact_errors))
        return False
    return True


with st.sidebar:
    st.markdown("### BOFiP Agentic RAG")
    st.caption("Prototype par Raphael Ifergan.")
    st.caption("Panneau secondaire : cache et limites.")
    if st.button("Vider le cache", use_container_width=True):
        st.session_state.result_cache = {}
        st.session_state.latest_results = None
        st.rerun()
    st.divider()
    st.caption("Anonymisez les cas réels avant usage.")
    st.caption("Prototype de recherche, pas conseil fiscal.")

render_app_shell()

query_col, config_col = st.columns([1.7, 0.7], gap="large")

with config_col:
    with st.container(border=True):
        st.markdown("### Connexion LLM")
        st.caption("Votre clé reste dans la session Streamlit.")
        provider_id = st.selectbox("Fournisseur", list(PROVIDERS.keys()), key="provider_select")
        provider = PROVIDERS[provider_id]
        api_key = st.text_input(
            f"Clé API ({provider['env_key']})",
            value="" if RUNNING_ON_SPACE else os.environ.get(provider["env_key"], ""),
            type="password",
            key="api_key_input",
            help="Sur HF, le champ reste vide. En local, .env.local peut pré-remplir la valeur.",
        )
        model_options = provider["models"]
        default_model = provider["default_model"]
        default_index = model_options.index(default_model) if default_model in model_options else 0
        model = st.selectbox(
            "Modèle",
            model_options,
            index=default_index,
            key=f"model_{provider_id}_select",
            help="Liste limitée aux modèles configurés pour ce prototype.",
        )
        st.caption(provider.get("note", ""))

    use_rewrite = True
    use_reranker = False

with query_col:
    with st.container(border=True):
        st.markdown("### Question fiscale")
        st.caption("Formulez le cas, puis contrôlez les passages BOFiP utilisés avant de lire la conclusion.")
        st.markdown(
            """
            <div class="inline-status">
              <span>Corpus : BOFiP commentaires</span>
              <span>Fraîcheur : 28/01/2026</span>
              <span>Reformulation : automatique</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        query = st.text_area(
            "Votre question",
            placeholder="Exemple : quel taux de TVA pour la pose d'une pompe à chaleur chez un particulier ?",
            height=150,
            label_visibility="collapsed",
            key="single_question",
        )
        submit = st.button("Analyser la question", type="primary", use_container_width=True)
        if submit:
            if not query.strip():
                st.warning("Saisissez une question avant de lancer l'analyse.")
            elif not api_key:
                st.warning("Saisissez une clé API dans le panneau Connexion LLM.")
            elif ensure_runtime_ready():
                rt = get_runtime(False, None)
                client = OpenAI(api_key=api_key, base_url=provider["base_url"])
                with st.spinner("Recherche, sélection des sources et génération..."):
                    st.session_state.latest_results = process_query(query.strip(), rt, client, model, True, False)

latest_results = st.session_state.get("latest_results")
if latest_results:
    display_results(latest_results)

st.markdown('<div class="app-footer">BOFiP Agentic RAG - prototype par Raphael Ifergan - sources BOFiP à vérifier avant usage professionnel.</div>', unsafe_allow_html=True)
