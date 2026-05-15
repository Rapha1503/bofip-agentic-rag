"""
BOFIP Agentic RAG — Streamlit application.
Clean, professional French fiscal aesthetic. No AI slop.
"""
import streamlit as st
import time
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

from bofip_agentic.agent_rag import AgenticRAG
from bofip_agentic.rag_runtime import RagRuntime

# ─── Page config ───
st.set_page_config(
    page_title="BOFIP Agentic RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Dark theme override ── */
.stApp {
    background: #0d1117;
}

section.main > div {
    padding-top: 1rem;
}

/* ── Header ── */
.main-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.main-header h1 {
    font-size: 1.75rem;
    font-weight: 600;
    color: #f0f0f0;
    letter-spacing: -0.02em;
    margin: 0;
}

.main-header span {
    font-size: 0.85rem;
    color: #8b949e;
    font-weight: 400;
}

/* ── Input area ── */
div[data-testid="stTextArea"] textarea {
    background: #161b22 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(200,169,81,0.4) !important;
    box-shadow: 0 0 0 2px rgba(200,169,81,0.1) !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #484f58 !important;
}

/* ── Button ── */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #1a3a6b, #1e4d8c) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #1e4d8c, #2563a8) !important;
    border-color: rgba(200,169,81,0.3) !important;
}

/* ── Results card ── */
.result-card {
    background: #161b22;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 1rem;
}

.status-supported {
    background: rgba(63,185,80,0.12);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.2);
}

.status-partial {
    background: rgba(210,153,34,0.12);
    color: #d29922;
    border: 1px solid rgba(210,153,34,0.2);
}

.status-insufficient {
    background: rgba(248,81,73,0.12);
    color: #f85149;
    border: 1px solid rgba(248,81,73,0.2);
}

/* ── Conclusion ── */
.conclusion-text {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #e6edf3;
    margin: 1rem 0;
}

/* ── Meta row ── */
.meta-row {
    display: flex;
    gap: 1.5rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}

.meta-item {
    font-size: 0.82rem;
    color: #8b949e;
}

.meta-item strong {
    color: #c9d1d9;
    font-weight: 500;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    margin: 0.5rem 0 !important;
}

[data-testid="stExpander"] summary {
    color: #c9d1d9 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Source list ── */
.source-item {
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 6px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    font-size: 0.82rem;
}

.source-item .ref {
    color: #58a6ff;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 0.78rem;
}

.source-item .title {
    color: #e6edf3;
    font-weight: 500;
    margin-top: 0.25rem;
}

.source-item .section {
    color: #8b949e;
    font-size: 0.75rem;
    margin-top: 0.25rem;
}

.source-item .text {
    color: #c9d1d9;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    line-height: 1.5;
    max-height: 120px;
    overflow-y: auto;
}

/* ── Trace steps ── */
.trace-step {
    border-left: 2px solid rgba(200,169,81,0.3);
    padding-left: 1rem;
    margin: 0.75rem 0;
}

.trace-step .step-label {
    color: #8b949e;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.trace-step .step-value {
    color: #e6edf3;
    font-size: 0.85rem;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] {
    color: #8b949e !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 1.5rem 0 !important;
}

/* ── Info boxes ── */
.info-box {
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.1);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.85rem;
    color: #c9d1d9;
}

/* ── Sidebar / footer ── */
.footer-meta {
    font-size: 0.72rem;
    color: #484f58;
    text-align: center;
    margin-top: 2rem;
}

/* Mobile tweaks */
@media (max-width: 640px) {
    .main-header h1 { font-size: 1.3rem; }
    .meta-row { gap: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ───
if "runtime" not in st.session_state:
    st.session_state.runtime = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "api_key" not in st.session_state:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    st.session_state.api_key = key


@st.cache_resource(show_spinner=False)
def get_runtime(device: str):
    return RagRuntime.from_local_corpus(corpus="commentary", device=device)


def init_agent():
    if st.session_state.runtime is None:
        with st.spinner("Chargement du moteur de recherche..."):
            device = "cuda"
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
            except:
                device = "cpu"
            st.session_state.runtime = get_runtime(device)
    if st.session_state.agent is None:
        st.session_state.agent = AgenticRAG(
            st.session_state.runtime,
            api_key=st.session_state.api_key,
            max_iterations=2,
        )


# ─── Header ───
st.markdown("""
<div class="main-header">
    <div>
        <h1>BOFIP Agentic RAG</h1>
        <span>Recherche fiscale augmentée • 5 666 documents • Analyse auto-évaluée</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── API Key input (if not set) ───
if not st.session_state.api_key:
    st.session_state.api_key = st.text_input(
        "Clé API DeepSeek",
        type="password",
        placeholder="sk-...",
        help="https://platform.deepseek.com/api_keys",
    )

# ─── Main input ───
question = st.text_area(
    "Votre question",
    placeholder="Ex: Puis-je récupérer la TVA sur l'achat d'une voiture de tourisme pour mon entreprise ?",
    height=100,
    label_visibility="collapsed",
    key="question_input",
)

col1, col2 = st.columns([3, 1])
with col1:
    pass
with col2:
    run = st.button("Analyser", use_container_width=True, disabled=not (question and st.session_state.api_key))

# ─── Process ───
if run and question:
    init_agent()

    t0 = time.time()
    with st.spinner("Analyse en cours..."):
        result = st.session_state.agent.run(question)
    elapsed = round(time.time() - t0, 1)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ─── Status badge ───
    status = result.get("answer_status", "partial")
    status_labels = {
        "supported": "Réponse complète",
        "partial": "Réponse partielle",
        "insufficient_evidence": "Informations insuffisantes",
    }
    badge_class = {
        "supported": "status-supported",
        "partial": "status-partial",
        "insufficient_evidence": "status-insufficient",
    }

    st.markdown("""
    <div class="result-card">
        <span class="status-badge {badge}">{label}</span>
        <div class="conclusion-text">{conclusion}</div>
    """.format(
        badge=badge_class.get(status, "status-partial"),
        label=status_labels.get(status, status),
        conclusion=result.get("conclusion", ""),
    ), unsafe_allow_html=True)

    # ─── Meta ───
    coverage = result.get("coverage", 0)
    iters = result.get("iterations", 1)
    chunks = result.get("chunks_used", 0)
    st.markdown("""
        <div class="meta-row">
            <div class="meta-item">Couverture <strong>{cov:.0%}</strong></div>
            <div class="meta-item">Itérations <strong>{iters}</strong></div>
            <div class="meta-item">Documents <strong>{chunks}</strong></div>
            <div class="meta-item">Temps <strong>{elapsed}s</strong></div>
        </div>
    </div>
    """.format(cov=coverage, iters=iters, chunks=chunks, elapsed=elapsed), unsafe_allow_html=True)

    # ─── Analyse détaillée (expander) ───
    with st.expander("Analyse détaillée", expanded=False):
        axes_requis = result.get("axes_requis", [])
        axes_couverts = result.get("axes_couverts", [])
        axes_manquants = result.get("axes_manquants", [])
        bullets = result.get("justification_bullets", [])
        limits = result.get("limits", "")

        if axes_requis:
            st.markdown("**Axes fiscaux requis**")
            for a in axes_requis:
                if a in axes_couverts:
                    st.markdown("✓ {}".format(a))
                else:
                    st.markdown("✗ {}".format(a))

        if bullets:
            st.markdown("**Justification**")
            for b in bullets:
                st.markdown("- {}".format(b))

        if axes_manquants:
            st.markdown("**Axes non couverts**")
            for a in axes_manquants:
                st.markdown("- {}".format(a))

        if limits:
            st.markdown("**Limites**")
            st.caption(limits)

    # ─── Sources (expander) ───
    with st.expander("Sources consultées", expanded=False):
        sources = result.get("sources", [])
        if sources:
            for i, s in enumerate(sources[:8]):
                ref = s.get("boi_reference", "")
                title = s.get("title", "")
                section = s.get("section_path", "")
                text = s.get("text", "")

                # Truncate long text intelligently
                if len(text) > 500:
                    text = text[:500] + "..."

                st.markdown("""
                <div class="source-item">
                    <div class="ref">[{n}] {ref}</div>
                    <div class="title">{title}</div>
                    <div class="section">{section}</div>
                    <div class="text">{text}</div>
                </div>
                """.format(n=i + 1, ref=ref, title=title, section=section, text=text), unsafe_allow_html=True)
        else:
            st.caption("Aucune source détaillée disponible.")

    # ─── Trace agent (expander) ───
    with st.expander("Trace de l'agent", expanded=False):
        trace = result.get("trace", [])
        for i, t in enumerate(trace):
            st.markdown('<div class="trace-step">', unsafe_allow_html=True)
            st.markdown('<span class="step-label">Itération {}</span>'.format(i + 1), unsafe_allow_html=True)
            st.markdown('<span class="step-value">Statut: {} | {} documents trouvés | {} extraits</span>'.format(
                t.get("answer_status", "?"),
                t.get("docs_found", 0),
                t.get("chunks_new", 0),
            ), unsafe_allow_html=True)
            if t.get("reformulated_query"):
                st.caption("Reformulation: {}".format(t["reformulated_query"]))
            st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ───
if not run:
    st.markdown("""
    <div class="info-box">
        Ce moteur de recherche utilise un agent qui s'auto-évalue. Il reformule automatiquement
        les questions en vocabulaire technique BOFIP quand la première recherche est incomplète.
    </div>
    <div class="footer-meta">
        BOFIP Agentic RAG · 5 666 documents BOFIP · E5-large + bge-reranker-v2-m3 · DeepSeek V4 Flash
    </div>
    """, unsafe_allow_html=True)
