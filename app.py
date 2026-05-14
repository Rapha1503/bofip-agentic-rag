"""BOFIP RAG — Assistant Fiscal (Streamlit UI v3 — multi-provider, deployable)"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st
from openai import OpenAI

from bofip_cleanroom.env_utils import load_default_env_files
from bofip_cleanroom.rag_runtime import RagRuntime

# ── Provider configuration ──────────────────────────────────────────
PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3-mini"],
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "Anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-opus-4-20250514"],
        "default_model": "claude-3-5-haiku-20241022",
        "env_key": "ANTHROPIC_API_KEY",
        "header_name": "x-api-key",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-large-latest", "mistral-small-latest", "codestral-latest"],
        "default_model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
    },
    "Google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "default_model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"],
        "default_model": "llama-4-scout-17b-16e-instruct",
        "env_key": "GROQ_API_KEY",
    },
    "Together": {
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "mistralai/Mixtral-8x22B-Instruct-v0.1"],
        "default_model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "env_key": "TOGETHER_API_KEY",
    },
}

st.set_page_config(page_title="BOFIP RAG", layout="wide")


@st.cache_resource(show_spinner="Chargement du runtime (~50s, une seule fois)...")
def get_runtime():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Auto-detect corpus: full (5666) or demo (200)
    from pathlib import Path as P
    corpus = "commentary" if (P("data/interim/raw_docs_sample_5666.jsonl").exists() and P("data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy").exists()) else "demo"
    st.caption(f"Corpus: {corpus} ({'GPU' if device == 'cuda' else 'CPU'})")
    return RagRuntime.from_local_corpus(corpus=corpus, device=device)


def build_prompt(query: str, chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c['rank']}] BOI: {c['boi_reference']}\n"
            f"Titre: {c['title']}\n"
            f"Date: {c['publication_date'] or 'inconnue'}\n"
            f"Section: {c['section_path'] or '(sans section)'}\n"
            f"Texte: {c['text']}"
        )
    return (
        "Question utilisateur:\n" + query + "\n\n"
        "Extraits BOFiP fournis:\n" + "\n\n".join(blocks) + "\n\n"
        "Instructions:\n"
        "- Tu es un assistant fiscal. Reponds UNIQUEMENT a partir des extraits fournis.\n"
        "- N'invente ni source, ni article, ni taux, ni reponse.\n"
        "- Renvoie un objet JSON valide et rien d'autre. Pas de markdown autour.\n\n"
        'Schema JSON: {"answer_status":"supported|partial|insufficient_evidence","axes_requis":["..."],"axes_couverts":["..."],"axes_manquants":["..."],"conclusion":"...","justification_bullets":["..."],"limits":"..."}\n\n'
        "Etape 1 - Identifier les axes fiscaux requis (1 a 5).\n"
        "Etape 2 - Verifier la couverture: supported (tous couverts) | partial (mixte) | insufficient_evidence.\n"
        "Etape 3 - 2-4 puces avec citations [n] pour axes couverts. Puce explicative pour chaque axe manquant.\n"
        "- limits obligatoire <= 40 mots. Lister axes manquants si partial.\n"
    )


def call_llm(prompt: str, client, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Tu es un assistant fiscal prudent. Schema JSON strict."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0, max_tokens=800,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return {
        "raw": content,
        "ptokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "ctokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


def rewrite_query_cached(query: str, client, model: str) -> str:
    system = (
        "Reecris cette question en francais administratif et fiscal formel. "
        "Developpe les sigles et abreviations. "
        "Reponds UNIQUEMENT avec la question reformulee, sans guillemets ni commentaire."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
        temperature=0.0, max_tokens=200,
    )
    return (resp.choices[0].message.content or "").strip() or query


def render_answer(parsed: dict) -> None:
    status = parsed.get("answer_status", "?").upper()
    conclusion = parsed.get("conclusion", "")
    bullets = parsed.get("justification_bullets", [])
    limits = parsed.get("limits", "")
    axes_req = parsed.get("axes_requis", [])
    axes_cov = parsed.get("axes_couverts", [])
    axes_manq = parsed.get("axes_manquants", [])

    color_map = {"SUPPORTED": "green", "PARTIAL": "orange", "INSUFFICIENT_EVIDENCE": "red"}
    st.markdown(f"### Statut: :{color_map.get(status, 'grey')}[**{status}**]")
    st.markdown(f"**{conclusion}**")

    if axes_req:
        cols = st.columns(3)
        cols[0].metric("Axes requis", len(axes_req))
        cols[1].metric("Axes couverts", len(axes_cov))
        cols[2].metric("Axes manquants", len(axes_manq))
        with st.expander("Détail des axes", expanded=False):
            st.caption("Requis: " + " | ".join(axes_req))
            st.caption("Couverts: " + (" | ".join(axes_cov) if axes_cov else "—"))
            st.caption("Manquants: " + (" | ".join(axes_manq) if axes_manq else "—"))

    st.markdown("**Justification:**")
    for b in bullets:
        st.markdown(f"- {b}")
    st.caption(f"Limites: {limits}")


def process_query(query: str, rt, client, llm_model: str, use_rewrite: bool) -> dict:
    """Run full pipeline. Returns dict with results or error."""
    results = {"query": query, "error": None}

    # 1. Rewrite
    if use_rewrite:
        try:
            rewritten = rewrite_query_cached(query, client, llm_model)
        except Exception as e:
            return {**results, "error": f"Erreur réécriture: {e}"}
    else:
        rewritten = query
    results["rewritten"] = rewritten

    # 2. Retrieval
    try:
        result = rt.retrieve(rewritten, top_docs=8)
    except Exception as e:
        return {**results, "error": f"Erreur retrieval: {e}"}
    results["stage1"] = result.stage1_hits

    # 3. Chunks
    chunks = [
        {"rank": c.rank, "boi_reference": c.boi_reference, "title": c.title,
         "publication_date": c.publication_date, "section_path": c.section_path,
         "text": c.text, "chunk_id": c.chunk_id, "score": getattr(c, "score", 0)}
        for c in result.stage2_chunks
    ]
    results["chunks"] = chunks

    if not chunks:
        results["parsed"] = {"answer_status": "insufficient_evidence", "conclusion": "Aucun extrait trouvé.",
                              "justification_bullets": ["La recherche n'a retourné aucun résultat."],
                              "limits": "Aucune source disponible.", "axes_requis": [], "axes_couverts": [], "axes_manquants": []}
        return results

    # 4. LLM
    prompt = build_prompt(query, chunks)
    results["prompt"] = prompt
    try:
        llm_result = call_llm(prompt, client, llm_model)
    except Exception as e:
        return {**results, "error": f"Erreur LLM: {e}"}
    results["llm_raw"] = llm_result["raw"]
    results["ptokens"] = llm_result["ptokens"]
    results["ctokens"] = llm_result["ctokens"]

    # 5. Parse
    try:
        parsed = json.loads(llm_result["raw"])
    except json.JSONDecodeError:
        parsed = None
    results["parsed"] = parsed

    return results


def display_results(results: dict):
    if results.get("error"):
        st.error(results["error"])
        return

    with st.expander("🔄 RÉÉCRITURE", expanded=True):
        st.info(results["rewritten"])
        if results["rewritten"] != results["query"]:
            st.caption(f"Original: « {results['query']} »")

    with st.expander(f"📚 DOCUMENTS — Stage 1 ({len(results.get('stage1',[]))} docs)", expanded=True):
        rows = [{"#": h.rank, "Score": f"{h.score:.4f}", "BOFIP": h.boi_reference, "Titre": h.title[:120]}
                for h in results.get("stage1", [])]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    chunks = results.get("chunks", [])
    with st.expander(f"✂️ CHUNKS — Stage 2 + Reranker ({len(chunks)} final)", expanded=True):
        for i, c in enumerate(chunks):
            bg = "#e8f5e9" if i == 0 else "#f0f2f6"
            st.markdown(
                f'<div style="background:{bg};padding:10px;border-radius:5px;margin-bottom:8px">'
                f'<b>[{c["rank"]}] {c["boi_reference"]}</b> — score: {c["score"]:.4f}<br>'
                f'<span style="font-size:12px;color:#666">📂 {c["section_path"]}</span><br>'
                f'<span style="font-size:13px">{c["text"][:300]}{"..." if len(c["text"])>300 else ""}</span>'
                f'</div>', unsafe_allow_html=True)

    with st.expander("🤖 PROMPT ENVOYÉ AU LLM", expanded=False):
        st.code(results.get("prompt", ""), language="text")

    with st.expander("✅ RÉPONSE", expanded=True):
        parsed = results.get("parsed")
        if parsed:
            render_answer(parsed)
        else:
            st.warning("JSON invalide")
            st.text(results.get("llm_raw", "")[:500])

    with st.expander("📋 DÉBOGAGE", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Prompt tokens", results.get("ptokens", "?"))
        c2.metric("Completion tokens", results.get("ctokens", "?"))
        st.code(results.get("llm_raw", ""), language="json")


# ── UI ───────────────────────────────────────────────────────────────
st.title("BOFIP RAG — Assistant Fiscal 🇫🇷")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    provider_id = st.selectbox("Fournisseur LLM", list(PROVIDERS.keys()), index=0)
    provider = PROVIDERS[provider_id]

    env_key = provider["env_key"]
    load_default_env_files()
    default_key = os.environ.get(env_key, "")

    api_key = st.text_input(
        f"Clé API ({env_key})",
        value=default_key,
        type="password",
        help=f"Chargée depuis .env.local si disponible. Non sauvegardée.",
    )

    model = st.selectbox("Modèle", provider["models"], index=0)

    use_rewrite = st.checkbox("Réécriture de la question", value=True,
                              help="Reformule la question en vocabulaire fiscal BOFIP avant la recherche.")

    st.divider()
    st.caption("Corpus: 5666 documents BOFIP (commentaire)")
    st.caption("Modèle doc: multilingual-e5-large")
    st.caption("Modèle chunk: multilingual-e5-base")
    st.caption("Reranker: bge-reranker-v2-m3")

if not api_key:
    st.warning(f"Entrez une clé API **{env_key}** dans la barre latérale pour commencer.")
    st.stop()

# Init runtime + client
rt = get_runtime()

base_url = provider["base_url"]
# Anthropic uses different auth header
if provider_id == "Anthropic":
    client = OpenAI(api_key=api_key, base_url=base_url, default_headers={"x-api-key": api_key})
else:
    client = OpenAI(api_key=api_key, base_url=base_url)

tab1, tab2 = st.tabs(["🔍 Question unique", "📋 Test par lot"])

with tab1:
    query = st.text_input("Votre question", placeholder="Quel taux de TVA pour une pompe à chaleur ?", key="single_query")
    if st.button("Rechercher", type="primary", disabled=not query.strip()):
        with st.spinner("Recherche en cours..."):
            results = process_query(query, rt, client, model, use_rewrite)
        display_results(results)

with tab2:
    st.caption("Collez plusieurs questions (une par ligne)")
    batch_text = st.text_area("Questions", height=120, placeholder="Quel taux de TVA pour une pompe à chaleur ?\nComment sont imposés les gains...")
    if st.button("Lancer le lot", type="primary", disabled=not batch_text.strip()):
        queries = [q.strip() for q in batch_text.strip().split("\n") if q.strip()]
        if queries:
            progress = st.progress(0)
            status_text = st.empty()
            all_results = []
            for i, q in enumerate(queries):
                status_text.text(f"[{i+1}/{len(queries)}] {q[:80]}...")
                progress.progress((i+1) / len(queries))
                all_results.append(process_query(q, rt, client, model, use_rewrite))
            progress.empty()
            status_text.empty()

            st.markdown("### Résumé")
            summary_rows = []
            for res in all_results:
                parsed = res.get("parsed")
                status = parsed.get("answer_status", "error") if parsed else "error"
                conclusion = (parsed.get("conclusion", "")[:80] if parsed else str(res.get("error", "")))
                summary_rows.append({"Question": res["query"][:80], "Statut": status, "Réponse": conclusion})
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)

            for i, res in enumerate(all_results):
                st.markdown(f"---\n### Q{i+1}: {res['query'][:100]}")
                display_results(res)
