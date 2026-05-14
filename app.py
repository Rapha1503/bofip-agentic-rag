"""BOFIP RAG — Assistant Fiscal (Streamlit UI v2)"""
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

BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

st.set_page_config(page_title="BOFIP RAG", layout="wide")

# --- Init ---
load_default_env_files()
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


@st.cache_resource(show_spinner="Chargement du runtime (~50s, une seule fois)...")
def get_runtime():
    return RagRuntime.from_local_corpus(corpus="commentary", device="cuda")


@st.cache_data(show_spinner=False, ttl=3600)
def rewrite_query_cached(query: str) -> str:
    if not API_KEY:
        return query
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    system = (
        "Reecris cette question en francais administratif et fiscal formel. "
        "Developpe les sigles et abreviations. "
        "Utilise le vocabulaire technique de la fiscalite francaise. "
        "Reponds UNIQUEMENT avec la question reformulee, sans guillemets ni commentaire."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
        temperature=0.0, max_tokens=200,
    )
    return (resp.choices[0].message.content or "").strip() or query


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
        "Etape 2 - Verifier la couverture: axes_couverts, axes_manquants.\n"
        "  supported: TOUS axes couverts | partial: COUVERT + MANQUANT | insufficient_evidence: RIEN couvert\n"
        "Etape 3 - Conclusion et justification:\n"
        "- conclusion <= 30 mots. Nuance si partial.\n"
        "- 2-4 puces. Citations [n] obligatoires pour axes couverts.\n"
        "- Si partial: puce expliquant chaque axe manquant.\n"
        "- limits obligatoire <= 40 mots. Lister axes manquants si partial.\n"
        "- Citations [n] referencent UNIQUEMENT les extraits fournis.\n"
    )


def call_llm(prompt: str) -> dict:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
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


def process_query(query: str, rt, progress_container=None):
    """Run full pipeline and return results for display."""
    results = {"query": query}

    # Rewrite
    rewritten = rewrite_query_cached(query)
    results["rewritten"] = rewritten

    # Retrieval
    result = rt.retrieve(rewritten, top_docs=8)
    results["stage1"] = result.stage1_hits

    # Build chunks
    chunks = [
        {"rank": c.rank, "boi_reference": c.boi_reference, "title": c.title,
         "publication_date": c.publication_date, "section_path": c.section_path,
         "text": c.text, "chunk_id": c.chunk_id, "score": getattr(c, "score", 0)}
        for c in result.stage2_chunks
    ]
    results["chunks"] = chunks

    # Prompt + LLM
    prompt = build_prompt(query, chunks)
    results["prompt"] = prompt
    llm_result = call_llm(prompt)
    results["llm_raw"] = llm_result["raw"]
    results["ptokens"] = llm_result["ptokens"]
    results["ctokens"] = llm_result["ctokens"]

    # Parse
    try:
        parsed = json.loads(llm_result["raw"])
    except json.JSONDecodeError:
        parsed = None
    results["parsed"] = parsed

    return results


def display_results(results: dict):
    """Display pipeline results in Streamlit."""
    # Rewrite
    with st.expander("🔄 RÉÉCRITURE", expanded=True):
        st.info(results["rewritten"])
        if results["rewritten"] != results["query"]:
            st.caption(f"Original: « {results['query']} »")

    # Stage 1 docs
    with st.expander(f"📚 DOCUMENTS — Stage 1 ({len(results['stage1'])} docs)", expanded=True):
        rows = [{"#": h.rank, "Score": f"{h.score:.4f}", "BOFIP": h.boi_reference, "Titre": h.title[:120]}
                for h in results["stage1"]]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Chunks
    chunks = results["chunks"]
    with st.expander(f"✂️ CHUNKS — Stage 2 + Reranker ({len(chunks)} final)", expanded=True):
        for i, c in enumerate(chunks):
            is_first = i == 0
            bg = "#e8f5e9" if is_first else "#f0f2f6"
            st.markdown(
                f'<div style="background:{bg};padding:10px;border-radius:5px;margin-bottom:8px">'
                f'<b>[{c["rank"]}] {c["boi_reference"]}</b> — score: {c["score"]:.4f}<br>'
                f'<span style="font-size:12px;color:#666">📂 {c["section_path"]}</span><br>'
                f'<span style="font-size:13px">{c["text"][:300]}{"..." if len(c["text"])>300 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Prompt
    with st.expander("🤖 PROMPT ENVOYÉ AU LLM", expanded=False):
        st.code(results["prompt"], language="text")

    # Answer
    with st.expander("✅ RÉPONSE", expanded=True):
        if results["parsed"]:
            render_answer(results["parsed"])
        else:
            st.warning("JSON invalide")
            st.text(results["llm_raw"][:500])

    # Raw + debug
    with st.expander("📋 DÉBOGAGE", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Prompt tokens", results.get("ptokens", "?"))
        c2.metric("Completion tokens", results.get("ctokens", "?"))
        st.code(results["llm_raw"], language="json")

        debug = {
            "query": results["query"],
            "rewritten": results["rewritten"],
            "stage1": [{"rank": h.rank, "ref": h.boi_reference, "title": h.title} for h in results["stage1"]],
            "chunks": [{"rank": c["rank"], "ref": c["boi_reference"], "section": c["section_path"],
                         "text": c["text"], "score": c["score"]} for c in results["chunks"]],
            "prompt": results["prompt"],
            "llm_raw": results["llm_raw"],
            "parsed": results["parsed"],
        }
        st.download_button("📥 Télécharger JSON", json.dumps(debug, indent=2, ensure_ascii=False),
                           file_name="bofip_debug.json", mime="application/json")


# --- UI ---
st.title("BOFIP RAG — Assistant Fiscal")

if not API_KEY:
    st.error("DEEPSEEK_API_KEY non trouvée. Vérifiez .env.local")
    st.stop()

rt = get_runtime()

tab1, tab2 = st.tabs(["🔍 Question unique", "📋 Test par lot"])

with tab1:
    query = st.text_input("Votre question", placeholder="Quel taux de TVA pour une pompe à chaleur ?", key="single_query")
    if st.button("Rechercher", type="primary", disabled=not query.strip()):
        with st.spinner("Recherche en cours..."):
            results = process_query(query, rt)
        display_results(results)

with tab2:
    st.caption("Collez plusieurs questions (une par ligne)")
    batch_text = st.text_area("Questions", height=150, placeholder="Quel taux de TVA pour une pompe à chaleur ?\nComment sont imposés les gains d'un compte-titres ordinaire ?\n...")
    if st.button("Lancer le lot", type="primary", disabled=not batch_text.strip()):
        queries = [q.strip() for q in batch_text.strip().split("\n") if q.strip()]
        if queries:
            all_results = []
            progress = st.progress(0)
            status_text = st.empty()
            for i, q in enumerate(queries):
                status_text.text(f"[{i+1}/{len(queries)}] {q[:80]}...")
                progress.progress((i+1) / len(queries))
                all_results.append(process_query(q, rt))

            progress.empty()
            status_text.empty()

            # Summary table
            st.markdown("### Résumé")
            summary_rows = []
            for res in all_results:
                parsed = res["parsed"]
                status = parsed.get("answer_status", "parse_error") if parsed else "parse_error"
                conclusion = (parsed.get("conclusion", "")[:80] if parsed else "JSON invalide")
                summary_rows.append({
                    "Question": res["query"][:100],
                    "Statut": status,
                    "Réponse": conclusion,
                    "Chunks": len(res["chunks"]),
                })
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)

            # Individual results
            for i, res in enumerate(all_results):
                st.markdown(f"---\n### Q{i+1}: {res['query'][:100]}")
                display_results(res)
