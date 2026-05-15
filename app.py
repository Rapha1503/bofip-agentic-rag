"""BOFIP RAG — Assistant Fiscal"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import streamlit as st
from openai import OpenAI
from bofip_cleanroom.env_utils import load_default_env_files
from bofip_cleanroom.rag_runtime import RagRuntime
from bofip_cleanroom.prompt_utils import build_prompt

PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "default_model": "deepseek-v4-flash",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.5", "gpt-5.4-mini", "gpt-5-mini", "gpt-4.1"],
        "default_model": "gpt-5.4-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "Anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"],
        "default_model": "claude-haiku-4-5",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-4", "mistral-large-3", "mistral-medium-3.5"],
        "default_model": "mistral-small-4",
        "env_key": "MISTRAL_API_KEY",
    },
    "Google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-3.1-flash-lite", "gemini-3.1-flash", "gemini-3.1-pro"],
        "default_model": "gemini-3.1-flash",
        "env_key": "GEMINI_API_KEY",
    },
}

st.set_page_config(page_title="BOFIP RAG", layout="wide")

@st.cache_resource(show_spinner="Chargement du runtime (~50s, une seule fois)...")
def get_runtime():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RagRuntime.from_local_corpus(corpus="commentary", device=device)

def rewrite_query(query, client, model):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"Reecris cette question en francais administratif et fiscal formel. Developpe les sigles et abreviations. Reponds UNIQUEMENT avec la question reformulee, sans guillemets ni commentaire."},
                  {"role":"user","content":query}],
        temperature=0.0, max_tokens=200,
    )
    return (resp.choices[0].message.content or "").strip() or query

def call_llm(prompt, client, model):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"Tu es un assistant fiscal prudent. Schema JSON strict."},
                  {"role":"user","content":prompt}],
        temperature=0.0, max_tokens=1200,
        response_format={"type":"json_object"},
    )
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return {"raw":content,"ptokens":getattr(usage,"prompt_tokens",None) if usage else None,
            "ctokens":getattr(usage,"completion_tokens",None) if usage else None}

def render_answer(parsed):
    status = parsed.get("answer_status","?").upper()
    c = parsed.get("conclusion","")
    bullets = parsed.get("justification_bullets",[])
    limits = parsed.get("limits","")
    axr, axc, axm = parsed.get("axes_requis",[]), parsed.get("axes_couverts",[]), parsed.get("axes_manquants",[])
    color = {"SUPPORTED":"green","PARTIAL":"orange","INSUFFICIENT_EVIDENCE":"red"}
    st.markdown(f"### Statut: :{color.get(status,'grey')}[**{status}**]")
    st.markdown(f"**{c}**")
    if axr:
        c1,c2,c3 = st.columns(3)
        c1.metric("Axes requis",len(axr)); c2.metric("Axes couverts",len(axc)); c3.metric("Axes manquants",len(axm))
        with st.expander("Détail des axes",expanded=False):
            st.caption("Requis: "+" | ".join(axr))
            st.caption("Couverts: "+(" | ".join(axc) if axc else "—"))
            st.caption("Manquants: "+(" | ".join(axm) if axm else "—"))
    st.markdown("**Justification:**")
    for b in bullets:
        st.markdown(f"- {b}")
    st.caption(f"Limites: {limits}")

def process_query(query, rt, client, llm_model, use_rewrite):
    results = {"query":query,"error":None}
    # Rewrite
    if use_rewrite:
        try:
            rewritten = rewrite_query(query, client, llm_model)
        except Exception as e:
            return {**results,"error":f"Erreur réécriture: {e}"}
    else:
        rewritten = query
    results["rewritten"] = rewritten
    # Retrieval
    try:
        result = rt.retrieve(rewritten, top_docs=8)
    except Exception as e:
        return {**results,"error":f"Erreur retrieval: {e}"}
    results["stage1"] = result.stage1_hits
    results["pipeline_log"] = getattr(result, "pipeline_log", {})
    # Chunks
    chunks = [{"rank":c.rank,"boi_reference":c.boi_reference,"title":c.title,
               "publication_date":c.publication_date,"section_path":c.section_path,
               "text":c.text,"chunk_id":c.chunk_id,"score":float(getattr(c,"score",0))}
              for c in result.stage2_chunks]
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
    try:
        parsed = json.loads(llm_r["raw"])
    except json.JSONDecodeError:
        parsed = None
    results["parsed"] = parsed
    return results

def display_results(results):
    if results.get("error"):
        st.error(results["error"]); return
    with st.expander("🔄 RÉÉCRITURE", expanded=True):
        st.info(results["rewritten"])
        if results["rewritten"] != results["query"]:
            st.caption(f"Original: « {results['query']} »")
    with st.expander(f"📚 DOCUMENTS — Stage 1 ({len(results.get('stage1',[]))} docs)", expanded=True):
        rows = [{"#":h.rank,"Score":f"{h.score:.4f}","BOFIP":h.boi_reference,"Titre":h.title[:120]} for h in results.get("stage1",[])]
        if rows: st.dataframe(rows, use_container_width=True, hide_index=True)
    chunks = results.get("chunks",[])
    with st.expander(f"✂️ CHUNKS — Stage 2 + Reranker ({len(chunks)} final)", expanded=True):
        for i,c in enumerate(chunks):
            bg = "#e8f5e9" if i==0 else "#f0f2f6"
            st.markdown(f'<div style="background:{bg};padding:10px;border-radius:5px;margin-bottom:8px">'
                        f'<b>[{c["rank"]}] {c["boi_reference"]}</b> — score: {c["score"]:.4f}<br>'
                        f'<span style="font-size:12px;color:#666">📂 {c["section_path"]}</span><br>'
                        f'<span style="font-size:13px">{c["text"][:300]}{"..." if len(c["text"])>300 else ""}</span></div>',
                        unsafe_allow_html=True)
    with st.expander("🤖 PROMPT ENVOYÉ AU LLM", expanded=False):
        st.code(results.get("prompt",""), language="text")
    with st.expander("✅ RÉPONSE", expanded=True):
        p = results.get("parsed")
        if p: render_answer(p)
        else: st.warning("JSON invalide"); st.text(results.get("llm_raw","")[:500])
    with st.expander("📋 DÉBOGAGE", expanded=False):
        plog = results.get("pipeline_log", {})
        if plog:
            c1,c2,c3 = st.columns(3)
            c1.metric("Docs uniques", plog.get("unique_docs_final","?"))
            c2.metric("Max/doc", plog.get("max_chunks_per_doc","?"))
            c3.metric("Candidats S2", plog.get("stage2_candidates","?"))
            dropped = plog.get("stage1_docs_dropped",[])
            if dropped:
                st.caption("Docs S1 non retenus: " + ", ".join(d[:30] for d in dropped))
            dist = plog.get("doc_distribution_final",{})
            if dist:
                st.caption("Distribution: " + str(dist))
        c1,c2 = st.columns(2)
        c1.metric("Prompt tokens", results.get("ptokens","?"))
        c2.metric("Completion tokens", results.get("ctokens","?"))
        st.code(results.get("llm_raw",""), language="json")

# ── UI ──
st.title("BOFIP RAG — Assistant Fiscal 🇫🇷")

with st.sidebar:
    st.header("⚙️ Configuration")
    provider_id = st.selectbox("Fournisseur LLM", list(PROVIDERS.keys()), key="provider_select")
    provider = PROVIDERS[provider_id]
    load_default_env_files()
    api_key = st.text_input(f"Clé API ({provider['env_key']})", value=os.environ.get(provider["env_key"],""),
                            type="password", key="api_key_input",
                            help="Chargée depuis .env.local si disponible. Non sauvegardée.")
    model = st.selectbox("Modèle", provider["models"], key=f"model_{provider_id}")
    use_rewrite = st.checkbox("Réécriture de la question", value=True,
                              help="Reformule la question en vocabulaire fiscal avant la recherche.")
    st.divider()
    st.caption("Corpus: 5666 documents BOFIP")
    st.caption("Modèles: E5-large (docs) / E5-base (chunks)")
    st.caption("Reranker: bge-reranker-v2-m3")

if not api_key:
    st.warning(f"Entrez une clé API **{provider['env_key']}** dans la barre latérale.")
    st.stop()

rt = get_runtime()
import torch
st.caption(f"Appareil: {'GPU' if torch.cuda.is_available() else 'CPU'}")

base_url = provider["base_url"]
client = OpenAI(api_key=api_key, base_url=base_url, **({"default_headers":{"x-api-key":api_key}} if provider_id=="Anthropic" else {}))

tab1, tab2 = st.tabs(["🔍 Question unique", "📋 Test par lot"])

with tab1:
    query = st.text_input("Votre question", placeholder="Quel taux de TVA pour une pompe à chaleur ?")
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
            progress = st.progress(0); status_text = st.empty(); all_results = []
            for i,q in enumerate(queries):
                status_text.text(f"[{i+1}/{len(queries)}] {q[:80]}...")
                progress.progress((i+1)/len(queries))
                all_results.append(process_query(q, rt, client, model, use_rewrite))
            progress.empty(); status_text.empty()
            st.markdown("### Résumé")
            rows = []
            for res in all_results:
                parsed = res.get("parsed")
                sts = parsed.get("answer_status","error") if parsed else "error"
                conc = (parsed.get("conclusion","")[:80] if parsed else str(res.get("error","")))
                rows.append({"Question":res["query"][:80],"Statut":sts,"Réponse":conc})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            for i,res in enumerate(all_results):
                st.markdown(f"---\n### Q{i+1}: {res['query'][:100]}")
                display_results(res)
