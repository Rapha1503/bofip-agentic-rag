"""BOFIP RAG — Assistant Fiscal"""
from __future__ import annotations
import hashlib, json, logging, os, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import streamlit as st
from openai import OpenAI
from bofip_cleanroom.env_utils import load_default_env_files
from bofip_cleanroom.rag_runtime import RagRuntime
from bofip_cleanroom.prompt_utils import build_prompt

# ── File logging ───────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_pipeline_logger = logging.getLogger("bofip_pipeline")
_pipeline_logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
_pipeline_logger.addHandler(_fh)
_pipeline_logger.propagate = False

def _log(step: str, data: dict):
    _pipeline_logger.info(f"[{step}] {json.dumps(data, ensure_ascii=False, default=str)[:2000]}")


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
    )
    content = (resp.choices[0].message.content or "").strip()
    # Strip markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(content)
        rewritten = data.get("rewritten_query", query) or query
        facets = data.get("facets", [])
        facet_queries = [f.get("query", rewritten) for f in facets if f.get("query")]
        return rewritten, facet_queries if facet_queries else [rewritten]
    except (json.JSONDecodeError, TypeError) as e:
        st.warning(f"Réécriture: JSON invalide ({e}). Question originale utilisée.")
        return query, [query]


def _detect_multi_axis(query: str) -> bool:
    """Heuristic: does this query need multi-facet retrieval?"""
    signals = 0
    qt = query.lower()
    # Multiple clauses
    if query.count("?") >= 2 or query.count(",") >= 2:
        signals += 1
    # Procedure/control keywords
    if any(w in qt for w in ("procédure", "redressement", "contrôle", "garantie", "sanction",
                              "délai", "prescription", "recours", "réclamation", "vérification")):
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

def process_query(query, rt, client, llm_model, use_rewrite, multi_axis="auto"):
    results = {"query":query,"error":None}

    # Cache check
    cache_key = hashlib.md5((query + llm_model + str(use_rewrite) + multi_axis).encode()).hexdigest()[:12]
    if "result_cache" not in st.session_state:
        st.session_state.result_cache = {}
    if cache_key in st.session_state.result_cache:
        cached = st.session_state.result_cache[cache_key]
        _log("CACHE_HIT", {"key": cache_key, "query": query[:80]})
        return cached

    t0 = time.time()
    _log("QUERY_START", {"query": query[:200], "model": llm_model, "rewrite": use_rewrite, "multi_axis": multi_axis})
    # Rewrite + optional facets
    if use_rewrite:
        try:
            rewritten, facet_queries = rewrite_query(query, client, llm_model)
        except Exception as e:
            return {**results,"error":f"Erreur réécriture: {e}"}
    else:
        rewritten, facet_queries = query, [query]
    # Auto-detect multi-axis
    if multi_axis == "auto" and len(facet_queries) <= 1:
        if _detect_multi_axis(query):
            facet_queries = [rewritten, query]
    elif multi_axis == "off":
        facet_queries = [rewritten]
    results["rewritten"] = rewritten
    results["facet_queries"] = facet_queries
    _log("REWRITE", {"original": query[:120], "rewritten": rewritten[:120], "facets": len(facet_queries)})
    # Retrieval — per facet, merge with diversity
    all_chunks_raw = []; all_stage1 = []; seen_docs = set(); main_log = {}
    for fq in facet_queries:
        try:
            res = rt.retrieve(fq, top_docs=5)
            for h in res.stage1_hits:
                if h.boi_reference not in seen_docs:
                    all_stage1.append(h); seen_docs.add(h.boi_reference)
            for c in res.stage2_chunks: all_chunks_raw.append(c)
            main_log = getattr(res, "pipeline_log", {})
        except Exception as e:
            return {**results,"error":f"Erreur retrieval: {e}"}
    # Post-merge diversity: max 3 chunks per document
    merged = []
    doc_counts = {}
    for c in all_chunks_raw:
        d = c.boi_reference
        doc_counts[d] = doc_counts.get(d, 0) + 1
        if doc_counts[d] <= 3:
            merged.append(c)
    all_chunks_raw = merged
    results["stage1"] = all_stage1[:8]
    results["pipeline_log"] = main_log
    _log("RETRIEVAL", {"stage1_docs": [h.boi_reference for h in all_stage1[:8]], 
          "merged_chunks": len(all_chunks_raw), "pipeline_log": main_log})
    chunks = [{"rank":c.rank,"boi_reference":c.boi_reference,"title":c.title,
               "publication_date":c.publication_date,"section_path":c.section_path,
               "text":c.text,"chunk_id":c.chunk_id,"score":float(getattr(c,"score",0))}
              for c in all_chunks_raw[:16]]
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
            bg = "#1a472a" if i==0 else "#1a1a2e"
            st.markdown(f'<div style="background:{bg};padding:10px;border-radius:5px;margin-bottom:8px;color:#e0e0e0">'
                        f'<b>[{c["rank"]}] {c["boi_reference"]}</b> — score: {c["score"]:.4f}<br>'
                        f'<span style="font-size:12px;color:#aab">📂 {c["section_path"]}</span><br>'
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
    multi_axis = st.selectbox("Recherche multi-axes", ["auto", "off", "on"], index=0,
                              help="Auto: activé si question complexe. On: toujours. Off: jamais.")
    st.divider()
    st.caption("Corpus: 5666 documents BOFIP")
    st.caption("Modèles: E5-large (docs) / E5-base (chunks)")
    st.caption("Reranker: bge-reranker-v2-m3")
    if st.button("🗑️ Vider le cache"):
        st.session_state.result_cache = {}
        st.rerun()

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
            results = process_query(query, rt, client, model, use_rewrite, multi_axis)
        display_results(results)

with tab2:
    st.caption("Collez plusieurs questions (une par ligne)")
    batch_text = st.text_area("Questions", height=120, placeholder="Quel taux de TVA pour une pompe à chaleur ?\n\nComment sont imposés les gains...", help="Séparez les questions par une ligne vide.")
    if st.button("Lancer le lot", type="primary", disabled=not batch_text.strip()):
        queries = [q.strip() for q in batch_text.strip().split("\n\n") if q.strip()]
        if queries:
            progress = st.progress(0); status_text = st.empty(); all_results = []
            for i,q in enumerate(queries):
                status_text.text(f"[{i+1}/{len(queries)}] {q[:80]}...")
                progress.progress((i+1)/len(queries))
                all_results.append(process_query(q, rt, client, model, use_rewrite, multi_axis))
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
                with st.expander(f"Afficher les détails", expanded=False):
                    display_results(res)
