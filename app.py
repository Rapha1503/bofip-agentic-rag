"""BOFiP Agentic RAG - Streamlit app."""
from __future__ import annotations
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
from bofip_cleanroom.rag_runtime import RagRuntime
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
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
        "default_model": "gpt-4.1-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "mistral-large-latest"],
        "default_model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
    },
    "Google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "default_model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
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


def _missing_runtime_paths() -> list[Path]:
    return [path for path in REQUIRED_RUNTIME_PATHS if not path.exists()]


st.set_page_config(page_title="BOFiP Agentic RAG", layout="wide")

@st.cache_resource(show_spinner="Chargement du runtime full corpus...")
def get_runtime(load_reranker: bool, reranker_model: str | None):
    import torch
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
    status = parsed.get("answer_status","?").upper()
    c = parsed.get("conclusion","")
    bullets = parsed.get("justification_bullets",[])
    limits = parsed.get("limits","")
    axr, axc, axm = parsed.get("axes_requis",[]), parsed.get("axes_couverts",[]), parsed.get("axes_manquants",[])
    color = {"SUPPORTED":"green","PARTIAL":"orange","INSUFFICIENT_EVIDENCE":"red"}
    st.markdown(f"### Statut: :{color.get(status,'grey')}[**{status}**]")
    st.markdown("---")
    st.markdown(f"### 📝 ANSWER")
    st.markdown(f"> {c}")
    st.markdown("")
    st.markdown("**Analyse détaillée :**")
    for b in bullets:
        st.markdown(f"- {b}")
    st.caption(f"*{limits}*")

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

def display_results(results):
    if results.get("error"):
        st.error(results["error"]); return

    # 1. Original question
    st.markdown(f"**❓ Question:** {results['query']}")

    # 2. Rewrite (only if different)
    if results.get("rewritten") and results["rewritten"] != results["query"]:
        st.caption(f"🔄 Reformulée: {results['rewritten'][:200]}")

    # 3. ANSWER
    p = results.get("parsed")
    if p:
        render_answer(p)
    else:
        st.warning("JSON invalide"); st.text(results.get("llm_raw","")[:500])
        p = {"answer_status": "parse_error"}

    # 4. Documents utilisés
    with st.expander(f"📚 Documents utilisés — Stage 1 ({len(results.get('stage1',[]))} docs)", expanded=False):
        rows = [{"#":h.rank,"Score":f"{h.score:.4f}","BOFIP":h.boi_reference,"Titre":h.title[:120]} for h in results.get("stage1",[])]
        if rows: st.dataframe(rows, use_container_width=True, hide_index=True)

    # 5. Technical details (collapsed)
    with st.expander("🔧 Détails techniques (chunks, prompt, debug)", expanded=False):
        chunks = results.get("chunks",[])
        st.caption(f"Chunks finaux: {len(chunks)}")
        for i,c in enumerate(chunks):
            bg = "#1a472a" if i==0 else "#1a1a2e"
            st.markdown(f'<div style="background:{bg};padding:10px;border-radius:5px;margin-bottom:8px;color:#e0e0e0">'
                        f'<b>[{c["rank"]}] {c["boi_reference"]}</b> — score: {c["score"]:.4f}<br>'
                        f'<span style="font-size:12px;color:#aab">📂 {c["section_path"]}</span><br>'
                        f'<span style="font-size:13px">{c["text"][:300]}{"..." if len(c["text"])>300 else ""}</span></div>',
                        unsafe_allow_html=True)
        with st.expander("🤖 Prompt LLM"):
            st.code(results.get("prompt",""), language="text")
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
        st.caption(f"Prompt tokens: {results.get('ptokens','?')} | Completion: {results.get('ctokens','?')}")
        st.code(results.get("llm_raw",""), language="json")

# ── UI ──
st.title("BOFiP Agentic RAG")
st.caption("Prototype de recherche par Rapha1503. Ne constitue pas un conseil fiscal.")

with st.sidebar:
    st.header("Configuration")
    st.caption("Projet portfolio de Rapha1503. Les cles API saisies ne sont pas sauvegardees par l'app.")
    provider_id = st.selectbox("Fournisseur LLM", list(PROVIDERS.keys()), key="provider_select")
    provider = PROVIDERS[provider_id]
    load_default_env_files()
    api_key = st.text_input(f"Clé API ({provider['env_key']})", value=os.environ.get(provider["env_key"],""),
                            type="password", key="api_key_input",
                            help="Chargée depuis .env.local si disponible. Non sauvegardée.")
    model = st.text_input(
        "Modele",
        value=provider["default_model"],
        key=f"model_{provider_id}",
        help="ID modifiable. Utilisez un modele disponible sur votre compte fournisseur.",
    )
    use_rewrite = st.checkbox("Réécriture de la question", value=True,
                              help="Reformule la question en vocabulaire fiscal avant la recherche.")
    reranker_available = RERANKER_MODEL_PATH.exists()
    use_reranker = st.checkbox(
        "Cross-encoder reranker",
        value=reranker_available,
        help="Ameliore le classement final. Si le modele local est absent, laissez desactive pour un demarrage CPU plus leger.",
    )
    if use_reranker and not reranker_available:
        st.warning("Modele reranker local absent: le chargement peut tenter un telechargement Hugging Face.")
    st.caption("En demo hebergee, votre cle et vos questions transitent par le serveur Streamlit et le fournisseur choisi.")
    st.divider()
    st.caption("Corpus: 5666 documents BOFIP")
    st.caption("Modeles: E5-large (docs) / E5-large (chunks)")
    st.caption("Reranker: bge-reranker-v2-m3 optionnel")
    if st.button("Vider le cache"):
        st.session_state.result_cache = {}
        st.rerun()

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

artifact_errors = validate_runtime_artifacts(PROJECT_ROOT)
if artifact_errors:
    st.error("Artefacts full-corpus invalides.")
    st.code("\n".join(artifact_errors))
    st.stop()

if not api_key:
    st.warning(f"Entrez une clé API **{provider['env_key']}** dans la barre latérale.")
    st.stop()

reranker_model = str(RERANKER_MODEL_PATH) if RERANKER_MODEL_PATH.exists() else None
rt = get_runtime(use_reranker, reranker_model)
import torch
st.caption(f"Appareil: {'GPU' if torch.cuda.is_available() else 'CPU'}")

base_url = provider["base_url"]
client = OpenAI(api_key=api_key, base_url=base_url)

tab1, tab2 = st.tabs(["Question unique", "Test par lot"])

with tab1:
    query = st.text_input("Votre question", placeholder="Quel taux de TVA pour une pompe à chaleur ?")
    if st.button("Rechercher", type="primary", disabled=not query.strip()):
        with st.spinner("Recherche en cours..."):
            results = process_query(query, rt, client, model, use_rewrite, use_reranker)
        st.markdown("---")
        display_results(results)

with tab2:
    st.caption("Collez plusieurs questions (une par ligne)")
    batch_text = st.text_area("Questions", height=120, placeholder="Quel taux de TVA pour une pompe à chaleur ?\n\nComment sont imposés les gains...", help="Séparez les questions par une ligne vide.")
    if st.button("Lancer le lot", type="primary", disabled=not batch_text.strip()):
        queries = [q.strip() for q in batch_text.strip().split("\n\n") if q.strip()]
        if len(queries) > 5:
            st.warning("Lot limite a 5 questions pour la demo publique.")
            queries = queries[:5]
        if queries:
            progress = st.progress(0); status_text = st.empty(); all_results = []
            for i,q in enumerate(queries):
                status_text.text(f"[{i+1}/{len(queries)}] {q[:80]}...")
                progress.progress((i+1)/len(queries))
                all_results.append(process_query(q, rt, client, model, use_rewrite, use_reranker))
            progress.empty(); status_text.empty()
            st.markdown("### Résumé")
            rows = []
            for res in all_results:
                parsed = res.get("parsed")
                sts = parsed.get("answer_status","error") if parsed else "error"
                conc = (parsed.get("conclusion","")[:80] if parsed else str(res.get("error","")))
                rows.append({"Question":res["query"][:80],"Statut":sts,"Réponse":conc})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            expand_all = st.checkbox("📂 Tout développer", value=False, key="expand_batch")
            for i,res in enumerate(all_results):
                st.markdown(f"---")
                with st.expander(f"### Q{i+1}: {res['query'][:100]}", expanded=expand_all):
                    display_results(res)
