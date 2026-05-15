"""Pipeline profiler — diverse queries with per-stage timing + answer audit."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from openai import OpenAI
from bofip_cleanroom.rag_runtime import RagRuntime
from bofip_cleanroom.env_utils import load_default_env_files

QUERIES = [
    # ── BOFIP doctrine (direct lookup) ──
    ("bo_tva", "Quel taux de TVA pour la pose d une pompe a chaleur chez un particulier ?", "TVA-LIQ"),

    # ── LPF only (procedure) ──
    ("lp_verif", "Je viens de recevoir un avis de verification de comptabilite. L administration peut-elle aussi examiner mes comptes bancaires personnels ?", "CF/LPF"),
    ("lp_presc", "J ai fait une erreur dans ma declaration de 2021 qui minore l impot du. En 2025, le fisc peut-il encore me redresser ?", "CF-PGR/LPF"),

    # ── CGI only (code articles) ──
    ("cg_div", "Je detiens des parts de SARL. L abattement de 40 pourcent sur les dividendes, comment ca marche ?", "RPPM/IR"),
    ("cg_seuil", "Je suis consultant en micro-entreprise, en dessous de quel CA je ne facture pas la TVA ?", "TVA/CGI"),

    # ── Mixed BOFIP + LPF ──
    ("mx_controle", "En cas de controle fiscal, mon entreprise peut-elle demander un delai supplementaire pour repondre au fisc ?", "CF/LPF"),
    ("mx_redress", "J ai recu une proposition de rectification. Quelles sont mes garanties pour me defendre ?", "CF-PGR/CTX"),

    # ── Mixed BOFIP + CGI ──
    ("mx_pv", "Je vends un terrain constructible herite il y a 15 ans. Comment calculer la plus-value imposable ?", "RFPI-PVI"),

    # ── All 3 mixed (BOFIP + CGI + LPF) ──
    ("mx_ifi", "L administration conteste la valeur de ma residence dans ma declaration d IFI. Quels sont mes droits pour contester ?", "PAT-IFI/CTX/LPF"),
]

load_default_env_files()
device = "cuda" if torch.cuda.is_available() else "cpu"
rt = RagRuntime.from_local_corpus(corpus="commentary", device=device)

# Build prompt inline (same as app.py)
def build_prompt(q, chunks):
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['rank']}] BOI: {c['boi_reference']}\nTitre: {c['title']}\nDate: {c['publication_date'] or 'inconnue'}\nSection: {c['section_path'] or '(sans section)'}\nTexte: {c['text']}")
    return (
        "Question utilisateur:\n" + q + "\n\nExtraits BOFiP fournis:\n" + "\n\n".join(blocks) + "\n\n"
        'Instructions: Reponds UNIQUEMENT a partir des extraits. Renvoie un JSON valide: {"answer_status":"supported|partial|insufficient_evidence","axes_requis":[],"axes_couverts":[],"axes_manquants":[],"conclusion":"...","justification_bullets":[],"limits":"..."}\n'
        "Etape 1 - Identifier les axes fiscaux requis.\nEtape 2 - Verifier couverture (supported/partial/insufficient).\nEtape 3 - 2-4 puces avec citations [n].\n"
    )

client = OpenAI(api_key=__import__("os").environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")

print(f"{'query':<14} {'stage1_ms':>9} {'rerank_ms':>9} {'llm_ms':>9} {'total_s':>7} {'status':<24} {'axes':<10} {'top_doc'}")
print("-" * 140)

for qid, query, domain in QUERIES:
    # Stage 1: retrieval without reranker
    t0 = time.time()
    r1 = rt.retrieve(query, top_docs=8, use_reranker=False)
    t_stage1 = time.time() - t0

    # Stage 2: reranker only (full pipeline minus stage1 time)
    t0 = time.time()
    r2 = rt.retrieve(query, top_docs=8, use_reranker=True)
    t_rerank = max(0, time.time() - t0 - t_stage1)

    # Build chunks + prompt + LLM
    chunks = [{"rank": c.rank, "boi_reference": c.boi_reference, "title": c.title,
               "publication_date": c.publication_date, "section_path": c.section_path,
               "text": c.text, "score": float(getattr(c, "score", 0))} for c in r2.stage2_chunks]
    prompt = build_prompt(query, chunks)

    t0 = time.time()
    resp = client.chat.completions.create(model="deepseek-chat",
        messages=[{"role":"system","content":"Tu es un assistant fiscal prudent. Schema JSON strict."},
                  {"role":"user","content":prompt}],
        temperature=0.0, max_tokens=300, response_format={"type":"json_object"})
    t_llm = time.time() - t0

    raw = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    status = parsed.get("answer_status", "parse_error")
    axes_c = len(parsed.get("axes_couverts", []))
    axes_m = len(parsed.get("axes_manquants", []))
    axes_str = f"{axes_c}c/{axes_m}m"
    top_doc = r2.stage1_hits[0].boi_reference if r2.stage1_hits else "NONE"
    total_s = t_stage1 + t_rerank + t_llm

    print(f"{qid:<14} {t_stage1*1000:>9.0f} {t_rerank*1000:>9.0f} {t_llm*1000:>9.0f} {total_s:>7.1f} {status:<24} {axes_str:<10} {top_doc}")

    # Show chunks
    dash = chr(45)
    print(f"  Chunks: {' | '.join(f'{c[\"boi_reference\"].split(dash)[0]}-{c[\"boi_reference\"].split(dash)[1]}' for c in chunks[:3])}")
    print(f"  Answer: {(parsed.get('conclusion','') or raw)[:120]}")
    print()
