from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from bofip_cleanroom.env_utils import load_default_env_files
from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.rag_runtime import RagRuntime
from bofip_cleanroom.prompt_utils import build_prompt
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs

BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"
REWRITE_MODEL = "deepseek-chat"
MAX_ATTEMPTS = 5
BASE_DELAY = 3.0
MAX_TOKENS = 800

_RETRY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)
_CITE_RE = re.compile(r"\[([0-9,\s]+)\]")
_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def rewrite_query(query: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    system = (
        "Reecris cette question en francais administratif et fiscal formel. "
        "Developpe les sigles et abreviations. "
        "Utilise le vocabulaire technique de la fiscalite francaise "
        "(ex: nom complet des taxes, termes du Code General des Impots). "
        "Reponds UNIQUEMENT avec la question reformulee, sans guillemets ni commentaire."
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=REWRITE_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            if rewritten:
                return rewritten
            return query
        except (RateLimitError, APITimeoutError) as e:
            if attempt >= MAX_ATTEMPTS:
                break
            delay = max(BASE_DELAY * attempt, _retry_delay(e))
            time.sleep(delay)
        except APIError as e:
            if getattr(e, "status_code", None) == 503 and attempt < MAX_ATTEMPTS:
                time.sleep(BASE_DELAY * attempt)
                continue
            return query
    return query


def _retry_delay(error):
    m = _RETRY_RE.search(str(error))
    return float(m.group(1)) if m else 0.0


def call_deepseek(prompt, api_key):
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    system = (
        "Tu es un assistant fiscal prudent. Reponds depuis les extraits BOFiP fournis. "
        "Schema JSON strict. Pas de citation inventee."
    )
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            return {
                "raw": content,
                "api": True,
                "attempts": attempt,
                "error": None,
                "finish": getattr(resp.choices[0], "finish_reason", None),
                "ptokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "ctokens": getattr(usage, "completion_tokens", None) if usage else None,
            }
        except (RateLimitError, APITimeoutError) as e:
            last = e
            if attempt >= MAX_ATTEMPTS:
                break
            d = max(BASE_DELAY * attempt, _retry_delay(e))
            time.sleep(d)
        except APIError as e:
            if getattr(e, "status_code", None) == 503 and attempt < MAX_ATTEMPTS:
                last = e
                time.sleep(BASE_DELAY * attempt)
                continue
            return {"raw": "", "api": False, "attempts": attempt, "error": f"APIError: {e}"}
    tag = type(last).__name__ if last else "Unknown"
    return {"raw": "", "api": False, "attempts": attempt, "error": f"{tag}: {last}"}


def parse_json(raw):
    candidates = [raw.strip()]
    for m in _JSON_RE.finditer(raw):
        c = m.group(1).strip()
        if c:
            candidates.append(c)
    for c in candidates:
        try:
            p = json.loads(c)
            if isinstance(p, dict):
                return p
        except json.JSONDecodeError:
            continue
    return None


def extract_citations(text):
    ids = []
    for block in _CITE_RE.findall(text or ""):
        for v in block.split(","):
            if v.strip().isdigit():
                ids.append(int(v.strip()))
    return ids


_GAP_MARKERS = (
    "manquant","insuffi","non couvert","non trouve","lacune","non traite",
    "absence","pas couvert","aucun","ne precise","ne permet","ne mentionne",
    "delai","non precise","n est pas","ne sont pas","ne figure","ne detaille",
    "reste a","impossible de","ne contient","ne contiennent","inconnu",
    "sous reserve","pourrait","eventuellement","non fourni",
)


def _is_gap_bullet(text):
    import unicodedata
    normalized = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return any(m in normalized for m in _GAP_MARKERS)


def validate(answer, n_chunks):
    errors = []
    status = str(answer.get("answer_status", "")).strip()
    if status not in ("supported", "insufficient_evidence", "partial"):
        errors.append("bad answer_status")
    conc = str(answer.get("conclusion", "")).strip()
    raw_bullets = answer.get("justification_bullets", [])
    bullets = [b for b in raw_bullets if isinstance(b, str) and b.strip()] if isinstance(raw_bullets, list) else []
    limits = str(answer.get("limits", "")).strip()
    if not conc:
        errors.append("conclusion empty")
    if not limits:
        errors.append("limits empty")
    if not bullets:
        errors.append("bullets empty")
    all_cites = []
    if status == "supported":
        if not 2 <= len(bullets) <= 4:
            errors.append(f"supported bullets count {len(bullets)} need 2-4")
        for i, b in enumerate(bullets, 1):
            cids = extract_citations(b)
            all_cites.extend(cids)
            if not cids:
                errors.append(f"bullet {i} no citation")
    elif status == "partial":
        if not 2 <= len(bullets) <= 4:
            errors.append(f"partial bullets count {len(bullets)} need 2-4")
        for i, b in enumerate(bullets, 1):
            cids = extract_citations(b)
            all_cites.extend(cids)
            # Missing citation on a bullet is OK for partial (gap explanation)
            if not cids and not _is_gap_bullet(b):
                errors.append(f"bullet {i} no citation (partial allows missing cites only on gap bullets)")
    else:
        if not 1 <= len(bullets) <= 2:
            errors.append(f"insufficient_evidence bullets count {len(bullets)} need 1-2")
        for b in bullets:
            all_cites.extend(extract_citations(b))
    bad = [c for c in all_cites if c < 1 or c > n_chunks]
    if bad:
        errors.append(f"out-of-range citations: {bad}")
    unique = sorted(set(all_cites)) if all_cites else []
    # Check axes fields presence
    axes_req = answer.get("axes_requis", [])
    axes_cov = answer.get("axes_couverts", [])
    axes_manq = answer.get("axes_manquants", [])
    if not isinstance(axes_req, list) or not isinstance(axes_cov, list) or not isinstance(axes_manq, list):
        errors.append("axes_requis|axes_couverts|axes_manquants must be arrays")
    elif status == "partial" and not axes_manq:
        errors.append("partial status requires non-empty axes_manquants")
    return {
        "valid": not errors,
        "status": status or None,
        "conclusion": bool(conc),
        "bullets": bool(bullets),
        "limits": bool(limits),
        "bullet_count": len(bullets),
        "cite_ids": unique,
        "cite_count": len(all_cites),
        "errors": errors,
        "axes_requis": axes_req if isinstance(axes_req, list) else [],
        "axes_manquants": axes_manq if isinstance(axes_manq, list) else [],
    }


def render(answer):
    s = str(answer.get("answer_status", "?"))
    c = str(answer.get("conclusion", ""))
    lines = [f"[{s.upper()}] {c}"]
    # Show axes for partial
    if s == "partial":
        miss = answer.get("axes_manquants", [])
        if miss and isinstance(miss, list):
            lines.append(f"  Axes manquants: {', '.join(miss)}")
    lines.append("")
    for b in answer.get("justification_bullets", []):
        lines.append(f"- {b}")
    lines.append("")
    lines.append(f"Limites: {str(answer.get('limits', ''))}")
    return "\n".join(lines)


@dataclass
class Case:
    case_id: str
    query: str
    category: str = ""
    note: str = ""


def main():
    p = argparse.ArgumentParser(description="BOFIP RAG preview with DeepSeek")
    p.add_argument("--query", type=str, default="")
    p.add_argument("--input", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--case-ids", type=str, default="")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    ensure_data_dirs()
    load_default_env_files()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("ERROR: DEEPSEEK_API_KEY not found")
        return 1

    cases = []
    if args.input:
        for r in read_jsonl(Path(args.input)):
            cases.append(Case(
                case_id=r.get("case_id", r.get("query_id", f"c{len(cases)}")),
                query=r.get("query", r.get("text", "")),
                category=r.get("category", ""),
                note=r.get("note", ""),
            ))
    elif args.query:
        cases = [Case(case_id="q000", query=args.query)]
    else:
        print("Need --query or --input")
        return 1

    if args.case_ids:
        sel = {v.strip() for v in args.case_ids.split(",") if v.strip()}
        cases = [c for c in cases if c.case_id in sel]
    if args.limit > 0:
        cases = cases[:args.limit]

    out = Path(args.output) if args.output else REPORTS_DIR / f"preview_batch_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    total = len(cases)
    done_ids = set()
    rows = []
    if args.resume:
        prev = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        for r in prev.get("rows", []):
            v = r.get("validation", {})
            if v.get("valid") and r.get("api_called"):
                done_ids.add(r["case_id"])
                rows.append(r)

    print("Loading runtime...", flush=True)
    rt = RagRuntime.from_local_corpus(corpus="commentary", device=args.device)

    for idx, case in enumerate(cases, 1):
        if case.case_id in done_ids:
            print(f"  [{idx}/{total}] {case.case_id} SKIP (resumed)", flush=True)
            continue

        rewritten = rewrite_query(case.query, key)
        if rewritten != case.query:
            print(f"  [{idx}/{total}] {case.case_id} rewriting...", flush=True)
        else:
            print(f"  [{idx}/{total}] {case.case_id} rewriting...", flush=True)
        print(f"    -> {rewritten[:120]}", flush=True)

        print(f"    retrieving...", flush=True)
        try:
            result = rt.retrieve(rewritten, top_docs=8)
        except Exception as e:
            rows.append({
                "case_id": case.case_id, "query": case.query, "category": case.category,
                "note": case.note, "api_called": False, "error": f"RetrievalError: {e}",
                "answer_text": str(e), "raw": "", "structured": None,
                "validation": {"valid": False, "errors": [str(e)]}, "chunks": [],
            })
            continue

        chunks = [
            {
                "rank": c.rank, "boi_reference": c.boi_reference, "title": c.title,
                "publication_date": c.publication_date, "section_path": c.section_path,
                "text": c.text, "chunk_id": c.chunk_id,
            }
            for c in result.stage2_chunks
        ]

        prompt = build_prompt(case.query, chunks)
        print(f"    {len(result.stage1_hits)} docs -> LLM...", flush=True)
        api = call_deepseek(prompt, key)

        if api["error"]:
            rows.append({"case_id":case.case_id,"query":case.query,"rewritten_query":rewritten,
                         "category":case.category,"note":case.note,"api_called":False,
                         "error":api["error"],"answer_text":api["error"],"raw":api.get("raw",""),
                         "structured":None,"validation":{"valid":False,"errors":[api["error"]]},
                         "chunks":chunks})
            print(f"    FAIL: {api['error'][:80]}", flush=True)
            continue

        parsed = parse_json(api["raw"])
        if parsed is None:
            rows.append({"case_id":case.case_id,"query":case.query,"rewritten_query":rewritten,
                         "category":case.category,"note":case.note,"api_called":True,"error":None,
                         "answer_text":api["raw"][:200],"raw":api["raw"],
                         "structured":None,"validation":{"valid":False,"errors":["unparseable JSON"]},
                         "chunks":chunks})
            print("    FAIL: unparseable JSON", flush=True)
            continue

        val = validate(parsed, len(chunks))
        rows.append({"case_id":case.case_id,"query":case.query,"rewritten_query":rewritten,
                     "category":case.category,"note":case.note,"api_called":True,
                     "attempts":api["attempts"],"answer_text":render(parsed),"raw":api["raw"],
                     "structured":parsed,"validation":val,"chunks":chunks})
        tag = "VALID" if val["valid"] else "INVALID"
        print(f"    {tag} ({val['status']})", flush=True)

        # Incremental save after each query
        report = {"generated_at":datetime.now(UTC).isoformat(),"provider":"deepseek","model":MODEL,
                  "case_count":len(rows),"valid_count":sum(1 for r in rows if r.get("validation",{}).get("valid")),"rows":rows}
        write_json(out, report)

    valid_n = sum(1 for r in rows if r.get("validation", {}).get("valid"))
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": "deepseek",
        "model": MODEL,
        "manifest": {
            "corpus": "commentary",
            "doc_count": 5666,
            "chunk_count": 66289,
            "chunk_strategy": "section_window",
            "doc_model": "intfloat/multilingual-e5-large",
            "chunk_model": "intfloat/multilingual-e5-base",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "llm_model": MODEL,
            "pipeline_version": "v2",
        },
        "case_count": len(rows),
        "valid_count": valid_n,
        "rows": rows,
    }
    out = Path(args.output) if args.output else REPORTS_DIR / f"preview_batch_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(out, report)
    print(f"\nDone. Valid: {valid_n}/{len(rows)}. Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
