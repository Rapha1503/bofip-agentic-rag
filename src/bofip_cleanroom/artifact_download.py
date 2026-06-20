from __future__ import annotations

import os
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_ARTIFACT_BASE_URL = (
    "https://github.com/Rapha1503/bofip-agentic-rag/releases/download/full-corpus-v1"
)

RUNTIME_ARTIFACTS = [
    "data/interim/raw_docs_sample_5666.jsonl",
    "data/interim/chunks_section_window_sample_5666.jsonl",
    "data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy",
    "data/interim/chunk_dense_cache_5666_full_e5large.npy",
]


def should_auto_download_artifacts() -> bool:
    explicit = os.environ.get("BOFIP_AUTO_DOWNLOAD_ARTIFACTS", "").strip().lower()
    if explicit in {"1", "true", "yes"}:
        return True
    if explicit in {"0", "false", "no"}:
        return False
    return bool(os.environ.get("SPACE_ID"))


def missing_runtime_artifacts(project_root: Path) -> list[Path]:
    return [project_root / rel_path for rel_path in RUNTIME_ARTIFACTS if not (project_root / rel_path).exists()]


def _manifest_artifacts(project_root: Path) -> dict[str, dict]:
    manifest_path = project_root / "docs" / "full_corpus_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {item["path"]: item for item in manifest["artifacts"]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_runtime_artifacts(project_root: Path, *, check_hashes: bool = True) -> list[str]:
    manifest = _manifest_artifacts(project_root)
    errors = []
    for rel_path in RUNTIME_ARTIFACTS:
        path = project_root / rel_path
        expected = manifest[rel_path]
        if not path.exists():
            errors.append(f"{rel_path}: missing")
            continue
        if path.stat().st_size != int(expected["size_bytes"]):
            errors.append(f"{rel_path}: size mismatch")
            continue
        expected_hash = expected.get("sha256") if check_hashes else None
        if expected_hash and _sha256(path) != expected_hash:
            errors.append(f"{rel_path}: sha256 mismatch")
    return errors


def download_missing_runtime_artifacts(project_root: Path, *, base_url: str | None = None) -> list[Path]:
    source = (base_url or os.environ.get("BOFIP_ARTIFACT_BASE_URL") or DEFAULT_ARTIFACT_BASE_URL).rstrip("/")
    manifest = _manifest_artifacts(project_root)
    downloaded = []
    for target in missing_runtime_artifacts(project_root):
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{source}/{target.name}"
        with tempfile.NamedTemporaryFile(delete=False, dir=str(target.parent)) as tmp:
            tmp_path = Path(tmp.name)
        try:
            urllib.request.urlretrieve(url, tmp_path)
            rel_path = target.relative_to(project_root).as_posix()
            expected_hash = manifest[rel_path].get("sha256")
            if expected_hash and _sha256(tmp_path) != expected_hash:
                raise ValueError(f"sha256 mismatch for {target.name}")
            tmp_path.replace(target)
            downloaded.append(target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    return downloaded
