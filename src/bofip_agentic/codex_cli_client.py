"""OpenAI-compatible shim around the local Codex CLI.

This is intended for local portfolio testing only. It uses the user's installed
Codex CLI/auth and is not expected to work on hosted deployments.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace


class CodexCliClient:
    """Minimal client exposing ``chat.completions.create`` like OpenAI."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        project_root: Path | str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.model = model
        self.project_root = Path(project_root or ".").resolve()
        self.timeout_seconds = timeout_seconds or int(os.environ.get("BOFIP_CODEX_TIMEOUT", "240"))
        self.chat = SimpleNamespace(completions=_CodexCliCompletions(self))


class _CodexCliCompletions:
    def __init__(self, client: CodexCliClient):
        self.client = client

    def create(self, **kwargs):
        prompt = _messages_to_prompt(
            kwargs.get("messages", []),
            json_mode=(kwargs.get("response_format") or {}).get("type") == "json_object",
        )
        content = _run_codex_exec(
            prompt,
            model=self.client.model,
            project_root=self.client.project_root,
            timeout_seconds=self.client.timeout_seconds,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _messages_to_prompt(messages: list[dict], *, json_mode: bool) -> str:
    parts = [
        "Tu es appelé comme moteur LLM local par une application RAG.",
        "N'utilise aucun outil. Ne lance aucune commande. Réponds uniquement au contenu demandé.",
    ]
    if json_mode:
        parts.append("La sortie finale doit être un seul objet JSON valide, sans markdown ni commentaire.")
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", ""))
        parts.append(f"\n[{role}]\n{content}")
    if json_mode:
        parts.append("\nRAPPEL FINAL: retourne uniquement un objet JSON valide.")
    return "\n".join(parts)


def _run_codex_exec(prompt: str, *, model: str, project_root: Path, timeout_seconds: int) -> str:
    codex_command = _resolve_codex_command()
    if codex_command is None:
        raise RuntimeError("Codex CLI introuvable dans le PATH.")

    command = [
        codex_command,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--json",
        "-m",
        model,
        "-C",
        str(project_root),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    content = _extract_agent_message(completed.stdout)
    if completed.returncode != 0 or not content:
        detail = "\n".join((completed.stdout or "", completed.stderr or "")).strip()
        raise RuntimeError(f"Codex CLI call failed: {detail[-1200:]}")
    return content


def _resolve_codex_command() -> str | None:
    """Return an executable Codex command path for the current platform."""
    if os.name == "nt":
        return shutil.which("codex.cmd") or shutil.which("codex.exe") or shutil.which("codex")
    return shutil.which("codex")


def _extract_agent_message(output: str) -> str:
    """Parse Codex JSONL output and return the last agent message."""
    last_message = ""
    for line in (output or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            last_message = str(item.get("text", "")).strip()
    return last_message
