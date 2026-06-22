from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_agentic.eval_schema import redact_secrets

PUBLIC_TEXT_SUFFIXES = {".md", ".json", ".csv", ".txt"}
FORBIDDEN_CREDENTIAL_LABELS = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "Authorization")
REQUIRED_PUBLIC_REPORTS = ("summary.json", "summary.md", "per_query_public.csv", "failure_review.md")


def command_environment() -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    parts = [str(SRC_ROOT)]
    if current_pythonpath:
        parts.append(current_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_command(args: list[str]) -> int:
    print(" ".join(args))
    return subprocess.call(args, cwd=PROJECT_ROOT, env=command_environment())


def scan_for_forbidden_public_content(root: Path) -> list[str]:
    problems: list[str] = []
    if not root.exists():
        return [f"Missing public evaluation directory: {root}"]

    for filename in REQUIRED_PUBLIC_REPORTS:
        expected = root / filename
        if not expected.exists():
            problems.append(f"Missing public evaluation report: {expected}")

    for path in root.rglob("*"):
        if path.is_dir() or path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if redact_secrets(text) != text:
            problems.append(f"Secret-like value in {path}")
        if any(label in text for label in FORBIDDEN_CREDENTIAL_LABELS):
            problems.append(f"Forbidden credential label in {path}")
    return problems


def default_chatgpt_bridge_script() -> Path:
    env_value = os.environ.get("CODEX_20X_CHATGPT_BRIDGE", "").strip()
    if env_value:
        return Path(env_value)
    return Path.home() / "Codex-20x" / "scripts" / "chatgpt-debate.ps1"


def main() -> int:
    parser = argparse.ArgumentParser(description="BOFiP Agentic RAG QA facade.")
    parser.add_argument("command", choices=["preflight", "unit", "smoke", "eval", "review", "release-check"])
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    if args.command == "preflight":
        return run_command([sys.executable, "scripts/check_setup.py", "--deep", "--skip-models"])
    if args.command == "unit":
        return run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    if args.command == "smoke":
        return run_command([sys.executable, "scripts/eval_run.py", "--limit", "3", "--provider", "codex", "--lexical-only"])
    if args.command == "eval":
        return run_command([sys.executable, "scripts/eval_run.py", "--limit", "50", "--provider", "deepseek"])
    if args.command == "review":
        if not args.run_dir:
            print("--run-dir is required for review")
            return 2
        code = run_command([sys.executable, "scripts/build_review_prompt.py", "--run-dir", args.run_dir])
        if code:
            return code
        return run_command(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "scripts/chatgpt_review.ps1",
                "-RunDir",
                args.run_dir,
                "-BridgeScript",
                str(default_chatgpt_bridge_script()),
            ]
        )
    if args.command == "release-check":
        problems = scan_for_forbidden_public_content(PROJECT_ROOT / "docs" / "evaluation" / "latest")
        if problems:
            print("\n".join(problems))
            return 1
        print("release-check OK")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
