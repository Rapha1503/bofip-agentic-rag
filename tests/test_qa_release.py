import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import call, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import qa
from scripts.qa import scan_for_forbidden_public_content


class QAReleaseTests(unittest.TestCase):
    def test_raw_eval_run_outputs_are_gitignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "output/eval-runs/example/traces/Q1.json"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_scan_rejects_secret_like_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs" / "evaluation" / "latest"
            docs.mkdir(parents=True)
            (docs / "summary.md").write_text("sk-1234567890abcdef", encoding="utf-8")

            problems = scan_for_forbidden_public_content(docs)

            self.assertTrue(problems)

    def test_public_scan_accepts_clean_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs" / "evaluation" / "latest"
            docs.mkdir(parents=True)
            for filename in qa.REQUIRED_PUBLIC_REPORTS:
                (docs / filename).write_text("Clean report", encoding="utf-8")

            self.assertEqual(scan_for_forbidden_public_content(docs), [])

    def test_public_scan_rejects_incomplete_report_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs" / "evaluation" / "latest"
            docs.mkdir(parents=True)
            (docs / ".gitkeep").write_text("", encoding="utf-8")

            problems = scan_for_forbidden_public_content(docs)

            self.assertTrue(any("summary.json" in item for item in problems))
            self.assertTrue(any("summary.md" in item for item in problems))

    def test_release_check_reports_missing_latest_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = StringIO()
            with patch.object(qa, "PROJECT_ROOT", root):
                with patch.object(sys, "argv", ["qa.py", "release-check"]):
                    with redirect_stdout(output):
                        returncode = qa.main()

            self.assertEqual(returncode, 1)
            stdout = output.getvalue()
            self.assertIn("docs", stdout)
            self.assertIn("evaluation", stdout)
            self.assertIn("latest", stdout)
            self.assertIn("Missing", stdout)

    def test_review_requires_run_dir(self):
        script = PROJECT_ROOT / "scripts" / "qa.py"
        result = subprocess.run(
            [sys.executable, str(script), "review"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--run-dir is required for review", result.stdout)

    def test_chatgpt_review_script_accepts_bridge_script_parameter(self):
        script = PROJECT_ROOT / "scripts" / "chatgpt_review.ps1"
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$null = [scriptblock]::Create((Get-Content -Raw -LiteralPath '{script}'))",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BridgeScript", script.read_text(encoding="utf-8"))

    def test_review_builds_prompt_then_runs_powershell_wrapper(self):
        with patch.object(sys, "argv", ["qa.py", "review", "--run-dir", "runs/eval1"]):
            with patch("scripts.qa.subprocess.call", side_effect=[0, 0]) as subprocess_call:
                self.assertEqual(qa.main(), 0)

        self.assertEqual(
            subprocess_call.call_args_list,
            [
                call(
                    [sys.executable, "scripts/build_review_prompt.py", "--run-dir", "runs/eval1"],
                    cwd=qa.PROJECT_ROOT,
                    env=qa.command_environment(),
                ),
                call(
                    [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        "scripts/chatgpt_review.ps1",
                        "-RunDir",
                        "runs/eval1",
                        "-BridgeScript",
                        str(qa.default_chatgpt_bridge_script()),
                    ],
                    cwd=qa.PROJECT_ROOT,
                    env=qa.command_environment(),
                ),
            ],
        )

    def test_unit_command_runs_with_src_on_pythonpath(self):
        with patch.object(sys, "argv", ["qa.py", "unit"]):
            with patch("scripts.qa.subprocess.call", return_value=0) as subprocess_call:
                self.assertEqual(qa.main(), 0)

        _, kwargs = subprocess_call.call_args
        pythonpath = kwargs["env"]["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(
            subprocess_call.call_args.args[0],
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        )
        self.assertIn(str(qa.SRC_ROOT), pythonpath)


if __name__ == "__main__":
    unittest.main()
