from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import check_setup as check_setup_script
from scripts import setup as setup_script
from scripts import sync as sync_script


SAMPLE_HTML = """
<html>
  <head><title>BOFiP test</title></head>
  <body>
    <h1>I. Section principale</h1>
    <p>Le premier paragraphe de doctrine est rattaché à la section principale.</p>
    <table>
      <tr><th>Seuil</th><th>Taux</th></tr>
      <tr><td>10 000 €</td><td>20 %</td></tr>
    </table>
    <h2>A. Sous-section</h2>
    <p>Le second paragraphe relève de la sous-section.</p>
  </body>
</html>
"""


class IngestionScriptTests(unittest.TestCase):
    def test_source_record_key_uses_permalink_pgp_to_avoid_bofip_id_collisions(self) -> None:
        first = {
            "identifiant_juridique": "BOI-IF-CFE-10-30-20",
            "debut_de_validite": "2019-06-26",
            "permalien": "https://bofip.impots.gouv.fr/bofip/4316-PGP.html/identifiant=BOI-IF-CFE-10-30-20-20190626",
        }
        second = {
            "identifiant_juridique": "BOI-IF-CFE-10-30-20",
            "debut_de_validite": "2019-06-26",
            "permalien": "https://bofip.impots.gouv.fr/bofip/284-PGP.html/identifiant=BOI-IF-CFE-10-30-20-20190626",
        }

        self.assertNotEqual(
            sync_script.source_record_key(first),
            sync_script.source_record_key(second),
        )
        self.assertEqual(
            "BOI-IF-CFE-10-30-20-20190626__PGP-4316",
            sync_script.source_record_key(first),
        )

    def test_diff_documents_preserves_duplicate_bofip_references_when_source_keys_differ(self) -> None:
        new_docs = [
            {
                "source_record_id": "BOI-TEST-10-20260101__PGP-1",
                "identifiant_juridique": "BOI-TEST-10",
                "contenu_html": "<p>Premier contenu</p>",
                "contenu": "Premier contenu",
            },
            {
                "source_record_id": "BOI-TEST-10-20260101__PGP-2",
                "identifiant_juridique": "BOI-TEST-10",
                "contenu_html": "<p>Second contenu</p>",
                "contenu": "Second contenu",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            diff = sync_script.diff_documents(new_docs, Path(tmp) / "missing_snapshot.jsonl")

        self.assertEqual(2, len(diff["new"]))
        self.assertEqual(2, diff["source_record_count"])
        self.assertEqual(2, diff["source_unique_count"])
        self.assertEqual(0, diff["source_collision_count"])

    def test_sync_parser_uses_unique_document_id_while_preserving_boi_reference(self) -> None:
        parsed = sync_script.parse_documents(
            [
                {
                    "source_record_id": "BOI-DUP-10-20260101__PGP-1",
                    "identifiant_juridique": "BOI-DUP-10",
                    "titre": "Premier document",
                    "serie": "TEST",
                    "contenu_html": "<p>Premier contenu doctrinal</p>",
                    "contenu": "Premier contenu doctrinal",
                    "permalien": "https://example.invalid/1",
                    "debut_de_validite": "2026-01-01",
                },
                {
                    "source_record_id": "BOI-DUP-10-20260101__PGP-2",
                    "identifiant_juridique": "BOI-DUP-10",
                    "titre": "Second document",
                    "serie": "TEST",
                    "contenu_html": "<p>Second contenu doctrinal</p>",
                    "contenu": "Second contenu doctrinal",
                    "permalien": "https://example.invalid/2",
                    "debut_de_validite": "2026-01-01",
                },
            ]
        )

        self.assertEqual(
            ["BOI-DUP-10-20260101__PGP-1", "BOI-DUP-10-20260101__PGP-2"],
            [document["document_id"] for document in parsed],
        )
        self.assertEqual(["BOI-DUP-10", "BOI-DUP-10"], [document["boi_reference"] for document in parsed])

    def test_sync_rebuild_helper_forces_build_without_source_changes(self) -> None:
        no_change_diff = {"new": [], "updated": [], "removed": [], "unchanged": [{"id": "doc"}]}
        changed_diff = {"new": [{"id": "new"}], "updated": [], "removed": [], "unchanged": []}

        self.assertFalse(sync_script.should_build_corpus(no_change_diff, rebuild=False))
        self.assertTrue(sync_script.should_build_corpus(no_change_diff, rebuild=True))
        self.assertTrue(sync_script.should_build_corpus(changed_diff, rebuild=False))

    def test_download_params_are_full_corpus_by_default(self) -> None:
        sync_params = sync_script.build_download_params(offset=0)
        setup_params = setup_script.build_download_params(offset=0)

        self.assertNotIn("where", sync_params)
        self.assertNotIn("where", setup_params)

    def test_download_params_can_apply_explicit_series_filter(self) -> None:
        params = sync_script.build_download_params(offset=100, series_filter=["IR", "AIS"])

        self.assertEqual(100, params["offset"])
        self.assertEqual("serie='IR' OR serie='AIS'", params["where"])

    def test_setup_no_longer_copies_legacy_project_artifacts(self) -> None:
        setup_source = Path(setup_script.__file__).read_text(encoding="utf-8")

        self.assertNotIn("--copy-from", setup_source)
        self.assertFalse(hasattr(setup_script, "copy_from_project"))

    def test_sync_preserves_eval_files_without_preserving_old_corpus_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "interim"
            target = Path(tmp) / "interim_tmp"
            source.mkdir()
            target.mkdir()
            (source / "eval_queries_v1.jsonl").write_text("eval", encoding="utf-8")
            (source / "passage_gold_v3.jsonl").write_text("gold", encoding="utf-8")
            (source / "raw_docs_sample_5666.jsonl").write_text("old corpus", encoding="utf-8")
            (target / "raw_docs.jsonl").write_text("new corpus", encoding="utf-8")

            copied = sync_script.preserve_interim_files(source, target)

            self.assertEqual(
                ["eval_queries_v1.jsonl", "passage_gold_v3.jsonl"],
                [path.name for path in copied],
            )
            self.assertEqual("eval", (target / "eval_queries_v1.jsonl").read_text(encoding="utf-8"))
            self.assertEqual("gold", (target / "passage_gold_v3.jsonl").read_text(encoding="utf-8"))
            self.assertFalse((target / "raw_docs_sample_5666.jsonl").exists())

    def test_sync_progress_file_reports_done_total_and_eta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.json"

            sync_script.write_sync_progress(
                progress_path,
                phase="embedding_chunks",
                done=25,
                total=100,
                started_at=10.0,
                now=20.0,
            )

            payload = json.loads(progress_path.read_text(encoding="utf-8"))

        self.assertEqual("embedding_chunks", payload["phase"])
        self.assertEqual(25, payload["done"])
        self.assertEqual(100, payload["total"])
        self.assertEqual(25.0, payload["percent"])
        self.assertEqual(10.0, payload["elapsed_s"])
        self.assertEqual(30.0, payload["eta_s"])

    def test_validate_corpus_rejects_previous_filtered_doc_count_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs = tmp_path / "raw_docs.jsonl"
            chunks = tmp_path / "chunks.jsonl"
            doc_npy = tmp_path / "doc_dense_cache.npy"
            chunk_npy = tmp_path / "chunk_dense_cache.npy"
            docs.write_text("\n".join("{}" for _ in range(7300)) + "\n", encoding="utf-8")
            chunks.write_text("\n".join("{}" for _ in range(79000)) + "\n", encoding="utf-8")
            np.save(doc_npy, np.zeros((7300, 1024), dtype=np.float32))
            np.save(chunk_npy, np.zeros((79000, 1024), dtype=np.float32))

            errors = sync_script.validate_corpus(docs, chunks, doc_npy, chunk_npy)

        self.assertIn("Only 7300 docs (min 9048)", errors)

    def test_check_setup_uses_manifest_exact_rows_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            manifest_path = docs_dir / "full_corpus_manifest.json"
            manifest_path.write_text(
                json.dumps({"artifacts": [{"path": "sample.jsonl", "rows": 2}]}),
                encoding="utf-8",
            )
            sample = tmp_path / "sample.jsonl"
            sample.write_text("{}\n", encoding="utf-8")

            previous_root = check_setup_script.PROJECT_ROOT
            previous_manifest = check_setup_script.MANIFEST_PATH
            try:
                check_setup_script.PROJECT_ROOT = tmp_path
                check_setup_script.MANIFEST_PATH = manifest_path
                artifacts = check_setup_script._manifest_artifacts()
                result = check_setup_script._check_file(
                    "sample",
                    Path("sample.jsonl"),
                    "sample",
                    deep=True,
                    manifest_artifacts=artifacts,
                )
            finally:
                check_setup_script.PROJECT_ROOT = previous_root
                check_setup_script.MANIFEST_PATH = previous_manifest

        self.assertFalse(result.ok)
        self.assertIn("manifest expects 2", result.detail)

    def test_setup_force_helper_removes_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "first.jsonl"
            second = tmp_path / "second.npy"
            missing = tmp_path / "missing.jsonl"
            first.write_text("x", encoding="utf-8")
            second.write_text("y", encoding="utf-8")

            removed = setup_script.remove_existing_files([first, second, missing])

            self.assertEqual([first, second], removed)
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())

    def test_setup_parser_preserves_tables_and_section_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.jsonl"
            out_path = tmp_path / "parsed.jsonl"
            payload = {
                "document_id": "BOI-TEST-10",
                "boi_reference": "BOI-TEST-10",
                "title": "Document test",
                "raw_html": SAMPLE_HTML,
                "raw_text": "fallback text",
                "publication_date": "2026-01-01",
                "source_url": "https://example.invalid",
                "subjects": ["TEST"],
                "category_path": ["Commentaire", "TEST"],
            }
            raw_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

            written = setup_script.parse_raw_docs(raw_path, out_path)
            parsed = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(written, 1)
        document = parsed[0]
        self.assertEqual(1, len(document["tables"]))
        self.assertIn("10 000", document["tables"][0]["linearized_text"])
        self.assertEqual(document["paragraphs"][0]["section_id"], document["sections"][0]["section_id"])
        self.assertEqual(document["tables"][0]["section_id"], document["sections"][0]["section_id"])
        self.assertEqual(document["paragraphs"][1]["section_id"], document["sections"][1]["section_id"])

    def test_sync_parser_preserves_tables_and_section_links(self) -> None:
        parsed = sync_script.parse_documents(
            [
                {
                    "identifiant_juridique": "BOI-TEST-20",
                    "titre": "Document sync test",
                    "serie": "TEST",
                    "contenu_html": SAMPLE_HTML,
                    "contenu": "fallback text",
                    "permalien": "https://example.invalid",
                    "debut_de_validite": "2026-01-01",
                }
            ]
        )

        document = parsed[0]
        self.assertEqual(1, len(document["tables"]))
        self.assertIn("20 %", document["tables"][0]["linearized_text"])
        self.assertEqual(document["paragraphs"][0]["section_id"], document["sections"][0]["section_id"])
        self.assertEqual(document["tables"][0]["section_id"], document["sections"][0]["section_id"])
        self.assertEqual(document["paragraphs"][1]["section_id"], document["sections"][1]["section_id"])

    def test_sync_parser_accepts_null_text_fields_from_api(self) -> None:
        parsed = sync_script.parse_documents(
            [
                {
                    "identifiant_juridique": "BOI-TEST-NULL",
                    "titre": None,
                    "serie": None,
                    "contenu_html": None,
                    "contenu": None,
                    "permalien": None,
                    "debut_de_validite": None,
                }
            ]
        )

        document = parsed[0]
        self.assertEqual("", document["title"])
        self.assertEqual(["Commentaire", ""], document["category_path"])
        self.assertEqual(0, document["raw_text_length"])


if __name__ == "__main__":
    unittest.main()
