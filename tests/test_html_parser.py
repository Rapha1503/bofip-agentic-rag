from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bofip_cleanroom.html_parser import parse_html_structure


HTML_FIXTURE = """\
<html>
  <head><title>Document de test</title></head>
  <body>
    <h1 id="sec-1">Section 1</h1>
    <p>10</p>
    <p id="p1">Le taux reduit prevu a l'article 279 du CGI s'applique.</p>
    <table>
      <tr><th>Colonne A</th><th>Colonne B</th></tr>
      <tr><td>10</td><td>20</td></tr>
    </table>
    <p id="p2"><a href="#sec-1">Voir section 1</a> pour plus de details.</p>
  </body>
</html>
"""


class HtmlParserTests(unittest.TestCase):
    def test_parse_html_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "data.html"
            path.write_text(HTML_FIXTURE, encoding="utf-8")
            payload = parse_html_structure(path, document_id="TESTDOC")

        self.assertEqual(payload["html_title"], "Document de test")
        self.assertEqual(len(payload["sections"]), 1)
        self.assertEqual(len(payload["paragraphs"]), 2)
        self.assertEqual(len(payload["tables"]), 1)
        self.assertEqual(payload["paragraphs"][0]["paragraph_number"], "10")
        self.assertTrue(any("article 279 du CGI" in ref for ref in payload["legal_refs"]))

    def test_parse_html_structure_creates_synthetic_root_section_for_orphans(self) -> None:
        html = """<html><head><title>Sans heading</title></head><body><p>Texte utile.</p></body></html>"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "data.html"
            path.write_text(html, encoding="utf-8")
            payload = parse_html_structure(path, document_id="NOHEAD")

        self.assertEqual(len(payload["sections"]), 1)
        self.assertEqual(payload["sections"][0]["level"], 0)
        self.assertEqual(payload["paragraphs"][0]["section_id"], payload["sections"][0]["section_id"])

    def test_parse_html_structure_tracks_nested_headings(self) -> None:
        html = """
        <html><head><title>Hiérarchie</title></head><body>
        <h1>Partie I</h1>
        <p>Intro.</p>
        <h2>A. Niveau 2</h2>
        <p>Texte 2.</p>
        <h3>1. Niveau 3</h3>
        <p>Texte 3.</p>
        <h4>a) Niveau 4</h4>
        <p>Texte 4.</p>
        </body></html>
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "data.html"
            path.write_text(html, encoding="utf-8")
            payload = parse_html_structure(path, document_id="NEST")

        self.assertEqual([section["level"] for section in payload["sections"]], [1, 2, 3, 4])
        self.assertEqual(payload["sections"][-1]["path"], ["Partie I", "A. Niveau 2", "1. Niveau 3", "a) Niveau 4"])


if __name__ == "__main__":
    unittest.main()
