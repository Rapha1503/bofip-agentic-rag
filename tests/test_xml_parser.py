from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bofip_cleanroom.xml_parser import parse_document_xml


XML_FIXTURE = """\
<document type="Contenu" xmlns:dc="http://purl.org/dc/elements/1.1" xmlns:bofip="https://bofip.impots.gouv.fr">
  <parts><part><dataRef>data.html</dataRef></part></parts>
  <dc:dublincore>
    <dc:title>BOI-TEST-0001</dc:title>
    <dc:date>2024-01-15</dc:date>
    <dc:language>fr</dc:language>
    <dc:subject>TVA</dc:subject>
    <dc:identifier>1234-PGP</dc:identifier>
    <dc:identifier>https://bofip.impots.gouv.fr/test</dc:identifier>
    <dc:relation type="parent">Contenu.Document:ROOT</dc:relation>
  </dc:dublincore>
  <bofip:bodgfip>
    <bofip:contenu_type>Commentaire</bofip:contenu_type>
    <bofip:contenu_id>BOI-TVA-TEST-20240115</bofip:contenu_id>
  </bofip:bodgfip>
</document>
"""


class XmlParserTests(unittest.TestCase):
    def test_parse_document_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "document.xml"
            path.write_text(XML_FIXTURE, encoding="utf-8")
            payload = parse_document_xml(path)

        self.assertEqual(payload["document_id"], "1234-PGP")
        self.assertEqual(payload["boi_reference"], "BOI-TVA-TEST-20240115")
        self.assertEqual(payload["content_type"], "Commentaire")
        self.assertEqual(payload["source_url"], "https://bofip.impots.gouv.fr/test")
        self.assertEqual(payload["relations"][0]["relation_type"], "parent")


if __name__ == "__main__":
    unittest.main()
