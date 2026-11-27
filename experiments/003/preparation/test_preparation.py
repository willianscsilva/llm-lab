import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from acquisition import Cache, LimitReached, MIB, sha
from prepare import Index, eligible, fingerprints, normalize, quality, split
from sources import WIKI, extract, wiki_source


class PreparationTests(unittest.TestCase):
    def test_nfc_and_paragraphs(self):
        self.assertEqual(normalize("Ação!\r\n\r\n— Olá, 2.\r\n\x00"), "Ação!\n\n— Olá, 2.\n")

    def test_html_keeps_prose(self):
        raw = ('<html class="vector-feature-toc"><script>{"wgRevisionId":123}</script>'
               '<div class="mw-parser-output">'
               '<p>O texto!<sup>ref</sup></p><table><tr><td>omitir</td></tr></table>'
               '<p>— Olá.</p><h2>Referências</h2><p>omitir</p></div></html>').encode()
        self.assertEqual(extract(raw,"wiki")[0], "O texto!\n\n— Olá.")

    def test_unknown(self):
        result = quality("a–a\n",set("a\n"))
        self.assertEqual(result["unknown"], {"–":1})

    def test_license(self):
        valid = wiki_source(WIKI, "Um tema", "enciclopédico")
        self.assertTrue(eligible(valid))
        for key in ("origin","license_basis","attribution"):
            self.assertFalse(eligible({**valid,key:""}))
        self.assertFalse(eligible({**valid,"license":"desconhecida"}))

    def test_exact_near_and_protection(self):
        words = " ".join("palavra"+str(i) for i in range(600))
        index = Index()
        index.add("holdout",sha(words.encode()),fingerprints(words),protected=True)
        self.assertEqual(index.match(sha(words.encode()),fingerprints(words)),("benchmark_idêntico","holdout"))
        changed = "Prefácio. " + words[100:] + " Epílogo."
        self.assertEqual(index.match(sha(changed.encode()),fingerprints(changed)),("benchmark_semelhante","holdout"))
        second = Index()
        second.add("obra",sha(words.encode()),fingerprints(words))
        self.assertEqual(second.match(sha(words.encode()),fingerprints(words)),("duplicata_exata","obra"))
        self.assertEqual(second.match(sha(changed.encode()),fingerprints(changed)),("duplicata_semelhante","obra"))
        self.assertIsNone(index.match(sha(b"sem encontro"),fingerprints("sem encontro")))

    def test_deterministic_cumulative_disjoint_work_groups(self):
        docs = [dict(id=str(i), work=str(i//2), characters=250, category="literatura", author=str(i%3))
                for i in range(30)]
        with patch("prepare.TOTALS",(2500,4000,4500)), patch("prepare.EVALUATION",1500):
            result = split(docs)
            self.assertEqual(result,split(docs))
        val, holdout, trains = result
        self.assertEqual(sum(d["characters"] for d in val),1500)
        self.assertEqual(sum(d["characters"] for d in holdout),1500)
        ids = [{d["id"] for d in t} for t in trains]
        self.assertTrue(ids[0] < ids[1] < ids[2])
        groups = [{d["work"] for d in t} for t in (val,holdout,trains[-1])]
        self.assertTrue(all(groups[i].isdisjoint(groups[j]) for i,j in ((0,1),(0,2),(1,2))))

    def test_cache_resume_integrity_and_limits(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            limits = dict(download_bytes=100,disk_bytes=MIB,memory_bytes=2048*MIB,seconds=10)
            cache = Cache(path/"download",path/"log",limits)
            with patch.object(cache,"request",return_value=(b"texto","revision")) as request:
                cache.get("https://example.org/test",robots=False)
                request.assert_called_once()
            resumed = Cache(path/"download",path/"log",limits)
            with patch.object(resumed,"request",side_effect=AssertionError("download repetido")):
                self.assertEqual(resumed.get("https://example.org/test",robots=False)[0],b"texto")
            with self.assertRaises(LimitReached):
                resumed.check(extra=101)
            with patch("acquisition.time.monotonic", return_value=resumed.started + 11):
                with self.assertRaises(LimitReached):
                    resumed.check()

    def test_robots_disallow_prevents_content_fetch(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            limits = dict(download_bytes=MIB,disk_bytes=MIB,memory_bytes=2048*MIB,seconds=10)
            cache = Cache(path/"download",path/"log",limits)
            with patch.object(cache,"request",return_value=(b"User-agent: *\nDisallow: /restrito",None)) as request:
                with self.assertRaisesRegex(ValueError,"robots.txt"):
                    cache.get("https://example.org/restrito")
                request.assert_called_once_with("https://example.org/robots.txt", 1)
