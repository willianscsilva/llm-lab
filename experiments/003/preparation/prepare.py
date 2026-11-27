"""Run from the repository root. Corpus preparation only."""

import csv
import json
import re
import resource
import subprocess
import sys
import time
import unicodedata as ud
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit

from acquisition import MIB, Cache, LimitReached, atomic, sha
from sources import BOOKS, PD, TARGETS, WIKI, WIKIBOOKS, WIKI_CATEGORIES, candidates, category_titles, extract, wiki_source


ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / "local"
LIMITS = dict(download_bytes=500*MIB, disk_bytes=2048*MIB, memory_bytes=2048*MIB, seconds=21600)
TOTALS = (5_000_000, 7_000_000, 10_000_000)
EVALUATION = 500_000
STOPWORDS = set("a o as os de do da em e é um uma que para por com se não ao dos das nos nas no na".split())
WORDS = re.compile(r"[^\W_]+")


def json_file(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def json_lines(path, rows):
    atomic(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode())


def normalize(text):
    text = ud.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(c for c in text if c == "\n" or ud.category(c) not in ("Cc", "Cf") or c == "\t")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def quality(text, vocabulary):
    characters = Counter(text)
    word_count = common = 0
    for word in WORDS.finditer(text.casefold()):
        word_count += 1
        common += word[0] in STOPWORDS
    length = len(text)
    fractions = {key: sum(n for c, n in characters.items() if ud.category(c).startswith(group))/length
                 for key, group in (("letters","L"), ("numbers","N"), ("punctuation","P"))}
    return dict(**fractions, pt_function_words=common/max(1, word_count),
                unknown={c: n for c, n in characters.items() if c not in vocabulary},
                invalid=characters["�"], residual_markup=len(re.findall(r"<[/!]?[a-z][^<>]*>", text, re.I)),
                longest_repetition=max((len(m[0]) for m in re.finditer(r"(\S)\1{5,}", text)), default=0))


def eligible(source):
    required = ("id", "title", "author", "category", "language", "source", "origin", "url",
                "license", "license_url", "license_basis", "attribution", "work")
    return all(source.get(key) for key in required) and source["category"] in TARGETS and (
        source["language"] == "pt" and source["license"] in {"domínio público", "CC BY-SA 4.0"}
        and all(source[k].startswith("https://") for k in ("origin", "url", "license_url", "license_basis"))
    )


def fingerprints(text):
    mask = (1 << 64) - 1
    result, words = set(), []
    for match in WORDS.finditer(text.casefold()):
        words.append(int(sha(match[0].encode())[:8], 16))
        if len(words) == 5:
            value = 0
            for word in words:
                value = ((value * 1000003) ^ word) & mask
            if value % 16 == 0:
                result.add(value)
            words.pop(0)
    paragraphs = {sha(re.sub(r"\s+", " ", p).strip().encode()): len(p)
                  for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 160}
    return result, paragraphs


class Index:
    def __init__(self):
        self.exact, self.sizes, self.protected = {}, {}, {}
        self.shingles, self.paragraphs = defaultdict(list), defaultdict(list)

    def add(self, name, digest, features, protected=False):
        words, paragraphs = features
        self.exact[digest] = name
        self.sizes[name], self.protected[name] = len(words), protected
        for key in words:
            self.shingles[key].append(name)
        for key in paragraphs:
            self.paragraphs[key].append(name)

    def match(self, digest, features):
        if digest in self.exact:
            other = self.exact[digest]
            return ("benchmark_idêntico" if self.protected[other] else "duplicata_exata", other)
        words, paragraphs = features
        hits, shared = Counter(), Counter()
        for key in words:
            hits.update(self.shingles.get(key, ()))
        for key, size in paragraphs.items():
            for other in self.paragraphs.get(key, ()):
                shared[other] += size
        for other in sorted(hits.keys() | shared.keys(), key=lambda k: (not self.protected[k], k)):
            overlap = hits[other] / max(1, min(len(words), self.sizes[other]))
            protected = self.protected[other]
            if (hits[other] >= 6 and overlap >= (.20 if protected else .65)) or (
                shared[other] >= (600 if protected else 1200)
                and shared[other] >= .25 * max(1, sum(paragraphs.values()))
            ):
                return ("benchmark_semelhante" if protected else "duplicata_semelhante", other)
        return None


def historical():
    books = "data/experiment-002"
    for path in sorted(Path(books, "train").glob("*.txt")):
        iracema = path.stem == "iracema"
        origin = "https://www.gutenberg.org/ebooks/" + ("67740" if iracema else "57001")
        yield path, dict(id="exp002-" + path.stem, title=path.stem, category="literatura", language="pt",
                         author="José de Alencar" if iracema else "Machado de Assis", source="Experimento 002",
                         origin=origin, url=origin, license="domínio público", license_url=PD,
                         license_basis=origin, attribution="Autoria e créditos no README do Experimento 002.",
                         format="historical", work="pg-67740" if iracema else "pg-57001", mandatory=True)


def grouped(documents):
    works = defaultdict(list)
    for doc in documents:
        works[doc["work"]].append(doc)
    return works


def choose(groups, target, initial=(), max_work_fraction=1.02, shares=TARGETS, category_limits=None):
    chosen = list(initial)
    counts = Counter()
    authors = Counter()
    for doc in chosen:
        counts[doc["category"]] += doc["characters"]
        authors[doc["author"]] += doc["characters"]
    size = sum(counts.values())
    remaining = dict(groups)
    while size < target:
        options = []
        for work, docs in remaining.items():
            amount = sum(d["characters"] for d in docs)
            key = docs[0]["category"]
            within_category = category_limits is None or counts[key] + amount <= category_limits[key]
            if within_category and size + amount <= target * 1.02 and amount <= target * max_work_fraction:
                deficit = (shares[key] * target - counts[key] - .45 * amount) / target
                concentration = authors[docs[0]["author"]] / target
                # Prefer the closest fit near the target and diversify sooner.
                score = -abs(target - size - amount) / target if size > .93 * target else deficit - .12 * concentration
                options.append((score, sha(("003:" + work).encode()), work, amount))
        if not options:
            break
        _, _, work, amount = max(options)
        docs = remaining.pop(work)
        for doc in docs:
            counts[doc["category"]] += doc["characters"]
            authors[doc["author"]] += doc["characters"]
        chosen.extend(docs)
        size += amount
    return chosen, remaining


def split(documents):
    mandatory = [d for d in documents if d.get("mandatory")]
    remaining = grouped(d for d in documents if not d.get("mandatory"))
    # On a partial acquisition leave meaningful data for the experiment.
    available = sum(d["characters"] for d in documents if not d.get("mandatory"))
    eval_target = min(EVALUATION, available // 5)
    limits = {key: 0 for key in TARGETS}
    for docs in remaining.values():
        for doc in docs:
            limits[doc["category"]] += .20 * doc["characters"]
    validation, remaining = choose(remaining, eval_target, max_work_fraction=.40, category_limits=limits)
    holdout, remaining = choose(remaining, eval_target, max_work_fraction=.40, category_limits=limits)
    trains, prior = [], mandatory
    val_size = sum(d["characters"] for d in validation)
    supply = Counter()
    for doc in mandatory + [doc for docs in remaining.values() for doc in docs]:
        supply[doc["category"]] += doc["characters"]
    capacity = min(TOTALS[-1] - val_size, supply.total())
    shares = {key: min(weight, supply[key]/max(1,capacity)) for key, weight in TARGETS.items()}
    headroom = {key: max(0, supply[key]/max(1,capacity) - shares[key]) for key in TARGETS}
    unfilled = 1 - sum(shares.values())
    if sum(headroom.values()):
        shares = {key: shares[key] + unfilled * headroom[key] / sum(headroom.values()) for key in TARGETS}
    for target in TOTALS:
        prior, remaining = choose(remaining, target - val_size, prior, shares=shares)
        trains.append(prior)
    return validation, holdout, trains


def summary(docs):
    chars, unknown = sum(d["characters"] for d in docs), Counter()
    dist = {key: Counter() for key in ("category", "source", "author", "license")}
    categories = defaultdict(Counter)
    for doc in docs:
        categories[doc["category"]]["characters"] += doc["characters"]
        for char, count in doc["quality"]["unknown"].items():
            unknown[char] += count
            categories[doc["category"]]["unknown"] += count
        for key, values in dist.items():
            values[doc[key]] += doc["characters"]
    return dict(documents=len(docs), characters=chars, unknown_count=unknown.total(),
                unknown_fraction=unknown.total()/max(chars,1), unknown_characters=dict(unknown.most_common()),
                unknown_by_category={k: dict(v, fraction=v["unknown"]/v["characters"]) for k,v in categories.items()},
                **{key: dict(value) for key, value in dist.items()})


def publish(documents, rejected, benchmarks, config, cache, acquired):
    validation, holdout, trains = split(documents)
    manifests, stats = {}, {}

    def manifest(name, docs, target, extra=None):
        result = summary(docs)
        entries = [{"id": d["id"], "sha256": d["sha256"], "characters": d["characters"]} for d in docs]
        out = dict(documents=entries, target=target, characters=result["characters"],
                   within_tolerance=abs(result["characters"]-target) <= .02*target, **(extra or {}))
        path = ROOT / f"{name}.json"
        json_file(path, out)
        stats[name] = result
        manifests[name] = {"sha256": sha(path.read_bytes()), "characters": result["characters"],
                           "within_tolerance": out["within_tolerance"]}

    manifest("validation", validation, EVALUATION)
    manifest("holdout", holdout, EVALUATION)
    val_chars = stats["validation"]["characters"]
    for target, train in zip(TOTALS, trains):
        name = f"corpus-{target//1_000_000}m"
        manifest(name, train, target-val_chars, {"validation_sha256": manifests["validation"]["sha256"],
                                                "combined_target": target,
                                                "combined_characters": val_chars + sum(d["characters"] for d in train)})
        manifests[name]["combined_characters"] = val_chars + stats[name]["characters"]
        manifests[name]["within_tolerance"] = abs(manifests[name]["combined_characters"]-target) <= .02*target
        stats[name+"-combined"] = summary(train + validation)
    sets = [{d["id"] for d in corpus} for corpus in trains]
    works = [{d["work"] for d in corpus} for corpus in (trains[-1], validation, holdout)]
    ids = [{d["id"] for d in corpus} for corpus in (trains[-1], validation, holdout)]
    assert all(not (sets_[i] & sets_[j]) for sets_ in (works, ids) for i,j in ((0,1),(0,2),(1,2)))
    assert sets[0] <= sets[1] <= sets[2]
    assert all(d["id"] in sets[0] for d in documents if d.get("mandatory"))
    repetitions, tables = [], []
    for doc in documents:
        contents = (LOCAL / "documents" / (doc["id"]+".txt")).read_bytes()
        assert sha(contents) == doc["sha256"]
        text = contents.decode("utf-8")
        paragraphs = [part.strip() for part in text.split("\n\n") if len(part.strip()) >= 80]
        repeated = (len(paragraphs) - len(set(paragraphs)))/max(1,len(paragraphs))
        lines = [line for line in text.splitlines() if line.strip()]
        tabular = sum(bool(re.search(r"\S(?:\t+| {3,})\S", line)) for line in lines)/max(1,len(lines))
        doc["quality"].update(repeated_long_paragraph_fraction=repeated, column_like_line_fraction=tabular)
        if repeated >= .10:
            repetitions.append(doc["id"])
        if tabular >= .50:
            tables.append(doc["id"])
    assert not {d["sha256"] for d in documents} & set(benchmarks.values())
    json_lines(ROOT / "sources.jsonl", [{k:v for k,v in d.items() if k != "quality"} for d in documents])
    json_lines(ROOT / "documents.jsonl", documents)
    json_lines(ROOT / "rejected-documents.jsonl", rejected)
    json_file(ROOT / "config.json", config)
    json_file(ROOT / "statistics.json", stats)
    with (ROOT / "statistics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("set", "group", "item", "characters", "unknown_count", "unknown_fraction"))
        for name, values in stats.items():
            writer.writerow((name,"total","",values["characters"],values["unknown_count"],values["unknown_fraction"]))
            for group in ("category","source","author","license"):
                for key, count in sorted(values[group].items()):
                    info = values["unknown_by_category"].get(key,{}) if group == "category" else {}
                    writer.writerow((name,group,key,count,info.get("unknown",""),info.get("fraction","")))
    attrib = ["# Atribuições", "",
              "A extração de prosa e a normalização estão descritas no README. Os arquivos de texto ficam locais para revisão de redistribuição.", ""]
    for d in documents:
        attrib.append(f"- {d['attribution']} [Fonte]({d['origin']}) · [Licença ou fundamento]({d['license_url']})")
    atomic(ROOT / "ATTRIBUTIONS.md", ("\n".join(attrib)+"\n").encode())
    reasons = Counter(row["reason"] for row in rejected)
    report = dict(manifests=manifests, strict_cumulative=sets[0] < sets[1] < sets[2], cumulative=True,
                  common_validation_identical=True, groups_disjoint=True, benchmarks=benchmarks,
                  benchmark_method="SHA-256, índice invertido com amostra de shingles de cinco palavras e hashes de parágrafos",
                  processed_hashes_verified=True, acquired_documents=len(acquired), accepted=len(documents),
                  historical_training_documents=sum(bool(d.get("mandatory")) for d in documents),
                  rejected=len(rejected), rejection_reasons=dict(reasons),
                  raw_characters=sum(acquired.values()), processed_characters=sum(d["characters"] for d in documents),
                  quality_review={"repeated_long_paragraphs_at_least_10_percent": repetitions,
                                  "column_like_lines_at_least_50_percent": tables},
                  download_bytes=cache.total, storage_bytes=sum(p.stat().st_size for p in LOCAL.rglob("*") if p.is_file()),
                  elapsed_seconds=time.monotonic()-cache.started,
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return report


def main():
    output = LOCAL / "documents"
    output.mkdir(parents=True, exist_ok=True)
    vocab_data = (ROOT/"vocabulary.json").read_bytes()
    vocabulary = set(json.loads(vocab_data)["entries"])
    assert len(vocabulary) == 101
    config = dict(origin_commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
                  python=sys.version.split()[0], dependencies={"pipeline":"biblioteca padrão","vocabulario":"PyTorch local"},
                  seed="003", vocabulary_sha256=sha(vocab_data), limits=LIMITS, http_concurrency=1,
                  requests_per_domain_min_seconds=1, targets=TOTALS, thematic_targets=TARGETS,
                  unknown_reference=.001, tolerance=.02, evaluation_target=EVALUATION,
                  evaluation_max_work_fraction=.40, evaluation_max_category_supply_fraction=.20)
    cache = Cache(LOCAL / "downloads", ROOT / "acquisition.log", LIMITS)
    index, benchmarks = Index(), {}
    for path in [*sorted(Path("data/experiment-002/validation").glob("*.txt")),
                 *sorted(Path("experiments/002/phase-03/external").glob("*.txt"))]:
        raw = path.read_bytes()
        benchmarks[str(path)] = sha(raw)
        index.add(str(path), sha(raw), fingerprints(raw.decode()), protected=True)
    assert len(benchmarks) == 6
    accepted, rejected, acquired, attempted, disabled = [], [], {}, set(), set()
    terminal = "candidatos esgotados"

    def reject(source, reason, **fields):
        rejected.append({"id": source["id"], "title": source.get("title"), "url": source.get("url"),
                         "reason": reason, **fields})
        cache.event("rejected", **rejected[-1])

    def handle(source, local_path=None):
        cache.check(measure_disk=True)
        if source["id"] in attempted:
            return
        attempted.add(source["id"])
        host = urlsplit(source["url"]).netloc
        if not eligible(source):
            reject(source, "procedência_ou_licença")
            return
        if not local_path and host in disabled:
            reject(source, "fonte_suspensa")
            return
        try:
            if local_path:
                raw = local_path.read_bytes()
                meta = {"sha256": sha(raw), "date": None, "revision": config["origin_commit"]}
            else:
                raw, meta = cache.get(source["url"])
                acquired[source["id"]] = len(raw.decode("utf-8", errors="replace"))
            if local_path:
                text = raw.decode("utf-8")
                detail = {"raw_characters": len(text), "edition": meta["revision"]}
            else:
                text, detail = extract(raw, source["format"])
                text = normalize(text)
        except HTTPError as error:
            if error.code in (403, 429):
                disabled.add(host)
            reject(source, "http_"+str(error.code))
            return
        except (ValueError, URLError, TimeoutError, OSError) as error:
            reject(source, "aquisição_ou_extração", error=type(error).__name__, detail=str(error))
            return
        if len(text) < 512:
            reject(source, "texto_curto")
            return
        profile = quality(text, vocabulary)
        if not local_path and (profile["invalid"] or profile["residual_markup"] or profile["letters"] < .60
                               or profile["pt_function_words"] < .11 or profile["longest_repetition"] >= 80):
            reject(source, "qualidade", quality=profile)
            return
        raw_text = text.encode()
        digest = sha(raw_text)
        features = fingerprints(text)
        if match := index.match(digest, features):
            reject(source, match[0], matched=match[1])
            return
        cache.check(extra=0, measure_disk=True)
        atomic(output / (source["id"]+".txt"), raw_text)
        doc = dict(**source, **detail, acquired_at=meta["date"], effective_url=str(local_path) if local_path else meta["url"],
                   http_etag=meta["revision"] if not local_path else None, raw_sha256=meta["sha256"], sha256=digest,
                   characters=len(text), unknown_fraction=sum(profile["unknown"].values())/len(text),
                   quality=profile, redistribution="revisão humana pendente")
        if source["format"] == "wiki":
            doc["attribution"] += f" Revisão {detail['edition']}."
            doc["revision_url"] = urlsplit(doc["origin"]).scheme + "://" + host + "/wiki/Special:PermanentLink/" + detail["edition"]
        index.add(source["id"], digest, features)
        accepted.append(doc)
        cache.event("accepted", id=doc["id"], characters=doc["characters"], sha256=digest)

    try:
        for path, source in historical():
            handle(source, path)
        assert sum(bool(d.get("mandatory")) for d in accepted) == 10
        for source in candidates():
            handle(source)
        for title in WIKIBOOKS:
            if "pt.wikibooks.org" in disabled:
                break
            try:
                data, _ = cache.get(wiki_source(BOOKS,title,"científico e educacional")["url"])
            except (ValueError, URLError, OSError):
                continue
            links = {unquote(link).replace("_", " ") for link in re.findall(
                r'href="/wiki/([^"#?]+)"',data.decode("utf-8"))}
            chapters = sorted(link for link in links if link.startswith(title + "/") and ":" not in link)
            for chapter in chapters[:12]:
                handle(wiki_source(BOOKS,chapter,"científico e educacional"))
        for category, names in WIKI_CATEGORIES.items():
            for name in names:
                if "pt.wikipedia.org" in disabled:
                    break
                url = f"{WIKI}/wiki/Categoria:{quote(name.replace(' ', '_'))}"
                try:
                    data, _ = cache.get(url)
                    for title in category_titles(data.decode("utf-8"))[:56]:
                        if re.search(r"\b(?:Quaresma|Papeis Avulsos|Papéis Avulsos|A chinela turca|O espelho|Polícrates)\b",
                                     title, re.I):
                            continue
                        handle(wiki_source(WIKI, title, category))
                        if sum(d["characters"] for d in accepted) >= 11_100_000:
                            break
                except (ValueError, URLError, OSError) as error:
                    cache.event("discovery_error", url=url, error=type(error).__name__)
                    if isinstance(error, HTTPError) and error.code in (403, 429):
                        disabled.add("pt.wikipedia.org")
                if sum(d["characters"] for d in accepted) >= 11_100_000:
                    break
            if sum(d["characters"] for d in accepted) >= 11_100_000:
                terminal = "estoque de candidatos suficiente para seleção"
                break
    except LimitReached as error:
        terminal = "limite operacional: " + str(error)

    report = publish(accepted, rejected, benchmarks, config, cache, acquired)
    report["termination"] = terminal
    json_file(ROOT / "validation-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def repartition():
    def read_rows(name):
        return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]

    path = ROOT / "validation-report.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config["evaluation_max_work_fraction"] = .40
    config["evaluation_max_category_supply_fraction"] = .20
    cache = Cache(LOCAL / "downloads", ROOT / "acquisition.log", LIMITS)
    result = publish(read_rows("documents.jsonl"), read_rows("rejected-documents.jsonl"),
                     previous["benchmarks"], config, cache, {})
    for field in ("acquired_documents", "raw_characters", "elapsed_seconds", "termination"):
        result[field] = previous[field]
    result["manifest_rebuild_seconds"] = time.monotonic() - cache.started
    result["peak_process_rss_kib"] = max(previous["peak_process_rss_kib"], result["peak_process_rss_kib"])
    json_file(path, result)


if __name__ == "__main__":
    if sys.argv[1:] == ["--manifests-only"]:
        repartition()
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("Uso: prepare.py [--manifests-only]")
