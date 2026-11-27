"""Bounded HTTP acquisition with a persistent, verified cache."""

import hashlib
import json
import resource
import shutil
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser


MIB = 1024 * 1024
AGENT = "LLMLabCorpus/1.0 (https://github.com/willianscsilva/llm-lab)"


class LimitReached(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


class Cache:
    def __init__(self, root, log, limits):
        self.root, self.log, self.limits = root, log, limits
        root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.last, self.robots = {}, {}
        self.opener = build_opener(NoRedirect)
        self.ledger = root / "download-bytes"
        self.total = int(self.ledger.read_text()) if self.ledger.exists() else 0
        self.check()

    def event(self, state, **fields):
        with self.log.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "event": state, **fields},
                                 ensure_ascii=False) + "\n")

    def check(self, extra=0, measure_disk=False):
        if time.monotonic() - self.started >= self.limits["seconds"]:
            raise LimitReached("tempo")
        if self.total + extra > self.limits["download_bytes"]:
            raise LimitReached("downloads")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= self.limits["memory_bytes"]:
            raise LimitReached("memória")
        if measure_disk:
            local = sum(p.stat().st_size for p in self.root.parent.rglob("*") if p.is_file())
            if local + extra > self.limits["disk_bytes"]:
                raise LimitReached("disco")
        if shutil.disk_usage(self.root).free < 64 * MIB + extra:
            raise LimitReached("espaço livre")

    def request(self, url, delay):
        host = urlsplit(url).netloc
        for attempt in range(3):
            self.check(measure_disk=True)
            time.sleep(max(0, self.last.get(host, 0) + delay - time.monotonic()))
            self.last[host] = time.monotonic()
            try:
                with self.opener.open(Request(url, headers={"User-Agent": AGENT, "Accept-Encoding": "identity"}),
                                      timeout=15) as response:
                    payload = bytearray()
                    while True:
                        remaining = self.limits["download_bytes"] - self.total
                        if remaining == 0:
                            raise LimitReached("downloads")
                        chunk = response.read1(min(65536, remaining))
                        if not chunk:
                            break
                        self.total += len(chunk)
                        atomic(self.ledger, str(self.total).encode())
                        self.check()
                        payload += chunk
                        if len(payload) > 8 * MIB:
                            raise ValueError("resposta individual maior que 8 MiB")
                    self.check(measure_disk=True)
                    return bytes(payload), response.headers.get("ETag")
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                code = getattr(error, "code", None)
                self.event("http_error", url=url, code=code, attempt=attempt + 1, error=type(error).__name__)
                if code in {401, 403, 429}:
                    raise
                if (code is not None and code < 500) or attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def get(self, url, robots=True):
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.username or parts.password:
            raise ValueError("URL de aquisição inválida")
        key = sha(url.encode())
        raw, metadata = self.root / (key + ".raw"), self.root / (key + ".json")
        if raw.is_file() and metadata.is_file():
            data, meta = raw.read_bytes(), json.loads(metadata.read_text())
            if meta["url"] == url and sha(data) == meta["sha256"]:
                self.event("cache_hit", url=url)
                return data, meta
        delay = 1
        if robots:
            origin = f"{parts.scheme}://{parts.netloc}"
            if origin not in self.robots:
                rules = RobotFileParser()
                try:
                    data, _ = self.get(origin + "/robots.txt", robots=False)
                except HTTPError as error:
                    if error.code not in (404, 410):
                        raise
                    rules.allow_all = True
                else:
                    # robotparser does not implement Allow-vs-Disallow longest matching.
                    rules.parse(data.decode("utf-8").splitlines())
                self.robots[origin] = rules
            rules = self.robots[origin]
            if not rules.can_fetch(AGENT, url):
                raise ValueError("robots.txt proíbe a URL")
            rate = rules.request_rate(AGENT)
            delay = max(1, rules.crawl_delay(AGENT) or 0, rate.seconds / rate.requests if rate else 0)
        data, revision = self.request(url, delay)
        disk = sum(p.stat().st_size for p in self.root.parent.rglob("*") if p.is_file())
        if disk + 2 * len(data) > self.limits["disk_bytes"]:
            raise LimitReached("disco")
        meta = {"url": url, "sha256": sha(data), "revision": revision,
                "date": datetime.now(timezone.utc).date().isoformat(), "bytes": len(data)}
        atomic(raw, data)
        atomic(metadata, json.dumps(meta).encode())
        self.event("download", **meta)
        return data, meta
