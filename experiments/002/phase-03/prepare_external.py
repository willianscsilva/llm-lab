"""Extract the three first chapters from the verified UTF-8 source supplied as argv[1]."""

import json
import re
import sys
import unicodedata
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path


def shingles(text):
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold())
    return set(zip(*(words[index:] for index in range(5))))


def main(source):
    source_hash = sha256(source.read_bytes()).hexdigest()
    assert source_hash == "4112224020706659b19d9873abea21d370bdc2188e0bdbf146393aeb226129fe"
    text = source.read_text(encoding="utf-8-sig")
    external = {}
    titles = ("A LIÇÃO DE VIOLÃO", "NO «SOCEGO»", "PATRIOTAS")
    for index, title in enumerate(titles, 1):
        start = text.index(f"\nI\n\n{title}\n")
        end = text.index("\nII\n", start)
        chapter = text[start:end].strip()
        name = f"parte-{index}-capitulo-1.txt"
        external[name] = unicodedata.normalize("NFC", chapter) + "\n"
    characters = sum(map(len, external.values()))
    assert 30000 <= characters <= 100000

    original = {
        path.as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(Path("data/experiment-002").glob("*/*.txt"))
    }
    assert len(original) == 13
    sequences = {name: shingles(doc) for name, doc in (original | external).items()}
    max_overlap, pair = -1, None
    for first, second in list(product(external, original)) + list(combinations(external, 2)):
        a, b = sequences[first], sequences[second]
        overlap = len(a & b) / min(len(a), len(b))
        assert external[first] != (original | external)[second]
        assert overlap < 0.8, (first, second, overlap)
        if overlap > max_overlap:
            max_overlap, pair = overlap, [first, second]

    target = Path(__file__).with_name("external")
    target.mkdir(exist_ok=True)
    documents = []
    for (filename, content), title in zip(external.items(), titles):
        encoded = content.encode("utf-8")
        (target / filename).write_bytes(encoded)
        documents.append({"filename": filename, "chapter_title": title, "characters": len(content),
                          "sha256": sha256(encoded).hexdigest()})
    metadata = {
        "author": "Lima Barreto (1881–1922)",
        "work": "Triste Fim de Polycarpo Quaresma (1915)",
        "edition": "Transcrição de Laura Natal Rodrigues, com imagens da Biblioteca Nacional do Brasil",
        "source": "https://www.gutenberg.org/ebooks/67535",
        "download": "https://raw.githubusercontent.com/GITenberg/Triste-Fim-de-Polycarpo-Quaresma_67535/master/67535-0.txt",
        "source_sha256": source_hash,
        "license": "Texto literário em domínio público no Brasil e nos EUA",
        "license_basis": ["https://www.planalto.gov.br/ccivil_03/leis/l9610.htm#art41",
                          "https://www.gutenberg.org/policy/license.html"],
        "normalization": "Somente os capítulos I de cada parte, com ortografia da transcrição, Unicode NFC e quebras LF",
        "characters": characters,
        "documents": documents,
        "overlap": {"criterion": "Sequências distintas de cinco palavras, divididas pelo menor conjunto",
                    "pairs_checked": len(external) * len(original) + 3,
                    "rejection_threshold": 0.8, "maximum": max_overlap, "pair": pair},
    }
    (target / "sources.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{characters} caracteres em {len(documents)} capítulos, sobreposição máxima {max_overlap:.4%}.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
