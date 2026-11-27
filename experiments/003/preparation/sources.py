import json
import re
from html.parser import HTMLParser
from urllib.parse import quote, unquote

from acquisition import sha


WIKI = "https://pt.wikipedia.org"
BOOKS = "https://pt.wikibooks.org"
CC = "https://creativecommons.org/licenses/by-sa/4.0/"
PD = "https://www.planalto.gov.br/ccivil_03/leis/l9610.htm#art41"
TERMS = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
TARGETS = {"literatura": .30, "enciclopédico": .25, "científico e educacional": .20,
           "histórico e institucional": .15, "diálogos e cotidiano": .10}

LITERATURE = (
    (62525, "Historias-Brazileiras", "Histórias brasileiras", "Alfredo d’Escragnolle Taunay", 1899),
    (62896, "Ao-Entardecer-Contos-Varios_62896", "Ao entardecer", "Alfredo d’Escragnolle Taunay", 1899),
    (55752, "Dom-Casmurro", "Dom Casmurro", "Machado de Assis", 1908),
    (54829, "Memorias-Posthumas-de-Braz-Cubas", "Memórias póstumas de Brás Cubas", "Machado de Assis", 1908),
    (56737, "Esau-e-Jacob", "Esaú e Jacó", "Machado de Assis", 1908),
    (55797, "Memorial-de-Ayres", "Memorial de Aires", "Machado de Assis", 1908),
    (67780, "Yaya-Garcia", "Yayá Garcia", "Machado de Assis", 1908),
    (69187, "O-Cortico", "O cortiço", "Aluísio Azevedo", 1913),
    (68541, "O-Atheneu-chronica-de-saudades", "O Ateneu", "Raul Pompéia", 1895),
    (69229, "A-fallencia", "A falência", "Júlia Lopes de Almeida", 1934),
    (30341, "Contos-Paraenses", "Contos paraenses", "João Marques de Carvalho", 1910),
    (27637, "Os-Maias-episodios-da-vida-romantica", "Os Maias", "Eça de Queirós", 1900),
    (18220, "A-Cidade-e-as-Serras", "A cidade e as serras", "Eça de Queirós", 1900),
)
WIKIBOOKS = (
    "Lógica", "Física", "Mecânica clássica", "Álgebra linear", "Química", "Astronomia",
    "Teoria musical", "Estatística", "Matemática", "Programar em Python", "História do Brasil",
    "Biologia", "Português", "Geologia", "Economia", "Eletricidade", "Fotografia",
    "Geometria", "Lógica/Proposições", "Física/Termodinâmica", "Matemática elementar",
    "Aprender a tocar guitarra", "Programação", "Manual do Linux", "Latim",
)
WIKI_CATEGORIES = {
    "enciclopédico": ("Geografia do Brasil", "Biomas do Brasil", "Arquitetura", "Vegetação",
                     "Música do Brasil", "Gêneros literários", "Cinema", "Culinária do Brasil",
                     "Ecossistemas", "Instrumentos musicais", "Astronomia"),
    "histórico e institucional": ("História do Brasil", "História do Brasil por período",
                                "Brasil Colônia", "Império do Brasil", "República Velha",
                                "História econômica do Brasil", "Patrimônio histórico do Brasil",
                                "História da ciência", "História da América do Sul",
                                "Instituições do Brasil"),
}
WIKI_PAGES = {
    "enciclopédico": ("Brasil", "Amazônia", "Mata Atlântica", "Pantanal", "Caatinga", "Cerrado",
                     "Pampa", "Floresta tropical", "Geografia do Brasil", "Clima do Brasil",
                     "Bacia do rio Amazonas", "Rio São Francisco", "Manguezal", "Pradaria",
                     "Música popular brasileira", "Samba", "Choro", "Bossa nova", "Maracatu",
                     "Literatura do Brasil", "Arquitetura do Brasil", "Arte plumária",
                     "Cultivo do café", "Cinema do Brasil", "Geologia", "Lua", "Sistema Solar",
                     "Terra", "Marte", "Júpiter", "Estrela", "Galáxia", "Via Láctea",
                     "Língua portuguesa", "Português brasileiro", "Culinária do Brasil",
                     "Cultura do Brasil", "Xadrez", "Voleibol", "Atletismo", "Futebol",
                     "Futsal", "Flora do Brasil", "Fauna do Brasil", "Coral", "Recife de coral"),
    "histórico e institucional": ("História do Brasil", "Brasil Colônia", "Império do Brasil",
                  "Primeira República Brasileira", "Era Vargas", "Independência do Brasil",
                  "Inconfidência Mineira", "Guerra do Paraguai", "Guerra de Canudos",
                  "Revolução Farroupilha", "Cabanagem", "Balaiada", "Revolta da Chibata",
                  "Guerra do Contestado", "Revolta da Vacina", "História de Portugal",
                  "História da América do Sul", "Revolução Industrial", "Revolução Francesa",
                  "Renascimento", "Iluminismo", "Revolução Científica", "História da matemática",
                  "História da tecnologia", "História da educação no Brasil", "História da imprensa no Brasil",
                  "História da moeda", "História do café no Brasil", "Fundação de Brasília",
                  "Capitanias hereditárias", "Declaração Universal dos Direitos Humanos",
                  "História da aviação", "História dos computadores"),
}


def wiki_source(domain, title, category):
    slug = quote(title.replace(" ", "_"), safe="/")
    source = "Wikivros" if domain == BOOKS else "Wikipédia em português"
    url = f"{domain}/wiki/{slug}"
    return dict(id=("wb-" if domain == BOOKS else "wp-") + sha(title.encode())[:20],
                title=title, author="Colaboradores de " + source, category=category,
                language="pt", source=source, origin=url, url=url, license="CC BY-SA 4.0",
                license_url=CC, license_basis=TERMS,
                attribution=f"Colaboradores de {source}, {title}. Histórico na página original. Texto extraído e normalizado.",
                format="wiki", work=f"{domain}/{title.split('/')[0].casefold()}")


def candidates():
    for number, repo, title, author, death in LITERATURE:
        repo = repo if repo.endswith(f"_{number}") else f"{repo}_{number}"
        origin = f"https://www.gutenberg.org/ebooks/{number}"
        yield dict(id=f"pg-{number}", title=title, author=author, category="literatura", language="pt",
                   source="Project Gutenberg via GITenberg", origin=origin,
                   url=f"https://raw.githubusercontent.com/GITenberg/{repo}/master/{number}-0.txt",
                   license="domínio público", license_url=PD, license_basis=origin,
                   attribution=f"{author} (falecido em {death}), {title}. Transcrição digital identificada na fonte.",
                   format="gutenberg", work=f"pg-{number}")
    for title in WIKIBOOKS:
        yield wiki_source(BOOKS, title, "científico e educacional")
    for category, titles in WIKI_PAGES.items():
        for title in titles:
            yield wiki_source(WIKI, title, category)


def category_titles(html):
    pages = re.search(r'<div[^>]+id="mw-pages"[^>]*>(.*?)(?:<div[^>]+id="mw-hidden|</main>|$)', html, re.S)
    if not pages:
        return []
    titles = set()
    for href in re.findall(r'href="/wiki/([^"#?]+)"', pages[1]):
        title = unquote(href).replace("_", " ")
        if ":" not in title and "/" not in title and not title.startswith(("Lista ", "Lista de ")):
            titles.add(title)
    return sorted(titles, key=lambda title: sha(("003:" + title).encode()))


class Prose(HTMLParser):
    void = set("area base br col embed hr img input link meta param source track wbr".split())
    stop_headings = re.compile(r"^(referências|bibliografia|ver também|ligações externas|notas|fontes)$", re.I)
    boilerplate = re.compile(r"(?:^|[ -])(navbox|infobox|sidebar|hatnote|mw-editsection|"
                             r"reference|reflist|metadata|toc|ambox|portable-infobox|"
                             r"gallery|mw-empty-elt)(?:[ -]|$)", re.I)

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.blocks, self.buffer = [], [], []
        self.block = self.stopped = False

    def handle_starttag(self, tag, attrs):
        names = dict(attrs)
        inside, skip = self.stack[-1][1:] if self.stack else (False, False)
        article = "mw-parser-output" in names.get("class", "").split()
        skip = skip and (inside or not article)
        inside = inside or article
        classes = names.get("class", "").split()
        skip = skip or (inside and (tag in {"table", "figure", "sup", "script", "style", "math", "nav", "aside"}
                or any(self.boilerplate.match(value) for value in classes)))
        if tag not in self.void:
            self.stack.append((tag, inside, skip))
        if inside and not skip and not self.stopped:
            if tag in {"p", "h2", "h3", "h4"}:
                self.flush()
                self.block = tag
            elif tag == "br" and self.block:
                self.buffer.append("\n")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.void:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                if tag == self.block and self.stack[index][1:] == (True, False):
                    self.flush()
                del self.stack[index:]
                break

    def handle_data(self, text):
        if self.block and self.stack and self.stack[-1][1:] == (True, False) and not self.stopped:
            self.buffer.append(text)

    def flush(self):
        line = "".join(self.buffer).strip()
        if self.block and self.block.startswith("h") and self.stop_headings.fullmatch(line):
            self.stopped = True
        if line and not self.stopped:
            self.blocks.append(line)
        self.buffer.clear()
        self.block = False


def extract(raw, kind):
    text = raw.decode("utf-8-sig")
    if kind == "gutenberg":
        start = re.search(r"\*\*\* START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.I | re.S)
        end = re.search(r"\*\*\* END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK", text, re.I)
        if not start or not end or start.end() > end.start() or re.search(
            r"posted with (?:the )?permission|used with permission|is copyrighted", text[:start.end()], re.I
        ):
            raise ValueError("envelope ou direitos da edição Gutenberg não confirmados")
        body = text[start.end():end.start()]
        body = re.sub(r"(?is)^\s*(?:Produced by|Credits:).*?(?:\r?\n){3,}", "", body, count=1)
        return body, {"edition": None, "raw_characters": len(text)}
    page = Prose()
    page.feed(text)
    page.close()
    page.flush()
    revision = re.search(r'\bwgRevisionId["\']?\s*:\s*(\d+)', text)
    if not revision:
        revision = re.search(r'oldid=(\d+)', text)
    if not revision:
        raise ValueError("revisão Wikimedia ausente")
    return "\n\n".join(page.blocks), {"edition": revision[1], "raw_characters": len(text)}
