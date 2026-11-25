import re
from hashlib import sha256

import torch


class Partition:
    def __init__(self, documents, vocabulary, context):
        self.data = torch.tensor(vocabulary.encode("".join(documents)))
        self.starts = None
        if len(documents) > 1:
            starts = []
            offset = 0
            for document in documents:
                starts.append(torch.arange(offset, offset + len(document) - context))
                offset += len(document)
            self.starts = torch.cat(starts)
        self.size = len(self.starts) if self.starts is not None else len(self.data) - context

    def sample(self, context, batch_size, rng):
        starts = torch.randint(self.size, (batch_size, 1), generator=rng)
        if self.starts is not None:
            starts = self.starts[starts]
        windows = self.data[starts + torch.arange(context + 1)]
        return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def read_documents(root, context):
    partitions = [sorted((root / name).glob("*.txt")) for name in ("train", "validation")]
    if not all(partitions):
        raise ValueError("Use ao menos um arquivo .txt em train/ e validation/.")

    documents, hashes, seen = [], {}, []
    for files in partitions:
        texts = []
        for path in files:
            text = path.read_text(encoding="utf-8-sig")
            name = path.relative_to(root).as_posix()
            if len(text) <= context:
                raise ValueError(f"{name} precisa ter pelo menos {context + 1} caracteres.")
            digest = sha256(text.encode("utf-8")).hexdigest()
            words = re.findall(r"[^\W_]+", text.casefold())
            shingles = set(zip(*(words[index:] for index in range(5))))
            for other, previous in seen:
                overlap = len(shingles & previous)
                smaller = min(len(shingles), len(previous))
                if digest == hashes[other] or (smaller and overlap / smaller >= 0.8):
                    raise ValueError(f"Documentos duplicados ou semelhantes: {other} e {name}.")
            seen.append((name, shingles))
            hashes[name] = digest
            texts.append(text)
        documents.append(texts)
    return documents[0], documents[1], hashes
