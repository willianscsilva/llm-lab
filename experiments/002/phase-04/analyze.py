"""Evaluate saved checkpoints, generate samples and measure verbatim matches."""

import csv
import json
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from generate import Vocabulary, generate, load_checkpoint
from model import CausalTransformer, Config
from run_training import OUTPUT, PHASE, PREVIOUS, SOURCE, digest, record, verify_hashes


PROMPTS = ("A manhã", "Era uma vez", "O homem", "Na cidade")
SEEDS = (42, 123, 2026)


def context_analysis(model, vocab, study):
    distributions, top = {}, {}
    groups = {}
    for group, entries in study.items():
        groups[group] = []
        for entry in entries:
            for prompt in entry["prompts"]:
                if prompt not in distributions:
                    tokens = torch.tensor([vocab.encode(prompt)])
                    probs = model(tokens)[0, -1].double().softmax(dim=-1)
                    distributions[prompt] = probs
                    values, indices = probs.topk(10)
                    top[prompt] = [
                        {"character": vocab.characters[index], "probability": value}
                        for index, value in zip(indices.tolist(), values.tolist())
                    ]
            p, q = (distributions[prompt] for prompt in entry["prompts"])
            mid = (p + q) / 2
            divergence = (torch.special.xlogy(p, p / mid).sum() + torch.special.xlogy(q, q / mid).sum()) / 2
            score = divergence.item()
            assert -1e-15 <= score <= math.log(2)
            groups[group].append({**entry, "jensen_shannon_nats": max(0.0, score)})
    distances = defaultdict(list)
    for row in groups["distance"]:
        distances[row["target_length"]].append(row["jensen_shannon_nats"])
    return {"pairs": groups, "top_10": top,
            "distance_means": {target: mean(values) for target, values in distances.items()}}


def historical_samples():
    expected = {}
    for line in (PREVIOUS / "samples.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("Passo "):
            step, count, seed = map(int, re.match(r"Passo (\d+) \| (\d+) caracteres \| seed (\d+)", line).groups())
        elif line and step == 5000:
            prompt, continuation = line.split(" → ", 1)
            expected[count, seed, prompt] = json.loads(continuation)
    assert len(expected) == 16
    return expected


def samples_for_step(model, vocab, step, roles):
    settings = [(40, 42)]
    if {"initial", "best", "final"}.intersection(roles):
        settings.extend((200, seed) for seed in SEEDS)
    rows = {}
    for count, seed in settings:
        for prompt in PROMPTS:
            continuation = generate(model, vocab, prompt, count, seed, 0.8)[len(prompt):]
            assert len(continuation) == count
            rows[count, seed, prompt] = continuation
    return rows


def checkpoint_roles(stopping):
    paths = {5000: SOURCE}
    roles = {5000: ["initial"]}
    for name, info in stopping["checkpoints"].items():
        if name == "latest.pt":
            continue
        step = info["step"]
        role = "snapshot" if name.startswith("model-") else name.removesuffix(".pt")
        paths.setdefault(step, OUTPUT / name)
        roles.setdefault(step, []).append(role)
    return paths, roles


def load_comparators(config):
    corpora = {"train": {}, "validation": {}, "external": {}}
    for name, expected in config["document_sha256"].items():
        content = (Path("data/experiment-002") / name).read_text(encoding="utf-8")
        assert sha256(content.encode("utf-8")).hexdigest() == expected
        corpora[name.split("/")[0]][name] = content
    for name, expected in config["external_sha256"].items():
        path = Path(name)
        assert digest(path) == expected
        corpora["external"][path.name] = path.read_text(encoding="utf-8")
    return {
        split: {name: SequenceMatcher(None, "", text, autojunk=False) for name, text in docs.items()}
        for split, docs in corpora.items()
    }


def verbatim(sample, documents):
    largest, location, literal = -1, "", ""
    for name, matcher in documents.items():
        matcher.set_seq1(sample)
        match = matcher.find_longest_match()
        if match.size > largest:
            largest, location = match.size, name
            literal = sample[match.a:match.a + match.size]
    covered = set()
    for start in range(len(sample) - 31):
        window = sample[start:start + 32]
        if any(window in matcher.b for matcher in documents.values()):
            covered.update(range(start, start + 32))
    return {"longest_chars": largest, "document": location, "longest_text": literal,
            "covered_fraction_32": len(covered) / len(sample)}


def memorization_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["step"], row["split"]].append(row)
    summaries = []
    for (step, split), group in groups.items():
        summary = {"step": step, "split": split, "samples": len(group), "roles": group[0]["roles"]}
        for field in ("longest_chars", "covered_fraction_32"):
            data = [row[field] for row in group]
            summary[field] = {"mean": mean(data), "median": median(data), "max": max(data)}
        summaries.append(summary)
    return summaries


@torch.inference_mode()
def main():
    started = perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = json.loads((PHASE / "config.json").read_text(encoding="utf-8"))
    stopping = json.loads((PHASE / "stopping.json").read_text(encoding="utf-8"))
    assert verify_hashes() == config["checkpoint_history_sha256"]
    paths, roles = checkpoint_roles(stopping)
    original = load_checkpoint(SOURCE)
    vocab = Vocabulary(original["vocabulary"])
    model = CausalTransformer(Config(**original["config"])).eval()
    prior_analysis = json.loads((PREVIOUS / "analysis.json").read_text(encoding="utf-8"))["prompt_comparison"]["5000"]
    comparators = load_comparators(config)
    context, matches = {}, []
    with (PHASE / "samples.txt").open("w", encoding="utf-8") as output:
        for step, path in sorted(paths.items()):
            saved = load_checkpoint(path)
            assert saved["step"] == step
            assert all(saved[key] == original[key] for key in
                       ("config", "vocabulary", "batch_size", "threads", "seed", "document_sha256"))
            model.load_state_dict(saved["model"])
            context[step] = context_analysis(model, vocab, config["context_prompts"])
            samples = samples_for_step(model, vocab, step, roles[step])
            if step == 5000:
                assert samples == historical_samples()
                assert context[step]["pairs"]["historical"] == prior_analysis["pairs"]
                assert all(context[step]["top_10"][prompt] == top for prompt, top in prior_analysis["top_10"].items())
            for (count, seed, prompt), continuation in samples.items():
                tags = ",".join(roles[step])
                print(f"{step} | {tags} | {count} caracteres | seed {seed} | temperatura 0,8 | "
                      f"{prompt} → {json.dumps(continuation, ensure_ascii=False)}", file=output)
                if count == 200:
                    for split, documents in comparators.items():
                        matches.append({"step": step, "roles": tags, "prompt": prompt, "seed": seed,
                                        "split": split, **verbatim(continuation, documents)})
            print(f"Passo {step}: {len(samples)} continuações e análise de contexto registradas.", flush=True)
    with (PHASE / "memorization.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=matches[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matches)
    record(PHASE / "analysis.json", {
        "context": context, "roles": roles, "historical_samples_verified": 16,
        "memorization": memorization_summary(matches), "analysis_seconds": perf_counter() - started,
    })
    print(f"Análise concluída em {perf_counter() - started:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
