"""Run from the repository root with PYTHONPATH=. after training."""

import csv
import json
import math
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch

from corpus import Partition, read_documents
from generate import Vocabulary, generate, load_checkpoint
from model import CausalTransformer, Config
from train import loss_on, validate_fingerprint


PHASE = Path(__file__).resolve().parent
CHECKPOINTS = Path("checkpoints")
STEPS = (0, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000)
PROMPTS = ("A manhã", "Era uma vez", "O homem", "Na cidade")
PAIRS = (("O homem", "Na cidade"), ("O homem", "A mulher"), ("A manhã", "A noite"))


def checkpoint(step):
    if step <= 100:
        return CHECKPOINTS / f"experiment-002-{step:04d}.pt"
    phase = 2 if step <= 1000 else 3
    return CHECKPOINTS / f"experiment-002-phase{phase}" / f"model-{step:04d}.pt"


def external_partition(vocab, context):
    directory = PHASE / "external"
    manifest = directory / "sources.json"
    if not manifest.is_file():
        return None, None
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    documents = []
    for entry in metadata["documents"]:
        data = (directory / entry["filename"]).read_bytes()
        assert sha256(data).hexdigest() == entry["sha256"]
        text = data.decode("utf-8")
        assert len(text) == entry["characters"] and len(text) > context
        documents.append(text)
    partition = Partition(documents, vocab, context)
    unknown = (partition.data == 0).sum().item()
    assert len(partition.data) == metadata["characters"]
    return partition, {
        "documents": len(documents), "characters": len(partition.data), "windows": partition.size,
        "unknown_characters": unknown, "unknown_fraction": unknown / len(partition.data),
    }


def evaluate_partition(model, data, seed, batch_size):
    rng = torch.Generator().manual_seed(seed)
    losses = [loss_on(model, data, batch_size, rng).item() for _ in range(64)]
    assert all(math.isfinite(loss) for loss in losses)
    return sum(losses) / 64, sum(losses[:4]) / 4


def context_analysis(model, vocab):
    distributions, top = {}, {}
    for prompt in dict.fromkeys(prompt for pair in PAIRS for prompt in pair):
        tokens = torch.tensor([vocab.encode(prompt)])
        probs = model(tokens)[0, -1].double().softmax(dim=-1)
        distributions[prompt] = probs
        values, indices = probs.topk(10)
        top[prompt] = [{"character": vocab.characters[index], "probability": value}
                       for index, value in zip(indices.tolist(), values.tolist())]
    pairs = []
    for first, second in PAIRS:
        p, q = distributions[first], distributions[second]
        mid = (p + q) / 2
        divergence = (torch.special.xlogy(p, p / mid).sum() + torch.special.xlogy(q, q / mid).sum()) / 2
        score = divergence.item()
        assert 0 <= score <= math.log(2)
        pairs.append({"prompts": [first, second], "jensen_shannon_nats": score})
    return {"top_10": top, "pairs": pairs}


def sample_rows(model, vocab, step):
    if step < 1000:
        return
    settings = [(40, 42)]
    if step in (1000, 5000):
        settings += [(200, seed) for seed in (42, 123, 2026)]
    for count, seed in settings:
        for prompt in PROMPTS:
            continuation = generate(model, vocab, prompt, count, seed, 0.8)[len(prompt):]
            assert len(continuation) == count
            yield {"step": step, "count": count, "seed": seed,
                   "temperature": 0.8, "prompt": prompt, "continuation": continuation}


def write_samples(path, samples):
    with path.open("w", encoding="utf-8") as output:
        previous = None
        for row in samples:
            group = row["step"], row["count"], row["seed"], row["temperature"]
            if group != previous:
                step, count, seed, temperature = group
                if previous:
                    print(file=output)
                print(f"Passo {step} | {count} caracteres | seed {seed} | temperatura {temperature}", file=output)
            previous = group
            print(f"{row['prompt']} → {json.dumps(row['continuation'], ensure_ascii=False)}", file=output)


@torch.inference_mode()
def main():
    started = perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    original = load_checkpoint(checkpoint(1000))
    config = Config(**original["config"])
    vocab = Vocabulary(original["vocabulary"])
    train_docs, val_docs, fingerprints = read_documents(Path("data/experiment-002"), config.context)
    train_data = Partition(train_docs, vocab, config.context)
    val_data = Partition(val_docs, vocab, config.context)
    external, external_summary = external_partition(vocab, config.context)
    partitions = [("train", train_data, 44), ("validation", val_data, 43)]
    if external is not None:
        partitions.append(("external", external, 45))
    model = CausalTransformer(config).eval()
    history = json.loads((PHASE.parent / "phase-02/results.json").read_text(encoding="utf-8"))
    phase2_validation = {row["step"]: row["validation_loss_64"] for row in history["checkpoint_evaluation"]}
    phase2_samples = [row for row in history["samples"] if row["step"] == 1000]

    metrics, samples, context = [], [], {}
    for step in STEPS:
        saved = load_checkpoint(checkpoint(step))
        assert saved["step"] == step and saved["config"] == original["config"]
        assert saved["vocabulary"] == vocab.characters
        assert all(saved[key] == original[key] for key in ("seed", "batch_size", "threads"))
        validate_fingerprint(saved, "documents", fingerprints)
        if step:
            assert saved["optimizer"]["param_groups"] == original["optimizer"]["param_groups"]
            assert all(state["step"].item() == step for state in saved["optimizer"]["state"].values())
        model.load_state_dict(saved["model"])
        for split, data, seed in partitions:
            loss, first_four = evaluate_partition(model, data, seed, saved["batch_size"])
            if split != "external":
                key = "val_loss" if split == "validation" else "train_loss"
                assert first_four == saved[key], (step, split, first_four, saved[key])
            if split == "validation" and step in phase2_validation:
                assert loss == phase2_validation[step]
            metrics.append({"step": step, "split": split, "batches": 64, "batch_size": saved["batch_size"],
                            "seed": seed, "loss": loss, "perplexity": math.exp(loss)})
            print(f"{step:4} {split:10} loss={loss:.6f} perplexity={math.exp(loss):.6f}", flush=True)
        new_samples = list(sample_rows(model, vocab, step))
        if step == 1000:
            assert new_samples == phase2_samples
        samples.extend(new_samples)
        if step in (1000, 5000):
            context[step] = context_analysis(model, vocab)

    assert len(samples) == 44
    with (PHASE / "evaluation-64.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=metrics[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    write_samples(PHASE / "samples.txt", samples)
    summary = {"external": external_summary, "external_seed": 45 if external is not None else None,
               "train_seed": 44, "validation_seed": 43, "prompt_comparison": context,
               "seconds_including_generation": perf_counter() - started}
    (PHASE / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(samples)} gerações registradas. Tempo de avaliação e geração: "
          f"{summary['seconds_including_generation']:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
