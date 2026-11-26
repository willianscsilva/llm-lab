"""Archive prior results and convert the original four-batch logs to CSV."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRIC = re.compile(
    r"^passo=(\d+)(?: minibatch=([\d.]+) tokens/s=(\d+))? "
    r"treino_médio=([\d.]+) validação=([\d.]+) tempo=([\d.]+)s$", re.MULTILINE,
)


def anonymize(text):
    return text.replace(str(Path.cwd()) + "/", "")


def log_metrics(text, destination, expected_steps):
    rows = []
    for step, minibatch, rate, train, val, elapsed in METRIC.findall(text):
        rows.append((int(step), minibatch, train, val, f"{math.exp(float(train)):.8f}",
                     f"{math.exp(float(val)):.8f}", rate, elapsed))
    assert [row[0] for row in rows] == list(expected_steps)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("step", "minibatch_loss", "train_loss_4", "validation_loss_4",
                         "train_perplexity_4", "validation_perplexity_4",
                         "update_tokens_per_second", "elapsed_seconds"))
        writer.writerows(rows)


def archive(phase, checkpoint_path, log, start, final, interval, save_interval):
    import torch

    from generate import load_checkpoint
    from model import CausalTransformer, Config

    torch.set_num_threads(2)
    target = ROOT / f"phase-{phase:02d}"
    target.mkdir(exist_ok=True)
    original = log.read_text(encoding="utf-8")
    (target / "training.log").write_text(anonymize(original), encoding="utf-8")
    first = start + interval if start else 0
    log_metrics(original, target / "metrics.csv", range(first, final + 1, interval))
    saved = load_checkpoint(checkpoint_path)
    assert saved["step"] == final
    config = {name: saved[name] for name in ("config", "document_sha256", "threads", "batch_size", "seed")}
    model = CausalTransformer(Config(**config["config"]))
    config.update(
        parameters=sum(p.numel() for p in model.parameters()),
        optimizer="AdamW", learning_rate=saved["optimizer"]["param_groups"][0]["lr"],
        start_step=start, final_step=final, evaluation_interval=interval, checkpoint_interval=save_interval,
        evaluation_batches=4, train_eval_seed=44, validation_seed=43, device="cpu",
        provenance="Configuração extraída do checkpoint final e dos comandos registrados na fase.",
    )
    (target / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def archive_prior(phase1_log):
    from evaluate_run import PROMPTS, write_samples

    import torch

    from generate import Vocabulary, generate, load_checkpoint
    from model import CausalTransformer, Config

    base = Path("checkpoints")
    historical = base / "experiment-002-phase2"
    first = archive(1, base / "experiment-002-0100.pt", phase1_log, 0, 100, 25, 50)
    second = archive(2, historical / "model-1000.pt", historical / "train.log", 100, 1000, 50, 250)
    for name in ("samples.txt", "results.json"):
        (second / name).write_bytes((historical / name).read_bytes())
    (second / "training-time.txt").write_text(anonymize((historical / "time.txt").read_text()), encoding="utf-8")

    torch.use_deterministic_algorithms(True)
    samples = []
    for step in (0, 50, 100):
        saved = load_checkpoint(base / f"experiment-002-{step:04d}.pt")
        model = CausalTransformer(Config(**saved["config"]))
        model.load_state_dict(saved["model"])
        vocab = Vocabulary(saved["vocabulary"])
        for prompt in PROMPTS:
            continuation = generate(model, vocab, prompt, 40, 42, 0.8)[len(prompt):]
            samples.append({"step": step, "count": 40, "seed": 42, "temperature": 0.8,
                            "prompt": prompt, "continuation": continuation})
    previous = json.loads((second / "results.json").read_text(encoding="utf-8"))
    assert samples[-4:] == [row for row in previous["samples"] if row["step"] == 100]
    write_samples(first / "samples.txt", samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-log", type=Path)
    args = parser.parse_args()
    if args.phase1_log:
        # The generation formatting and historical prompts are shared with phase 3.
        import sys
        sys.path.insert(0, str(ROOT / "phase-03"))
        archive_prior(args.phase1_log)
    else:
        directory = ROOT / "phase-03"
        log = directory / "training.log"
        content = anonymize(log.read_text(encoding="utf-8"))
        log.write_text(content, encoding="utf-8")
        log_metrics(content, directory / "metrics.csv", range(1100, 5001, 100))
        timing = directory / "training-time.txt"
        timing.write_text(anonymize(timing.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
