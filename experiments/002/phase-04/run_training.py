"""Resume the original step 5000 checkpoint from the repository root."""

import csv
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch

from context_prompts import study_prompts
from corpus import Partition, read_documents
from generate import Vocabulary, load_checkpoint
from model import CausalTransformer, Config
from train import loss_on, validate_fingerprint

PHASE = Path("experiments/002/phase-04")
PREVIOUS = PHASE.parent / "phase-03"
SOURCE = Path("checkpoints/experiment-002-phase3/model-5000.pt")
OUTPUT = Path("checkpoints/experiment-002-phase4")
HISTORY = (PREVIOUS / "checkpoints-before.sha256", PREVIOUS / "checkpoints-phase3.sha256")
SEEDS = {"train": 44, "validation": 43, "external": 45}
SNAPSHOTS = (7500, 10000, 12500, 15000, 17500, 20000)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_hashes():
    hashes = {}
    for manifest in HISTORY:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, name = line.split(maxsplit=1)
            path = Path(name)
            if path != SOURCE and not path.exists():
                print(f"Checkpoint histórico indisponível: {path}", flush=True)
                continue
            if digest(path) != expected:
                raise ValueError(f"SHA-256 divergente: {path}")
            hashes[name] = expected
    if str(SOURCE) not in hashes:
        raise ValueError("Checkpoint 5000 ausente do manifesto da fase 3.")
    return hashes


def finite(value):
    if not math.isfinite(value):
        raise FloatingPointError("Loss não finita.")
    return value


@dataclass
class StopState:
    best_step: int
    best_loss: float
    significant_step: int
    significant_loss: float
    overfit_count: int = 0

    def update(self, step, loss):
        finite(loss)
        improved = loss < self.best_loss
        if improved:
            self.best_step, self.best_loss = step, loss
        self.overfit_count = self.overfit_count + 1 if loss >= self.best_loss + 0.02 else 0
        if loss <= self.significant_loss - 0.005:
            self.significant_step, self.significant_loss = step, loss
        reason = None
        if self.overfit_count >= 4:
            reason = "overfitting_sustentado"
        elif step - self.significant_step >= 3000:
            reason = "saturacao"
        return improved, reason


@torch.inference_mode()
def evaluate_fixed(model, data, batch_size, seed, batches):
    was_training = model.training
    try:
        model.eval()
        rng = torch.Generator().manual_seed(seed)
        losses = [finite(loss_on(model, data, batch_size, rng).item()) for _ in range(batches)]
        return sum(losses[:4]) / 4, sum(losses) / batches
    finally:
        model.train(was_training)


def restore(saved):
    model = CausalTransformer(Config(**saved["config"]))
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if optimizer.state_dict()["param_groups"] != saved["optimizer"]["param_groups"]:
        raise ValueError("Hiperparâmetros do AdamW divergentes.")
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    rng = torch.Generator().set_state(saved["train_rng"])
    return model, optimizer, rng


def update_weights(model, optimizer, data, batch_size, rng):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_on(model, data, batch_size, rng)
    measured = finite(loss.item())
    loss.backward()
    optimizer.step()
    return measured


def measure(model, data, batches):
    four, wide = {}, {}
    for split in (SEEDS if batches == 64 else ("train", "validation")):
        small, large = evaluate_fixed(model, data[split], 8, SEEDS[split], batches)
        if split != "external":
            four[split] = small
        if batches == 64:
            wide[split] = large
    return four, wide


def record(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_link(source, destination):
    temporary = destination.with_suffix(".tmp")
    os.link(source, temporary)
    temporary.replace(destination)


def save_state(output, state, improved, snapshot=False):
    latest = output / "latest.pt"
    temporary = latest.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(latest)
    if improved:
        atomic_link(latest, output / "best.pt")
    if snapshot:
        atomic_link(latest, output / f"model-{state['step']:05d}.pt")


def preflight():
    historical_hashes = verify_hashes()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    saved = load_checkpoint(SOURCE)
    assert (saved["step"], saved["threads"], saved["batch_size"], saved["seed"]) == (5000, 2, 8, 42)
    assert Config(**saved["config"]) == Config(101)
    train_docs, val_docs, hashes = read_documents(Path("data/experiment-002"), 128)
    validate_fingerprint(saved, "documents", hashes)
    assert len(hashes) == 13
    assert saved["vocabulary"] == ["�", *sorted(set("".join(train_docs)) - {"�"})]
    vocab = Vocabulary(saved["vocabulary"])
    external_dir = PREVIOUS / "external"
    external = json.loads((external_dir / "sources.json").read_text(encoding="utf-8"))
    external_docs, external_hashes = [], {}
    for row in external["documents"]:
        path = external_dir / row["filename"]
        content = path.read_bytes()
        assert sha256(content).hexdigest() == row["sha256"], path
        decoded = content.decode("utf-8")
        assert len(decoded) == row["characters"] and len(decoded) > 128
        external_hashes[path.as_posix()] = row["sha256"]
        external_docs.append(decoded)
    model, optimizer, _ = restore(saved)
    assert sum(p.numel() for p in model.parameters()) == 835685
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert len(optimizer.state) == len(list(model.parameters()))
    for state in optimizer.state.values():
        assert state["step"].item() == 5000 and all(torch.isfinite(t).all() for t in state.values())
    data = {name: Partition(docs, vocab, 128) for name, docs in
            zip(SEEDS, (train_docs, val_docs, external_docs))}
    four, wide = measure(model, data, 64)
    assert four == {"train": saved["train_loss"], "validation": saved["val_loss"]}
    with (PREVIOUS / "evaluation-64.csv").open(newline="", encoding="utf-8") as stream:
        expected = {row["split"]: float(row["loss"]) for row in csv.DictReader(stream) if row["step"] == "5000"}
    assert wide == expected
    configuration = {key: saved[key] for key in ("config", "document_sha256", "threads", "batch_size", "seed")}
    configuration.update(
        origin_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch_version=str(torch.__version__), parameters=835685, device="cpu", optimizer="AdamW",
        learning_rate=0.0003, adamw={k: v for k, v in optimizer.param_groups[0].items() if k != "params"},
        resume_checkpoint=str(SOURCE), start_step=5000, maximum_step=20000, snapshots=SNAPSHOTS,
        evaluation_intervals={"4": 250, "64": 500}, evaluation_seeds=SEEDS,
        external_sha256=external_hashes, checkpoint_history_sha256=historical_hashes,
        stop={"split": "validation", "batches": 64, "overfit_margin": 0.02, "overfit_count": 4,
              "significant_improvement": 0.005, "steps_without_significant_improvement": 3000},
        context_prompts=study_prompts(vocab.ids, 128),
    )
    return saved, data, four, wide, configuration


def csv_row(stream, row):
    writer = csv.DictWriter(stream, fieldnames=row.keys(), lineterminator="\n")
    if stream.tell() == 0:
        writer.writeheader()
    writer.writerow(row)
    stream.flush()


def run(saved, data, initial_four, initial_wide, historical_hashes):
    model, optimizer, rng = restore(saved)
    metadata = {key: saved[key] for key in ("config", "vocabulary", "seed", "batch_size", "threads",
                                           "data_mode", "document_sha256")}
    baseline = initial_wide["validation"]
    stop = StopState(5000, baseline, 5000, baseline)
    history, checkpoints = [], {}
    started = perf_counter()
    update_seconds, minibatch = 0.0, None
    last_completed = 5000

    def control(step, reason):
        record(PHASE / "stopping.json", {
            "evaluations": history, "checkpoints": checkpoints, "last_completed_step": step,
            "reason": reason, **asdict(stop),
        })

    try:
        with (PHASE / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics, \
             (PHASE / "evaluation-64.csv").open("w", newline="", encoding="utf-8") as extended:
            for step in range(5000, 20001):
                if step > 5000:
                    tick = perf_counter()
                    minibatch = update_weights(model, optimizer, data["train"], 8, rng)
                    update_seconds += perf_counter() - tick
                    last_completed = step
                if step % 250:
                    continue
                four, wide = (initial_four, initial_wide) if step == 5000 else measure(
                    model, data, 64 if step % 500 == 0 else 4,
                )
                rate = (step - 5000) * 1024 / update_seconds if update_seconds else None
                elapsed = perf_counter() - started
                csv_row(metrics, {"step": step, "minibatch_loss": minibatch, "train_loss_4": four["train"],
                                  "validation_loss_4": four["validation"], "train_perplexity_4": math.exp(four["train"]),
                                  "validation_perplexity_4": math.exp(four["validation"]),
                                  "update_tokens_per_second": rate, "elapsed_seconds": elapsed})
                print(f"passo={step} minibatch={minibatch} treino4={four['train']:.6f} "
                      f"val4={four['validation']:.6f} tokens/s={rate or 0:.0f} tempo={elapsed:.2f}s", flush=True)
                if not wide:
                    continue
                for split, loss in wide.items():
                    csv_row(extended, {"step": step, "split": split, "batches": 64, "batch_size": 8,
                                       "seed": SEEDS[split], "loss": loss, "perplexity": math.exp(loss)})
                improved, reason = (True, None) if step == 5000 else stop.update(step, wide["validation"])
                if step == 20000 and reason is None:
                    reason = "limite_maximo"
                event = {"step": step, "validation_loss_64": wide["validation"], "new_best": improved,
                         "steps_without_significant_improvement": step - stop.significant_step,
                         **asdict(stop), "reason": reason}
                history.append(event)
                state = {**metadata, "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "torch_rng": torch.get_rng_state(), "train_rng": rng.get_state(),
                         "train_loss": four["train"], "val_loss": four["validation"],
                         "losses_64": wide, "stopping": event}
                snapshot = step in SNAPSHOTS
                save_state(OUTPUT, state, improved, snapshot)
                info = {"step": step, "validation_loss_64": wide["validation"]}
                checkpoints["latest.pt"] = info
                if improved:
                    checkpoints["best.pt"] = info
                if snapshot:
                    checkpoints[f"model-{step:05d}.pt"] = info
                print(f"val64 passo={step} treino={wide['train']:.6f} validação={wide['validation']:.6f} "
                      f"externa={wide['external']:.6f} melhor={stop.best_loss:.6f}@{stop.best_step} "
                      f"overfit={stop.overfit_count} sem_melhora={step - stop.significant_step}", flush=True)
                if reason:
                    if verify_hashes() != historical_hashes:
                        raise ValueError("Inventário histórico mudou durante o treinamento.")
                    atomic_link(OUTPUT / "latest.pt", OUTPUT / "final.pt")
                    checkpoints["final.pt"] = info
                control(step, reason)
                if reason:
                    print(f"Encerramento: {reason} no passo {step}.", flush=True)
                    return
    except Exception as error:
        control(last_completed, f"falha_operacional: {type(error).__name__}")
        raise


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Diretório de saída já existe: {OUTPUT}")
    saved, data, four, wide, configuration = preflight()
    record(PHASE / "config.json", configuration)
    OUTPUT.mkdir()
    print(f"Checkpoint 5000 validado. Avaliações reproduzidas: quatro={four}, 64={wide}.", flush=True)
    run(saved, data, four, wide, configuration["checkpoint_history_sha256"])


if __name__ == "__main__":
    main()
