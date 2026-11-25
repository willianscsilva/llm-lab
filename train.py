import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from corpus import Partition, read_documents
from generate import Vocabulary, generate, load_checkpoint
from model import CausalTransformer, Config


def loss_on(model, data, batch_size, rng):
    x, y = data.sample(model.config.context, batch_size, rng)
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())


@torch.inference_mode()
def evaluate(model, data, batch_size, seed):
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    return sum(loss_on(model, data, batch_size, rng).item() for _ in range(4)) / 4


def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--steps", type=int, help="Passo final acumulado")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--corpus", type=Path, default=Path(__file__).with_name("data") / "corpus.txt")
    inputs.add_argument("--documents", type=Path, help="Diretório com train/ e validation/")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--keep-checkpoints", action="store_true", help="Guardar também um arquivo por passo salvo")
    args = parser.parse_args()
    smoke = args.preset == "smoke"
    final_step = args.steps if args.steps is not None else (3 if smoke else 1000)
    batch_size, interval = (2, 1) if smoke else (8, 50)
    eval_interval = interval if args.eval_interval is None else args.eval_interval
    save_interval = interval if args.save_interval is None else args.save_interval
    if min(eval_interval, save_interval) < 1:
        parser.error("Os intervalos devem ser >= 1.")
    saved = load_checkpoint(args.resume) if args.resume else None
    step = saved["step"] if saved else 0
    if final_step <= step or args.threads < 1:
        parser.error("O passo final deve superar o passo salvo e threads deve ser >= 1.")
    seed = saved["seed"] if saved else args.seed
    if saved:
        batch_size = saved["batch_size"]
        if args.threads != saved["threads"]:
            parser.error("Use o mesmo número de threads do checkpoint.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train_rng = torch.Generator().manual_seed(seed)
    context = saved["config"]["context"] if saved else Config(0).context
    try:
        if args.documents:
            train_docs, val_docs, hashes = read_documents(args.documents, context)
            fingerprint = {"document_sha256": hashes}
        else:
            text = args.corpus.read_text(encoding="utf-8")
            split = int(len(text) * 0.9)
            train_docs, val_docs = [text[:split]], [text[split:]]
            fingerprint = {"corpus_sha256": sha256(text.encode("utf-8")).hexdigest()}
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if saved and any(saved.get(key) != value for key, value in fingerprint.items()):
        parser.error("O corpus difere do corpus usado no checkpoint.")
    if any(len(doc) <= context for doc in train_docs + val_docs):
        parser.error("Cada documento ou partição deve conter pelo menos contexto + 1 caracteres.")
    characters = saved["vocabulary"] if saved else ["�", *sorted(set("".join(train_docs)) - {"�"})]
    vocabulary = Vocabulary(characters)
    config = Config(**saved["config"]) if saved else Config(len(characters))
    train_data = Partition(train_docs, vocabulary, context)
    val_data = Partition(val_docs, vocabulary, context)
    model = CausalTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        train_rng.set_state(saved["train_rng"])

    print(f"CPU | parâmetros={sum(p.numel() for p in model.parameters()):,} | "
          f"vocabulário={len(characters)} | "
          f"passo_inicial={step}", flush=True)
    for name, docs, partition in (("treino", train_docs, train_data), ("validação", val_docs, val_data)):
        unknown = (partition.data == 0).sum().item()
        size = len(partition.data)
        print(f"{name} | documentos={len(docs)} | caracteres={size} | "
              f"desconhecidos={unknown}/{size} ({unknown / size:.4%}) | janelas={partition.size}", flush=True)
    prompt = "A manhã"
    if not args.documents:
        print(f"Antes: {generate(model, vocabulary, prompt, 40, seed)!r}", flush=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    first_step = step + 1 if saved else 0
    update_seconds = 0.0
    for step in range(first_step, final_step + 1):
        minibatch = ""
        if step:
            model.train()
            tick = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(model, train_data, batch_size, train_rng)
            loss.backward()
            optimizer.step()
            update_seconds += perf_counter() - tick
            rate = (step - max(first_step - 1, 0)) * batch_size * context / update_seconds
            minibatch = f" minibatch={loss.item():.4f} tokens/s={rate:.0f}"
        saving = step % save_interval == 0 or step == final_step
        evaluating = step % eval_interval == 0 or saving
        if evaluating:
            train_loss = evaluate(model, train_data, batch_size, seed + 2)
            val_loss = evaluate(model, val_data, batch_size, seed + 1)
            print(f"passo={step}{minibatch} treino_médio={train_loss:.4f} validação={val_loss:.4f} "
                  f"tempo={perf_counter() - started:.2f}s", flush=True)
        if saving and (step or args.keep_checkpoints):
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(config), "vocabulary": characters, "step": step,
                "seed": seed, "batch_size": batch_size, "threads": args.threads,
                **fingerprint, "torch_rng": torch.get_rng_state(),
                "train_rng": train_rng.get_state(),
                "train_loss": train_loss, "val_loss": val_loss,
            }
            if args.keep_checkpoints:
                snapshot = args.checkpoint.with_name(f"{args.checkpoint.stem}-{step:04d}{args.checkpoint.suffix}")
                save_checkpoint(snapshot, state)
            if step:
                save_checkpoint(args.checkpoint, state)
    if not args.documents:
        print(f"Depois: {generate(model, vocabulary, prompt, 40, seed)!r}", flush=True)
    print(f"Checkpoint: {args.checkpoint} | tempo total={perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
