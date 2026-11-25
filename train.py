import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from generate import Vocabulary, generate, load_checkpoint
from model import CausalTransformer, Config


def sample(data, context, batch_size, rng):
    starts = torch.randint(len(data) - context, (batch_size, 1), generator=rng)
    windows = data[starts + torch.arange(context + 1)]
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def loss_on(model, data, batch_size, rng):
    x, y = sample(data, model.config.context, batch_size, rng)
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())


@torch.inference_mode()
def evaluate(model, data, batch_size, seed):
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    return sum(loss_on(model, data, batch_size, rng).item() for _ in range(4)) / 4


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--steps", type=int, help="Passo final acumulado")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).with_name("data") / "corpus.txt")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    smoke = args.preset == "smoke"
    final_step = args.steps if args.steps is not None else (3 if smoke else 1000)
    batch_size, interval = (2, 1) if smoke else (8, 50)
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
    text = args.corpus.read_text(encoding="utf-8")
    digest = sha256(text.encode("utf-8")).hexdigest()
    if saved and digest != saved["corpus_sha256"]:
        parser.error("O corpus difere do corpus usado no checkpoint.")
    # Divide o texto antes de extrair qualquer janela.
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    characters = saved["vocabulary"] if saved else ["�", *sorted(set(train_text) - {"�"})]
    vocabulary = Vocabulary(characters)
    config = Config(**saved["config"]) if saved else Config(len(characters))
    if min(len(train_text), len(val_text)) <= config.context:
        parser.error("Cada partição deve conter pelo menos contexto + 1 caracteres.")
    train_data = torch.tensor(vocabulary.encode(train_text))
    val_data = torch.tensor(vocabulary.encode(val_text))
    model = CausalTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        train_rng.set_state(saved["train_rng"])

    print(f"CPU | parâmetros={sum(p.numel() for p in model.parameters()):,} | "
          f"treino={len(train_data)} | validação={len(val_data)} | vocabulário={len(characters)} | "
          f"passo_inicial={step}", flush=True)
    prompt = "A manhã"
    print(f"Antes: {generate(model, vocabulary, prompt, 40, seed)!r}", flush=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    for step in range(step + 1, final_step + 1):
        model.train()
        tick = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(model, train_data, batch_size, train_rng)
        loss.backward()
        optimizer.step()
        rate = batch_size * config.context / (perf_counter() - tick)
        if step % interval == 0 or step == final_step:
            val_loss = evaluate(model, val_data, batch_size, seed + 1)
            print(f"passo={step} treino={loss.item():.4f} validação={val_loss:.4f} "
                  f"tokens/s={rate:.0f} tempo={perf_counter() - started:.2f}s", flush=True)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(config), "vocabulary": characters, "step": step,
                "seed": seed, "batch_size": batch_size, "threads": args.threads,
                "corpus_sha256": digest, "torch_rng": torch.get_rng_state(),
                "train_rng": train_rng.get_state(),
            }, temporary)
            temporary.replace(args.checkpoint)
    print(f"Depois: {generate(model, vocabulary, prompt, 40, seed)!r}", flush=True)
    print(f"Checkpoint: {args.checkpoint} | tempo total={perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
