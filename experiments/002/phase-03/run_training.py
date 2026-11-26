"""Run from the repository root with PYTHONPATH=. using the project environment."""

import json
import sys
from pathlib import Path

import torch

import train
from corpus import Partition, read_documents
from generate import Vocabulary, load_checkpoint
from model import CausalTransformer, Config


SOURCE = Path("checkpoints/experiment-002-phase2/model-1000.pt")
OUTPUT = Path("checkpoints/experiment-002-phase3")
CORPUS = Path("data/experiment-002")


def validate():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    saved = load_checkpoint(SOURCE)
    assert (saved["step"], saved["threads"], saved["batch_size"], saved["seed"]) == (1000, 2, 8, 42)
    config = Config(**saved["config"])
    assert config == Config(101)
    train_docs, val_docs, fingerprints = read_documents(CORPUS, config.context)
    assert len(fingerprints) == 13
    train.validate_fingerprint(saved, "documents", fingerprints)
    characters = ["�", *sorted(set("".join(train_docs)) - {"�"})]
    assert saved["vocabulary"] == characters
    model = CausalTransformer(config)
    model.load_state_dict(saved["model"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 835685
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    assert optimizer.state_dict()["param_groups"] == saved["optimizer"]["param_groups"]
    optimizer.load_state_dict(saved["optimizer"])
    assert len(optimizer.state) == len(list(model.parameters()))
    for state in optimizer.state.values():
        assert state["step"].item() == 1000
        assert all(torch.isfinite(value).all() for value in state.values())
    torch.Generator().set_state(saved["torch_rng"])
    torch.Generator().set_state(saved["train_rng"])
    vocab = Vocabulary(characters)
    for docs, seed, metric in ((train_docs, 44, "train_loss"), (val_docs, 43, "val_loss")):
        assert train.evaluate(model, Partition(docs, vocab, config.context), 8, seed) == saved[metric]
    return saved, parameters


def main():
    saved, parameters = validate()
    OUTPUT.mkdir(exist_ok=False)
    configuration = {name: saved[name] for name in ("config", "document_sha256", "threads", "batch_size", "seed")}
    configuration.update(parameters=parameters, optimizer="AdamW", learning_rate=3e-4,
                         resume_checkpoint=str(SOURCE), start_step=1000, final_step=5000,
                         evaluation_interval=100, checkpoint_interval=1000, evaluation_batches=4,
                         train_eval_seed=44, validation_seed=43, device="cpu", torch_version=str(torch.__version__))
    Path(__file__).with_name("config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print("Checkpoint 1000 validado: 835685 parâmetros, 101 entradas, 13 fingerprints, AdamW e RNGs.", flush=True)

    original_loss_on = train.loss_on

    def checked_loss_on(*args, **kwargs):
        loss = original_loss_on(*args, **kwargs)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Loss não finita. Treinamento interrompido.")
        return loss

    train.loss_on = checked_loss_on
    sys.argv = [
        "train.py", "--documents", str(CORPUS), "--preset", "full", "--resume", str(SOURCE),
        "--steps", "5000", "--threads", "2", "--eval-interval", "100", "--save-interval", "1000",
        "--keep-checkpoints", "--checkpoint", str(OUTPUT / "model.pt"),
    ]
    train.main()


if __name__ == "__main__":
    main()
