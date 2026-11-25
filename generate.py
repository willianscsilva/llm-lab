import argparse
import math

import torch

from model import CausalTransformer, Config


class Vocabulary:
    def __init__(self, characters):
        self.characters = characters
        self.ids = {char: i for i, char in enumerate(characters)}

    def encode(self, text):
        return [self.ids.get(char, 0) for char in text]

    def decode(self, ids):
        return "".join(self.characters[i] for i in ids)


@torch.inference_mode()
def generate(model, vocabulary, prompt, count, seed, temperature=1.0):
    model.eval()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.tensor([vocabulary.encode(prompt)], dtype=torch.long)
    for _ in range(count):
        logits = model(tokens[:, -model.config.context:])[:, -1]
        next_token = torch.multinomial((logits / temperature).softmax(dim=-1), 1, generator=rng)
        tokens = torch.cat((tokens, next_token), dim=1)
    return prompt + vocabulary.decode(tokens[0, len(prompt):].tolist())


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="A manhã")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.prompt or args.tokens < 0 or args.threads < 1:
        parser.error("Informe um prompt não vazio, tokens >= 0 e threads >= 1.")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("A temperatura deve ser finita e maior que zero.")
    torch.set_num_threads(args.threads)
    checkpoint = load_checkpoint(args.checkpoint)
    model = CausalTransformer(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    print(generate(model, Vocabulary(checkpoint["vocabulary"]), args.prompt, args.tokens, args.seed, args.temperature))


if __name__ == "__main__":
    main()
