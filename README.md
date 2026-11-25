# Experimento 001 — Transformer causal em português

Requer Python 3.10 ou superior. Dentro do projeto, crie o ambiente e instale o PyTorch para CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Treine por três passos, gere texto a partir do checkpoint e retome até o quarto passo acumulado.

```bash
python train.py
python generate.py checkpoints/latest.pt --prompt "A manhã"
python train.py --resume checkpoints/latest.pt --steps 4
```

Para iniciar um treino de 1.000 passos com lote de oito janelas, use `python train.py --preset full --checkpoint checkpoints/full.pt`. Ajuste o passo final com `--steps` e as threads com `--threads` (padrão de duas). O corpus é original. Para usar outro arquivo UTF-8, passe `--corpus` e mantenha o mesmo arquivo ao retomar.
