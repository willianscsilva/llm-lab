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

## Experimento 002 — Padrões do português

```bash
python train.py --documents data/experiment-002 --preset full --steps 100 \
  --eval-interval 25 --save-interval 50 --keep-checkpoints \
  --checkpoint checkpoints/experiment-002.pt
python generate.py checkpoints/experiment-002-0100.pt \
  --prompt "A manhã" --tokens 40 --seed 42 --temperature 0.8
```

Em `--documents`, coloque um documento UTF-8 por arquivo `.txt` em `train/` ou `validation/`. Cada documento precisa ter pelo menos 129 caracteres. As janelas ficam dentro de cada documento. O treino rejeita duplicatas e documentos com sobreposição de pelo menos 80% das sequências distintas de cinco palavras do documento com menos sequências. Na retomada com `--resume`, mantenha os mesmos documentos, nomes, partições e número de threads.

O treino registra as losses médias no passo zero, nos intervalos de avaliação e a cada salvamento. A avaliação usa quatro lotes fixos por partição. Com `--keep-checkpoints`, os arquivos numerados desta execução correspondem aos passos 0, 50 e 100.

O corpus tem 444.089 caracteres e 13 documentos. Inclui 12 contos de Machado de Assis em *Papéis Avulsos* (1882), com transcrição de Laura Natal Rodrigues e Marc D’Hooghe e imagens da Biblioteca Brasiliana Guita e José Mindlin, e a narrativa de *Iracema* (1865), de José de Alencar, com transcrição de Laura Natal Rodrigues e imagens da HathiTrust Digital Library. A validação contém *A chinela turca*, *O anel de Polícrates* e *O espelho*. O texto literário mantém a ortografia da transcrição, com Unicode NFC e quebras LF. Os arquivos contêm os textos extraídos das edições, com a autoria e os créditos de digitalização registrados aqui. As obras estão em domínio público no Brasil e nos EUA. [Papéis Avulsos, Project Gutenberg](https://www.gutenberg.org/ebooks/57001) · [Iracema, Project Gutenberg](https://www.gutenberg.org/ebooks/67740) · [Lei 9.610/1998, art. 41](https://www.planalto.gov.br/ccivil_03/leis/l9610.htm#art41) · [Política do Project Gutenberg](https://www.gutenberg.org/policy/license.html)

### Resultados em CPU

| Fase | Passos acumulados | Loss média de treino | Loss de validação (4 lotes) | Perplexidade por caractere (validação) | Tempo total | Tokens/s nas atualizações | Pico de memória |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 → 100 | 4,7797 → 2,6049 | 4,7717 → 2,6456 | 118,12 → 14,09 | 34,04 s | 4.011 | 425 MiB |
| 2 | 100 → 1.000 | 2,6049 → 2,3513 | 2,6456 → 2,3816 | 14,09 → 10,82 | 221,25 s | 4.454 | 435 MiB |

A fase 2 retomou o checkpoint 100 com o AdamW e os geradores aleatórios restaurados. Em 64 lotes fixos de validação, a loss foi 4,7734 no passo 0, 2,6715 no passo 100 e 2,4212 no passo 1.000. A perplexidade final nessa avaliação foi 11,26. As duas avaliações medem previsão de caracteres em prosa literária do século XIX.
