# Experimento 002, fase 3

A execução retomou o checkpoint do passo 1.000 em `main`, no commit `ac4e60f`, e terminou no passo 5.000. Os cinco testes unitários passaram. A validação prévia conferiu 835.685 parâmetros, 101 entradas no vocabulário, 13 fingerprints, os estados do AdamW e dos geradores aleatórios e as losses salvas no passo 1.000.

## Treinamento

A fase fez 4.000 atualizações em CPU com duas threads, lote 8, contexto 128, AdamW com taxa de aprendizado de 0,0003 e seed 42. O processo completo levou **15 min 58,89 s**, incluindo a verificação inicial. O temporizador de `train.py` registrou 954,43 s. A taxa média das atualizações foi **4.383 tokens/s**, com pico de memória residente de **443,57 MiB**. O treino avaliou a cada 100 passos e salvou snapshots em 2.000, 3.000, 4.000 e 5.000.

| Passo | Loss do minibatch | Loss média de treino | Loss média de validação | Perplexidade de validação |
|---:|---:|---:|---:|---:|
| 1.000 | 2,3341 | 2,3513 | 2,3816 | 10,82 |
| 2.000 | 2,1504 | 2,1278 | 2,1847 | 8,89 |
| 3.000 | 2,0164 | 1,9711 | 2,0741 | 7,96 |
| 4.000 | 1,8752 | 1,8627 | 1,9948 | 7,35 |
| 5.000 | 1,9672 | 1,7792 | 1,9292 | 6,88 |

O [CSV de treino](metrics.csv) traz todas as 40 medições desta fase. O [log](training.log) e o [relatório de tempo](training-time.txt) preservam as medições originais. A [configuração](config.json) inclui os fingerprints dos documentos de treino e validação.

## Avaliação de 64 lotes

Cada conjunto usa 512 janelas de 128 caracteres, sorteadas com reposição e respeitando os limites dos documentos. A avaliação usou lote 8 e seeds 44 no treino, 43 na validação e 45 no corpus externo. A loss de validação dos seis checkpoints históricos coincidiu exatamente com a avaliação complementar da fase 2. A avaliação e a geração levaram **2 min 42,20 s** no total, segundo o [relatório de tempo](evaluation-time.txt).

Cada célula da tabela mostra **loss / perplexidade por caractere**. A perplexidade é a exponencial da loss média. [CSV com precisão integral](evaluation-64.csv)

| Passo | Treino | Validação original | Avaliação externa |
|---:|---:|---:|---:|
| 0 | 4,7826 / 119,42 | 4,7734 / 118,32 | 4,7786 / 118,94 |
| 100 | 2,6395 / 14,01 | 2,6715 / 14,46 | 2,6633 / 14,34 |
| 250 | 2,4985 / 12,16 | 2,5295 / 12,55 | 2,5214 / 12,45 |
| 500 | 2,4383 / 11,45 | 2,4826 / 11,97 | 2,4697 / 11,82 |
| 750 | 2,4027 / 11,05 | 2,4491 / 11,58 | 2,4382 / 11,45 |
| 1.000 | 2,3688 / 10,68 | 2,4212 / 11,26 | 2,4084 / 11,12 |
| 2.000 | 2,1397 / 8,50 | 2,2168 / 9,18 | 2,2105 / 9,12 |
| 3.000 | 1,9891 / 7,31 | 2,0972 / 8,14 | 2,0946 / 8,12 |
| 4.000 | 1,8797 / 6,55 | 2,0127 / 7,48 | 2,0204 / 7,54 |
| 5.000 | 1,7923 / 6,00 | 1,9431 / 6,98 | 1,9578 / 7,08 |

O corpus externo contém o primeiro capítulo de cada uma das três partes de *Triste Fim de Polycarpo Quaresma*, de Lima Barreto. São **79.538 caracteres** e **79.154 janelas candidatas**, com **2 caracteres desconhecidos (0,0025%)** no vocabulário original. A checagem de 42 pares encontrou sobreposição máxima de **0,0227%** das sequências distintas de cinco palavras. Os textos são de domínio público e foram usados apenas para avaliação. [Origem, créditos e SHA-256 dos capítulos](external/sources.json)

A diferença entre as losses de treino e validação de 64 lotes subiu de **0,0524** a **0,1508** entre os passos 1.000 e 5.000. A validação original e a externa continuaram melhorando em todos os checkpoints avaliados, sem reversão sustentada. A melhora por mil passos diminuiu. Na validação original, foi de 0,2044 no primeiro intervalo e 0,0696 no último. Os três capítulos externos pertencem ao mesmo romance, o que limita generalizações sobre outros autores e gêneros.

## Geração e dependência do contexto

As [44 continuações](samples.txt) incluem 20 amostras de 40 caracteres nos cinco checkpoints e 24 amostras de 200 caracteres nos passos 1.000 e 5.000. As 16 amostras do passo 1.000 coincidiram exatamente com os resultados arquivados da fase 2. No passo 5.000 aparecem sequências reconhecíveis como “outro com o vinho” e “da morte amigo”, além de parágrafos e pontuação. Persistem pseudopalavras, construções incoerentes e trechos repetidos entre prompts com a mesma seed.

| Par de prompts | Jensen–Shannon no passo 1.000 (nats) | Passo 5.000 (nats) |
|---|---:|---:|
| O homem / Na cidade | 0,1597 | 0,1781 |
| O homem / A mulher | 0,0955 | 0,1406 |
| A manhã / A noite | 0,5547 | 0,1375 |

A análise usa probabilidades do próximo caractere sem ajuste de temperatura. Em “A manhã”, por exemplo, `o` tinha probabilidade 0,8463 no passo 1.000. No passo 5.000, espaço ficou em primeiro com 0,3463 e `o` caiu para 0,2731. Os pares têm finais diferentes e as divergências medem a distribuição inteira. Um teste com sufixos iguais e diferenças no trecho inicial isolaria melhor o efeito do contexto mais distante. [Dez caracteres mais prováveis por prompt, probabilidades e divergências](analysis.json)

## Reprodução

Com os checkpoints das fases anteriores e o ambiente do projeto, execute a partir da raiz de uma cópia de trabalho destinada à reprodução.

```bash
set -euo pipefail
phase=experiments/002/phase-03
test ! -e checkpoints/experiment-002-phase3
PYTHONPATH=. /usr/bin/time -v -o "$phase/training-time.txt" \
  .venv/bin/python "$phase/run_training.py" 2>&1 | tee "$phase/training.log"
PYTHONPATH=. .venv/bin/python experiments/002/collect_artifacts.py
PYTHONPATH=. /usr/bin/time -v -o "$phase/evaluation-time.txt" \
  .venv/bin/python "$phase/evaluate_run.py"
sha256sum -c "$phase/checkpoints-before.sha256" "$phase/checkpoints-phase3.sha256"
```

O script de treino usa `train.py` com uma verificação de loss finita em cada cálculo. As losses permaneceram finitas e o processo concluiu os 4.000 passos sem erro operacional. A checagem de SHA-256 confirmou os 12 [checkpoints anteriores](checkpoints-before.sha256) e os cinco [arquivos binários da fase 3](checkpoints-phase3.sha256) após a avaliação. Os hashes correspondem aos binários desta execução.

Para reconstruir o corpus externo, baixe a transcrição indicada em [sources.json](external/sources.json) e passe o arquivo a `prepare_external.py`. O script verifica o SHA-256 da fonte antes da extração.
