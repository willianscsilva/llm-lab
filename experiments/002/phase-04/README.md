# Experimento 002, fase 4

## Resultado

A execução retomou o snapshot 5.000 na branch `main`, no commit `7391639`, e fez 15.000 atualizações em CPU até o limite de **20.000**. Passaram **11 testes** (5 existentes e 6 da fase). Pesos, otimizador e ambos os geradores aleatórios coincidiram exatamente no teste de duas atualizações. As avaliações de quatro e 64 lotes reproduziram os valores históricos do passo 5.000, incluindo a avaliação externa. Também conferiram os 835.685 parâmetros, as 101 entradas do vocabulário, os 13 fingerprints e os SHA-256 dos três capítulos externos.

O melhor checkpoint foi o **19.500**, com loss de validação **1,678863** em 64 lotes. Esse passo renovou a referência de melhora significativa. O processo terminou pelo **limite máximo**, 500 passos depois. O maior contador de saturação observado foi de 2.000 passos, e o contador de overfitting ficou em zero. Os seis snapshots permanentes foram salvos. `latest.pt` e `final.pt` correspondem ao passo 20.000. [Decisões e checkpoints](stopping.json)

O processo de treino levou **1 h 19 min 30 s**, incluindo a preparação, com média de **3.643 tokens/s** durante as atualizações e pico de memória residente de **411,43 MiB**. As 61 medições de quatro lotes incluem a loss do minibatch, as médias, perplexidades e velocidade. A loss de validação de quatro lotes passou de **1,9292 para 1,6629**. [Métricas de quatro lotes](metrics.csv) · [Log](training.log) · [Tempo e memória](training-time.txt)

### Avaliação de 64 lotes

As células dos conjuntos mostram **loss / perplexidade por caractere**. O gap é a diferença entre as losses de validação original e treino.

| Passo | Treino | Validação original | Lima Barreto | Gap |
|---:|---:|---:|---:|---:|
| 5.000 | 1,7923 / 6,00 | 1,9431 / 6,98 | 1,9578 / 7,08 | 0,1508 |
| 7.500 | 1,6422 / 5,17 | 1,8363 / 6,27 | 1,8784 / 6,54 | 0,1941 |
| 10.000 | 1,5448 / 4,69 | 1,7763 / 5,91 | 1,8266 / 6,21 | 0,2315 |
| 12.500 | 1,4767 / 4,38 | 1,7366 / 5,68 | 1,7957 / 6,02 | 0,2599 |
| 15.000 | 1,4294 / 4,18 | 1,7058 / 5,51 | 1,7723 / 5,88 | 0,2764 |
| 17.500 | 1,3847 / 3,99 | 1,6875 / 5,41 | 1,7592 / 5,81 | 0,3028 |
| 19.500 | 1,3612 / 3,90 | 1,6789 / 5,36 | 1,7492 / 5,75 | 0,3177 |
| 20.000 | 1,3560 / 3,88 | 1,6820 / 5,38 | 1,7553 / 5,78 | 0,3260 |

A melhora de validação diminuiu de 0,1068 nos primeiros 2.500 passos para 0,0055 nos últimos 2.500. O gap cresceu de 0,1508 para 0,3260. Do melhor checkpoint ao final, o treino melhorou e as duas losses fora do treino subiram ligeiramente. Esses dados indicam ganhos decrescentes e uma diferença crescente de generalização. O limiar definido para overfitting sustentado nunca foi atingido. O corpus externo representa três capítulos do mesmo romance. [Todas as 31 avaliações de 64 lotes](evaluation-64.csv)

### Gerações e correspondências literais

Foram registradas **68 continuações** em oito checkpoints distintos, 32 de 40 caracteres e 36 de 200 caracteres. As 16 amostras do passo 5.000 reproduziram exatamente as da fase 3. No passo final aparecem sequências como “a mão da porta” e “o guerreiro branco do barbeiro”, além de parágrafos e pontuação. Ainda ocorrem pseudopalavras, trechos repetidos entre prompts e mudanças incoerentes de assunto. A análise pós-treino levou **2 min 23,01 s**. [Todas as continuações](samples.txt) · [Tempo da análise](analysis-time.txt)

A tabela resume o comprimento da maior correspondência literal de cada uma das 12 amostras longas por checkpoint. As células mostram **média / mediana / máximo**, em caracteres.

| Passo | Treino | Validação | Lima Barreto |
|---:|---:|---:|---:|
| 5.000 | 11,08 / 11 / 13 | 9,58 / 10 / 11 | 9,58 / 9,5 / 11 |
| 19.500 | 15,08 / 14 / 21 | 11,75 / 12 / 13 | 11,42 / 11 / 13 |
| 20.000 | 16,08 / 15 / 21 | 12,50 / 12,5 / 16 | 12,42 / 12 / 15 |

A cobertura por janelas literais de 32 caracteres ficou em zero nos três conjuntos e nos três checkpoints. As correspondências curtas cresceram em todos os conjuntos. O corpus de treino é maior e oferece mais oportunidades de coincidência. Este teste cobre apenas cópia literal nas amostras observadas. [108 comparações individuais com documento e trecho](memorization.csv)

### Dependência do contexto

Para os três pares históricos, as divergências de Jensen–Shannon do passo 5.000 foram reproduzidas exatamente. Do passo 5.000 ao 20.000, elas passaram de 0,1781 a 0,0656 em “O homem” / “Na cidade”, de 0,1406 a 0,0316 em “O homem” / “A mulher” e de 0,1375 a 0,0207 em “A manhã” / “A noite”.

| Tamanho-alvo do sufixo | Divergência média no passo 5.000 (nats) | Passo 19.500 | Passo 20.000 |
|---:|---:|---:|---:|
| 8 | 0,000512 | 0,000777 | 0,001299 |
| 16 | 0,000131 | 0,000175 | 0,000166 |
| 32 | 0,000080 | 0,000038 | 0,000065 |
| 64 | 0,000013 | 0,000014 | 0,000014 |
| 96 | 0,000014 | 0,000020 | 0,000014 |

No passo final, os quatro pares de sufixo idêntico tiveram divergências entre 0,000069 e 0,003287 nats. Para os prefixos testados, as diferenças nas distribuições foram menores com sufixos longos. Cada distância usa um sufixo natural diferente, então conteúdo e distância variam juntos. A análise mede a distribuição do próximo caractere em um conjunto pequeno de prompts. [Pares individuais, médias e dez caracteres mais prováveis](analysis.json)

Os hashes dos **17 binários históricos** e dos **9 arquivos de checkpoint da fase 4** conferiram após a análise. O snapshot do passo 20.000, `latest.pt` e `final.pt` compartilham o mesmo estado salvo. Os arquivos binários ficam no diretório local ignorado pelo Git. [Manifesto da fase 4](checkpoints-phase4.sha256) · [Hashes de entrada e configuração](config.json)

## Protocolo e reprodução

A fase retoma o snapshot original do passo 5.000 da fase 3. O executável usa as mesmas funções de amostragem, modelo e loss do treino original. O teste de equivalência compara duas atualizações com o `train.py` original e inclui uma avaliação de 64 lotes entre as atualizações no fluxo novo.

O checkpoint de partida e os demais binários históricos disponíveis são verificados pelos manifestos da fase 3. A preparação exige que as losses de quatro e 64 lotes do passo 5.000 sejam reproduzidas exatamente. Os arquivos externos são verificados com os SHA-256 registrados na fase 3.

O treinamento avalia quatro lotes a cada 250 passos e 64 lotes a cada 500 passos, com as seeds 44, 43 e 45 para treino, validação e corpus externo. Nas avaliações de 64 lotes, os primeiros quatro lotes alimentam a série de quatro lotes. A mesma seed recria as mesmas janelas de cada conjunto em todas as avaliações.

O critério de encerramento usa a validação original de 64 lotes. Quatro avaliações consecutivas pelo menos 0,02 acima do melhor valor acionam o critério de overfitting. A condição de saturação exige 3.000 passos sem redução de pelo menos 0,005 desde a última referência de melhora significativa. O passo 5.000 é o primeiro candidato a melhor checkpoint. `stopping.json` registra o estado dos critérios e o passo e a loss de cada checkpoint.

Em uma cópia de trabalho com os checkpoints históricos disponíveis, execute a partir da raiz do repositório:

```bash
set -euo pipefail
phase=experiments/002/phase-04
test ! -e checkpoints/experiment-002-phase4
export PYTHONPATH=.
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s "$phase" -v
/usr/bin/time -v -o "$phase/training-time.txt" \
  .venv/bin/python "$phase/run_training.py" 2>&1 | sed -u "s|$PWD/||g" | tee "$phase/training.log"
.venv/bin/python "$phase/analyze.py"
```

As comparações de memorização usam as continuações de 200 caracteres, com correspondência literal, sensível a maiúsculas e dentro de cada documento. A cobertura de 32 caracteres conta a união das posições cobertas por janelas encontradas no conjunto de referência. O maior trecho é escolhido entre os documentos de cada conjunto. Em caso de empate, vale o primeiro documento na ordem registrada.

Os prompts definitivos e os sufixos constam em `config.json`. O par do soldado usa “Um pintor” para igualar o comprimento e manter “ pegou a”. A análise de distância usa três pares de prefixos por sufixo. Os pares de comprimento igual controlam o sufixo e o tamanho da entrada. A divergência de Jensen–Shannon usa as distribuições do próximo caractere sem ajuste de temperatura.
