# Experimento 002

Transformer causal de caracteres treinado em CPU com 444.089 caracteres de prosa brasileira. As quatro fases usam o mesmo corpus, arquitetura e vocabulário. Após o treino inicial, cada fase retoma a anterior a partir do checkpoint indicado.

| Fase | Passos acumulados | Artefatos |
|---|---:|---|
| 1 | 0–100 | [Métricas](phase-01/metrics.csv) · [Amostras](phase-01/samples.txt) · [Configuração](phase-01/config.json) · [Log](phase-01/training.log) |
| 2 | 100–1.000 | [Métricas](phase-02/metrics.csv) · [Avaliação de 64 lotes e amostras estruturadas](phase-02/results.json) · [Amostras](phase-02/samples.txt) · [Configuração](phase-02/config.json) · [Log](phase-02/training.log) |
| 3 | 1.000–5.000 | [Relatório e reprodução](phase-03/README.md) · [Métricas](phase-03/metrics.csv) · [Avaliação de 64 lotes](phase-03/evaluation-64.csv) · [Amostras](phase-03/samples.txt) · [Configuração](phase-03/config.json) · [Log](phase-03/training.log) |
| 4 | 5.000–20.000 | [Relatório e reprodução](phase-04/README.md) · [Métricas](phase-04/metrics.csv) · [Avaliação de 64 lotes](phase-04/evaluation-64.csv) · [Amostras](phase-04/samples.txt) · [Contexto](phase-04/analysis.json) · [Memorização](phase-04/memorization.csv) · [Configuração](phase-04/config.json) · [Log](phase-04/training.log) |

Os CSVs de treino das fases 1 a 3 foram extraídos dos logs reais e usam a loss arredondada para calcular a perplexidade. A fase 4 gravou os CSVs durante a execução, com precisão integral. O passo inicial de uma fase carece de loss de minibatch e tokens/s quando nenhuma atualização ocorreu nessa fase. As configurações das fases 1 e 2 foram recuperadas dos checkpoints e dos comandos de execução.

As amostras da fase 1 foram reproduzidas a partir dos checkpoints 0, 50 e 100 com as opções originais. As amostras do passo 100 coincidem com o arquivo preservado da fase 2. Na fase 2, o JSON e as amostras foram copiados dos resultados originais. Os logs arquivados omitem apenas caminhos absolutos da máquina. Os binários ficam no diretório local ignorado `checkpoints/`.
