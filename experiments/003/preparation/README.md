# Experimento 003 · Etapa 0

Preparação de corpus de caracteres em português. Os textos brutos, o cache e os documentos processados ficam em `local/`, ignorado pelo Git. Os manifestos apontam para documentos únicos por ID e SHA-256. Nenhum checkpoint ou código de treinamento é modificado pela preparação.

## Reprodução

Execute da raiz do repositório, com o ambiente Python existente. O artefato `vocabulary.json` preserva em ordem as 101 entradas do checkpoint verificado do passo 19.500 do Experimento 002. O pipeline é implementado com a biblioteca padrão e trata o vocabulário somente como instrumento de diagnóstico.

```bash
phase=experiments/003/preparation
sha256sum -c "$phase/vocabulary.sha256"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s "$phase" -v
/usr/bin/time -v -o "$phase/runtime.txt" .venv/bin/python "$phase/prepare.py"
```

A execução reaproveita respostas com SHA-256 verificado. Para reproduzir os manifestos de uma aquisição específica, preserve também o cache local. Páginas públicas podem mudar de revisão entre aquisições novas.

Para reconstruir a seleção com os documentos já preparados, execute `.venv/bin/python experiments/003/preparation/prepare.py --manifests-only`. Essa opção verifica os hashes dos textos locais e recompõe os manifestos sem fazer requisições HTTP.

## Métodos

- **Aquisição.** Um pedido por vez, intervalo mínimo de um segundo por domínio, regras de robôs verificadas, timeout de 15 s e até três tentativas para erros transitórios. Respostas e hashes são gravados atomicamente no cache. Redirecionamentos HTTP ficam registrados como falha. A aquisição exclui conversas privadas e rotas autenticadas.
- **Licenças.** A Wikipédia e o Wikilivros distribuem o texto de colaboradores sob CC BY-SA 4.0, com atribuição e preservação dos termos aplicáveis. Obras históricas passam por seleção a partir de autoria e edição de domínio público. Para material Gutenberg, a referência de domínio público no Brasil é o art. 41 da Lei 9.610/1998. O envelope de distribuição fica no cache. Os documentos processados aguardam revisão humana antes de qualquer redistribuição pública. [Termos da Wikimedia](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · [Lei 9.610/1998](https://www.planalto.gov.br/ccivil_03/leis/l9610.htm#art41) · [Política Gutenberg](https://www.gutenberg.org/policy/license.html)
- **Extração.** UTF-8, Unicode NFC e quebras LF. O extrator HTML retém parágrafos e títulos da prosa, omite navegação, tabelas, legendas e referências finais. A extração de livros remove o envelope Gutenberg quando identificado. A etapa preserva grafia, acentos e pontuação. Os dez documentos de treino anteriores são copiados byte a byte.
- **Qualidade.** O filtro usa uma heurística de palavras funcionais do português, proporção de letras, sequências repetidas, presença de marcação residual, caracteres inválidos e tamanho. É uma triagem heurística, com diferenças de cobertura entre formatos. Caracteres desconhecidos continuam nos documentos e são contabilizados por conjunto e por categoria.
- **Deduplicação e vazamento.** SHA-256 para igualdade exata. Um índice invertido de fingerprints amostrados de sequências de cinco palavras e de parágrafos longos encontra candidatos por sobreposição sem comparar todos os pares. Os seis benchmarks entram primeiro no índice, com limiar de sobreposição mais restritivo. O método detecta republicações com trechos preservados e pode perder edições muito reescritas. O particionamento conserva os grupos de obra conhecidos, inclusive capítulos do Wikilivros.
- **Seleção.** Validação e holdout são escolhidos em grupos disjuntos antes dos treinos, com prioridade para diversidade temática e de autoria e desempate por hash. Uma obra pode ocupar até 40% do alvo de cada conjunto reservado. Os manifestos de treino reaproveitam as seleções anteriores. O holdout fica reservado à comparação final. As metas são calculadas em caracteres Unicode e excluem o holdout e os benchmarks históricos.

## Artefatos

- [Configuração](config.json), [fontes](sources.jsonl), [atribuições](ATTRIBUTIONS.md) e [documentos](documents.jsonl).
- [5 milhões](corpus-5m.json), [7 milhões](corpus-7m.json), [10 milhões](corpus-10m.json), [validação comum](validation.json) e [holdout](holdout.json).
- [Estatísticas por conjunto e distribuição](statistics.json), [CSV](statistics.csv), [rejeições](rejected-documents.jsonl), [aquisição](acquisition.log) e [validação e hashes](validation-report.json).
- [SHA-256 dos cinco manifestos](manifests.sha256) e [SHA-256 do vocabulário](vocabulary.sha256).

## Resultado da execução

O pipeline foi executado na `main` sobre o commit `02817c6`. Passaram cinco testes existentes e oito novos. A reconstrução sem rede produziu os mesmos cinco hashes. O treino de 5 milhões é subconjunto estrito do de 7 milhões, que é subconjunto estrito do de 10 milhões. Os três manifestos referenciam o mesmo SHA-256 da validação. Treino, validação comum e holdout são disjuntos por ID e grupo de obra.

| Meta | Documentos de treino | Caracteres de treino | Validação comum | Total | Fora do vocabulário no total |
|---:|---:|---:|---:|---:|---:|
| 5 milhões | 366 | 4.515.664 | 501.280 | 5.016.944 | 0,1618% |
| 7 milhões | 416 | 6.526.605 | 501.280 | 7.027.885 | 0,1534% |
| 10 milhões | 455 | 9.564.501 | 501.280 | 10.065.781 | 0,1636% |

A validação comum tem 223 documentos e 0,1983% de caracteres desconhecidos. O holdout tem 174 documentos, 500.836 caracteres e 0,1711% de desconhecidos. O holdout contém prosa enciclopédica, histórica e educacional. Nenhuma obra literária coube na seleção do holdout com as restrições por obra e por categoria adotadas.

| Categoria | Meta | 5 milhões | 7 milhões | 10 milhões |
|---|---:|---:|---:|---:|
| Literatura | 30% | 37,1% | 32,7% | 27,5% |
| Enciclopédico | 25% | 29,9% | 31,5% | 33,5% |
| Científico e educacional | 20% | 3,0% | 2,1% | 1,5% |
| Histórico e institucional | 15% | 30,0% | 33,6% | 37,5% |
| Diálogos e cotidiano | 10% | 0,0% | 0,0% | 0,0% |

Foram adquiridos **960 documentos novos**. Desses, 848 foram aceitos e 112 descartados (104 curtos, 5 duplicatas semelhantes e 3 por qualidade). Outros 19 candidatos retornaram HTTP 404 antes da aquisição. Com os dez documentos históricos de treino, o acervo aceito tem **858 documentos** e **11.106.321 caracteres processados**. As respostas brutas dos documentos novos somaram **169.502.557 caracteres**, incluindo HTML. Nenhum dos seis benchmarks históricos foi incluído. A revisão heurística de tabelas sinalizou um artigo de prosa sobre Dom Pedro II na Guerra do Paraguai por espaços consecutivos. A inspeção confirmou que o texto é prosa.

No conjunto de 10 milhões, a Wikipédia em português representa 71,0% dos caracteres e inclui tanto a categoria enciclopédica quanto a temática histórica. O Wikilivros representa 1,5%, o Project Gutenberg via GITenberg 23,7% e o treino legado 3,8%. Machado de Assis é o autor individual mais representado, com 879.088 caracteres. As distribuições integrais de autoria, categoria, fonte e licença constam em `statistics.json`. A atribuição coletiva das páginas Wikimedia está vinculada à página e à revisão registradas. A preparação não encontrou um acervo de diálogos naturais com procedência e licença verificadas no trabalho realizado.

Todos os corpora excederam a referência de 0,1% de caracteres desconhecidos. No corpus de 10 milhões, os mais frequentes são `à` (4.967), travessão (1.847), espaço inseparável (1.290), `%` (1.036) e `W` (920). Antes de autorizar treino, é possível avaliar o impacto por categoria e revisar separadamente a convenção de espaços e a representação de caracteres em uma futura política de vocabulário. Nenhuma substituição foi aplicada nesta etapa.

A execução corrigida levou 15 min 47,32 s e usou pico de 66,1 MiB de memória residente. Os downloads somaram 165,3 MiB no contador persistente, incluindo respostas aproveitadas da tentativa inicial. O armazenamento local ficou em 176,4 MiB. A primeira tentativa foi interrompida para corrigir o extrator HTML. A retomada reutilizou o cache e reconstruiu os documentos. A seleção final limitou cada conjunto reservado a 40% do alvo por obra e a 20% da oferta por categoria, para preservar material escasso no treino. A etapa encerrou sem iniciar treinamento.
