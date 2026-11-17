# Síntese da análise pareada de IoU — Testes A e A2

**Manuscrito:** Access-2026-27912
**Data desta síntese:** 2026-11-16
**Escopo:** consolidação das explicações e conclusões discutidas sobre os testes pareados A e A2. Não houve novo treinamento nem nova inferência para este documento.

## Resumo executivo

Foram comparados checkpoints base, treinados somente com dados reais, e checkpoints aumentados, treinados com dados reais e sintéticos, nos mesmos 800 pares de imagens reais de teste. Os resultados mostram diferenças favoráveis aos checkpoints aumentados em algumas medidas, mas devem ser interpretados com cautela:

- No Teste A, a diferença média de IoU foi positiva (`+0.0293`), mas a mediana foi zero.
- No Teste A2 (brightness/contrast), a diferença média foi `+0.0413`, mas a mediana no conjunto completo também foi zero.
- Em ambos, 313 máscaras de referência vazias resultaram em IoU zero para os dois modelos pela convenção usada, mesmo quando uma predição continha falsos positivos.
- No A2, a análise complementar das 487 imagens com sal teve mediana da diferença de IoU `+0.0067`.
- Os testes fornecem evidência estatística de resultados direcionais favoráveis aos checkpoints aumentados sob as hipóteses testadas. Não demonstram que a augmentation causou a diferença: os checkpoints comparados foram treinados com seeds diferentes.

## Dados e definição da comparação

Para cada imagem, calculou-se a diferença pareada:

```text
d = IoU_aumentado - IoU_base
```

Uma diferença positiva favorece o aumentado; uma negativa favorece o base; zero indica empate. Como as comparações são pareadas, cada imagem é avaliada pelos dois checkpoints.

### Convenção para máscaras vazias

A implementação usa:

```text
IoU = intersection / (union + 1e-8)
```

Quando a máscara de referência não contém sal, a interseção é zero. Assim, o IoU fica zero tanto se a predição também estiver vazia quanto se contiver pixels previstos como sal. Essa convenção não distingue nem penaliza falsos positivos nos 313 exemplos sem sal. A análise restrita a imagens com sal responde a uma pergunta diferente e deve ser apresentada como análise complementar, não como substituição silenciosa da avaliação completa.

## Resultados descritivos

| Execução e população | Pares | Diferença média de IoU | Mediana de `d` | Aumentado melhor | Base melhor | Empates |
|---|---:|---:|---:|---:|---:|---:|
| Teste A — conjunto completo | 800 | +0.0293 | 0.0000 | 298 | 152 | 350 |
| Teste A — referência com sal | 487 | — | — | 298 | 152 | 37 |
| Teste A2 — conjunto completo | 800 | +0.0413 | 0.0000 | 304 | 144 | 352 |
| Teste A2 — referência com sal | 487 | +0.0678 | +0.0067 | 304 | 144 | 39 |

No Teste A2, ao excluir as 313 máscaras vazias, a média do IoU por modelo também aumenta, pois os valores zero são retirados: base de `0.3988` para `0.6551`, e aumentado de `0.4400` para `0.7229`. Isso não muda os IoUs dos 487 pares restantes. Entre esses pares, a média da diferença é `+0.0678` e a mediana `+0.0067`.

## Assimetria e interpretação da média e da mediana

No Teste A2, as diferenças no conjunto completo têm skewness `2.7045`, indicando assimetria positiva acentuada: há uma cauda de ganhos grandes. Das 800 imagens, 352 empataram. Entre diferenças não nulas, houve 304 ganhos e 144 perdas. O ganho médio entre os casos favoráveis foi aproximadamente `+0.138`, enquanto a perda média entre os casos desfavoráveis foi aproximadamente `-0.062`.

Assim, ganhos expressivos podem elevar a média (`+0.0413`) enquanto a mediana do conjunto completo permanece zero. A mediana do subconjunto com sal é `+0.0067`, um ganho central observado positivo, porém pequeno.

O Shapiro-Wilk reportado para o A2 tem `p=8.484e-39`; ele testa normalidade, não simetria diretamente. A assimetria é indicada pelo skewness e pela distribuição observada. Portanto, o p-valor do Wilcoxon não deve ser descrito, sem qualificação, como prova de que a mediana seja positiva.

## Wilcoxon e teste exato dos sinais

### Wilcoxon signed-rank

No A2, o Wilcoxon unilateral reportado usa a alternativa de que os IoUs do aumentado tendem a ser maiores. As 352 diferenças zero são excluídas do ranqueamento, restando 448 diferenças não nulas. O teste ordena as diferenças pelo valor absoluto e compara as somas dos postos positivos e negativos:

| Estatística do A2 | Valor |
|---|---:|
| Soma dos postos positivos (`W+`) | 73011 |
| Soma dos postos negativos (`W-`) | 27565 |
| Estatística reportada (`W`) | 73011 |
| `p` unilateral por aproximação | `5.7926e-17` |
| Correlação bisserial por postos | 0.4519 |

Esse p-valor é evidência contra a hipótese nula do Wilcoxon, na direção especificada, sob as premissas do teste. Ele não é a probabilidade de a hipótese nula ser verdadeira, não mede a importância prática do ganho e não estabelece causalidade. O teste usa a magnitude relativa das diferenças para formar postos; seu p-valor não expressa o ganho em unidades de IoU. A interpretação clássica como teste de deslocamento/mediana depende de condições sobre a distribuição das diferenças, em particular simetria.

Como o teste descarta diferenças zero, remover as 313 máscaras vazias do A2 não altera o ranqueamento Wilcoxon: essas imagens eram empates, e os 448 pares não nulos permanecem os mesmos. No subconjunto com sal, há 39 empates, também descartados.

### Teste exato dos sinais

O teste dos sinais utiliza somente a direção das diferenças, não sua magnitude. No A2, entre as 448 imagens sem empate, o aumentado venceu 304 e perdeu 144. Sob a hipótese nula de chances iguais de vitória e derrota, o p-valor unilateral exato para observar 304 ou mais vitórias é aproximadamente `1.56e-14`.

Esse resultado sustenta a conclusão de que, neste conjunto pareado, o checkpoint aumentado venceu mais frequentemente do que o esperado sob a hipótese nula do teste dos sinais. Não informa o tamanho dos ganhos e não testa diretamente uma diferença média ou mediana. Para uma hipótese bilateral de diferença em qualquer direção, o p-valor seria aproximadamente `3.12e-14`.

O teste dos sinais e o Wilcoxon respondem a perguntas relacionadas, mas distintas: o primeiro conta vitórias e derrotas; o segundo também usa a ordenação das magnitudes absolutas.

## H1 e alcance das conclusões

No código do experimento, a alternativa unilateral do Wilcoxon foi definida como `IoU aumentado > IoU base`. Como o p-valor é muito menor que `alpha=0.05`, há evidência estatística favorável à alternativa direcional formulada pelo teste, condicionada às suas premissas.

Isso não equivale a confirmar qualquer formulação possível de H1. Em particular:

- Se H1 for especificamente `mediana(d) > 0` para as 800 imagens, a mediana observada do A2 é zero; o resultado não demonstra essa afirmação.
- Se a população-alvo for apenas as imagens com sal, a mediana observada de `d` no A2 é `+0.0067`; isso descreve o subconjunto, mas o p-valor do Wilcoxon é o mesmo do conjunto completo, pois os empates foram excluídos. Essa mediana positiva não deve ser confundida com uma conclusão inferencial baseada num teste separado da mediana.
- A significância estatística não implica que o ganho seja grande ou relevante na prática. Para comunicar magnitude, reportam-se diferenças em IoU, estatísticas descritivas e, idealmente, intervalos de confiança.

## Causalidade e desenho experimental

O Wilcoxon pareado avalia diferenças observadas entre dois checkpoints nas mesmas imagens; não identifica a causa dessas diferenças. No A2, o checkpoint base é da seed 42 e o aumentado da seed 123. Assim, condição de treinamento e seed não foram controladas como fatores separados. A diferença pode refletir a augmentation, a variação entre seeds ou ambas.

Para evidência causal mais forte sobre a augmentation, seriam necessários múltiplos treinamentos por condição, com seeds pareadas e as demais condições mantidas iguais, alterando apenas a augmentation. As diferenças entre execuções correspondentes poderiam então ser avaliadas, além das diferenças por imagem. Mesmo esse desenho produz evidência sob um protocolo específico, não uma prova absoluta válida para qualquer modelo ou conjunto de dados.

## Conclusão

Os resultados do A e A2 mostram tendência estatística favorável aos checkpoints aumentados nas avaliações pareadas. No A2, essa tendência aparece tanto nas somas dos postos do Wilcoxon quanto na maior contagem de vitórias no teste dos sinais. Ainda assim, a mediana de `d` no conjunto completo é zero, há assimetria acentuada, a métrica de IoU não discrimina os modelos nas máscaras vazias e os checkpoints A2 usam seeds diferentes. A conclusão apropriada é relatar a evidência direcional observada com essas limitações, sem afirmar melhora mediana no conjunto completo nem atribuição causal à augmentation.

## Arquivos de referência

- [Relatório Teste A](./relatorio-teste-pareado-teste-a.md)
- [Relatório Teste A2 — brightness/contrast](../relatorio-teste-pareado-teste-a2-123random_brightness_contrast.md)
- [Métricas por imagem do A2](../../resultsHPC/paired_iou_test_a2-123random_brightness_contrast/per_image_metrics.csv)
- [Resumo JSON do A2](../../resultsHPC/paired_iou_test_a2-123random_brightness_contrast/summary.json)
