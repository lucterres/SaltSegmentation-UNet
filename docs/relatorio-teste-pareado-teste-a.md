# Relatório do Teste A - Comparação Pareada por Imagem

**Manuscrito:** Access-2026-27912  
**Data:** 2026-11-06  
**Experimento:** Teste A - Wilcoxon signed-rank pareado  
**Execução:** `resultsHPC/paired_iou_test_a/`

## Resumo executivo

A análise avaliou os mesmos 800 pares reais de teste com dois checkpoints já treinados: um
modelo base, treinado somente com dados reais, e um modelo aumentado, treinado com dados
reais e sintéticos. A diferença média de IoU por imagem foi positiva (`+0.0293`), mas a
mediana das diferenças foi `0.0000`; somente 298 das 800 imagens favoreceram o modelo
 aumentado.

O Wilcoxon unilateral produziu `W=70464`, `p=4.448e-13` e correlação bisserial por postos
`r=0.3888`. Entretanto, a distribuição das diferenças é fortemente assimétrica (Shapiro-Wilk
`p=8.833e-39`, skewness `2.796`) e contém 350 pares sem diferença. Por isso, o resultado
significativo do teste de postos não demonstra que a mediana do ganho seja positiva. A
hipótese formulada como `Mediana(d) > 0` não é sustentada por este resultado: a mediana
observada é zero.

Há ainda uma limitação importante na métrica usada para reproduzir o código de treinamento:
313 máscaras de referência vazias tiveram diferença de IoU igual a zero, independentemente
de a predição conter falsos positivos. Esses pares não distinguem os modelos pela IoU. O
resultado deve ser reportado com essa ressalva e não como evidência conclusiva de melhora
geral do modelo aumentado.

## Protocolo

| Item | Configuração |
|---|---|
| Modelo base | `resultsHPC/scenario_A_seed42_nreal1200_train/best_model.pth` |
| Modelo aumentado | `resultsHPC/scenario_B_seed42_nreal1200_train_pairs1600_seismic_ns1200/best_model.pth` |
| Test set canônico | `D:\dataset\tgs-salt\subset_split\test` (Windows); `/var/tmp/cym7/datasets/subset_split/test` (HPC) |
| Identidade do test set | 800 IDs, iguais aos de `test_set.csv`; IDs verificados contra a pasta canônica local |
| Amostras de teste | 800 imagens reais, pareadas por ID |
| Threshold de binarização | 0.5 |
| Teste | Wilcoxon signed-rank, unilateral, alternativa `IoU aumentada > IoU base` |
| Nível de significância | `alpha = 0.05` |
| Zeros e empates | Diferenças zero excluídas do ranqueamento (`zero_method=wilcox`); postos médios para empates |
| Dispositivo | CPU |
| Batch size | 16 |
| Pré-processamento | Grayscale, resize para 128x128, máscaras com nearest neighbor |
| Convenção de IoU | `intersection / (union + 1e-8)`; união vazia resulta em IoU 0 |

Não houve novo treinamento. A inferência usou os `state_dict` dos checkpoints indicados e o
mesmo conjunto de imagens e máscaras para ambos os modelos. O manifesto
`Salt-Segmentation-UNet/output/test_paths.txt` não foi usado: ele representa outro split.

## Resultados descritivos

| Métrica | Base: média | Base: mediana | Base: IQR | Aumentado: média | Aumentado: mediana | Aumentado: IQR |
|---|---:|---:|---:|---:|---:|---:|
| IoU | 0.3988 | 0.1465 | 0.8871 | 0.4280 | 0.3204 | 0.9125 |
| Dice | 0.4354 | 0.2554 | 0.9402 | 0.4641 | 0.4853 | 0.9542 |

| Diferença pareada (aumentado - base) | Valor |
|---|---:|
| IoU média | +0.0293 |
| IoU mediana | 0.0000 |
| Dice média | +0.0288 |
| Imagens com IoU aumentada maior | 298/800 (37.25%) |
| Imagens com IoU aumentada menor | 152/800 (19.00%) |
| Imagens com IoUs iguais | 350/800 (43.75%) |

Das 487 imagens com máscara de referência não vazia, 298 favoreceram o modelo aumentado,
152 favoreceram a base e 37 tiveram IoUs iguais. As outras 313 imagens tinham máscara vazia;
todas produziram diferença de IoU zero com a convenção descrita abaixo.

## Teste inferencial

| Estatística | Resultado |
|---|---:|
| Pares não nulos usados no ranqueamento | 450 |
| Soma dos postos positivos (`W+`) | 70464 |
| Soma dos postos negativos (`W-`) | 31011 |
| Estatística reportada (`W`) | 70464 |
| `p` unilateral | 4.448074e-13 |
| `alpha` | 0.05 |
| Correlação bisserial por postos | 0.3888 |
| Shapiro-Wilk (`W`) | 0.6159 |
| Shapiro-Wilk (`p`) | 8.833175e-39 |
| Skewness de Fisher-Pearson | 2.7960 |

O valor de `p` é inferior a `alpha` para o teste signed-rank unilateral. A distribuição das
diferenças, porém, não é aproximadamente simétrica: o teste de Shapiro-Wilk rejeita
normalidade e a assimetria é acentuada. A rejeição de normalidade, por si só, não invalida o
Wilcoxon; a falta de simetria impede interpretar diretamente este resultado como teste da
mediana. Além disso, a mediana observada das diferenças pareadas é exatamente zero.

Portanto, o relatório não conclui que a mediana do IoU melhorou. O resultado é compatível
com uma tendência positiva nas somas dos postos dos 450 pares não nulos, mas não confirma a
hipótese de ganho mediano definida para o Teste A.

## Limitação da IoU para máscaras vazias

A implementação usou a convenção do `train.py` para manter comparabilidade:

```text
IoU = intersection / (union + 1e-8)
```

Se a máscara de referência está vazia, `intersection` é zero. Quando a predição também está
vazia, a união é zero e o epsilon produz IoU zero; quando a predição contém pixels positivos,
a interseção continua zero e a IoU também é zero. Assim, a métrica atribui IoU zero a ambas
as predições e não penaliza falsos positivos nesses 313 exemplos. Isso explica 313 dos 350
pares empatados e reduz a capacidade deste teste de discriminar os modelos no conjunto
completo.

A função de exemplo no documento do Teste A usa outra convenção (IoU 1 para vazio-vazio).
Este relatório preserva o cálculo que foi efetivamente executado, mas recomenda uma análise
de sensibilidade com uma regra explícita para máscaras vazias antes de sustentar uma
conclusão global sobre superioridade. Uma reanálise com métrica diferente deve ser reportada
separadamente, sem substituir silenciosamente estes resultados.

## Rastreabilidade

- Ambiente: Python 3.10.20, PyTorch 2.5.1, NumPy 2.0.1, SciPy 1.15.3 e OpenCV 5.0.0.
- SHA-256 dos IDs ordenados: `835d74e6244dd874202a843557259213d66b0880bd9b67d005ae0b7755520bc2`.
- [Métricas por imagem](../resultsHPC/paired_iou_test_a/per_image_metrics.csv)
- [Resumo JSON](../resultsHPC/paired_iou_test_a/summary.json)
- [Gráfico pareado](../resultsHPC/paired_iou_test_a/paired_iou.png)
- [Distribuição das diferenças](../resultsHPC/paired_iou_test_a/iou_difference.png)
