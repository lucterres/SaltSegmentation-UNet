# Relatório do Teste A2 — Brightness/Contrast

**Manuscrito:** Access-2026-27912
**Data:** 2026-11-07
**Experimento:** Comparação pareada por imagem — modelo aumentado com random brightness/contrast
**Execução:** `resultsHPC/paired_iou_test_a2-123random_brightness_contrast/`

## Resumo executivo

A análise comparou os mesmos 800 pares reais de teste usando um checkpoint base
treinado somente com dados reais e um checkpoint aumentado treinado com dados reais
e sintéticos, com a transformação random brightness/contrast. Na avaliação completa,
a diferença média de IoU por imagem (aumentado − base) foi `+0.0413`; a mediana foi
`0.0000`. O modelo aumentado teve IoU maior em 304 imagens, menor em 144 e igual em
352.

Das 352 diferenças iguais, 313 vieram de imagens cuja máscara de referência não
continha sal. Com a convenção de IoU usada, essas imagens recebem IoU zero para ambos
os modelos, mesmo se a predição produzir falsos positivos. Excluindo-as para uma
análise complementar das 487 imagens com sal, a diferença média de IoU foi `+0.0678`
e a mediana foi `+0.0067`; 304 imagens favoreceram o modelo aumentado, 144 a base e
39 empataram.

O Wilcoxon unilateral reportado foi significativo (`W=73011`, `p=5.793e-17`), mas
a distribuição das diferenças é fortemente assimétrica (Shapiro-Wilk
`p=8.484e-39`, skewness `2.7045`). O resultado do teste de postos não deve ser
interpretado isoladamente como prova de melhora da mediana. A mediana observada é
zero no conjunto completo e positiva, embora pequena, no subconjunto com sal.

Esta comparação usa um checkpoint base de seed 42 e um checkpoint aumentado de
seed 123. Portanto, a diferença observada não isola o efeito da transformação:
também pode refletir a variação entre seeds.

## Protocolo

| Item | Configuração |
|---|---|
| Modelo base | `resultsHPC/scenario_A_seed42_nreal1200_train/best_model.pth` |
| Modelo aumentado | `resultsHPC/scenario_B_seed123_nreal1200_train_random_brightness_contrast1600clean_ns1200/best_model.pth` |
| Seeds comparadas | Base: 42; aumentado: 123 |
| Test set | 800 pares reais; IDs verificados pelo SHA-256 abaixo |
| Threshold de binarização | 0.5 |
| Teste | Wilcoxon signed-rank pareado, unilateral (`IoU aumentada > IoU base`) |
| Nível de significância | `alpha = 0.05` |
| Zeros e empates | Diferenças zero excluídas do ranqueamento (`zero_method=wilcox`); postos médios para empates |
| Dispositivo | CPU |
| Batch size | 16 |
| Pré-processamento | Grayscale, resize para 128x128, máscaras com nearest neighbor |
| Convenção de IoU | `intersection / (union + 1e-8)`; união vazia resulta em IoU 0 |

Não houve novo treinamento. Foram avaliados os checkpoints já existentes, nos mesmos
800 pares. A análise filtrada exclui somente imagens com `ground_truth_empty=True`;
ela é complementar e não substitui a avaliação do conjunto completo.

## Resultados no conjunto completo

| Métrica | Base: média | Base: mediana | Base: IQR | Aumentado: média | Aumentado: mediana | Aumentado: IQR |
|---|---:|---:|---:|---:|---:|---:|
| IoU | 0.3988 | 0.1465 | 0.8871 | 0.4400 | 0.3640 | 0.9217 |
| Dice | 0.4354 | 0.2554 | 0.9402 | 0.4740 | 0.5336 | 0.9593 |

| Diferença pareada (aumentado − base) | Valor |
|---|---:|
| IoU média | +0.0413 |
| IoU mediana | 0.0000 |
| Dice média | +0.0386 |
| Imagens com IoU aumentada maior | 304/800 (38.00%) |
| Imagens com IoU aumentada menor | 144/800 (18.00%) |
| Imagens com IoUs iguais | 352/800 (44.00%) |

## Análise complementar: máscaras com sal

Ao excluir as 313 imagens com máscara de referência vazia, restam 487 pares:

| Métrica | Base: média | Base: mediana | Base: IQR | Aumentado: média | Aumentado: mediana | Aumentado: IQR |
|---|---:|---:|---:|---:|---:|---:|
| IoU | 0.6551 | 0.8303 | 0.5519 | 0.7229 | 0.8904 | 0.3637 |
| Dice | 0.7152 | 0.9073 | 0.4016 | 0.7786 | 0.9420 | 0.2310 |

| Diferença pareada (aumentado − base) | Valor |
|---|---:|
| IoU média | +0.0678 |
| IoU mediana | +0.0067 |
| Dice média | +0.0634 |
| Imagens com IoU aumentada maior | 304/487 (62.42%) |
| Imagens com IoU aumentada menor | 144/487 (29.57%) |
| Imagens com IoUs iguais | 39/487 (8.01%) |

A remoção das máscaras vazias eleva as médias e altera a mediana da diferença de IoU
de zero para `+0.0067`. Isso ocorre porque as 313 imagens removidas eram todas
empates de IoU; os valores dos outros 487 pares não mudam. Entre as imagens com sal,
o modelo aumentado teve IoU maior em aproximadamente 62% dos casos, mas ainda houve
144 imagens em que a base foi melhor.

## Teste inferencial e interpretação

| Estatística | Resultado |
|---|---:|
| Pares no conjunto completo | 800 |
| Pares com diferença não nula | 448 |
| Soma dos postos positivos (`W+`) | 73011 |
| Soma dos postos negativos (`W-`) | 27565 |
| Estatística reportada (`W`) | 73011 |
| `p` unilateral (aproximação) | 5.792567e-17 |
| `alpha` | 0.05 |
| Correlação bisserial por postos | 0.4519 |
| Shapiro-Wilk (`W`) | 0.6154 |
| Shapiro-Wilk (`p`) | 8.484015e-39 |
| Skewness de Fisher-Pearson | 2.7045 |

O resultado do Wilcoxon permanece igual após a remoção das máscaras vazias: o teste
descarta diferenças zero, e as 313 imagens removidas tinham diferença zero. No
subconjunto com sal há mais 39 empates; também são excluídos do ranqueamento, restando
os mesmos 448 pares não nulos.

O teste fornece evidência de uma tendência positiva nas somas dos postos dos pares
não empatados. Contudo, a assimetria acentuada limita a interpretação do Wilcoxon
como teste da mediana. A mediana observada é `0.0000` no conjunto completo e
`+0.0067` no subconjunto com sal. Assim, deve-se reportar essa diferença entre
populações e evitar afirmar que o p-valor, por si só, demonstra uma melhora geral
ou mediana positiva no conjunto completo.

## Limitações

Com `IoU = intersection / (union + 1e-8)`, toda imagem com referência vazia tem
interseção zero. O IoU resulta em zero tanto quando a predição também está vazia
quanto quando contém pixels previstos como sal. Portanto, a métrica não distingue
nem penaliza falsos positivos nas 313 imagens sem sal. A análise filtrada responde
à pergunta sobre segmentação nas imagens com sal, mas não avalia o comportamento dos
modelos nas imagens sem sal. Esse comportamento deve ser medido separadamente por
uma métrica de falsos positivos, como a quantidade ou proporção de pixels previstos
como sal.

Além disso, os checkpoints comparados usam seeds diferentes (42 e 123), e há apenas
um checkpoint por condição. Para atribuir diferenças à transformação brightness/
contrast, é necessário comparar seeds pareadas ou múltiplas execuções por condição.

## Rastreabilidade

- Ambiente da execução: Python 3.10.20, PyTorch 2.5.1, NumPy 2.0.1, SciPy 1.15.3 e OpenCV 5.0.0.
- SHA-256 dos 800 IDs ordenados: `835d74e6244dd874202a843557259213d66b0880bd9b67d005ae0b7755520bc2`.
- Contagens, médias, medianas e IQRs deste relatório foram recalculados a partir de [métricas por imagem](../resultsHPC/paired_iou_test_a2-123random_brightness_contrast/per_image_metrics.csv).
- Estatísticas do teste e ambiente foram conferidos no [resumo JSON da execução](../resultsHPC/paired_iou_test_a2-123random_brightness_contrast/summary.json).
- [Gráfico pareado](../resultsHPC/paired_iou_test_a2-123random_brightness_contrast/paired_iou.png)
- [Distribuição das diferenças](../resultsHPC/paired_iou_test_a2-123random_brightness_contrast/iou_difference.png)
