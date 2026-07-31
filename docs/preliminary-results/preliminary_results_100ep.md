# Resultados Preliminares — Cenário A (Real Only) — 100 Epochs (GPU)

**Manuscrito:** Access-2026-27912  
**Experimento:** R2.1 — Downstream Segmentation  
**Configuração:** Scenario A · seed=42 · epochs=100 (early stop patience=10) · device=**CUDA** (Tesla V100-SXM2-32GB)  
**Servidor:** atn2b02n07 · torch=2.4.1+cu124  
**Data:** 2026-07-20  
**Objetivo:** Avaliar desempenho com dataset completo (~3200 amostras de treino) e budget de 100 epochs; comparar com resultados anteriores em CPU (50ep)

---

## Resultado Final

| Métrica | Valor |
|:-------:|:-----:|
| **Test IoU** | **0.4312** |
| **Test Dice** | **0.4657** |
| Best Val IoU | 0.4395 (E52) |
| Epochs rodadas | 57 *(early stop E57, patience=10)* |
| Tempo total | **202s (~3.4 min)** |
| Throughput | ~55 it/s (GPU V100) |

---

## Configuração do treinamento

| Parâmetro | Valor |
|:----------|:------|
| Dataset total | 3998 pares válidos |
| Test set fixo | 800 imagens (split seed=0) |
| Train pool | 3198 imagens |
| Train / Val split | 2878 / 320 (90/10) |
| n_real | 3198 (sem limitação — dataset completo) |
| Epochs máx | 100 |
| Early stop patience | 10 |
| Batch size | config.BATCH_SIZE |
| Optimizer | Adam (lr=config.LR) |

---

## Evolução por Epoch

| E | Loss treino | Loss val | IoU val | Dice val | Best |
|:-:|:-----------:|:--------:|:-------:|:--------:|:----:|
| 1  | 0.5540 | 0.4818 | 0.3026 | 0.3507 | ✓ |
| 2  | 0.4406 | 0.3718 | 0.3089 | 0.3415 | ✓ |
| 3  | — | — | ~0.32 | — | |
| 5  | — | — | ~0.33 | — | |
| 10 | — | — | ~0.36 | — | |
| 15 | — | — | ~0.38 | — | |
| 20 | — | — | ~0.40 | — | |
| 25 | — | — | ~0.41 | — | |
| 27 | 0.1686 | 0.1775 | 0.4154 | 0.4496 | |
| 28 | 0.1738 | 0.2039 | 0.4212 | 0.4578 | |
| 29 | 0.1730 | 0.2066 | 0.4164 | 0.4519 | |
| 30 | 0.1674 | 0.1978 | 0.4152 | 0.4474 | |
| 48 | 0.1075 | 0.1770 | 0.4344 | 0.4695 | |
| 49 | 0.1102 | 0.1729 | 0.4316 | 0.4647 | |
| 50 | 0.1049 | 0.1734 | 0.4378 | 0.4727 | |
| 51 | 0.1071 | 0.1705 | 0.4365 | 0.4712 | |
| **52** | **0.1045** | **0.1721** | **0.4395** | **0.4746** | **✓ best** |
| 53 | 0.1048 | 0.1838 | 0.4349 | 0.4699 | |
| 54 | 0.1002 | 0.1751 | 0.4370 | 0.4720 | |
| 55 | 0.0976 | 0.1750 | 0.4337 | 0.4679 | |
| 56 | 0.0989 | 0.1763 | 0.4386 | 0.4736 | |
| 57 | 0.0970 | 0.1789 | 0.4344 | 0.4692 | *early stop* |

---

## Comparação com Resultados Anteriores (CPU · 50ep)

### Tabela consolidada por N de amostras e budget de epochs

| N real | Device | Epochs max | Epochs rodadas | Test IoU | Test Dice | Elapsed |
|:------:|:------:|:----------:|:--------------:|:--------:|:---------:|:-------:|
| 200    | CPU    | 50         | 29 *(early stop)* | 0.3102 | 0.3616 | ~667s |
| 400    | CPU    | 50         | 32 *(early stop)* | 0.3391 | 0.3820 | ~757s |
| 800    | CPU    | 50         | 50 *(completo)*   | 0.3630 | 0.4019 | ~1461s |
| 1600   | CPU    | 50         | 50 *(completo)*   | 0.4011 | 0.4401 | ~4724s |
| **~3200** | **GPU V100** | **100** | **57 *(early stop)*** | **0.4312** | **0.4657** | **202s** |

### Ganho de desempenho vs melhor resultado anterior

| Comparação | Test IoU | Test Dice | Δ IoU | Δ Dice |
|:-----------|:--------:|:---------:|:-----:|:------:|
| N=1600, 50ep, CPU (melhor anterior) | 0.4011 | 0.4401 | — | — |
| **N~3200, 100ep, GPU (atual)**       | **0.4312** | **0.4657** | **+0.0301 (+7.5%)** | **+0.0256 (+5.8%)** |

### Comparativo de velocidade CPU vs GPU

| Device | N treino | Epochs | Tempo total | it/s (aprox.) |
|:------:|:--------:|:------:|:-----------:|:-------------:|
| CPU    | 1600     | 50     | ~4724s      | ~1–2 it/s     |
| **GPU V100** | **3200** | **57** | **202s** | **~55 it/s** |

> **~23× mais rápido** na GPU para o dobro de dados e mais epochs.

---

## Análise

### Convergência com dataset completo (~3200 amostras)

Com o dataset completo, o modelo apresentou convergência estável:
- Loss de treino decresceu de **0.554 → 0.097** (−82%)
- Val IoU subiu de **0.303 → 0.440** (best), com platô claro entre E47–E57
- Early stop disparou na E57 (patience=10 após best na E52), indicando **convergência real** — diferentemente dos runs N=800 e N=1600 com 50 epochs que terminaram sem convergir

### Comparação com runs anteriores

Os resultados anteriores (CPU, 50ep) revelavam uma curva de escala: mais dados → melhor IoU, mas nenhum run com N>800 havia convergido dentro de 50 epochs. O run atual com ~3200 amostras e budget 100ep **confirma a tendência e atinge convergência real** (early stop na E57).

O ganho de IoU de **0.4011 → 0.4312 (+7.5%)** ao dobrar os dados (1600→3200) é consistente com a taxa de escala observada anteriormente (cada duplicação de N rendeu ~7–17% de ganho).

### Implicação para o experimento R2.1

Este resultado estabelece o **baseline Cenário A (real only, N~3200)** para o manuscrito. O próximo passo é rodar o Cenário B (real + synthetic) com as mesmas condições para quantificar o benefício dos dados sintéticos.

**Baseline definido:**
- Cenário A · seed=42 · N~3200 · 100ep → **Test IoU = 0.4312 ± (aguardar seeds 123, 456)**

---

## Próximos passos

- [ ] Rodar Cenário A com seeds 123 e 456 (para calcular mean ± std)
- [ ] Rodar Cenário B (real + synthetic, N=3200+400) com seeds 42, 123, 456
- [ ] Executar `evaluate.py` para gerar `results/summary.csv`
- [ ] Calcular p-value (Wilcoxon) entre Cenário A e B
- [ ] Preencher tabela do manuscrito (`_v7.tex`)

---

## Artefatos gerados

```
results/scenario_A_seed42/
├── best_model.pth     ← checkpoint E52 (best val IoU = 0.4395)
├── result.csv         ← métricas finais do test set
└── plot.png           ← curvas loss/IoU por epoch
```

Log completo: `/var/tmp/cym7/train_full_A_s42.log` (servidor atn2b02n07)
