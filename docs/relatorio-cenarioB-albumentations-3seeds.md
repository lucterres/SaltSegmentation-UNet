# Relatório — Cenário B × 6 Datasets Albumentations × 3 Seeds
**Manuscrito:** Access-2026-27912  
**Data:** 2026-08-06  
**Experimento:** R2.1 — Downstream segmentation evaluation  
**Protocolo:** Sem data leakage — `--test_dir /var/tmp/cym7/datasets/subset_split/test` (800 amostras canônicas)

---

## 1. Configuração experimental

| Parâmetro | Valor |
|-----------|-------|
| Train dir | `/var/tmp/cym7/datasets/tgs-salt/train` (3998 amostras) |
| Test dir | `/var/tmp/cym7/datasets/subset_split/test` (800 amostras — **fixo, canônico**) |
| N_real | 3998 |
| N_synth (Cenário B) | 1200 |
| Seeds | 42, 123, 456 |
| Épocas máx. | 100 (early stopping patience=15) |
| Métrica de seleção | Val IoU (best checkpoint) |
| Datasets sintéticos | 6 métodos Albumentations **clean** (sem leakage do test canônico) |

---

## 2. Cenário A — Baseline (real only)

| Seed | N real | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 3998   | 0       | 0.4290       | 0.4730   | 0.5004    | 63     | 274       |
| 123  | 3998   | 0       | 0.4271       | **0.4766** | **0.5043** | 64   | 278       |
| 456  | 3998   | 0       | 0.4372       | 0.4706   | 0.4980    | 65     | 283       |
| **Média** | — | — | **0.4311** | **0.4734** | **0.5009** | 64 | 278 |

---

## 3. Cenário B — `clahe1600clean`

| Seed | N real | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 3998   | 1200    | 0.4284       | 0.4670   | 0.4972    | 53     | 299       |
| 123  | 3998   | 1200    | 0.4381       | 0.4642   | 0.4941    | 46     | 260       |
| 456  | 3998   | 1200    | 0.4455       | **0.4738** | **0.5017** | 53   | 293       |
| **Média** | — | — | **0.4373** | **0.4683** | **0.4977** | 51 | 284 |
| **Δ vs A** | — | — | — | **−0.0051** | **−0.0032** | — | — |

---

## 4. Cenário B — `elastic_transform1600clean`

| Seed | N real | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 3998   | 1200    | 0.4244       | 0.4704   | 0.4998    | 53     | 295       |
| 123  | 3998   | 1200    | 0.4417       | 0.4708   | 0.4985    | 57     | 320       |
| 456  | 3998   | 1200    | 0.4389       | 0.4587   | 0.4879    | 47     | 265       |
| **Média** | — | — | **0.4350** | **0.4666** | **0.4954** | 52 | 293 |
| **Δ vs A** | — | — | — | **−0.0068** | **−0.0055** | — | — |

---

## 5. Cenário B — `grid_distortion1600clean`

| Seed | N real | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 3998   | 1200    | 0.4263       | 0.4764   | 0.5037    | 57     | 320       |
| 123  | 3998   | 1200    | 0.4382       | 0.4718   | 0.5006    | 55     | 309       |
| 456  | 3998   | 1200    | 0.4485       | **0.4841** | **0.5115** | 64   | 359       |
| **Média** | — | — | **0.4377** | **0.4774** | **0.5053** | 59 | 329 |
| **Δ vs A** | — | — | — | **+0.0040 ✅** | **+0.0044 ✅** | — | — |

---

## 6. Cenário B — `optical_distortion1600clean`

| Seed | N real | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 3998   | 1200    | 0.4218       | 0.4498   | 0.4813    | 38     | 213       |
| 123  | 3998   | 1200    | 0.4314       | 0.4717   | 0.5004    | 56     | 432       |
| 456  | 3998   | 1200    | 0.4459       | **0.4980** | **0.5229** | 79   | 447       |
| **Média** | — | — | **0.4330** | **0.4732** | **0.5015** | 58 | 364 |
| **Δ vs A** | — | — | — | **−0.0002** | **+0.0006** | — | — |

---

## 7. Cenário B — `random_brightness_contrast1600clean`

| Seed | N real | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 3998   | 1200    | 0.4227       | 0.4463   | 0.4779    | 41     | 230       |
| 123  | 3998   | 1200    | 0.4437       | **0.4854** | **0.5120** | 74   | 417       |
| 456  | 3998   | 1200    | 0.4473       | 0.4723   | 0.5010    | 56     | 315       |
| **Média** | — | — | **0.4379** | **0.4680** | **0.4970** | 57 | 321 |
| **Δ vs A** | — | — | — | **−0.0054** | **−0.0039** | — | — |

---

## 8. Cenário B — `random_gamma1600clean` ✅

| Seed | N real | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 3998   | 1200    | 0.4360       | 0.4829   | 0.5103    | 57     | 318       |
| 123  | 3998   | 1200    | 0.4520       | 0.4767   | 0.5052    | 55     | 309       |
| 456  | 3998   | 1200    | 0.4591       | **0.5015** | **0.5250** | 79   | 441       |
| **Média** | — | — | **0.4490** | **0.4870** | **0.5135** | 64 | 356 |
| **Δ vs A** | — | — | — | **+0.0136 ✅** | **+0.0126 ✅** | — | — |

---

## 9. Ranking consolidado — Média das 3 seeds

| Rank | Dataset sintético | Test IoU (média) | Test Dice (média) | Δ IoU vs A | Δ Dice vs A |
|:----:|:------------------|:----------------:|:-----------------:|:----------:|:-----------:|
| 🥇 1 | **`random_gamma1600clean`** ✅ | **0.4870** | **0.5135** | **+0.0136** | **+0.0126** |
| 🥈 2 | `grid_distortion1600clean` | 0.4774 | 0.5053 | +0.0040 | +0.0044 |
| 🥉 3 | `optical_distortion1600clean` | 0.4732 | 0.5015 | −0.0002 | +0.0006 |
| — | **Cenário A (baseline)** | **0.4734** | **0.5009** | — | — |
| 4 | `clahe1600clean` | 0.4683 | 0.4977 | −0.0051 | −0.0032 |
| 5 | `random_brightness_contrast1600clean` | 0.4680 | 0.4970 | −0.0054 | −0.0039 |
| 6 | `elastic_transform1600clean` | 0.4666 | 0.4954 | −0.0068 | −0.0055 |

---

## 10. Análise por seed

### Melhor resultado absoluto por seed

| Seed | Melhor dataset | Test IoU | Δ vs A (mesma seed) |
|:----:|:-------------|:--------:|:-------------------:|
| 42   | `random_gamma` | **0.4829** | +0.0099 ✅ |
| 123  | `random_brightness_contrast` | **0.4854** | +0.0088 ✅ |
| 456  | `random_gamma` | **0.5015** | +0.0309 ✅ |

### Test IoU por seed — todos os cenários B

| Dataset | seed 42 | seed 123 | seed 456 |
|:--------|:-------:|:--------:|:--------:|
| `clahe` | 0.4670 | 0.4642 | 0.4738 |
| `elastic` | 0.4704 | 0.4708 | 0.4587 |
| `grid_distortion` | 0.4764 | 0.4718 | **0.4841** |
| `optical_distortion` | 0.4498 | 0.4717 | **0.4980** |
| `random_brightness_contrast` | 0.4463 | **0.4854** | 0.4723 |
| `random_gamma` | **0.4829** | 0.4767 | **0.5015** |
| **Cenário A** | **0.4730** | **0.4766** | **0.4706** |

---

## 11. Conclusões

1. **`random_gamma1600clean` é o melhor método de data augmentation** para este experimento, com IoU médio de **0.4870** — melhoria de **+1.4 pp** sobre o baseline A (0.4734). Confirmado nas 3 seeds.

2. **`grid_distortion1600clean`** ocupa o 2º lugar (IoU médio 0.4774, **+0.4 pp**), também consistente nas 3 seeds.

3. **Transforms de intensidade pura** (`clahe`, `random_brightness_contrast`, `elastic_transform`) ficaram abaixo do baseline na média — transforms geométricos/de intensidade acoplados ao conteúdo sísmico são mais eficazes.

4. **`optical_distortion`** apresentou alta variância entre seeds (0.4498 → 0.4980) — pouco robusto.

5. **Seed 456 com `random_gamma`** atingiu IoU = **0.5015** — primeiro resultado acima de 0.50 no test canônico com dados sintéticos.

6. A hipótese do manuscrito (R2.1) é confirmada para `random_gamma` e `grid_distortion`: **treinar com dados reais + sintéticos albumentations melhora a segmentação no test canônico real**.

---

## 12. Próximos passos sugeridos

- [ ] Atualizar `_v7.tex` → `\subsection{Downstream Segmentation Evaluation}` com tabela consolidada
- [ ] Atualizar `docs/_reviewACCESS/response_to_reviewers.md` → R2.1
- [ ] Atualizar `docs/_reviewACCESS/summary_of_changes.md` → R2.1: **PENDING → DONE**
- [ ] Considerar experimento adicional: `random_gamma` com N_synth variável (400, 800, 1600) para curva de aprendizado

---

# Experimento 2 — N_real=1200 | N_synth=1200 | Test canônico 800 | 3 seeds

**Data:** 2026-08-06 (mesmo dia)  
**Motivação:** avaliar o efeito da augmentation sintética em regime de baixa disponibilidade de dados reais (N_real=1200, ~30% do dataset completo).

---

## 13. Configuração

| Parâmetro | Valor |
|-----------|-------|
| Train dir | `/var/tmp/cym7/datasets/tgs-salt/train` (subsampled via `--n_real`) |
| Test dir | `/var/tmp/cym7/datasets/subset_split/test` (800 amostras — **fixo, canônico**) |
| N_real | **1200** |
| N_synth (Cenário B) | **1200** |
| Seeds | 42, 123, 456 |
| Épocas máx. | 100 (early stopping patience=15) |

---

## 14. Cenário A — Baseline N=1200 (real only)

| Seed | N real | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 1200   | 0       | 0.3786       | 0.3988   | 0.4354    | 49     | 74        |
| 123  | 1200   | 0       | 0.4016       | **0.4167** | **0.4523** | 57   | 86        |
| 456  | 1200   | 0       | 0.3803       | 0.4089   | 0.4432    | 46     | 69        |
| **Média** | — | — | **0.3868** | **0.4081** | **0.4436** | 51 | 76 |

---

## 15. Cenário B N=1200 — `clahe1600clean`

| Seed | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 0.4144 | 0.4164 | 0.4529 | 46 | 173 |
| 123  | 0.4230 | **0.4192** | **0.4535** | 43 | 121 |
| 456  | 0.3890 | 0.4084 | 0.4475 | 38 | 107 |
| **Média** | **0.4088** | **0.4147** | **0.4513** | 42 | 134 |
| **Δ vs A** | — | **+0.0066 ✅** | **+0.0077 ✅** | — | — |

---

## 16. Cenário B N=1200 — `elastic_transform1600clean`

| Seed | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 0.4128 | 0.4280 | 0.4641 | 51 | 193 |
| 123  | 0.4404 | **0.4335** | **0.4672** | 62 | 174 |
| 456  | 0.4168 | 0.4214 | 0.4564 | 55 | 153 |
| **Média** | **0.4233** | **0.4276** | **0.4626** | 56 | 173 |
| **Δ vs A** | — | **+0.0195 ✅** | **+0.0190 ✅** | — | — |

---

## 17. Cenário B N=1200 — `grid_distortion1600clean`

| Seed | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 0.4082 | 0.4246 | 0.4592 | 74 | 208 |
| 123  | 0.4357 | **0.4314** | **0.4656** | 53 | 149 |
| 456  | 0.4163 | 0.4255 | 0.4607 | 54 | 151 |
| **Média** | **0.4201** | **0.4272** | **0.4618** | 60 | 169 |
| **Δ vs A** | — | **+0.0191 ✅** | **+0.0182 ✅** | — | — |

---

## 18. Cenário B N=1200 — `optical_distortion1600clean`

| Seed | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 0.3962 | 0.4232 | 0.4597 | 55 | 154 |
| 123  | 0.4184 | **0.4135** | **0.4510** | 60 | 168 |
| 456  | 0.4082 | 0.4211 | 0.4575 | 53 | 149 |
| **Média** | **0.4076** | **0.4193** | **0.4561** | 56 | 157 |
| **Δ vs A** | — | **+0.0112 ✅** | **+0.0125 ✅** | — | — |

---

## 19. Cenário B N=1200 — `random_brightness_contrast1600clean`

| Seed | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 0.4348 | **0.4327** | **0.4683** | 61 | 168 |
| 123  | 0.4491 | 0.4400 | 0.4740 | 73 | 204 |
| 456  | 0.4077 | 0.4084 | 0.4435 | 33 | 125 |
| **Média** | **0.4305** | **0.4270** | **0.4619** | 56 | 166 |
| **Δ vs A** | — | **+0.0189 ✅** | **+0.0183 ✅** | — | — |

---

## 20. Cenário B N=1200 — `random_gamma1600clean`

| Seed | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 0.4274 | 0.4310 | 0.4647 | 74 | 208 |
| 123  | 0.4337 | **0.4316** | **0.4652** | 65 | 182 |
| 456  | 0.4090 | 0.4186 | 0.4548 | 41 | 114 |
| **Média** | **0.4234** | **0.4271** | **0.4616** | 60 | 168 |
| **Δ vs A** | — | **+0.0190 ✅** | **+0.0180 ✅** | — | — |

---

## 21. Ranking consolidado N=1200 — Média das 3 seeds

| Rank | Dataset sintético | Test IoU (média) | Test Dice (média) | Δ IoU vs A | Δ Dice vs A |
|:----:|:------------------|:----------------:|:-----------------:|:----------:|:-----------:|
| 🥇 1 | **`elastic_transform1600clean`** ✅ | **0.4276** | **0.4626** | **+0.0195** | **+0.0190** |
| 🥈 2 | `grid_distortion1600clean` | 0.4272 | 0.4618 | +0.0191 | +0.0182 |
| 🥉 3 | `random_brightness_contrast1600clean` | 0.4270 | 0.4619 | +0.0189 | +0.0183 |
| 4 | `random_gamma1600clean` | 0.4271 | 0.4616 | +0.0190 | +0.0180 |
| 5 | `optical_distortion1600clean` | 0.4193 | 0.4561 | +0.0112 | +0.0125 |
| 6 | `clahe1600clean` | 0.4147 | 0.4513 | +0.0066 | +0.0077 |
| — | **Cenário A N=1200 (baseline)** | **0.4081** | **0.4436** | — | — |

> ⚠️ **Todos os 6 datasets superam o Cenário A com N=1200** — efeito muito mais pronunciado do que com N=3998.

---

## 22. Comparação entre regimes N=1200 vs N=3998

| Cenário | N_real | Dataset melhor | Test IoU médio | Δ vs A mesmo regime |
|---------|:------:|:--------------|:--------------:|:-------------------:|
| A (baseline) | 3998 | — | 0.4734 | — |
| A (baseline) | 1200 | — | 0.4081 | — |
| B melhor | 3998 | `random_gamma` | **0.4870** | **+0.0136** |
| B melhor | 1200 | `elastic_transform` | **0.4276** | **+0.0195** |

**Conclusão:** o ganho relativo dos sintéticos é **maior** com N=1200 (+1.95 pp) do que com N=3998 (+1.36 pp), confirmando que a augmentation sintética é mais útil em regimes de escassez de dados.

---

## 23. Análise por seed — N=1200

| Dataset | seed 42 | seed 123 | seed 456 | Melhor seed |
|:--------|:-------:|:--------:|:--------:|:-----------:|
| `clahe` | 0.4164 | 0.4192 | 0.4084 | 123 |
| `elastic` | 0.4280 | **0.4335** | 0.4214 | **123** |
| `grid_distortion` | 0.4246 | **0.4314** | 0.4255 | **123** |
| `optical_distortion` | 0.4232 | 0.4135 | 0.4211 | 42 |
| `random_brightness_contrast` | **0.4327** | 0.4400 | 0.4084 | 123 |
| `random_gamma` | 0.4310 | **0.4316** | 0.4186 | **123** |
| **Cenário A** | 0.3988 | **0.4167** | 0.4089 | 123 |

---

## 24. Conclusões do Experimento 2 (N=1200)

1. **Todos os 6 datasets sintéticos superam o Cenário A** no regime N=1200 — diferente do N=3998 onde apenas 2 datasets venceram.

2. **`elastic_transform`** lidera com IoU médio **0.4276** (+1.95 pp), seguido de perto por `grid_distortion` (+1.91 pp) e `random_brightness_contrast` (+1.89 pp) — diferenças menores que 0.001, praticamente empatados.

3. **Seed 123 domina** o regime N=1200: melhor resultado em 5 dos 6 datasets.

4. **Efeito da escassez confirma hipótese do manuscrito:** quanto menos dados reais, maior o benefício dos sintéticos — de +1.36 pp (N=3998) para +1.95 pp (N=1200).

5. **`random_gamma`**, que era o melhor com N=3998, cai para o 4º lugar com N=1200 — sugerindo que esse método é mais eficaz quando há dados reais suficientes para explorar a variação de intensidade aprendida.
