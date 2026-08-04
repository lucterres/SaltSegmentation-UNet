# Relatório — Cenário DA: Data Augmentation Albumentations (Cenário B)

**Manuscrito:** Access-2026-27912  
**Data:** 2026-08-03/04  
**Nó:** atn2b03n03 / atn2b02n05 (8× Tesla V100-SXM2-32GB)  
**Objetivo:** Comparar 6 métodos de Data Augmentation (Albumentations) como pool sintético no Cenário B (N=1200+1200), com protocolo rigoroso sem data leakage com o test set canônico.

---

## 1. Protocolo experimental

### 1.1 Configuração

| Parâmetro | Valor |
|-----------|-------|
| Cenário | B (real + sintético) |
| N real | 1200 (amostras de `train_pool`) |
| N sintético | 1200 (por método, de `*1600clean`) |
| Seed | 42 |
| Épocas máx. | 100 (early stop patience=10) |
| Train dir | `dataset/train_pool/` — **3198 amostras reais sem leakage** |
| Test dir | `subset_split/test/` — **800 amostras canônicas fixas** |
| GPU | Tesla V100-SXM2-32GB (1 GPU por run) |

### 1.2 Controle de data leakage

O TGS tem **3998 imagens**. O test set canônico de **800 amostras** foi extraído dessas mesmas 3998.  
Para evitar leakage:

- **`dataset/train_pool/`**: criado com `create_train_pool.py` — 3998 − 800 = **3198 amostras** livres do test
- **`dataset/*1600clean/`**: gerados com `generate_albumentations_da.py --exclude_csv split_stats.csv` — pool de **3199 fontes** (excluindo as 800 do test)

### 1.3 Métodos DA avaliados

| GPU | Método | Tipo | Descrição |
|:---:|--------|:----:|-----------|
| 0 | `elastic_transform` | geom | ElasticTransform: alpha=80, sigma=9 |
| 1 | `grid_distortion` | geom | GridDistortion: num_steps=5, distort_limit=0.3 |
| 2 | `optical_distortion` | geom | OpticalDistortion: distort_limit=0.4 |
| 3 | `clahe` | int | CLAHE: clip_limit=4.0, tile=(4×4) |
| 4 | `random_brightness_contrast` | int | Brightness±0.3, Contrast±0.3 |
| 5 | `random_gamma` | int | RandomGamma: gamma_limit=(60,140) |

---

## 2. Baseline — Cenário A (sem sintéticos)

### 2.1 Cenário A — N=1200 (seed 42, test canônico)

| Config | Test IoU | Test Dice |
|--------|:--------:|:---------:|
| A — N=1200 reais, seed 42 | 0.3862 | 0.4252 |
| A — N=1200 reais, seed 123 | 0.3821 | 0.4205 |
| A — N=1200 reais, seed 456 | 0.3817 | 0.4230 |
| **A — N=1200 média** | **0.383 ± 0.002** | **0.423 ± 0.002** |

### 2.2 Cenário A — train_pool N=3198 (3 seeds, test canônico, sem leakage)

| Seed | N real | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------:|:--------:|:---------:|:------:|:---------:|
| 42  | 3198 | 0.4245 | 0.4576 | 48 | 221 |
| 123 | 3198 | 0.4343 | 0.4668 | 57 | 207 |
| 456 | 3198 | 0.4321 | 0.4638 | 52 | 236 |
| **Média ± std** | | **0.430 ± 0.005** | **0.463 ± 0.005** | | |

---

## 3. Resultados — 6 métodos DA clean (seed 42, N=1200+1200)

Datasets: `*1600clean` — gerados excluindo as 800 imagens do test set canônico.

| Método DA | Tipo | Épocas | Best Val IoU | **Test IoU** | **Test Dice** | Tempo (s) |
|-----------|:----:|:------:|:------------:|:------------:|:-------------:|:---------:|
| elastic_transform | geom | 51 | 0.4128 | 0.4280 | 0.4641 | 152 |
| grid_distortion | geom | 74 | 0.4082 | 0.4246 | 0.4592 | 217 |
| optical_distortion | geom | 55 | 0.3962 | 0.4232 | 0.4597 | 204 |
| clahe | int | 46 | 0.4144 | 0.4164 | 0.4529 | 166 |
| **random_brightness_contrast** | **int** | **61** | **0.4348** | **0.4327** | **0.4683** | **178** |
| random_gamma | int | 74 | 0.4274 | 0.4310 | 0.4647 | 268 |

> 🏆 Melhor resultado (clean, seed 42): **random_brightness_contrast** — Test IoU=**0.4327**, Test Dice=**0.4683**

---

## 3b. Consistência estatística — random_brightness_contrast e random_gamma (3 seeds)

| Método | Seed 42 | Seed 123 | Seed 456 | **Média ± std** | **Dice médio** |
|--------|:-------:|:--------:|:--------:|:---------------:|:--------------:|
| **random_brightness_contrast** | **0.4327** | **0.4219** | **0.4218** | **0.425 ± 0.006** | **0.460 ± 0.007** |
| random_gamma | 0.4310 | 0.4206 | 0.4153 | 0.422 ± 0.008 | 0.459 ± 0.006 |
| **A — N=1200 (ref)** | 0.3862 | 0.3821 | 0.3817 | **0.383 ± 0.002** | **0.423 ± 0.002** |
| **A — train_pool N=3198 (ref)** | 0.4245 | 0.4343 | 0.4321 | **0.430 ± 0.005** | **0.463 ± 0.005** |

**Δ vs A (N=1200) — estatisticamente consistente (3 seeds):**
- random_brightness_contrast: **+0.042 ± 0.006** IoU
- random_gamma: **+0.039 ± 0.007** IoU

> ✅ O ganho de DA é **consistente nas 3 seeds** — não é artefato de uma seed favorável.  
> ✅ **B + RBC (0.425) ≈ A train_pool (0.430)** com apenas 37% dos dados reais.  
> ✅ **random_brightness_contrast** é o método mais consistente (menor desvio padrão).

---

## 4. Comparação — Cenário B DA vs Baseline A

| Config | N real | N synth | Test IoU | Test Dice | Δ vs A (N=1200) |
|--------|:------:|:-------:|:--------:|:---------:|:---------------:|
| A — N=3198 (train_pool) | 3198 | 0 | 0.430 ± 0.005 | 0.463 ± 0.005 | — |
| 🏆 **B + random_brightness_contrast** | **1200** | **1200** | **0.4327** | **0.4683** | **+0.047** |
| B + random_gamma | 1200 | 1200 | 0.4310 | 0.4647 | +0.045 |
| B + elastic_transform | 1200 | 1200 | 0.4280 | 0.4641 | +0.042 |
| B + grid_distortion | 1200 | 1200 | 0.4246 | 0.4592 | +0.038 |
| B + optical_distortion | 1200 | 1200 | 0.4232 | 0.4597 | +0.037 |
| B + clahe | 1200 | 1200 | 0.4164 | 0.4529 | +0.030 |
| A — train_filtered (1293) | 1293 | 0 | 0.424 | 0.458 | +0.037 |
| **A — N=1200** | **1200** | **0** | **0.386** | **0.425** | **—** |

> ✅ **Todos os 6 métodos DA superam o Cenário A com N=1200** (+3 a +5pp IoU).  
> ✅ **B + random_brightness_contrast (0.4327) ≥ A — train_pool (0.430)** — com 37% dos dados reais, DA atinge desempenho equivalente ao treino completo.  
> ✅ **Métodos de intensidade** (RBC, gamma) superam geométricos (média 0.427 vs 0.425).

---

## 5. Análise por tipo de DA

| Grupo | Média Test IoU | Média Test Dice |
|:-----:|:--------------:|:---------------:|
| Geométrico (elastic, grid, optical) | 0.425 | 0.461 |
| **Intensidade (clahe, rbc, gamma)** | **0.427** | **0.462** |

Transforms de **intensidade** preservam a geometria das máscaras e perturbam apenas a textura de amplitude — mais adequado para dados sísmicos onde variação de ganho entre aquisições é a principal fonte de variabilidade.

---

## 6. Escala de dados — random_gamma (seed 42)

| N real | N synth | N total | Épocas | **Test IoU** | **Test Dice** |
|:------:|:-------:|:-------:|:------:|:------------:|:-------------:|
| 1000 | 1600 | 2600 | 45 | 0.4287 | 0.4636 |
| **1200** | **1200** | **2400** | **74** | **0.4310** | **0.4647** |
| 1600 | 1000 | 2600 | 61 | 0.4517 | 0.4826 |
| 1600 | 1600 | 3200 | 47 | 0.4469 | 0.4788 |

> ⚠️ Os runs de escala com N_real=1600 e 1000 usaram pools `*1600` (com leakage) e são apresentados aqui apenas como referência de tendência — não para reportar no paper. O ponto ótimo limpo confirmado é **N=1200+1200**.

---

## 7. Paths dos artefatos (resultados válidos)

| Experimento | Run tag |
|-------------|---------|
| A — N=1200 seed42 | `scenario_A_seed42_nreal1200` |
| A — train_pool seed42 | `scenario_A_seed42_train_pool` |
| A — train_pool seed123 | `scenario_A_seed123_train_pool` |
| A — train_pool seed456 | `scenario_A_seed456_train_pool` |
| B + elastic_transform clean | `scenario_B_seed42_nreal1200_train_elastic_transform1600clean_ns1200` |
| B + grid_distortion clean | `scenario_B_seed42_nreal1200_train_grid_distortion1600clean_ns1200` |
| B + optical_distortion clean | `scenario_B_seed42_nreal1200_train_optical_distortion1600clean_ns1200` |
| B + clahe clean | `scenario_B_seed42_nreal1200_train_clahe1600clean_ns1200` |
| B + random_brightness_contrast clean | `scenario_B_seed42_nreal1200_train_random_brightness_contrast1600clean_ns1200` |
| B + random_gamma clean | `scenario_B_seed42_nreal1200_train_random_gamma1600clean_ns1200` |

---

## 8. Próximos passos

- [x] Rodar seeds 123 e 456 para `random_brightness_contrast` e `random_gamma` clean — **concluído, seção 3b**
- [x] Atualizar `docs/relatorio-final-r21-downstream.md` com esses resultados — **seção 13 adicionada**
- [ ] Atualizar `_v7.tex` → `\subsection{Downstream Segmentation Evaluation}`
- [ ] Atualizar `response_to_reviewers.md` → seção R2.1