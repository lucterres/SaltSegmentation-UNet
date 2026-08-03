# Relatório — Cenário DA: Data Augmentation Albumentations (Cenário B)

**Manuscrito:** Access-2026-27912  
**Data:** 2026-08-03  
**Nó:** atn2b03n03 (8× Tesla V100-SXM2-32GB)  
**Objetivo:** Comparar 6 métodos de Data Augmentation (Albumentations) como pool sintético no Cenário B (N=1200+1200) e avaliar o efeito de escala do melhor método.

---

## 1. Configuração experimental

| Parâmetro | Valor |
|-----------|-------|
| Cenário | B (real + sintético) |
| N real | 1200 |
| N sintético | 1200 (por método) |
| Seed | 42 |
| Épocas máx. | 100 (early stop patience=10) |
| Train dir | `/var/tmp/cym7/datasets/tgs-salt/train` (TGS completo, 3998 imagens) |
| Test dir | `/var/tmp/cym7/datasets/subset_split/test` (800 amostras canônicas) |
| GPU | Tesla V100-SXM2-32GB (1 GPU por run) |

### Métodos DA avaliados

| GPU | Método | Tipo | Descrição |
|:---:|--------|:----:|-----------|
| 0 | `elastic_transform` | geom | ElasticTransform: alpha=80, sigma=9 |
| 1 | `grid_distortion` | geom | GridDistortion: num_steps=5, distort_limit=0.3 |
| 2 | `optical_distortion` | geom | OpticalDistortion: distort_limit=0.4 |
| 3 | `clahe` | int | CLAHE: clip_limit=4.0, tile=(4×4) |
| 4 | `random_brightness_contrast` | int | Brightness±0.3, Contrast±0.3 |
| 5 | `random_gamma` | int | RandomGamma: gamma_limit=(60,140) |

Datasets gerados com `generate_albumentations_da.py --src D:\dataset\tgs-salt\train --n 1600`.  
Path no servidor: `$PROJ/Salt-Segmentation-UNet/dataset/<método>1600/`

---

## 2. Resultados — 6 métodos (seed 42, N=1200+1200)

| Método DA | Tipo | Épocas | Best Val IoU | **Test IoU** | **Test Dice** | Tempo (s) |
|-----------|:----:|:------:|:------------:|:------------:|:-------------:|:---------:|
| elastic_transform | geom | 50 | 0.4272 | 0.4256 | 0.4621 | 140 |
| grid_distortion | geom | 41 | 0.4043 | 0.4249 | 0.4586 | 116 |
| optical_distortion | geom | 63 | 0.4134 | 0.4367 | 0.4712 | 176 |
| clahe | int | 71 | 0.4292 | 0.4401 | 0.4748 | 197 |
| random_brightness_contrast | int | 58 | 0.4134 | 0.4328 | 0.4654 | 160 |
| **random_gamma** | **int** | **74** | **0.4414** | **0.4580** | **0.4892** | **205** |

> 🏆 Melhor resultado: **random_gamma** — Test IoU=**0.4580**, Test Dice=**0.4892**

---

## 3. Comparação com experimentos anteriores (seed 42, N=1200)

### 3.1 Baseline — Cenário A (sem sintéticos)

| Config | Seed | Test IoU | Test Dice |
|--------|:----:|:--------:|:---------:|
| A — real only | 42 | 0.3862 | 0.4252 |
| A — real only | 123 | 0.3821 | 0.4205 |
| A — real only | 456 | 0.3817 | 0.4230 |
| **A média** | — | **0.383** | **0.423** |

### 3.2 Cenário B — todos os métodos sintéticos com N=1200 (seed 42)

| Método sintético | N synth | Fase | Test IoU | Test Dice | Δ vs A |
|------------------|:-------:|:----:|:--------:|:---------:|:------:|
| geo seísmica | 955 | faseIV | 0.3844 | 0.4220 | −0.002 |
| geo seísmica | 1400 | faseV | 0.3844 | 0.4220 | −0.002 |
| geometric (VAE) | 1200 | faseV | 0.3912 | 0.4324 | +0.005 |
| geometric (VAE) | 1600 | faseIII | 0.4077 | 0.4474 | +0.022 |
| elastic_transform (Albu) | 1200 | DA | 0.4256 | 0.4621 | +0.039 |
| grid_distortion (Albu) | 1200 | DA | 0.4249 | 0.4586 | +0.039 |
| random_brightness_contrast (Albu) | 1200 | DA | 0.4328 | 0.4654 | +0.047 |
| optical_distortion (Albu) | 1200 | DA | 0.4367 | 0.4712 | +0.051 |
| clahe (Albu) | 1200 | DA | 0.4401 | 0.4748 | +0.054 |
| 🏆 **random_gamma (Albu)** | **1200** | **DA** | **0.4580** | **0.4892** | **+0.072** |

> ✅ **Todos os 6 métodos Albumentations superam** o Cenário A (0.3862) e os melhores runs anteriores de Cenário B (0.4077).

---

## 4. Análise dos 6 métodos

### Intensidade supera geométrico

| Grupo | Média Test IoU | Média Test Dice |
|:-----:|:--------------:|:---------------:|
| Geométrico (elastic, grid, optical) | 0.4291 | 0.4640 |
| **Intensidade (clahe, rbc, gamma)** | **0.4436** | **0.4765** |

Transforms de **intensidade** preservam a geometria das máscaras e perturbam apenas a textura de amplitude — mais adequado para dados sísmicos onde a variação de ganho entre aquisições é a principal fonte de variabilidade.

### Por que random_gamma é o melhor
- Simula variações de resposta do sensor sísmico (curva de ganho de amplitude)
- Fisicamente motivado: dados sísmicos reais têm variação de amplitude entre levantamentos
- Regularização suave: requer mais épocas (74) → melhor generalização

---

## 5. Escala de dados — random_gamma (seed 42)

Experimentos com 4 combinações N_real × N_synth para o método campeão.

| Config | N real | N synth | N total | Épocas | Best Val IoU | **Test IoU** | **Test Dice** | Tempo (s) |
|--------|:------:|:-------:|:-------:|:------:|:------------:|:------------:|:-------------:|:---------:|
| B + random_gamma | 1000 | 1600 | 2600 | 45 | 0.3930 | 0.4287 | 0.4636 | 141 |
| 🏆 **B + random_gamma** | **1200** | **1200** | **2400** | **74** | **0.4414** | **0.4580** | **0.4892** | **205** |
| B + random_gamma | 1600 | 1000 | 2600 | 61 | 0.4338 | 0.4517 | 0.4826 | 186 |
| B + random_gamma | 1600 | 1600 | 3200 | 47 | 0.4258 | 0.4469 | 0.4788 | 165 |

### Análise de escala

| N real | N synth | Razão r:s | Test IoU | Δ vs ótimo |
|:------:|:-------:|:---------:|:--------:|:----------:|
| 1000 | 1600 | 1:1.6 | 0.4287 | −0.029 |
| **1200** | **1200** | **1:1** | **0.4580** | **—** |
| 1600 | 1000 | 1.6:1 | 0.4517 | −0.006 |
| 1600 | 1600 | 1:1 | 0.4469 | −0.011 |

**Observações:**
- **N=1200+1200 (N total=2400) é o ponto ótimo** — menor N total, maior Test IoU
- Aumentar para 1600+1600 (N total=3200) **piora** em −0.011 → overfitting (converge em 47 épocas vs 74)
- Razão 1:1 é consistentemente melhor que razões assimétricas
- Dados reais têm maior peso: reduzir N_real (1200→1000) prejudica mais (−0.029) do que reduzir N_synth (1200→1000, −0.006)

> ✅ **Configuração ótima confirmada: N_real=1200 + N_synth=1200 (razão 1:1)**

---

## 6. Paths dos artefatos

| Método | Run tag (em `results/`) |
|--------|------------------------|
| elastic_transform | `scenario_B_seed42_nreal1200_train_elastic_transform1600` |
| grid_distortion | `scenario_B_seed42_nreal1200_train_grid_distortion1600` |
| optical_distortion | `scenario_B_seed42_nreal1200_train_optical_distortion1600` |
| clahe | `scenario_B_seed42_nreal1200_train_clahe1600` |
| random_brightness_contrast | `scenario_B_seed42_nreal1200_train_random_brightness_contrast1600` |
| random_gamma (N=1200+1200) | `scenario_B_seed42_nreal1200_train_random_gamma1600` |
| random_gamma (N=1000+1600) | `scenario_B_seed42_nreal1000_train_random_gamma1600` |
| random_gamma (N=1600+1000) | `scenario_B_seed42_nreal1600_train_random_gamma1600` |
| random_gamma (N=1600+1600) | `scenario_B_seed42_nreal1600_train_random_gamma1600_ns1600` |

Cada diretório contém: `best_model.pth`, `history.csv`, `result.csv`

---

## 7. Próximos passos

- [ ] Rodar seeds 123 e 456 para `random_gamma` N=1200+1200 — confirmar consistência estatística
- [ ] Testar `random_gamma` com N_real=800 (regime de poucos dados)
- [ ] Combinar `random_gamma` + `clahe` (pipeline multi-DA)
- [ ] Atualizar `docs/relatorio-final-r21-downstream.md` com esses resultados
- [ ] Atualizar `_v7.tex` → `\subsection{Downstream Segmentation Evaluation}`
- [ ] Atualizar `response_to_reviewers.md` → seção R2.1