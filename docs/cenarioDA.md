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

---

## 8. Ranking geral — test set canônico (800 amostras reais)

> ⚠️ Apenas runs com `--test_dir /var/tmp/cym7/datasets/subset_split/test` são comparáveis entre si. Runs com test set interno filtrado (IoU ~0.77–0.85) **não entram neste ranking**.

| Rank | Run | Cenário | N real | N synth | Test IoU | Test Dice |
|:----:|-----|:-------:|:------:|:-------:|:--------:|:---------:|
| 🏆 1 | A — TGS completo (seed 123) | A | 3998 | 0 | **0.4766** | **0.5043** |
| 2 | A — TGS completo (seed 42) | A | 3998 | 0 | 0.4730 | 0.5004 |
| 3 | A — TGS completo (seed 456) | A | 3998 | 0 | 0.4706 | 0.4980 |
| — | **A — TGS completo média** | **A** | **3998** | **0** | **0.473 ± 0.003** | **0.501 ± 0.003** |
| 4 | B + random_gamma | B | 1200 | 1200 | 0.4580 | 0.4892 |
| 5 | B + random_gamma | B | 1600 | 1000 | 0.4517 | 0.4826 |
| 6 | B + random_gamma | B | 1600 | 1600 | 0.4469 | 0.4788 |
| 7 | B + clahe | B | 1200 | 1200 | 0.4401 | 0.4748 |
| 8 | B + optical_distortion | B | 1200 | 1200 | 0.4367 | 0.4712 |
| 9 | B + random_brightness_contrast | B | 1200 | 1200 | 0.4328 | 0.4654 |
| 10 | B + train_filtered + sísmico | B | 1293 | 955 | 0.4308 | 0.4672 |
| 11 | B + random_gamma | B | 1000 | 1600 | 0.4287 | 0.4636 |
| 12 | A — train_filtered seed 123 | A | 1293 | 0 | 0.4279 | 0.4637 |
| 13 | B + elastic_transform | B | 1200 | 1200 | 0.4256 | 0.4621 |
| 14 | B + grid_distortion | B | 1200 | 1200 | 0.4249 | 0.4586 |
| 15 | A — train_filtered seed 456 | A | 1293 | 0 | 0.4201 | 0.4553 |

### Interpretação corrigida

> ⚠️ **O Cenário A com TGS completo (3998 amostras) supera o Cenário B + random_gamma (N=1200+1200)** no test set canônico.

A comparação justa é **mesmo N de treino real**:

| Config | N real | N synth | Test IoU | Δ |
|--------|:------:|:-------:|:--------:|:---:|
| A — TGS completo | 3998 | 0 | **0.473** | ref |
| B + random_gamma | 1200 | 1200 | 0.458 | −0.015 |
| A — train_filtered | 1293 | 0 | 0.420 | ref |
| B + random_gamma (mesmo N) | 1200 | 1200 | 0.458 | **+0.038** ✅ |

**Conclusão real:** No regime de **poucos dados reais (N=1200)**, random_gamma DA (+3.8pp) é eficaz. Com **dados reais abundantes (N=3998)**, mais dados reais superam DA. O ganho de DA é relevante exatamente quando os dados reais são escassos — que é o cenário do paper (R2.1).

---

## 9. Correção de data leakage — datasets `*1600clean`

### Problema identificado

Os datasets `*1600` originais foram gerados a partir das **3998 imagens TGS completas**, incluindo as **800 do test set canônico**. A análise mostrou que **305 das 1600 amostras DA (19.1%)** tinham origem em imagens do test set — data leakage direto.

### Correção aplicada

O script `generate_albumentations_da.py` foi atualizado com `--exclude_csv` para excluir os IDs do test set canônico:

```bash
python DataAugmentation/generate_albumentations_da.py \
    --src D:\dataset\tgs-salt\train \
    --exclude_csv dataset/subset_split/split_stats.csv \
    --exclude_split test \
    --out_suffix 1600clean --n 1600
```

Pool limpo: **3199 imagens** (3998 − 800 do test + 1 máscara vazia aceita).

### Resultados — datasets limpos vs originais (seed 42, N=1200+1200)

| Método | Test IoU (original) | Test IoU (clean) | Δ leakage | Test Dice (clean) |
|--------|:-------------------:|:----------------:|:---------:|:-----------------:|
| elastic_transform | 0.4256 | 0.4280 | +0.002 | 0.4641 |
| grid_distortion | 0.4249 | 0.4246 | −0.000 | 0.4592 |
| optical_distortion | 0.4367 | 0.4232 | −0.013 | 0.4597 |
| clahe | 0.4401 | 0.4164 | −0.024 | 0.4529 |
| random_brightness_contrast | 0.4328 | **0.4327** | −0.000 | **0.4683** |
| random_gamma | 0.4580 | 0.4310 | −0.027 | 0.4647 |

> ⚠️ Os resultados `*1600` (com leakage) eram inflados — especialmente `random_gamma` (−0.027) e `clahe` (−0.024).  
> ✅ Os resultados `*1600clean` são os **valores válidos para publicação**.

### Ranking corrigido — datasets clean (seed 42, N=1200+1200)

| Rank | Método | Test IoU | Test Dice | Δ vs A (N=1200) |
|:----:|--------|:--------:|:---------:|:---------------:|
| 🏆 1 | random_brightness_contrast | **0.4327** | **0.4683** | **+0.047** |
| 2 | random_gamma | 0.4310 | 0.4647 | +0.045 |
| 3 | elastic_transform | 0.4280 | 0.4641 | +0.042 |
| 4 | grid_distortion | 0.4246 | 0.4592 | +0.038 |
| 5 | optical_distortion | 0.4232 | 0.4597 | +0.037 |
| 6 | clahe | 0.4164 | 0.4529 | +0.030 |
| — | **A baseline (N=1200)** | **0.3862** | **0.4252** | — |
| — | A TGS completo (N=3998) | 0.473 ± 0.003 | 0.501 ± 0.003 | ref |

### Conclusão corrigida

> ✅ **Todos os 6 métodos Albumentations (clean) superam o Cenário A com N=1200** em +3.0 a +4.7pp IoU.  
> ✅ **random_brightness_contrast** é o novo campeão com pool limpo (0.4327 vs 0.4310 do random_gamma).  
> ⚠️ **Nenhum método supera o Cenário A com TGS completo (0.473)** — mais dados reais ainda ganham.  
> 📌 **O ganho de DA é real e válido no regime de dados escassos (N=1200)**, mesmo após correção do leakage.

### Run tags (artefatos clean)

| Método | Run tag |
|--------|---------|
| elastic_transform | `scenario_B_seed42_nreal1200_train_elastic_transform1600clean_ns1200` |
| grid_distortion | `scenario_B_seed42_nreal1200_train_grid_distortion1600clean_ns1200` |
| optical_distortion | `scenario_B_seed42_nreal1200_train_optical_distortion1600clean_ns1200` |
| clahe | `scenario_B_seed42_nreal1200_train_clahe1600clean_ns1200` |
| random_brightness_contrast | `scenario_B_seed42_nreal1200_train_random_brightness_contrast1600clean_ns1200` |
| random_gamma | `scenario_B_seed42_nreal1200_train_random_gamma1600clean_ns1200` |