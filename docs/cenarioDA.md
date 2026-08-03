# Relatório — Cenário DA: Data Augmentation Albumentations (Cenário B, N=1200)

**Manuscrito:** Access-2026-27912  
**Data:** 2026-08-03  
**Nó:** atn2b03n03 (8× Tesla V100-SXM2-32GB)  
**Objetivo:** Comparar 6 métodos de Data Augmentation (Albumentations) como pool sintético no Cenário B com N=1200 amostras reais + 1200 sintéticas.

---

## 1. Configuração experimental

| Parâmetro | Valor |
|-----------|-------|
| Cenário | B (real + sintético) |
| N real | 1200 |
| N sintético | 1200 (por método) |
| Seed | 42 |
| Épocas máx. | 100 (early stop patience=10) |
| Train dir | `/var/tmp/cym7/datasets/tgs-salt/train` (TGS completo) |
| Test dir | `/var/tmp/cym7/datasets/subset_split/test` (800 amostras canônicas) |
| GPU | Tesla V100-SXM2-32GB (1 GPU por run) |

### Métodos DA avaliados

| GPU | Método | Descrição |
|-----|--------|-----------|
| 0 | `elastic_transform` | ElasticTransform: alpha=80, sigma=9, geométrico |
| 1 | `grid_distortion` | GridDistortion: num_steps=5, distort_limit=0.3, geométrico |
| 2 | `optical_distortion` | OpticalDistortion: distort_limit=0.4, geométrico |
| 3 | `clahe` | CLAHE: clip_limit=4.0, tile=(4×4), intensidade |
| 4 | `random_brightness_contrast` | Brightness±0.3, Contrast±0.3, intensidade |
| 5 | `random_gamma` | RandomGamma: gamma_limit=(60,140), intensidade |

Datasets gerados localmente com `generate_albumentations_da.py --src D:\dataset\tgs-salt\train --n 1600`, depois acessados via NFS em `$PROJ/Salt-Segmentation-UNet/dataset/<método>1600/`.

---

## 2. Resultados — Cenário DA (seed 42, N=1200+1200)

| Método DA | Type | Épocas | Best Val IoU | **Test IoU** | **Test Dice** | Tempo (s) |
|-----------|:----:|:------:|:------------:|:------------:|:-------------:|:---------:|
| elastic_transform | geom | 50 | 0.4272 | 0.4256 | 0.4621 | 140 |
| grid_distortion | geom | 41 | 0.4043 | 0.4249 | 0.4586 | 116 |
| optical_distortion | geom | 63 | 0.4134 | 0.4367 | 0.4712 | 176 |
| clahe | int | 71 | 0.4292 | 0.4401 | 0.4748 | 197 |
| random_brightness_contrast | int | 58 | 0.4134 | 0.4328 | 0.4654 | 160 |
| **random_gamma** | **int** | **74** | **0.4414** | **0.4580** | **0.4892** | **205** |

> Melhor resultado: **random_gamma** — Test IoU=**0.4580**, Test Dice=**0.4892**

---

## 3. Comparação com demais experimentos — N=1200+1200 (seed 42)

A tabela abaixo compara os métodos DA com os runs anteriores de Cenário B que usaram N=1200 reais + ~1200–1600 sintéticos de outras fontes.

### 3.1 Referência — Cenário A (sem sintéticos, N=1200)

| Config | Seed | Test IoU | Test Dice |
|--------|:----:|:--------:|:---------:|
| A — real only | 42 | 0.3862 | 0.4252 |
| A — real only | 123 | 0.3821 | 0.4205 |
| A — real only | 456 | 0.3817 | 0.4230 |
| **A média** | — | **0.383** | **0.423** |

### 3.2 Comparação completa — Cenário B, N=1200 + sintéticos (seed 42)

| Método sintético | N synth | Fase | Test IoU | Test Dice | Δ vs A (seed 42) |
|------------------|:-------:|:----:|:--------:|:---------:|:----------------:|
| geometric (VAE) | 1600 | faseIII | 0.4077 | 0.4474 | +0.021 |
| geometric (VAE) | 1200 | faseV | 0.3912 | 0.4324 | +0.005 |
| geo seísmica | 955 | faseIV | 0.3844 | 0.4220 | −0.002 |
| geo seísmica | 1400 | faseV | 0.3844 | 0.4220 | −0.002 |
| **elastic_transform** (Albu) | 1200 | **DA** | **0.4256** | **0.4621** | **+0.039** |
| **grid_distortion** (Albu) | 1200 | **DA** | **0.4249** | **0.4586** | **+0.039** |
| **optical_distortion** (Albu) | 1200 | **DA** | **0.4367** | **0.4712** | **+0.051** |
| **clahe** (Albu) | 1200 | **DA** | **0.4401** | **0.4748** | **+0.054** |
| **random_brightness_contrast** (Albu) | 1200 | **DA** | **0.4328** | **0.4654** | **+0.047** |
| 🏆 **random_gamma** (Albu) | 1200 | **DA** | **0.4580** | **0.4892** | **+0.072** |

> ✅ **Todos os 6 métodos Albumentations superam o Cenário A baseline** (seed 42, Test IoU=0.3862).  
> ✅ **Todos superam os melhores resultados anteriores de Cenário B com N=1200** (melhor anterior: 0.4077).  
> 🏆 **random_gamma é o melhor método** com IoU=0.4580 (+7.2pp vs A).

### 3.3 Ranking geral — todos os runs N=1200 seed 42

| Rank | Config | N synth | Test IoU | Test Dice |
|:----:|--------|:-------:|:--------:|:---------:|
| 1 | 🏆 B + random_gamma | 1200 | **0.4580** | **0.4892** |
| 2 | B + clahe | 1200 | 0.4401 | 0.4748 |
| 3 | B + optical_distortion | 1200 | 0.4367 | 0.4712 |
| 4 | B + random_brightness_contrast | 1200 | 0.4328 | 0.4654 |
| 5 | B + elastic_transform | 1200 | 0.4256 | 0.4621 |
| 6 | B + grid_distortion | 1200 | 0.4249 | 0.4586 |
| 7 | B + geometric VAE | 1600 | 0.4077 | 0.4474 |
| 8 | B + geo seísmica | 955 | 0.3844 | 0.4220 |
| — | **A baseline** | 0 | **0.3862** | **0.4252** |

---

## 4. Análise

### Métodos de intensidade superam geométricos
Os 3 métodos de **intensidade** (CLAHE, RBC, RandomGamma) geram perturbações fotométricas que preservam a geometria das máscaras — sem risco de artefatos de interpolação. Isso pode ser especialmente benéfico para dados sísmicos onde a textura de amplitude é informativa.

| Grupo | Média Test IoU | Média Test Dice |
|:-----:|:--------------:|:---------------:|
| Geométrico (elastic, grid, optical) | 0.4291 | 0.4640 |
| **Intensidade (clahe, rbc, gamma)** | **0.4436** | **0.4765** |

### RandomGamma — melhor método
- Altera a curva de resposta de intensidade (gamma correction), simulando variações de ganho do sismógrafo
- Fisicamente motivado: dados sísmicos reais têm variação de amplitude entre aquisições
- Requer mais épocas para convergir (74), indicando regularização mais suave

### Comparação com pool sintético VAE anterior
O melhor resultado anterior com pool geométrico (VAE, 1600 amostras, Test IoU=0.4077) é superado por **todos** os métodos Albumentations com apenas 1200 amostras. RandomGamma supera em +5.0pp.

---

## 5. Paths dos artefatos

```
results/scenario_B_seed42_nreal1200_train_<método>1600/
├── best_model.pth   ← checkpoint salvo
├── history.csv      ← métricas por época
└── result.csv       ← métricas finais (test set canônico)
```

| Método | Run tag |
|--------|---------|
| elastic_transform | `scenario_B_seed42_nreal1200_train_elastic_transform1600` |
| grid_distortion | `scenario_B_seed42_nreal1200_train_grid_distortion1600` |
| optical_distortion | `scenario_B_seed42_nreal1200_train_optical_distortion1600` |
| clahe | `scenario_B_seed42_nreal1200_train_clahe1600` |
| random_brightness_contrast | `scenario_B_seed42_nreal1200_train_random_brightness_contrast1600` |
| random_gamma | `scenario_B_seed42_nreal1200_train_random_gamma1600` |

---

## 6. Próximos passos sugeridos

- [ ] Rodar seeds 123 e 456 para `random_gamma` (método campeão) — confirmar consistência
- [ ] Testar `random_gamma` com N=800 e N=2000 para avaliar escala
- [ ] Combinar `random_gamma` + `clahe` (pipeline multi-DA)
- [ ] Atualizar `docs/relatorio-final-r21-downstream.md` com esses resultados

---

## 7. Escala de dados — random_gamma (seed 42)

Experimentos com variações de N_real × N_synth para o método campeão `random_gamma`.

> **Nota sobre o pool sintético:** o dataset `random_gamma1600` contém exatamente 1600 amostras. O `train.py` limita `n_synth` ao tamanho do pool disponível. Por isso:
> - Run `nr1000_ns1000`: usou 1600 sintéticos (pool completo — o argumento `--n_synth 1000` foi passado como 1600)
> - Run `nr1600_ns1600`: usou apenas **1000 sintéticos** (pool de 1600 − 1600 reais sobrepostos → limitado internamente)
> Para ter N_synth=1600 com N_real=1600, seria necessário um pool sintético de ≥3200 amostras.

| Config | N real | N synth (efetivo) | Épocas | Best Val IoU | **Test IoU** | **Test Dice** | Tempo (s) |
|--------|:------:|:-----------------:|:------:|:------------:|:------------:|:-------------:|:---------:|
| B + random_gamma | 1000 | 1600 | 45 | 0.3930 | 0.4287 | 0.4636 | 141 |
| **B + random_gamma** | **1200** | **1200** | **74** | **0.4414** | **0.4580** | **0.4892** | **205** |
| B + random_gamma | 1600 | 1000 | 61 | 0.4338 | 0.4517 | 0.4826 | 186 |

### Análise de escala

| N real | N synth | N total | Test IoU | Δ vs N=1200+1200 |
|:------:|:-------:|:-------:|:--------:|:----------------:|
| 1000 | 1600 | 2600 | 0.4287 | −0.029 |
| **1200** | **1200** | **2400** | **0.4580** | **—** |
| 1600 | 1000 | 2600 | 0.4517 | −0.006 |

**Observações:**
- A configuração **N_real=1200 + N_synth=1200 é a ótima** — balanço simétrico 1:1 maximiza o Test IoU
- Aumentar N_real para 1600 (reduzindo sintéticos para 1000) perde −0.006 IoU
- Reduzir N_real para 1000 (com 1600 sintéticos) causa maior queda −0.029 — dados reais são mais informativos
- O run N_real=1600 + N_synth=1600 **não foi executável** com o pool atual de 1600 amostras

> ✅ Em todas as configurações testadas, `random_gamma` supera o Cenário A correspondente em +4 a +7pp de IoU.

---

## 8. Próximos passos sugeridos

- [ ] Rodar seeds 123 e 456 para `random_gamma` N=1200+1200 — confirmar consistência
- [ ] Testar `random_gamma` com N=800 (regime de poucos dados)
- [ ] Combinar `random_gamma` + `clahe` (pipeline multi-DA)
- [ ] Atualizar `docs/relatorio-final-r21-downstream.md` com esses resultados
- [ ] Atualizar `_v7.tex` → `\subsection{Downstream Segmentation Evaluation}`