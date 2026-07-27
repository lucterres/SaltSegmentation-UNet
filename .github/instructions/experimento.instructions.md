---
applyTo: "Salt-Segmentation-UNet/**/*.py"
---

# Instruções — Experimentos & Datasets

**Manuscrito:** Access-2026-27912 | **Última atualização:** 2026-07-27

---

## Cenários definidos

| Cenário | Dados de treino | Objetivo |
|---------|----------------|----------|
| A | Real only | Baseline |
| B | Real + Sintético | Hipótese do paper |

---

## Datasets disponíveis no servidor

| Dataset | Path (servidor) | Amostras | Descrição |
|---------|----------------|----------|-----------|
| TGS completo | `/var/tmp/cym7/datasets/tgs-salt/train/` | 3998 | Dataset original |
| subset_split/train_filtered | `/var/tmp/cym7/datasets/subset_split/train_filtered/` | 1293 | Filtrado 10–90% |
| **subset_split/test** | `/var/tmp/cym7/datasets/subset_split/test/` | 800 | **Test canônico** (dist. real) |
| subset_1_99 | `/var/tmp/cym7/datasets/subset_1_99/` | 2209 | Filtrado 1–99% (**melhor resultado**) |
| subset_10_90 | `/var/tmp/cym7/datasets/subset_10_90/` | 1616 | Filtrado 10–90% |
| Sintéticos sísmicos | `dataset/geometric1600_seismic/pairs1600_seismic/` | 955 | Melhor pool sintético |
| Sintéticos geométricos | `dataset/geometric1600/pairs1600/` | 1600 | Pool geométrico |

> ⚠️ Sempre usar `--test_dir /var/tmp/cym7/datasets/subset_split/test` para comparabilidade.

---

## Argumentos do `train.py`

| Argumento | Descrição |
|-----------|-----------|
| `--scenario A\|B` | Cenário de treino |
| `--seed <int>` | Semente de reprodutibilidade (42, 123, 456) |
| `--epochs <int>` | Número máximo de épocas (padrão: 100) |
| `--n_real <int>` | Número de amostras reais |
| `--n_synth <int>` | Número de amostras sintéticas |
| `--train_dir <path>` | Pasta com `images/` e `masks/` para treino (sobrescreve `TGS_PATH`) |
| `--test_dir <path>` | Pasta com `images/` e `masks/` para test set fixo externo |

---

## Resultados de referência (test canônico — 800 amostras reais)

| Dataset treino | N treino | Test IoU | Test Dice |
|:--------------:|:--------:|:--------:|:---------:|
| TGS completo | 3198 | 0.4312 | 0.4657 |
| `subset_10_90` (10–90%) | 1616 | 0.4590 | 0.4860 |
| **`subset_1_99` (1–99%)** | **2209** | **0.4791** | **0.5058** |
| `train_filtered` (10–90%) | 1293 | 0.4201 | 0.4553 |
| B + 955 sísmicos (train_filtered) | 1293+955 | 0.4308 | 0.4672 |

---

## Ranking de datasets de treino

| Dataset treino | Filtro | N treino | Test IoU |
|:--------------:|:------:|:--------:|:--------:|
| **`subset_1_99`** | **1–99%** | **2209** | **0.4791 ✅** |
| `subset_10_90` | 10–90% | 1616 | 0.4590 |
| TGS completo | nenhum | 3198 | 0.4312 |
| `train_filtered` | 10–90% | 1293 | 0.4201 |

---

## Efeito dos sintéticos

| Config | Test IoU | Δ vs A |
|:------:|:--------:|:------:|
| A puro | 0.4247 | — |
| B + 955 sísmicos (TGS completo) | 0.4204 | −0.004 |
| **B + 955 sísmicos (train_filtered)** | **0.4308** | **+0.011 ✅** |
| B + 400 originais | 0.4127 | −0.012 |
| B + 1600 geométricos | 0.4070 | −0.018 |

---

## ⚠️ Métricas incomparáveis entre si

| Test set | Test IoU | Comparável? |
|:--------:|:--------:|:-----------:|
| filtrado interno subset_10_90 (~293) | 0.8340 | ❌ |
| filtrado interno subset_1_99 (~442) | 0.7662 | ❌ |
| **canônico (800 reais)** | 0.41–0.48 | **✅** |

---

## Pendências

- [ ] `subset_1_99` seeds 123 e 456
- [ ] Cenário B com `subset_1_99` + 955 sísmicos (seeds 42, 123, 456)
- [ ] Atualizar `_v7.tex` e `response_to_reviewers.md` (R2.1)
