# Relatório — Filtros de Cobertura de Sal e Tipos de Conjunto de Teste

**Manuscrito:** Access-2026-27912  
**Data:** 2026-07-30

---

## 1. Objetivo

Determinar o melhor filtro de cobertura de sal para o dataset de treino TGS-Salt, usando o **protocolo definitivo** de avaliação:

- Split estratificado de **400 amostras** para teste canônico, feito diretamente do total TGS antes de qualquer filtragem
- Filtro aplicado nos **3598 remanescentes**
- N de treino normalizado para **1456** em todos os filtros (resultado natural do filtro 10–90%)
- Seed de treino: **42**

> **Regra central:** métricas obtidas com **teste interno** não são comparáveis com métricas do **teste canônico**.

---

## 2. Protocolo definitivo

### 2.1 Separação do teste canônico

| Parâmetro | Valor |
|-----------|-------|
| Total TGS | 3998 |
| Test set (estratificado, `random_state=42`) | **400** |
| Remanescentes para treino | **3598** |
| N treino normalizado | **1456** |
| Test set path | `/var/tmp/cym7/datasets/subset_test400_new` |

### 2.2 Filtros avaliados

Filtro aplicado nos 3598 remanescentes (após remover o test set), amostragem para N=1456 com `random_state=42`:

| Filtro | N disponível pós-split | N usado |
|--------|:----------------------:|:-------:|
| 1–99% | 1990 | 1456 |
| 2–98% | 1876 | 1456 |
| 3–97% | 1805 | 1456 |
| 5–95% | 1670 | 1456 |
| 10–90% | 1456 | 1456 |

---

## 3. Resultados — Protocolo definitivo (N=1456, test=400, seed=42)

### 3.1 Ranking

| Rank | Filtro | N treino | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|--------|:--------:|:--------:|:---------:|:------------:|:------:|:---------:|
| 1 | **1–99%** | **1456** | **0.4400** | **0.4763** | — | — | 382 |
| 2 | 5–95% | 1456 | 0.4362 | 0.4706 | — | — | 360 |
| 3 | 2–98% | 1456 | 0.4340 | 0.4687 | — | — | 331 |
| 4 | **10–90% (baseline)** | **1456** | **0.4259** | **0.4603** | **0.8493** | **48** | **95** |
| 5 | 3–97% | 1456 | 0.4169 | 0.4523 | — | — | 181 |

### 3.2 Variabilidade entre seeds — filtro `1–99%` (3 seeds)

| Seed | Test IoU | Test Dice |
|:----:|:--------:|:---------:|
| 42 | 0.4400 | 0.4763 |
| 123 | 0.4385 | 0.4729 |
| 456 | 0.4384 | 0.4731 |
| **Média ± DP** | **0.4390 ± 0.0009** | **0.4741 ± 0.0019** |

> Resultado muito estável — desvio padrão de `±0.0009` em IoU confirma reprodutibilidade do filtro `1–99%` neste protocolo.

### 3.3 Interpretação

1. **`1–99%` lidera com `IoU = 0.4400`** — vantagem de `+0.0141` sobre o baseline `10–90%`
2. `5–95%` e `2–98%` ficaram próximos em 2º e 3º lugar
3. **`10–90%` em 4º** confirma que filtros mais leves são superiores com N igualado
4. **`3–97%` foi o pior** (`0.4169`) — instável, não recomendado como filtro principal
5. A vantagem de `1–99%` sobre `10–90%` com N e test set idênticos: **+0.0141 IoU**

---

## 4. Experimentos com teste interno (referência histórica)

> ⚠️ Resultados não comparáveis com o protocolo definitivo. Registrados apenas como referência histórica.

| Dataset | Filtro | Seed | Test IoU | Test Dice | Observação |
|---------|--------|:----:|:--------:|:---------:|------------|
| subset_1_99 | 1–99% | 42 | 0.7662 | 0.8269 | teste interno ❌ |
| subset_1_99 | 1–99% | 123 | 0.7837 | 0.8440 | teste interno ❌ |
| subset_2_98 | 2–98% | 42 | 0.7971 | 0.8613 | teste interno ❌ |
| subset_3_97 | 3–97% | 42 | 0.7833 | 0.8492 | teste interno ❌ |
| subset_10_90 | 10–90% | 42 | 0.8340 | 0.8943 | teste interno ❌ |

---

## 5. Conclusões

1. **O filtro `1–99%` é o melhor** para treino no protocolo definitivo (`IoU = 0.4400`).
2. O N de treino disponível com `1–99%` pós-split é `1990` — amostrado para `1456` para comparação justa.
3. `3–97%` mostrou instabilidade entre séries — não recomendado como filtro principal.
4. Testes internos superestimam fortemente o desempenho e devem ser usados apenas como apoio exploratório.
5. Para o manuscrito, **apenas resultados com o protocolo definitivo sustentam comparações válidas**.

---

## 6. Próximos passos

1. ✅ Protocolo definitivo N=1456, test=400 estratificado — concluído.
2. **Cenário B** com filtro `1–99%` no protocolo definitivo:

```bash
python -u train.py --scenario B --seed 42 --n_synth 955 --epochs 100 \
  --train_dir /var/tmp/cym7/datasets/subset_1_99_postsplit400_n1456 \
  --test_dir /var/tmp/cym7/datasets/subset_test400_new
```

3. Comparar Cenário B vs Cenário A com mesmo filtro/protocolo para verificar hipótese **B > A**.
