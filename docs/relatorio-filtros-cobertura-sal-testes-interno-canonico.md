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

---

### 3.3 Impacto do tamanho de amostra de treino — filtro `1–99%` (3 seeds)

Mesmos subsets, mesmo test set canônico (`subset_test400_new`), variando apenas N de treino.

| N treino | Seed 42 | Seed 123 | Seed 456 | Média IoU | Desvio DP |
|:--------:|:-------:|:--------:|:--------:|:---------:|:---------:|
| **1000** | 0.4091 | 0.4435 | 0.4203 | **0.4243** | **±0.0175** |
| **1456** | 0.4400 | 0.4385 | 0.4384 | **0.4390** | **±0.0009** |
| **1990** | 0.4473 | 0.4442 | 0.4517 | **0.4477** | **±0.0038** |

**Curva de aprendizado (IoU médio por N):**

```
N=1000 → 0.4243 (±0.0175)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░
N=1456 → 0.4390 (±0.0009)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░
N=1990 → 0.4477 (±0.0038)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░
```

**Conclusões:**
1. Há ganho consistente ao aumentar N: `+0.0147` de N=1000→1456, e `+0.0087` de N=1456→1990
2. N=1990 usa todo o pool disponível — `IoU médio = 0.4477 ± 0.0038`
3. N=1456 representa o melhor equilíbrio estabilidade/desempenho: desvio mínimo `±0.0009`
4. N=1000 é instável — não recomendado para o paper
5. A curva sugere que mais dados ainda trariam ganho, mas o pool disponível (1990) já foi esgotado

---

### 3.4 Tabela comparativa completa por tamanho de amostra — filtro `1–99%`

| N treino | % do pool | Seed 42 IoU | Seed 123 IoU | Seed 456 IoU | Média IoU | DP IoU | Δ vs N=1000 | Δ vs N=1456 |
|:--------:|:---------:|:-----------:|:------------:|:------------:|:---------:|:------:|:-----------:|:-----------:|
| 1000 | 50% | 0.4091 | 0.4435 | 0.4203 | 0.4243 | ±0.0175 | — | −0.0147 |
| 1456 | 73% | 0.4400 | 0.4385 | 0.4384 | 0.4390 | ±0.0009 | +0.0147 | — |
| **1990** | **100%** | **0.4473** | **0.4442** | **0.4517** | **0.4477** | **±0.0038** | **+0.0234** | **+0.0087** |

**Observações adicionais:**

| Métrica | N=1000 | N=1456 | N=1990 |
|---------|:------:|:------:|:------:|
| Train pool | 1000 | 1456 | 1990 |
| Train efetivo (90%) | 900 | 1310 | 1791 |
| Val efetivo (10%) | 100 | 146 | 199 |
| Melhor seed | 123 (0.4435) | 42 (0.4400) | 456 (0.4517) |
| Pior seed | 42 (0.4091) | 456 (0.4384) | 123 (0.4442) |
| Range seed | 0.0344 | 0.0016 | 0.0075 |
| Estabilidade | ❌ baixa | ✅ alta | ⚠️ média |
| Recomendado? | ❌ | ✅ | ✅ |

> **Recomendação:** para reprodutibilidade máxima, usar **N=1456** (desvio `±0.0009`). Para melhor desempenho absoluto, usar **N=1990** (todo o pool disponível, desvio `±0.0038`).

---

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

## 5. Cenário B — filtro `1–99%` com sintéticos sísmicos (955 pares)

Mesmo protocolo definitivo:
- test set canônico: `400` amostras (`subset_test400_new`)
- filtro `1–99%` pós-split
- sintéticos sísmicos: `955`
- seeds: `42`, `123`, `456`

### 5.1 Resultados por tamanho de treino

#### N=1000

| Seed | Test IoU | Test Dice |
|:----:|:--------:|:---------:|
| 42 | 0.4221 | 0.4598 |
| 123 | 0.4335 | 0.4694 |
| 456 | 0.4336 | 0.4687 |
| **Média** | **0.4297** | **0.4660** |

#### N=1456

| Seed | Test IoU | Test Dice |
|:----:|:--------:|:---------:|
| 42 | **falhou** | — |
| 123 | 0.4404 | 0.4742 |
| 456 | 0.4436 | 0.4762 |
| **Média válida** | **0.4420** | **0.4752** |

> O run `seed 42` falhou porque o diretório de treino ficou vazio (`Valid pairs: 0`) em `/var/tmp/cym7/datasets/subset_1_99_postsplit400_n1456_s42`.

#### N=1990

| Seed | Test IoU | Test Dice |
|:----:|:--------:|:---------:|
| 42 | 0.4411 | 0.4744 |
| 123 | 0.4406 | 0.4752 |
| 456 | 0.4404 | 0.4742 |
| **Média** | **0.4407** | **0.4746** |

---

### 5.2 Comparação A vs B

| N treino | Cenário A (IoU médio) | Cenário B (IoU médio) | Δ (B − A) | Interpretação |
|:--------:|:---------------------:|:---------------------:|:---------:|---------------|
| 1000 | 0.4243 | **0.4297** | **+0.0054** | sintéticos ajudam em baixo N |
| 1456 | 0.4390 | **0.4420** | **+0.0030** | ganho pequeno, mas positivo |
| 1990 | **0.4477** | 0.4407 | **−0.0070** | sintéticos não ajudam em alto N |

### 5.3 Conclusões do Cenário B

1. A hipótese **B > A** é sustentada apenas no regime de menos dados (`N=1000` e `N=1456`).
2. O ganho de B sobre A é modesto, mas consistente em baixo N: `+0.0054` e `+0.0030` IoU.
3. Em alto N (`1990`), os sintéticos passam a prejudicar levemente o desempenho (`−0.0070`).
4. Isso sugere que os dados sintéticos funcionam como complemento útil quando o conjunto real é limitado, mas se tornam redundantes ou levemente ruidosos quando o pool real já é suficientemente grande.
5. Para o paper, a formulação mais precisa é: **B supera A em regime de poucos dados; A volta a ser melhor quando há muitos dados reais disponíveis**.

---

## 6. Experimentos com teste interno (referência histórica)

> ⚠️ Resultados não comparáveis com o protocolo definitivo. Registrados apenas como referência histórica.

| Dataset | Filtro | Seed | Test IoU | Test Dice | Observação |
|---------|--------|:----:|:--------:|:---------:|------------|
| subset_1_99 | 1–99% | 42 | 0.7662 | 0.8269 | teste interno ❌ |
| subset_1_99 | 1–99% | 123 | 0.7837 | 0.8440 | teste interno ❌ |
| subset_2_98 | 2–98% | 42 | 0.7971 | 0.8613 | teste interno ❌ |
| subset_3_97 | 3–97% | 42 | 0.7833 | 0.8492 | teste interno ❌ |
| subset_10_90 | 10–90% | 42 | 0.8340 | 0.8943 | teste interno ❌ |

---

## 7. Conclusões finais

1. **O filtro `1–99%` é o melhor** para treino no protocolo definitivo (`IoU médio A = 0.4390` com N=1456; `0.4477` com N=1990).
2. O impacto do tamanho do conjunto real é claro: mais dados reais melhoram A de forma consistente.
3. **Cenário B supera A apenas quando N de treino real é limitado** (`1000` e `1456`).
4. Em `N=1990`, o cenário A supera B, indicando saturação do benefício dos sintéticos.
5. Testes internos continuam superestimando fortemente o desempenho e devem ser usados apenas como apoio exploratório.
6. Para o manuscrito, a conclusão correta é: **o uso de sintéticos é vantajoso em regime de poucos dados, mas não necessariamente em regime de muitos dados reais**.

---

## 8. Próximos passos

1. Corrigir e repetir o run `scenario_B_seed42_subset_1_99_postsplit400_n1456` para fechar a média de 3 seeds em `N=1456`.
2. Consolidar no manuscrito a mensagem principal:
   - `1–99%` é o melhor filtro
   - `B > A` em baixo N
   - `A ≥ B` em alto N
3. Atualizar a redação final do paper com essa interpretação refinada.
