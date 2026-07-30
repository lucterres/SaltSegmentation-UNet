# Relatório — Filtros de Cobertura de Sal e Tipos de Conjunto de Teste

**Manuscrito:** Access-2026-27912  
**Data:** 2026-07-29

---

## 1. Objetivo

Consolidar os experimentos realizados nesta sessão com filtros de cobertura de sal no dataset TGS-Salt, destacando a diferença entre dois protocolos de avaliação:

- **Teste canônico** — conjunto externo fixo com 800 amostras reais (`subset_split/test`)
- **Teste interno** — split 80/20 gerado a partir do próprio dataset filtrado

> **Regra central:** métricas obtidas com **teste interno** não são comparáveis com métricas do **teste canônico**.

---

## 2. Definição dos protocolos de teste

### 2.1 Teste canônico

- Origem: `/var/tmp/cym7/datasets/subset_split/test`
- Tamanho: **800 amostras reais**
- Distribuição: próxima da distribuição real do problema
- Uso: comparação principal do paper
- Vantagem: permite comparação justa entre datasets e cenários

### 2.2 Teste interno

- Origem: split 80/20 do próprio dataset filtrado
- Tamanho: depende do filtro aplicado
- Distribuição: mesma distribuição do treino filtrado
- Uso: exploração rápida / diagnóstico interno
- Limitação: tende a produzir métricas infladas

---

## 3. Experimentos com teste canônico

### 3.1 Tabela consolidada

| Dataset | Filtro de cobertura de sal | Cenário | Seed | N treino | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|---------|----------------------------|:-------:|:----:|:--------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|
| TGS completo (GEM Desktop) | nenhum | A | 42 | 3198 | 0 | 0.4417 | 0.4270 | 0.4609 | 50 | 17608.7 |
| **subset_1_99** | **1–99%** | **A** | **42** | **2209** | **0** | **0.7826** | **0.4791** | **0.5058** | **55** | **142.4** |
| **subset_2_98** | **2–98%** | **A** | **42** | **2080** | **0** | **0.8063** | **0.4761** | **0.5034** | **49** | **116.0** |
| train_filtered | 10–90% | A | 42 | 1293 | 0 | 0.8517 | 0.4201 | 0.4553 | 56 | 213.7 |
| train_filtered | 10–90% | A | 123 | 1293 | 0 | 0.8507 | 0.4286 | 0.4621 | 54 | 208.6 |
| train_filtered | 10–90% | A | 456 | 1293 | 0 | 0.8371 | 0.4223 | 0.4559 | 38 | 164.9 |
| train_filtered + sísmicos | 10–90% | B | 42 | 1293 | 955 | 0.8514 | 0.4308 | 0.4672 | 47 | 121.9 |

---

### 3.2 Ranking no teste canônico

| Rank | Dataset | Filtro | Test IoU | Test Dice | Observação |
|------|---------|--------|----------|-----------|------------|
| 1 | **subset_1_99** | **1–99%** | **0.4791** | **0.5058** | melhor resultado desta sessão ✅ |
| 2 | subset_2_98 | 2–98% | 0.4761 | 0.5034 | muito próximo do melhor |
| 3 | train_filtered + sísmicos | 10–90% | 0.4308 | 0.4672 | melhor B observado |
| 4 | train_filtered (seed 123) | 10–90% | 0.4286 | 0.4621 | melhor seed do train_filtered |
| 5 | TGS completo | nenhum | 0.4270 | 0.4609 | baseline de referência |
| 6 | train_filtered (seed 456) | 10–90% | 0.4223 | 0.4559 | abaixo do baseline |
| 7 | train_filtered (seed 42) | 10–90% | 0.4201 | 0.4553 | abaixo do baseline |

---

### 3.3 Interpretação

1. **Filtros leves funcionam melhor**: `subset_1_99` e `subset_2_98` superam claramente o TGS completo.
2. **O melhor resultado observado foi o filtro `1–99%`** com `Test IoU = 0.4791`.
3. **O filtro `2–98%` ficou extremamente próximo** (`0.4761`), sugerindo robustez da ideia de remover extremos de cobertura.
4. **Filtros mais agressivos (10–90%) perdem generalização** no teste canônico.

---

## 4. Experimentos com teste interno

### 4.1 Tabela consolidada

| Dataset | Filtro de cobertura de sal | Seed | N total | N treino pool | N treino final | N val | N test | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|---------|----------------------------|:----:|:-------:|:-------------:|:--------------:|:-----:|:------:|:------------:|:--------:|:---------:|:------:|:---------:|
| subset_1_99 | 1–99% | 123 | ~2209 | 1767 | — | — | ~442 | 0.7852 | 0.7837 | 0.8440 | 45 | 91.3 |
| subset_2_98 | 2–98% | 42 | 2080 | 1664 | 1497 | 167 | 416 | 0.8216 | 0.7971 | 0.8613 | 59 | 114.8 |
| subset_3_97 | 3–97% | 42 | 1996 | 1596 | 1436 | 160 | 400 | 0.8043 | 0.7833 | 0.8492 | 40 | 79.5 |
| subset_10_90 (`salt10-90_1600`) | 10–90% | 42 | ~1616 | ~1292 | — | — | ~323 | 0.8069 | 0.8340 | 0.8943 | 44 | 183.4 |

> Observação: em teste interno, `n_real` registrado no `result.csv` corresponde ao tamanho do **train pool** após a separação treino/teste, não ao total bruto do dataset filtrado.

---

### 4.2 Interpretação

1. **Os valores de IoU interno são muito mais altos** (`0.78–0.83`) do que no teste canônico (`0.42–0.48`).
2. Isso ocorre porque treino e teste vêm da **mesma distribuição filtrada**, tornando a tarefa mais fácil.
3. O run `seed123_subset_1_99` (`best_val_iou = 0.7852`, `test_iou = 0.7837`) é consistente com **teste interno**, pois o desempenho de teste ficou praticamente igual ao de validação.
4. À medida que o filtro foi apertado de `2–98%` para `3–97%`, o IoU interno caiu de `0.7971` para `0.7833`, indicando perda de diversidade útil.
5. Mesmo quando o teste interno mostra números muito altos, isso **não implica melhor desempenho no cenário real**.

---

## 5. Comparação entre teste canônico e teste interno

| Dataset | Filtro | Tipo de teste | Test IoU | Test Dice | Comparável ao paper? |
|---------|--------|---------------|----------|-----------|----------------------|
| subset_1_99 | 1–99% | canônico | 0.4791 | 0.5058 | ✅ |
| subset_1_99 | 1–99% | interno | 0.7837 | 0.8440 | ❌ |
| subset_2_98 | 2–98% | canônico | 0.4761 | 0.5034 | ✅ |
| subset_2_98 | 2–98% | interno | 0.7971 | 0.8613 | ❌ |
| subset_3_97 | 3–97% | interno | 0.7833 | 0.8492 | ❌ |

### Diferença crítica

Para o mesmo conceito de filtro (`subset_2_98`), o IoU muda de:

- **0.4761** no **teste canônico**
- **0.7971** no **teste interno**

No caso de `subset_1_99`, a diferença também é grande:

- **0.4791** no **teste canônico** (`seed42_subset_1_99`)
- **0.7837** no **teste interno** (`seed123_subset_1_99`)

Isso mostra que o tipo de conjunto de teste altera drasticamente a interpretação do experimento.

---

## 6. Conclusões desta sessão

1. **O melhor filtro observado no teste canônico foi `1–99%`** (`IoU = 0.4791`).
2. **O filtro `2–98%` ficou muito próximo** (`IoU = 0.4761`) e confirma que remover extremos ajuda.
3. **`subset_1_99` possui também um run interno (`seed123`)**, com `IoU = 0.7837`, que não deve ser comparado ao ranking principal.
4. **O filtro `3–97%` só foi avaliado com teste interno**, portanto não pode entrar no ranking principal do paper.
5. **Testes internos superestimam fortemente o desempenho** e devem ser usados apenas como apoio exploratório.
6. Para o manuscrito, **apenas o teste canônico deve sustentar comparações e conclusões principais**.

---

## 7. Próximos passos recomendados

1. Rodar `subset_1_99` com seed `456` no **teste canônico**.
2. Registrar explicitamente nos nomes de pasta ou no relatório quando um run for **interno** vs **canônico**.
3. Rodar `subset_2_98` com seeds adicionais apenas se for necessário estimar variabilidade.
4. Priorizar o experimento principal do paper:

```bash
python -u train.py --scenario B --seed 42 --n_synth 955 --epochs 100 \
  --train_dir /var/tmp/cym7/datasets/subset_1_99 \
  --test_dir /var/tmp/cym7/datasets/subset_split/test
```

Esse é o experimento mais relevante para verificar a hipótese **B > A**.
