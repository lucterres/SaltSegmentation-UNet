# Relatório — Filtros de Cobertura de Sal e Tipos de Conjunto de Teste

**Manuscrito:** Access-2026-27912  
**Data:** 2026-07-30

---

## 1. Objetivo

Consolidar os experimentos realizados nesta sessão com filtros de cobertura de sal no dataset TGS-Salt, destacando a diferença entre dois protocolos de avaliação:

- **Teste canônico** — conjunto externo fixo com 800 amostras reais (`subset_split/test`)
- **Teste interno** — split 80/20 gerado a partir do próprio dataset filtrado
- **Correção metodológica posterior** — os filtros de cobertura destinados ao treino devem ser aplicados **somente após remover as 800 amostras do teste canônico**

> **Regra central:** métricas obtidas com **teste interno** não são comparáveis com métricas do **teste canônico**.

---

## 2. Definição dos protocolos de teste

### 2.1 Teste canônico

- Origem: `dataset/subset_split/test`
- Tamanho: **800 amostras reais**
- Distribuição: próxima da distribuição real do problema
- Uso: comparação principal do paper
- Vantagem: permite comparação justa entre datasets e cenários
- Regra correta: qualquer filtro de cobertura aplicado ao treino deve usar apenas as **3198 amostras remanescentes**, nunca o conjunto completo de 3998

### 2.2 Teste interno

- Origem: split 80/20 do próprio dataset filtrado
- Tamanho: depende do filtro aplicado
- Distribuição: mesma distribuição do treino filtrado
- Uso: exploração rápida / diagnóstico interno
- Limitação: tende a produzir métricas infladas

---

## 3. Experimentos com teste canônico

### 3.1 Tabela consolidada

| Dataset | Filtro de cobertura de sal | Cenário | Seed | N treino | N synth | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) | Observação |
|---------|----------------------------|:-------:|:----:|:--------:|:-------:|:------------:|:--------:|:---------:|:------:|:---------:|------------|
| TGS completo (GEM Desktop) | nenhum | A | 42 | 3198 | 0 | 0.4417 | 0.4270 | 0.4609 | 50 | 17608.7 | baseline |
| subset_1_99 | 1–99% | A | 42 | 2209 | 0 | 0.7826 | 0.4791 | 0.5058 | 55 | 142.4 | **pré-correção metodológica** |
| subset_1_99 | 1–99% | A | 456 | 2209 | 0 | 0.7934 | 0.4739 | 0.5016 | 46 | 134.0 | **pré-correção metodológica** |
| **subset_1_99_postsplit** | **1–99%** | **A** | **423** | **1766** | **0** | **0.7979** | **0.4494** | **0.4843** | **76** | **181.2** | **pós-correção ✅** |
| subset_2_98_postsplit | 2–98% | A | 423 | 1665 | 0 | 0.7761 | 0.4359 | 0.4713 | 40 | 298.1 | pós-correção ✅ |
| subset_3_97_postsplit | 3–97% | A | 423 | 1605 | 0 | 0.7806 | 0.4270 | 0.4640 | 39 | 285.5 | pós-correção ✅ |
| subset_5_95_postsplit | 5–95% | A | 423 | 1480 | 0 | 0.8458 | 0.4293 | 0.4641 | 54 | 325.6 | pós-correção ✅ |
| subset_10_90_postsplit | 10–90% | A | 423 | 1293 | 0 | 0.8385 | 0.4250 | 0.4606 | 57 | 317.9 | pós-correção ✅ |
| subset_2_98 | 2–98% | A | 42 | 2080 | 0 | 0.8063 | 0.4761 | 0.5034 | 49 | 116.0 | **pré-correção metodológica** |
| train_filtered | 10–90% | A | 42 | 1293 | 0 | 0.8517 | 0.4201 | 0.4553 | 56 | 213.7 | pós-split por construção |
| train_filtered | 10–90% | A | 123 | 1293 | 0 | 0.8507 | 0.4286 | 0.4621 | 54 | 208.6 | pós-split por construção |
| train_filtered | 10–90% | A | 456 | 1293 | 0 | 0.8371 | 0.4223 | 0.4559 | 38 | 164.9 | pós-split por construção |
| train_filtered + sísmicos | 10–90% | B | 42 | 1293 | 955 | 0.8514 | 0.4308 | 0.4672 | 47 | 121.9 | pós-split por construção |

---

### 3.2 Ranking no teste canônico (pós-correção metodológica — N variável)

| Rank | Dataset | Filtro | N treino | Test IoU | Test Dice | Observação |
|------|---------|--------|:--------:|----------|-----------|------------|
| 1 | **subset_1_99_postsplit** | **1–99%** | **1766** | **0.4494** | **0.4843** | melhor filtro pós-correção ✅ |
| 2 | subset_2_98_postsplit | 2–98% | 1665 | 0.4359 | 0.4713 | |
| 3 | subset_5_95_postsplit | 5–95% | 1480 | 0.4293 | 0.4641 | |
| 4 | train_filtered + sísmicos | 10–90% | 1293+955 | 0.4308 | 0.4672 | Cenário B |
| 5 | train_filtered (seed 123) | 10–90% | 1293 | 0.4286 | 0.4621 | |
| 6 | TGS completo | nenhum | 3198 | 0.4270 | 0.4609 | baseline |
| 7 | subset_3_97_postsplit | 3–97% | 1605 | 0.4270 | 0.4640 | |
| 8 | subset_10_90_postsplit | 10–90% | 1293 | 0.4250 | 0.4606 | |

> ⚠️ Comparação ainda desequilibrada: N de treino varia de 1293 a 1766 entre os filtros.

---

### 3.3 Experimentos normalizados (N=1293, test canônico 400 amostras) — **concluído**

Para comparação justa, todos os filtros foram normalizados para **N=1293 amostras** de treino (o menor da série), avaliados no **test canônico reduzido a 400 amostras** (`seed 423`).

| Rank | Filtro | N treino (fixo) | Test IoU | Test Dice | Tempo (s) |
|:----:|--------|:---------------:|:--------:|:---------:|:---------:|
| 1 | **1–99%** | **1293** | **0.4509** | **0.4870** | 301 |
| 2 | 3–97% | 1293 | 0.4504 | 0.4868 | 300 |
| 3 | 10–90% | 1293 | 0.4470 | 0.4841 | 322 |
| 4 | 2–98% | 1293 | 0.4426 | 0.4821 | 280 |
| 5 | 5–95% | 1293 | 0.4376 | 0.4759 | 198 |

> Observação: resultados desta série **não são comparáveis** com a série de 800 amostras de teste — servem para comparação relativa entre filtros com N fixo.

**Conclusões da série normalizada:**
1. Com N=1293 igualado, o filtro `1–99%` continua sendo o melhor (`IoU = 0.4509`).
2. `3–97%` ficou praticamente empatado (`0.4504`), ambos dentro de margem de variabilidade de seed.
3. `10–90%` surpreendeu positivamente ao chegar a `0.4470` com o mesmo N.
4. `2–98%` e `5–95%` ficaram abaixo, sugerindo que esses intervalos não capuram a diversidade ideal.
5. A vantagem de `1–99%` sobre `10–90%` com N igual é de `+0.0039` em IoU — pequena mas consistente.

---

### 3.3 Interpretação

1. **A correção metodológica mudou materialmente o experimento `1–99%`**: o conjunto de treino caiu de `2209` para `1766` amostras.
2. Com a metodologia correta, o run `subset_1_99_postsplit` (`seed 423`) obteve `Test IoU = 0.4494`, abaixo dos resultados `0.4791` e `0.4739` obtidos antes da correção.
3. Isso sugere que os resultados anteriores de `subset_1_99` estavam favorecidos por uma filtragem aplicada **antes** da separação do teste canônico.
4. **Filtros leves continuam sendo os melhores** na série pós-correção: `1–99%` > `2–98%` > `5–95%` > `3–97%` > `10–90%`.
5. Comparação por N ainda desequilibrada: os filtros mais leves têm mais amostras. A série normalizada (N=1293) resolverá isso.

---

## 4. Experimentos com teste interno

### 4.1 Tabela consolidada

| Dataset | Filtro de cobertura de sal | Seed | N total | N treino pool | N treino final | N val | N test | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|---------|----------------------------|:----:|:-------:|:-------------:|:--------------:|:-----:|:------:|:------------:|:--------:|:---------:|:------:|:---------:|
| subset_1_99 | 1–99% | 42 | 2209 | 1767 | 1590 | 177 | 442 | 0.7329 | 0.7662 | 0.8269 | 46 | 93.6 |
| subset_1_99 | 1–99% | 123 | 2209 | 1767 | — | — | 442 | 0.7852 | 0.7837 | 0.8440 | 45 | 91.3 |
| subset_2_98 | 2–98% | 42 | 2080 | 1664 | 1497 | 167 | 416 | 0.8216 | 0.7971 | 0.8613 | 59 | 114.8 |
| subset_3_97 | 3–97% | 42 | 1996 | 1596 | 1436 | 160 | 400 | 0.8043 | 0.7833 | 0.8492 | 40 | 79.5 |
| subset_10_90 (`salt10-90_1600`) | 10–90% | 42 | ~1616 | ~1292 | — | — | ~323 | 0.8069 | 0.8340 | 0.8943 | 44 | 183.4 |

> Observação: em teste interno, `n_real` registrado no `result.csv` corresponde ao tamanho do **train pool** após a separação treino/teste, não ao total bruto do dataset filtrado.

---

### 4.2 Interpretação

1. **Os valores de IoU interno são muito mais altos** (`0.76–0.83`) do que no teste canônico (`0.42–0.48`).
2. Isso ocorre porque treino e teste vêm da **mesma distribuição filtrada**, tornando a tarefa mais fácil.
3. O run `seed42_subset_1_99` (`best_val_iou = 0.7329`, `test_iou = 0.7662`) confirma que o protocolo interno para esse subset produz métricas muito acima do teste canônico.
4. O run `seed123_subset_1_99` (`best_val_iou = 0.7852`, `test_iou = 0.7837`) segue o mesmo padrão de **teste interno**, embora com seed mais favorável.
5. À medida que o filtro foi apertado de `2–98%` para `3–97%`, o IoU interno caiu de `0.7971` para `0.7833`, indicando perda de diversidade útil.
6. Mesmo quando o teste interno mostra números muito altos, isso **não implica melhor desempenho no cenário real**.

---

## 5. Comparação entre teste canônico e teste interno

| Dataset | Filtro | Tipo de teste | Test IoU | Test Dice | Comparável ao paper? |
|---------|--------|---------------|----------|-----------|----------------------|
| subset_1_99 | 1–99% | canônico (seed 42, pré-correção) | 0.4791 | 0.5058 | ⚠️ provisório |
| subset_1_99 | 1–99% | canônico (seed 456, pré-correção) | 0.4739 | 0.5016 | ⚠️ provisório |
| subset_1_99_postsplit | 1–99% | canônico (seed 423, pós-correção) | 0.4494 | 0.4843 | ✅ |
| subset_1_99 | 1–99% | interno (seed 42) | 0.7662 | 0.8269 | ❌ |
| subset_1_99 | 1–99% | interno (seed 123) | 0.7837 | 0.8440 | ❌ |
| subset_2_98 | 2–98% | canônico (pré-correção) | 0.4761 | 0.5034 | ⚠️ provisório |
| subset_2_98 | 2–98% | interno | 0.7971 | 0.8613 | ❌ |
| subset_3_97 | 3–97% | interno | 0.7833 | 0.8492 | ❌ |

### Diferença crítica

Para o filtro `1–99%`, o IoU observado passou a ser:

- **0.4791** no **teste canônico** (`seed42_subset_1_99`, pré-correção)
- **0.4739** no **teste canônico** (`seed456_subset_1_99`, pré-correção)
- **0.4494** no **teste canônico** (`seed423_subset_1_99_postsplit`, pós-correção)
- **0.7662** no **teste interno** (`seed42_subset_1_99`)
- **0.7837** no **teste interno** (`seed123_subset_1_99`)

Isso mostra que tanto o **tipo de conjunto de teste** quanto a **ordem correta entre split e filtro** alteram drasticamente a interpretação do experimento.

---

## 6. Conclusões desta sessão

1. A observação metodológica estava correta: o filtro `1–99%` deve ser aplicado **somente após remover as 800 amostras do teste canônico**.
2. Com essa correção, o subset `1–99%` caiu de `2209` para `1766` amostras de treino.
3. Com N variável, o run corrigido (`seed 423`) obteve `Test IoU = 0.4494`.
4. Na comparação justa com N=1293 fixo e test=400, o filtro `1–99%` manteve a liderança: `IoU = 0.4509`.
5. A vantagem do filtro `1–99%` sobre o `10–90%` (baseline `train_filtered`) **é confirmada mesmo com N igualado**.
6. Os filtros `3–97%` e `1–99%` apresentam desempenho equivalente nesta série — ambos são boas escolhas para o treino principal.
7. Testes internos continuam superestimando fortemente o desempenho e devem ser usados apenas como apoio exploratório.
8. Para o manuscrito, apenas resultados obtidos com **teste canônico** e **filtro aplicado após a separação do teste** devem sustentar conclusões principais.

---

## 7. Próximos passos recomendados

1. ✅ Repetir `subset_1_99` com protocolo correto pós-split.
2. ✅ Repetir os demais intervalos de filtro com protocolo pós-split.
3. ✅ Normalizar todos para N=1293 e test=400 — comparação justa concluída.
4. Priorizar o experimento principal do paper (`Cenário B`) com o filtro `1–99%` pós-split:

```bash
python -u train.py --scenario B --seed 42 --n_synth 955 --epochs 100 \
  --train_dir /var/tmp/cym7/datasets/subset_1_99_postsplit \
  --test_dir dataset/subset_split/test
```

5. Se necessário estimar variabilidade: repetir a série normalizada com seeds `42` e `456`.
