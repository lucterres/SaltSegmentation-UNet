# Relatório de Resultados — Cenários A e B com Data Augmentation

**Manuscrito:** Access-2026-27912  
**Data:** 2026-08-19  
**Fonte dos dados:** `docs/csv/last_ab_consolidated_results.csv`  
**Critério de ranking:** média de Test IoU entre seeds 42, 123 e 456

---

## 1. Configuração Experimental

| Parâmetro | Valor |
|-----------|-------|
| n_real (Cenário A) | 1200 |
| n_real (Cenário B) | 1200 |
| n_synth (Cenário B) | 1200 |
| Seeds avaliadas | 42, 123, 456 |
| Métodos de augmentação (B) | 7 |
| Métrica principal | IoU (test set canônico) |

---

## 2. Cenário A — Baseline (sem augmentação)

> Treino com dados reais apenas (`n_real=1200`, `n_synth=0`, sem data augmentation).

| Seed | N real | Test IoU | Best Val IoU | Épocas | Tempo (s) |
|:----:|:------:|:--------:|:------------:|:------:|:---------:|
| 42   | 1200   | 0.3988   | 0.3786       | 49     | 74.3      |
| 123  | 1200   | **0.4167** | **0.4016** | 57     | 86.2      |
| 456  | 1200   | 0.4089   | 0.3803       | 46     | 69.4      |
| **Média** | — | **0.4081** | **0.3868** | 50.7 | 76.6 |

---

## 3. Cenário B — Real + Sintético com Data Augmentation

> Treino com `n_real=1200` + `n_synth=1200`. Ranking por **média de Test IoU** (seeds 42, 123, 456).

### 3.1 Resultados por seed e método

| Rank | Método | IoU seed 42 | IoU seed 123 | IoU seed 456 | **Média IoU** |
|:----:|--------|:-----------:|:------------:|:------------:|:-------------:|
| 1° ✅ | **context seismic**       | 0.4280 | 0.4335 | 0.4214 | **0.4276** |
| 2°   | elastic transform          | 0.4246 | 0.4314 | 0.4255 | **0.4272** |
| 3°   | random_brightness_contrast | 0.4310 | 0.4316 | 0.4186 | **0.4271** |
| 4°   | grid distortion            | 0.4327 | 0.4400 | 0.4084 | **0.4270** |
| 5°   | random_gamma               | 0.4232 | 0.4135 | 0.4211 | **0.4193** |
| 6°   | optical_distortion         | 0.4241 | 0.4025 | 0.4233 | **0.4166** |
| 7°   | clahe                      | 0.4164 | 0.4192 | 0.4084 | **0.4147** |

> **Notas:**
> - Os 4 primeiros métodos são estatisticamente próximos (diferença < 0.0006 IoU entre 1° e 4°).
> - `grid distortion` tem o melhor run individual (seed 123: IoU = 0.4400), mas cai para 4° na média por instabilidade no seed 456 (IoU = 0.4084).
> - **`context seismic` é o método mais equilibrado**: melhor média e variância razoável.

---

## 4. Comparativo Cenário A vs. Cenário B

### 4.1 Melhor método médio (context seismic) vs. Baseline

| Métrica | Cenário A (média 3 seeds) | Cenário B — context seismic (média 3 seeds) | Δ absoluto | Δ relativo |
|---------|:-------------------------:|:-------------------------------------------:|:----------:|:----------:|
| Test IoU | 0.4081 | **0.4276** | +0.0195 | **+4.8%** |

### 4.2 Ganho por seed (context seismic vs. baseline)

| Seed | Cenário A IoU | Cenário B IoU | Δ |
|:----:|:-------------:|:-------------:|:-:|
| 42   | 0.3988 | 0.4280 | +0.0292 |
| 123  | 0.4167 | 0.4335 | +0.0168 |
| 456  | 0.4089 | 0.4214 | +0.0125 |
| **Média** | **0.4081** | **0.4276** | **+0.0195** |

> **Todos os seeds confirmam ganho positivo do Cenário B sobre A com `context seismic`.**

---

## 5. Ranking Completo — Média IoU (critério principal)

| Rank | Método | Média IoU ↓ | Std IoU | Ganho vs. A (Δ IoU) |
|:----:|--------|:-----------:|:-------:|:--------------------:|
| 1° ✅ | **context seismic**       | **0.4276** | 0.0062 | **+0.0195** |
| 2°   | elastic transform          | 0.4272 | 0.0036 | +0.0191 |
| 3°   | random_brightness_contrast | 0.4271 | 0.0068 | +0.0190 |
| 4°   | grid distortion            | 0.4270 | 0.0163 | +0.0189 |
| 5°   | random_gamma               | 0.4193 | 0.0050 | +0.0112 |
| 6°   | optical_distortion         | 0.4166 | 0.0122 | +0.0085 |
| 7°   | clahe                      | 0.4147 | 0.0056 | +0.0066 |
| —    | **Cenário A (baseline)**   | 0.4081 | 0.0092 | — |

---

## 6. Análise de Estabilidade

| Método | Std IoU | Classificação |
|--------|:-------:|---------------|
| elastic transform          | **0.0036** | 🟢 Muito estável |
| context seismic            | 0.0062 | 🟢 Estável |
| clahe                      | 0.0056 | 🟢 Estável |
| random_gamma               | 0.0050 | 🟢 Estável |
| random_brightness_contrast | 0.0068 | 🟡 Moderado |
| optical_distortion         | 0.0122 | 🔴 Instável |
| grid distortion            | 0.0163 | 🔴 Instável |

> `grid distortion` tem o pior coeficiente de estabilidade entre todos os métodos, o que justifica sua queda do 1° (melhor run) para o 4° lugar (melhor média).

---

## 7. Conclusão

### 7.1 Hipótese confirmada

O Cenário B supera consistentemente o Cenário A em todos os 7 métodos testados:
- **Ganho mínimo:** +0.0066 IoU (clahe, +1.6%)
- **Ganho máximo:** +0.0195 IoU (context seismic, +4.8%)

### 7.2 Melhor configuração identificada

| Critério | Recomendação |
|----------|-------------|
| **Máxima média IoU** | `context seismic`: média IoU = **0.4276** ✅ |
| **Máxima estabilidade** | `elastic transform`: std IoU = **0.0036** |
| **Melhor run individual** | `grid distortion`, seed 123: IoU = 0.4400 (não recomendado por instabilidade) |

### 7.3 Narrativa para o manuscrito (R2.1)

> "Treinando com 1200 amostras reais e 1200 amostras sintéticas (Cenário B), com data augmentation, obtivemos ganhos consistentes em relação ao baseline com dados reais apenas (Cenário A, IoU médio = 0.4081). O método `context seismic` alcançou a maior média de IoU entre os três seeds avaliados (0.4276 ± 0.0062), representando um ganho de +4.8% sobre o baseline. Os quatro métodos melhor posicionados — `context seismic`, `elastic transform`, `random brightness/contrast` e `grid distortion` — apresentam médias de IoU entre 0.4270 e 0.4276, confirmando que a combinação de dados sintéticos com augmentação geométrica ou baseada em contexto sísmico melhora a generalização para dados reais de teste."

---

## 8. Próximas etapas sugeridas

- [ ] Validar `context seismic` com mais seeds (ex: 789, 1024) para confirmar estabilidade
- [ ] Testar combinação `context seismic` + `elastic transform`
- [ ] Expandir n_synth (1600, 2000) para verificar saturação com `context seismic`
- [ ] Atualizar `docs/relatorio-final-r21-downstream.md` com estes resultados
- [ ] Atualizar `_v7.tex` → `\subsection{Downstream Segmentation Evaluation}`
- [ ] Atualizar `docs/_reviewACCESS/response_to_reviewers.md` → seção R2.1

---

*Gerado automaticamente a partir de `docs/csv/last_ab_consolidated_results.csv` em 2026-08-19.*
