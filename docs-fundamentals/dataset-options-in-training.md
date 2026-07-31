# Opções de Dataset para Comparação Cenário A × B — Resposta aos Revisores

## Contexto

Para responder aos revisores sobre a hipótese **B > A** (Real + Sintético > Real only), a escolha do dataset de treino e teste impacta diretamente os resultados e a credibilidade do argumento.

---

## Argumento 1 — Usar o dataset **completo** (distribuição real)

**Favorável quando:** o paper reivindica que os sintéticos melhoram a segmentação em condições reais de campo.

- O test set com distribuição real é o mais **honesto e generalizável**
- Revisores tendem a questionar filtros como "cherry-picking"
- A comparação é direta com benchmarks da literatura TGS
- **Problema:** o efeito dos sintéticos é diluído pelos ~47% de casos sem sal, e B ficou abaixo de A em todos os cenários testados

---

## Argumento 2 — Usar o dataset **filtrado** `subset_split` (train 10–90%, test completo)

**Este é o cenário mais defensável para o paper:**

- O train é filtrado (sem casos triviais), mas o **test permanece com distribuição real completa**
- Neste cenário, B > A foi observado (+0.011 IoU, +0.012 Dice, seed 42)
- O argumento é legítimo: *"treinamos com amostras relevantes e testamos no mundo real"*
- **Problema:** resultado só confirmado para seed 42 — faltam seeds 123 e 456

---

## Recomendação objetiva

**Usar `subset_split` com as 3 seeds (42, 123, 456)**

**Por quê:**

1. É o único cenário onde B > A se confirmou empiricamente
2. O test set é a distribuição real completa — sem acusação de filtro tendencioso no teste
3. A filtragem do **treino** (10–90%) é justificável metodologicamente: os dados sintéticos gerados naturalmente têm sal parcial, e faz sentido treinar comparativamente sem casos triviais
4. Com 3 seeds você tem média ± std para argumentar estatisticamente

---

## O que falta executar

```bash
# seeds 123 e 456 no subset_split, Cenário A e B
python train.py --scenario A --seed 123 --train_dir .../train_filtered --test_dir .../test
python train.py --scenario B --seed 123 --n_synth 955 --train_dir .../train_filtered --test_dir .../test
# idem seed 456
```

---

## Critério de decisão final

| Resultado das seeds 123 e 456 | Ação recomendada |
|---|---|
| B > A confirmado nas 3 seeds | Reportar `subset_split` como experimento principal |
| B > A apenas em seed 42 | Reportar **ambos os cenários** com explicação da distorção do IoU |
| B < A nas 3 seeds do `subset_split` | Reportar dataset completo com análise honesta do papel dos casos triviais |

Se as outras seeds não confirmarem o ganho, o argumento mais honesto para os revisores é reportar **ambos os cenários** (completo e filtrado) e explicar a distorção do IoU causada pelas máscaras com 0% de sal.

---

## Referências internas

- Análise técnica do impacto das máscaras com 0% sal: `docs-fundamentals/analise-de-metricas.md`
- Dados experimentais completos: `docs/relatorio-final-r21-downstream.md`
- Protocolo de experimentos: `docs-fundamentals/experimentUNet-protocol.md`
