# Fase IV (Cenário B)

> **Objetivo:** Verificar se sintéticos **geométricos puros** (sem síntese sísmica) superam o Cenário A — isola o efeito da textura sísmica vs. apenas aumentação geométrica.

---

## Resultados — Fase IV: Cenário B, TGS não filtrado + 1600 sintéticos geométricos (2026-08-02)

> **Dataset real:** TGS full não filtrado (~3998 imgs)  
> **Sintéticos:** 1600 imagens geométricas puras (`geometric1600/pairs1600` — sem síntese sísmica)  
> **Seeds:** 42, 123 e 456 (3 seeds)  
> **Test set:** split interno estratificado seed=0, 20% (~800 amostras) — **idêntico às fases anteriores**  
> **Early stop:** patience=10 (val IoU)

### N = 1200 + 1600 sintéticos geométricos

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | 0.3680   | 0.4044    | 0.3308       | 30     | 111.2     |
| 123  | **0.3816** | **0.4179** | 0.3879      | 69     | 238.9     |
| 456  | 0.3810   | 0.4205    | 0.3620       | 39     | 154.7     |
| **média ± dp** | **0.3769 ± 0.0076** | **0.4143 ± 0.0086** | | | |

### N = 1600 + 1600 sintéticos geométricos

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | 0.3810   | 0.4190    | 0.3876       | 62     | 423.3     |
| 123  | **0.3976** | **0.4379** | 0.4216      | 45     | 176.6     |
| 456  | 0.3894   | 0.4282    | 0.3700       | 48     | 324.8     |
| **média ± dp** | **0.3893 ± 0.0083** | **0.4284 ± 0.0095** | | | |

### N = 2000 + 1600 sintéticos geométricos

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | **0.3906** | **0.4269** | 0.3989      | 64     | 268.6     |
| 123  | 0.3988   | 0.4377    | 0.4278       | 53     | 329.3     |
| 456  | 0.3872   | 0.4247    | 0.4263       | 27     | 220.5     |
| **média ± dp** | **0.3922 ± 0.0059** | **0.4298 ± 0.0069** | | | |

---

## Comparativo Consolidado — Fases I, III e IV (sísmico vs geométrico)

| N real | Cenário | Pool sintético | Test IoU (média ± dp) | Test Dice (média ± dp) | Δ IoU vs A |
|:------:|:-------:|:--------------:|:---------------------:|:----------------------:|:----------:|
| 1200 | A | — | 0.3833 ± 0.0025 | 0.4229 ± 0.0024 | — |
| 1200 | **B (F-III)** ✅ | sísmico 1600 | **0.3953 ± 0.0121** | **0.4337 ± 0.0126** | **+0.0120 (+3.1%)** |
| 1200 | B (F-IV) ❌ | geométrico 1600 | 0.3769 ± 0.0076 | 0.4143 ± 0.0086 | −0.0064 (−1.7%) |
| 1600 | A | — | 0.3994 ± 0.0073 | 0.4361 ± 0.0082 | — |
| 1600 | **B (F-III)** ✅ | sísmico 1600 | **0.4099 ± 0.0022** | **0.4489 ± 0.0031** | **+0.0105 (+2.6%)** |
| 1600 | B (F-IV) ❌ | geométrico 1600 | 0.3893 ± 0.0083 | 0.4284 ± 0.0095 | −0.0101 (−2.5%) |
| 2000 | **A** | — | **0.4089 ± 0.0042** | **0.4450 ± 0.0047** | — |
| 2000 | B (F-III) ✅ | sísmico 1600 | 0.4094 ± 0.0057 | 0.4466 ± 0.0061 | +0.0005 (+0.1%) |
| 2000 | B (F-IV) ❌ | geométrico 1600 | 0.3922 ± 0.0059 | 0.4298 ± 0.0069 | −0.0167 (−4.1%) |

---

## Análise da Fase IV

### Sintéticos geométricos puros **não superam** o Cenário A ❌

- **N=1200:** F-IV (−1.7%) vs F-III (+3.1%) — diferença de **4.8 p.p.** em favor do pool sísmico
- **N=1600:** F-IV (−2.5%) vs F-III (+2.6%) — diferença de **5.1 p.p.** em favor do pool sísmico
- **N=2000:** F-IV (−4.1%) vs F-III (+0.1%) — diferença de **4.2 p.p.** em favor do pool sísmico

### A síntese sísmica é determinante ✅

O pool geométrico puro **piora** o modelo em relação ao Cenário A em todos os N. A textura sísmica adicionada ao pool da Fase III é o fator responsável pelo ganho — não apenas o volume ou a geometria das imagens sintéticas.

### Conclusão da Fase IV

❌ **Sintéticos geométricos puros não superam o Cenário A** — B (F-IV) < A em todos os N.  
✅ **A síntese sísmica é essencial** — sem ela, os sintéticos prejudicam o modelo.  
✅ **Resultado confirma e fortalece a Fase III:** o ganho observado com `pairs1600_seismic` é atribuído à qualidade sísmica dos sintéticos, não apenas ao volume.  
➡️ **Para o manuscrito:** usar contraste F-III vs F-IV como evidência da importância da síntese sísmica na geração de dados sintéticos.