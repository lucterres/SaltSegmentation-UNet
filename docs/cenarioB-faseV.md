Executar Cenário B (real + 1200 sintéticos de  Salt-Segmentation-UNet\dataset\geometric1600_seismic) para N=1200,1600 e 2000 com seeds 42, 123, 456 para verificar se a adição de dados sintéticos supera o baseline do Cenário A estabelecido aqui. uma gpu para cada run


# Fase V (Cenário B)

> **Objetivo:** Completar a curva de volume sintético — 1200 sintéticos sísmicos (entre 955 da F-II e 1600 da F-III) para verificar se o ganho é monotônico com o volume.

---

## Resultados — Fase V: Cenário B, TGS não filtrado + 1200 sintéticos sísmicos (2026-08-02)

> **Dataset real:** TGS full não filtrado (~3998 imgs)  
> **Sintéticos:** 1200 imagens sísmicas (`pairs1600_seismic`, subamostrado)  
> **Seeds:** 42, 123 e 456 (3 seeds)  
> **Test set:** split interno estratificado seed=0, 20% (~800 amostras) — **idêntico às fases anteriores**  
> **Early stop:** patience=10 (val IoU)

### N = 1200 + 1200 sintéticos

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | **0.3912** | **0.4324** | 0.3875       | 37     | 109.3     |
| 123  | 0.3755   | 0.4138    | 0.3846       | 21     | 61.8      |
| 456  | 0.3969   | 0.4357    | 0.3779       | 41     | 121.4     |
| **média ± dp** | **0.3879 ± 0.0111** | **0.4273 ± 0.0118** | | | |

### N = 1600 + 1200 sintéticos

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | **0.4087** | **0.4475** | 0.3965       | 56     | 335.3     |
| 123  | 0.4049   | 0.4436    | 0.4521       | 61     | 201.5     |
| 456  | 0.3999   | 0.4379    | 0.3895       | 43     | 260.2     |
| **média ± dp** | **0.4045 ± 0.0045** | **0.4430 ± 0.0049** | | | |

### N = 2000 + 1200 sintéticos

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | **0.4114** | **0.4508** | 0.4271       | 50     | 189.2     |
| 123  | 0.4099   | 0.4478    | 0.4279       | 46     | 340.3     |
| 456  | 0.4111   | 0.4485    | 0.4543       | 61     | 393.5     |
| **média ± dp** | **0.4108 ± 0.0008** | **0.4490 ± 0.0016** | | | |

---

## Curva de Volume Sintético — Fases II, V e III (sísmico, 3 seeds cada)

| N real | A (ref) | B 955 sint. (F-II) | B 1200 sint. (F-V) | B 1600 sint. (F-III) |
|:------:|:-------:|:------------------:|:------------------:|:--------------------:|
| 1200 | 0.3833 | 0.3911 (+2.0%) | 0.3879 (+1.2%) | **0.3953 (+3.1%)** |
| 1600 | 0.3994 | 0.4077 (+2.1%) | 0.4045 (+1.3%) | **0.4099 (+2.6%)** |
| 2000 | 0.4089 | 0.4058 (−0.8%) | **0.4108 (+0.5%)** | 0.4094 (+0.1%) |

### Análise da curva de volume

- **N=1200 e N=1600:** ganho **não é monotônico** — 1200 sint. (F-V) fica *abaixo* de 955 sint. (F-II), e 1600 sint. (F-III) é o melhor. Isso sugere que a subamostagem aleatória de 1200 de um pool de 1600 pode remover exemplos importantes presentes no subconjunto de 955.
- **N=2000:** F-V (1200 sint.) é ligeiramente melhor que F-III (1600 sint.) e F-II (955 sint.) — tendência invertida, mas diferenças marginais (dp sobrepostos).
- **Variabilidade:** F-V com N=2000 tem dp=0.0008 — a mais baixa de todo o experimento, indicando convergência muito estável com 1200 sintéticos + 2000 reais.
- **Conclusão:** o volume ótimo de sintéticos sísmicos está em **1600 (pool completo)** para N≤1600; para N=2000 os três volumes são equivalentes dentro da margem de erro.

### Conclusão da Fase V

✅ **B > A confirmado para todos os N** com 1200 sintéticos (exceto N=1200 onde F-V < F-II).  
⚠️ **Ganho não é estritamente monotônico com volume** — o pool completo (1600) é o mais robusto.  
➡️ **Recomendação:** usar o **pool completo (1600)** para maximizar resultado — subamostrar o pool não traz benefício consistente.