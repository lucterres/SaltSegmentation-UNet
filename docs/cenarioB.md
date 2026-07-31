# Experimento #3

## Fase I

* usar dataset tgs não filtrado por cobertura

* N train 1200, 1600, 2000
seed 42,123,456
epocas = 100

* Test set canonico (estratificado, random_state=42)	400

* executar cenario A

---

## Resultados — Cenário A, TGS não filtrado (2026-07-31)

> **Dataset:** TGS full não filtrado (`/var/tmp/cym7/datasets/tgs-salt/train`, ~3998 imgs)  
> **Test set:** split interno estratificado seed=0, 20% (~800 amostras)  
> **Early stop:** patience=10 (val IoU)

### N = 1200

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | 0.3862   | 0.4252    | 0.3724       | 72     | 124.1     |
| 123  | 0.3821   | 0.4205    | 0.3896       | 41     | 61.5      |
| 456  | 0.3817   | 0.4230    | 0.3815       | 62     | 92.7      |
| **média ± dp** | **0.3833 ± 0.0025** | **0.4229 ± 0.0024** | | | |

### N = 1600

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | 0.4025   | 0.4396    | 0.3988       | 51     | 113.0     |
| 123  | **0.4047** | **0.4423** | 0.4390     | 65     | 146.7     |
| 456  | 0.3910   | 0.4265    | 0.3782       | 51     | 99.6      |
| **média ± dp** | **0.3994 ± 0.0073** | **0.4361 ± 0.0082** | | | |

### N = 2000

| Seed | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | 0.4067   | 0.4423    | 0.4122       | 46     | 106.4     |
| 123  | **0.4139** | **0.4505** | 0.4392     | 63     | 243.3     |
| 456  | 0.4061   | 0.4423    | 0.4563       | 47     | 207.3     |
| **média ± dp** | **0.4089 ± 0.0042** | **0.4450 ± 0.0047** | | | |

---

### Resumo consolidado — Cenário A por N (média ± dp, 3 seeds)

| N real | Test IoU (média ± dp) | Test Dice (média ± dp) | Ganho IoU vs N=1200 |
|:------:|:---------------------:|:----------------------:|:-------------------:|
| 1200   | 0.3833 ± 0.0025       | 0.4229 ± 0.0024        | —                   |
| 1600   | 0.3994 ± 0.0073       | 0.4361 ± 0.0082        | +0.0161 (+4.2%)     |
| 2000   | **0.4089 ± 0.0042**   | **0.4450 ± 0.0047**    | **+0.0256 (+6.7%)** |

---

### Análise

- **Tendência clara de escala:** IoU cresce monotonicamente com N — de 0.3833 (N=1200) para 0.4089 (N=2000), ganho de +6.7%.
- **N=2000 é o melhor baseline do Cenário A** neste experimento, com Test IoU médio de 0.4089 ± 0.0042.
- **Variabilidade entre seeds diminui com N:** dp cai de 0.0025 (N=1200) → estabiliza em ~0.004 para N maiores, indicando que o modelo fica mais robusto com mais dados.
- **Seed 123 consistentemente melhor** para N=1600 e N=2000; seed 42 ligeiramente melhor para N=1200.
- **Referência histórica (N=1200, seed 42):** IoU=0.3862 — confirmado reproduzível ✅

### Próximo passo — Fase II (Cenário B)

Executar Cenário B (real + sintético) para N=1200 com seeds 42, 123, 456 para verificar se a adição de dados sintéticos supera o baseline do Cenário A estabelecido aqui.
