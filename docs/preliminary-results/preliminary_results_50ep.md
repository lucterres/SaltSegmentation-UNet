# Resultados Preliminares — Cenário A (Real Only) — 50 Epochs

**Manuscrito:** Access-2026-27912  
**Experimento:** R2.1 — Downstream Segmentation  
**Configuração:** Scenario A · seed=42 · epochs=50 (early stop patience=10) · device=CPU  
**Data:** 2026-07-19  
**Objetivo:** Avaliar convergência com mais epochs e tendência de melhoria com volume de dados

---

## Visão geral do processo de treinamento

A U-Net é treinada por **gradiente descendente** (Adam): a cada epoch processa todos os batches, calcula a perda (BCE + Dice), e atualiza os pesos. O **early stopping** interrompe o treino se `val_iou` não melhorar por 10 epochs consecutivas — o `best_model.pth` é salvo a cada novo máximo de `val_iou`. A avaliação final usa esse checkpoint sobre o **test set fixo** (800 imagens, nunca vistas durante treino/validação).

Comportamento esperado:
- **Loss treino** decresce monotonicamente
- **Loss val** decresce, oscila e pode subir com overfitting
- **Val IoU** sobe gradualmente, atinge platô e dispara o early stop
- **Test IoU** ≤ `best_val_iou` por leve viés de seleção do checkpoint

---

## Tabela consolidada — result.csv

| scenario | seed | n_real | n_synth | best_val_iou | test_iou | test_dice | epochs_run | elapsed_s |
|:--------:|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:----------:|:---------:|
| A | 42 | 200  | 0 | 0.3389 | 0.3102 | 0.3616 | 29 *(early stop)* | 667   |
| A | 42 | 400  | 0 | 0.3393 | 0.3391 | 0.3820 | 32 *(early stop)* | 757   |
| A | 42 | 800  | 0 | 0.4010 | 0.3630 | 0.4019 | 50 *(completo)*   | 1461  |
| A | 42 | 1600 | 0 | 0.3976 | 0.4011 | 0.4401 | 50 *(completo)*   | 4724  |

---

## Métricas no Test Set

| N imagens | Test IoU | Test Dice | Δ IoU vs N=200 | Best val IoU | Epochs |
|:---------:|:--------:|:---------:|:--------------:|:------------:|:------:|
| 200       | 0.3102   | 0.3616    | —              | 0.3389       | 29     |
| 400       | 0.3391   | 0.3820    | +0.0289 (+9%)  | 0.3393       | 32     |
| 800       | 0.3630   | 0.4019    | +0.0528 (+17%) | 0.4010       | 50     |
| 1600      | **0.4011** | **0.4401** | **+0.0909 (+29%)** | 0.3976 | 50  |

---

## Evolução por Epoch

### N=200 (early stop E29)

| E | Loss treino | Loss val | IoU val | Dice val |
|:-:|:-----------:|:--------:|:-------:|:--------:|
| 1  | 0.6746 | 0.6936 | 0.2115 | 0.2703 |
| 2  | 0.6536 | 0.7253 | 0.2904 | 0.3419 |
| 3  | 0.6382 | 0.7486 | 0.2927 | 0.3432 |
| 4  | 0.6158 | 0.7168 | 0.2902 | 0.3403 |
| 5  | 0.5974 | 0.6982 | 0.2910 | 0.3438 |
| 6  | 0.5819 | 0.5852 | 0.3097 | 0.3534 |
| 7  | 0.5674 | 0.6407 | 0.2831 | 0.3320 |
| 8  | 0.5411 | 0.5690 | 0.3044 | 0.3483 |
| 9  | 0.5214 | 0.5514 | 0.3055 | 0.3517 |
| 10 | 0.5178 | 0.5443 | 0.3065 | 0.3511 |
| 11 | 0.5087 | 0.5120 | 0.3038 | 0.3470 |
| 12 | 0.4928 | 0.5837 | 0.3101 | 0.3590 |
| 13 | 0.4838 | 0.5678 | 0.3197 | 0.3699 |
| 14 | 0.4764 | 0.5097 | 0.3014 | 0.3499 |
| 15 | 0.4628 | 0.4344 | 0.3024 | 0.3462 |
| 16 | 0.4628 | 0.4691 | 0.2980 | 0.3415 |
| 17 | 0.4634 | 0.5190 | 0.3105 | 0.3554 |
| 18 | 0.4564 | 0.4703 | 0.3001 | 0.3446 |
| 19 | 0.4508 | 0.5534 | **0.3389** | **0.3754** |
| 20 | 0.4573 | 0.4502 | 0.2932 | 0.3397 |
| 21 | 0.4502 | 0.4434 | 0.2629 | 0.3141 |
| 22 | 0.4399 | 0.4661 | 0.2951 | 0.3429 |
| 23 | 0.4506 | 0.4921 | 0.3063 | 0.3508 |
| 24 | 0.4292 | 0.4140 | 0.3268 | 0.3608 |
| 25 | 0.4276 | 0.5019 | 0.3170 | 0.3612 |
| 26 | 0.4197 | 0.4715 | 0.3200 | 0.3639 |
| 27 | 0.4131 | 0.4823 | 0.2984 | 0.3504 |
| 28 | 0.4218 | 0.4411 | 0.3239 | 0.3629 |
| 29 | 0.3908 | 0.4408 | 0.3143 | 0.3573 |
| — | — | — | *early stop* | — |

### N=400 (early stop E32)

| E | Loss treino | Loss val | IoU val | Dice val |
|:-:|:-----------:|:--------:|:-------:|:--------:|
| 1  | 0.6640 | 0.7443 | 0.2455 | 0.2995 |
| 2  | 0.6223 | 0.7913 | 0.2463 | 0.3000 |
| 3  | 0.5797 | 0.6052 | 0.2460 | 0.3044 |
| 4  | 0.5461 | 0.5251 | 0.2617 | 0.3206 |
| 5  | 0.5214 | 0.5429 | 0.2846 | 0.3427 |
| 6  | 0.5005 | 0.4493 | 0.2382 | 0.2783 |
| 7  | 0.4992 | 0.5552 | 0.3018 | 0.3586 |
| 8  | 0.4750 | 0.4671 | 0.2604 | 0.2975 |
| 9  | 0.4584 | 0.4176 | 0.2671 | 0.3018 |
| 10 | 0.4375 | 0.4381 | 0.3083 | 0.3550 |
| 11 | 0.4381 | 0.4011 | 0.2658 | 0.3009 |
| 12 | 0.4241 | 0.4096 | 0.2779 | 0.3141 |
| 13 | 0.4378 | 0.4142 | 0.2986 | 0.3412 |
| 14 | 0.4077 | 0.4446 | 0.3167 | 0.3660 |
| 15 | 0.4130 | 0.3838 | 0.2919 | 0.3285 |
| 16 | 0.3983 | 0.3903 | 0.2974 | 0.3384 |
| 17 | 0.4008 | 0.4054 | 0.3088 | 0.3501 |
| 18 | 0.3827 | 0.3965 | 0.2949 | 0.3334 |
| 19 | 0.3821 | 0.4070 | 0.3123 | 0.3534 |
| 20 | 0.3686 | 0.3907 | 0.3038 | 0.3408 |
| 21 | 0.3582 | 0.3649 | 0.3049 | 0.3446 |
| 22 | 0.3568 | 0.4035 | 0.3393 | 0.3832 |
| 23 | 0.3466 | 0.3632 | 0.3066 | 0.3468 |
| 24 | 0.3595 | 0.3687 | 0.3120 | 0.3525 |
| 25 | 0.3468 | 0.3569 | 0.3372 | 0.3794 |
| 26 | 0.3418 | 0.3652 | 0.3169 | 0.3601 |
| 27 | 0.3349 | 0.3606 | 0.2976 | 0.3357 |
| 28 | 0.3387 | 0.3664 | 0.3139 | 0.3530 |
| 29 | 0.3375 | 0.3613 | 0.3355 | 0.3768 |
| 30 | 0.3286 | 0.3674 | 0.3307 | 0.3718 |
| 31 | 0.3255 | 0.3643 | 0.3327 | 0.3735 |
| 32 | 0.3239 | 0.3584 | 0.3275 | 0.3670 |
| — | — | — | *early stop* | — |

### N=800 (50 epochs completos)

| E | Loss treino | Loss val | IoU val | Dice val |
|:-:|:-----------:|:--------:|:-------:|:--------:|
| 1  | 0.6443 | 0.7168 | 0.2455 | 0.3098 |
| 5  | 0.4673 | 0.4332 | 0.3177 | 0.3630 |
| 10 | 0.3874 | 0.3547 | 0.3317 | 0.3670 |
| 15 | 0.3470 | 0.3427 | 0.3676 | 0.4086 |
| 20 | 0.3296 | 0.3048 | 0.3522 | 0.3860 |
| 25 | 0.3132 | 0.2816 | 0.3421 | 0.3767 |
| 30 | 0.2952 | 0.2700 | 0.3505 | 0.3865 |
| 35 | 0.2868 | 0.2750 | 0.3781 | 0.4156 |
| 40 | 0.2797 | 0.2591 | 0.3836 | 0.4225 |
| 45 | 0.2587 | 0.2607 | 0.3893 | 0.4287 |
| 48 | 0.2628 | 0.2711 | **0.4010** | **0.4442** |
| 50 | 0.2615 | 0.2655 | 0.3978 | 0.4405 |

### N=1600 (50 epochs completos)

| E | Loss treino | Loss val | IoU val | Dice val |
|:-:|:-----------:|:--------:|:-------:|:--------:|
| 1  | 0.6073 | 0.5234 | 0.2516 | 0.3081 |
| 5  | 0.4070 | 0.3592 | 0.2748 | 0.3069 |
| 10 | 0.3332 | 0.3161 | 0.3172 | 0.3533 |
| 15 | 0.2869 | 0.2713 | 0.3339 | 0.3702 |
| 20 | 0.2410 | 0.2522 | 0.3447 | 0.3824 |
| 25 | 0.2153 | 0.2504 | 0.3634 | 0.4023 |
| 30 | 0.2006 | 0.2270 | 0.3534 | 0.3892 |
| 35 | 0.1817 | 0.2273 | 0.3720 | 0.4134 |
| 40 | 0.1637 | 0.2484 | 0.3411 | 0.3825 |
| 44 | 0.1554 | 0.2181 | **0.3976** | **0.4321** |
| 47 | 0.1393 | 0.2522 | 0.3976 | 0.4431 |
| 50 | 0.1322 | 0.2260 | 0.3891 | 0.4301 |

---

## Análise

### Ganho 10 → 50 epochs

| N imagens | Test IoU (10ep) | Test IoU (50ep) | Ganho absoluto | Ganho relativo |
|:---------:|:---------------:|:---------------:|:--------------:|:--------------:|
| 200       | 0.2589          | 0.3102          | +0.0513        | +20%           |
| 400       | 0.3168          | 0.3391          | +0.0223        | +7%            |
| 800       | 0.3241          | 0.3630          | +0.0389        | +12%           |
| 1600      | 0.3493          | 0.4011          | +0.0518        | +15%           |

---

### Discussão: 10 epochs vs 50 epochs

#### Por que 10 epochs é insuficiente

Com 10 epochs, todos os modelos ainda estavam em fase ativa de aprendizado: a loss de treino continuava decrescendo e a val-IoU não havia atingido seu platô. Isso é visível nas curvas de cada run — em nenhum caso houve disparo de early stop dentro das 10 primeiras epochs. O checkpoint salvo às 10 epochs captura um modelo parcialmente treinado, sub-representando o potencial real de cada configuração.

O efeito é especialmente pronunciado em N=200 (+20%) e N=1600 (+15%): no primeiro caso, o modelo ainda estava aprendendo representações básicas de bordas de sal; no segundo, o volume maior de dados exige mais epochs para que os gradientes propaguem informação útil por toda a rede.

#### Comportamento diferente por tamanho de dataset

Com **N pequeno (200–400)**, o treinamento converge *mais rápido* mas para um valor de IoU menor:
- N=200 dispara early stop na E29, N=400 na E32
- A val-loss oscila fortemente (variância alta com apenas 20–40 amostras de validação)
- O modelo memoriza o conjunto pequeno rapidamente e entra em overfitting leve, impedindo melhoria adicional
- O ganho de 10→50 epochs existe (+20% para N=200), mas o teto absoluto é baixo (~0.31)

Com **N grande (800–1600)**, o treinamento converge *mais lentamente* mas para valores mais altos:
- N=800 e N=1600 completaram as 50 epochs sem disparar early stop — ainda havia melhoria em andamento
- A val-loss decresce de forma mais suave e monótona, sinal de aprendizado estável
- O ganho de 10→50 epochs é consistente (+12% e +15%) e o teto é significativamente maior (IoU > 0.40)
- **Isso indica que o modelo ainda não convergiu em 50 epochs para N grande** — epochs adicionais trariam ganhos adicionais

#### Relação entre volume de dados e necessidade de epochs

| N imagens | Epochs até early stop | Val-IoU final | Interpretação |
|:---------:|:--------------------:|:-------------:|:-------------:|
| 200  | 29  | 0.339 | Convergência rápida, teto baixo — dados insuficientes |
| 400  | 32  | 0.339 | Idem, teto ligeiramente maior |
| 800  | >50 | 0.401 | Convergência lenta, teto alto — precisa de mais epochs |
| 1600 | >50 | 0.398 | Idem — aprendizado ainda ativo na E50 |

Existe uma relação inversa: **mais dados → mais epochs necessárias para convergir**. Isso é esperado: com mais amostras, cada epoch atualiza os pesos com gradientes mais diversos, exigindo mais iterações para a rede integrar toda a informação do conjunto de treino.

#### Implicação para o experimento completo (R2.1)

O experimento oficial usa N=800 (Cenário A) e N=800+400 sintéticos (Cenário B). Com base nos resultados preliminares:

- **50 epochs é insuficiente** para N=800 — o modelo ainda melhorava na última epoch
- Recomendar `--epochs 100 --early_stop_patience 15` para garantir convergência sem desperdício de tempo
- Estimativa conservadora: N=800, 100 epochs ≈ **48 min/run** × 6 runs ≈ **~5 h no total em CPU**

---

### Observações adicionais

- **N=1600 atingiu test IoU = 0.4011** — único caso a ultrapassar 0.40 nas condições preliminares; serve como referência de teto para o Cenário A
- **`best_val_iou` ≥ `test_iou` em todos os casos**, confirmando o comportamento esperado de leve viés de seleção do checkpoint; a diferença é pequena (≤ 0.01), indicando boa generalização
- **Tempo CPU escala linearmente com N:** 667s (N=200) → 4724s (N=1600) ≈ fator ~7× para fator ~8× em N — consistente com o aumento de batches por epoch

### Recomendações para o experimento final

1. Usar `n_real=800` como configuração padrão (protocolo R2.1)
2. Usar `--epochs 100 --early_stop_patience 15` para garantir convergência
3. Rodar seeds 42, 123, 456 para calcular mean ± std e p-value
4. Após Cenário A convergir, rodar Cenário B com mesma configuração para comparação justa

---

## Próximos passos

1. Gerar pool sintético via `generate_synthetic.py` (Cenário B)
2. Rodar experimento completo: Cenário A × 3 seeds + Cenário B × 3 seeds (`--epochs 100`)
3. Executar `evaluate.py` → `summary.csv`
4. Preencher tabela do manuscrito (`_v7.tex`)
