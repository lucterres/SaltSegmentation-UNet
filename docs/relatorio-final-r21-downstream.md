# Relatório Final — Experimento Downstream R2.1

**Manuscrito:** Access-2026-27912  
**Comentário:** Reviewer 2, Issue 1 — Downstream segmentation experiment  
**Data:** 2026-07-23  
**Nó GPU:** `atn2b03n01`  
**Hardware:** 8 × Tesla V100-SXM2-32GB  
**Ambiente:** PyTorch 2.4.1+cu124, Python 3.8.16, venv em `/var/tmp/cym7/venvs/salt-unet/`

---

## 1. Objetivo

Avaliar se o treinamento com **dados reais + sintéticos** melhora a segmentação downstream de salt domes em relação ao treinamento com **dados reais apenas** no dataset TGS.

Foram testados três cenários:

- **Cenário A** — Real only
- **Cenário B** — Real + Synthetic (400 imagens sintéticas)
- **Cenário B'** — Real + Synthetic (1600 imagens sintéticas geométricas)

A métrica principal é **IoU**. A métrica secundária é **Dice**. O critério de early stopping é **val IoU**.

---

## 2. Protocolo executado

### 2.1 Cenário A — Dataset completo (~3200 amostras reais)

Seeds executadas: **42, 123, 456**

| Seed | N real | N synth | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42  | 3198 | 0 | 0.4312 | 0.4657 | 0.4426 | 57 | 202.1 |
| 123 | 3198 | 0 | 0.4190 | 0.4544 | 0.4269 | 52 | 185.7 |
| 456 | 3198 | 0 | 0.4240 | 0.4593 | 0.4460 | 54 | 191.2 |
| **Média** |  |  | **0.4247** | **0.4598** |  | **54.3** |  |

### 2.2 Cenário B — Dataset completo (~3200 reais + 400 sintéticos)

Seeds executadas: **42, 123, 456**

| Seed | N real | N synth | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42  | 3198 | 400 | 0.4090 | 0.4455 | 0.4286 | 44 | 539.4 |
| 123 | 3198 | 400 | 0.4092 | 0.4460 | 0.4255 | 52 | 612.1 |
| 456 | 3198 | 400 | 0.4198 | 0.4585 | 0.4387 | 67 | 671.6 |
| **Média** |  |  | **0.4127** | **0.4500** |  | **54.3** |  |

### 2.3 Cenário B' — Dataset completo (~3200 reais + 1600 sintéticos geométricos)

Seeds executadas: **42, 123, 456**  
Fonte sintética: `dataset/pairs1600.tar` → `dataset/geometric1600/pairs1600/` → symlink `dataset/synthetic`

| Seed | N real | N synth | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:------:|:-------:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42  | 3198 | 1600 | 0.4086 | 0.4441 | 0.4358 | 53 | 766 |
| 123 | 3198 | 1600 | 0.3989 | 0.4345 | 0.4120 | 31 | 521 |
| 456 | 3198 | 1600 | 0.4135 | 0.4485 | 0.4339 | 57 | 787 |
| **Média** |  |  | **0.4070** | **0.4424** |  | **47.0** |  |

---

## 3. Comparação principal — Cenário A vs Cenário B

| Cenário | Configuração | Test IoU médio | Test Dice médio |
|---------|--------------|----------------|-----------------|
| **A — Real only** | 3198 reais | **0.4247** | **0.4598** |
| **B — Real + 400 sintéticos** | 3198 reais + 400 sintéticos | 0.4127 | 0.4500 |
| **B — Real + 1600 sintéticos geométricos** | 3198 reais + 1600 sintéticos | 0.4070 | 0.4424 |

### Diferença média

- **B (400 synth) vs A**
  - IoU: $0.4127 - 0.4247 = -0.0120$
  - Dice: $0.4500 - 0.4598 = -0.0098$

- **B (1600 geometric) vs A**
  - IoU: $0.4070 - 0.4247 = -0.0177$
  - Dice: $0.4424 - 0.4598 = -0.0174$

- **B (1600 geometric) vs B (400 synth)**
  - IoU: $0.4070 - 0.4127 = -0.0057$
  - Dice: $0.4424 - 0.4500 = -0.0076$

### Interpretação

**Nenhuma variante do Cenário B superou o Cenário A.**  
Além disso, aumentar o número de amostras sintéticas de **400 para 1600** piorou ainda mais o desempenho médio.

Portanto:

1. **A hipótese do revisor (B > A) não se confirmou**.
2. **Mais dados sintéticos geométricos agravaram a queda de desempenho**.
3. O principal fator de ganho observado no experimento foi o aumento de **dados reais**, e não a adição de dados sintéticos.

---

## 4. Escala de dados — Cenário A

Além do cenário completo, foi avaliado o efeito do número de amostras reais no Cenário A.

### 4.1 Seed 42 — evolução com N real

| N real | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:------:|:--------:|:---------:|:------------:|:------:|:---------:|
| 200  | 0.2587 | 0.3178 | 0.3100 | 10 | 4.5 |
| 400  | 0.3198 | 0.3623 | 0.3111 | 10 | 6.6 |
| 800  | 0.3771 | 0.4168 | 0.4165 | 75 | 79.2 |
| 1200 | 0.3862 | 0.4252 | 0.3724 | 72 | 103.8 |
| 2000 | 0.4067 | 0.4423 | 0.4122 | 46 | 109.5 |
| 3198 | **0.4312** | **0.4657** | **0.4426** | 57 | 202.1 |

### 4.2 Comparação por seed — N = 800

| Seed | Test IoU | Test Dice | Épocas |
|:----:|:--------:|:---------:|:------:|
| 42  | 0.3771 | 0.4168 | 75 |
| 123 | 0.3874 | 0.4249 | 72 |
| 456 | 0.3812 | 0.4219 | 63 |
| **Média** | **0.3819** | **0.4212** | **70.0** |

### 4.3 Comparação por seed — N = 1200

| Seed | Test IoU | Test Dice | Épocas |
|:----:|:--------:|:---------:|:------:|
| 42  | 0.3862 | 0.4252 | 72 |
| 123 | 0.3821 | 0.4205 | 41 |
| 456 | 0.3817 | 0.4230 | 62 |
| **Média** | **0.3833** | **0.4229** | **58.3** |

### 4.4 Comparação por seed — N ≈ 3200

| Seed | Test IoU | Test Dice | Épocas |
|:----:|:--------:|:---------:|:------:|
| 42  | 0.4312 | 0.4657 | 57 |
| 123 | 0.4190 | 0.4544 | 52 |
| 456 | 0.4240 | 0.4593 | 54 |
| **Média** | **0.4247** | **0.4598** | **54.3** |

### Interpretação da escala de dados

Há um crescimento consistente do desempenho com o aumento do número de amostras reais:

- **800 → 1200:** ganho marginal de IoU ($+0.0014$)
- **1200 → 3200:** ganho expressivo de IoU ($+0.0414$)
- **800 → 3200:** ganho total de IoU ($+0.0428$), cerca de **11% relativo**

Isso indica que, para esta arquitetura, o ganho de generalização é mais pronunciado quando se aproxima do uso do conjunto completo de treino real.

---

## 5. Comportamento de treinamento e validação

A análise dos `history.csv` mostrou:

1. **Convergência estável** em todos os runs
2. **Early stopping** funcionando como esperado
3. **Val IoU** cresce de forma monotônica nas primeiras épocas e entra em plateau próximo do término
4. Para **N≈3200**, a curva inicia em patamar superior e converge com menos variabilidade entre seeds
5. O Cenário B completo apresentou desempenho de validação competitivo, mas **não converteu essa vantagem em melhor resultado de teste**

Em outras palavras: os dados sintéticos não causaram colapso do treinamento, mas também **não melhoraram a generalização** no test set real.

---

## 6. Conclusão final

### Achado principal

**O treinamento com dados reais + sintéticos não superou o treinamento apenas com dados reais no dataset TGS.**

Resultado médio final:

- **Cenário A:** IoU = **0.4247**, Dice = **0.4598**
- **Cenário B:** IoU = **0.4127**, Dice = **0.4500**

Logo, para o pool sintético atual e a configuração utilizada, a hipótese **B > A** foi **refutada**.

### Implicação para a resposta ao revisor

A seção R2.1 deve reportar o experimento de forma transparente, destacando que:

- o experimento downstream foi implementado e executado com múltiplas seeds;
- a adição de imagens sintéticas **não trouxe ganho** de desempenho no teste real;
- o principal fator de melhoria observado foi o aumento da quantidade de **dados reais**.

---

## 7. Arquivos de referência

- `results/scenario_A_seed42/result.csv`
- `results/scenario_A_seed123/result.csv`
- `results/scenario_A_seed456/result.csv`
- `results/scenario_B_seed42/result.csv`
- `results/scenario_B_seed123/result.csv`
- `results/scenario_B_seed456/result.csv`
- `results/scenario_A_seed42_nreal800/result.csv`
- `results/scenario_A_seed123_nreal800/result.csv`
- `results/scenario_A_seed456_nreal800/result.csv`
- `results/scenario_A_seed42_nreal1200/result.csv`
- `results/scenario_A_seed123_nreal1200/result.csv`
- `results/scenario_A_seed456_nreal1200/result.csv`
- `results/relatorio_cenarioA_800_1200_3200.md`
