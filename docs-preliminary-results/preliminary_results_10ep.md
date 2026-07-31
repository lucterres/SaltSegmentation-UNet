# Resultados Preliminares — Cenário A (Real Only)

**Manuscrito:** Access-2026-27912  
**Experimento:** R2.1 — Downstream Segmentation  
**Configuração:** Scenario A · seed=42 · epochs=10/50 · device=CPU  
**Data:** 2026-07-19  
**Objetivo:** Validar pipeline e observar tendência de melhoria com mais dados reais

---

## Visão geral do processo de treinamento

### Como a rede aprende

A U-Net é treinada por **gradiente descendente** (otimizador Adam): a cada epoch, ela processa todos os batches do conjunto de treino, compara sua predição com a máscara ground truth por meio da função de perda (BCE + Dice), calcula os gradientes e atualiza os pesos. Esse ciclo se repete epoch após epoch.

O comportamento típico ao longo do treinamento é:

- **Loss de treino** decresce monotonicamente — a rede memoriza padrões progressivamente
- **Loss de validação** decresce inicialmente, mas pode oscilar ou subir quando a rede começa a superajustar (*overfitting*) ao conjunto de treino
- **IoU e Dice de validação** sobem gradualmente, atingem um pico e podem estabilizar ou cair levemente após esse ponto

### Critério de parada e checkpoint

O treinamento usa **early stopping**: se o `val_iou` não melhorar por `EARLY_STOP_PATIENCE=10` epochs consecutivas, o treino é interrompido antes de atingir o número máximo de epochs. A cada nova melhoria de `val_iou`, o estado atual dos pesos é salvo como `best_model.pth` — este é o checkpoint que será usado na avaliação final.

```
Epoch N   → val_iou melhora  → salva checkpoint  ✓ best
Epoch N+1 → val_iou piora    → descarta, mantém checkpoint anterior
...
Epoch N+10 → 10 epochs sem melhoria → [Early stop]
```

### Da validação ao teste

Ao final do treino, o `best_model.pth` (selecionado pelo maior `val_iou`) é carregado e avaliado no **test set fixo** (800 imagens reservadas antes de qualquer treinamento e nunca vistas durante o treino ou validação). Essa separação garante que o `test_iou` e `test_dice` reportados sejam estimativas não-enviesadas da capacidade de generalização do modelo.

É esperado e normal que:

- `best_val_iou` ≥ `test_iou` — o checkpoint foi selecionado *com base* na validação, introduzindo um leve viés de seleção
- A diferença entre os dois diminui à medida que o tamanho do conjunto de treino aumenta, pois o modelo generaliza melhor

### Efeito do volume de dados

Com poucos dados de treino (N pequeno), a rede tem capacidade limitada de aprender a distribuição real das imagens sísmicas: converge mais rápido, mas com `val_iou` mais baixo e maior variância entre epochs. Com mais dados, o aprendizado é mais estável, a curva de validação oscila menos e o modelo generaliza melhor para o test set.

---

## Introdução

### Tarefa

Segmentação semântica binária de *salt domes* em imagens sísmicas 2D do dataset **TGS Salt Identification Challenge** (Kaggle, 2018). Cada imagem possui 101×101 pixels (grayscale) e uma máscara binária correspondente indicando regiões de sal (1) e não-sal (0). A U-Net é treinada para prever essa máscara pixel a pixel.

### Arquitetura

**U-Net** com progressão de canais `(1 → 16 → 32 → 64)` adaptada para entrada grayscale de canal único. Modificação principal: `padding=1` nas convoluções para preservar dimensão espacial sem necessidade de *CenterCrop*.

### Parâmetros de treinamento

| Parâmetro | Valor | Descrição |
|-----------|:-----:|-----------|
| `EPOCHS` | 10 (prelim.) / 50 (completo) | Número máximo de épocas |
| `BATCH_SIZE` | 16 | Amostras por iteração |
| `LR` | 1e-4 | Taxa de aprendizado (Adam) |
| `THRESHOLD` | 0.5 | Limiar para binarização da saída sigmoide |
| `EARLY_STOP_PATIENCE` | 10 | Épocas sem melhoria de val-IoU para parar |
| `TEST_SPLIT` | 0.20 | 800 imagens reservadas como test set fixo |
| `VAL_SPLIT` | 0.10 | 10% do pool de treino usado para validação |
| `INPUT_SIZE` | 128×128 | Imagem redimensionada via padding para divisões limpas de MaxPool2d |
| `seed` | 42 | Semente para reprodutibilidade (numpy, torch, random) |
| `device` | CPU | Ryzen 8700G — sem GPU NVIDIA/CUDA |

### Função de perda

Combinação de **Binary Cross-Entropy (BCE)** e **Dice Loss**:

$$\mathcal{L} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$$

$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum p_i \, g_i + \varepsilon}{\sum p_i + \sum g_i + \varepsilon}$$

onde $p_i \in [0,1]$ é a predição sigmoide e $g_i \in \{0,1\}$ é a ground truth, com $\varepsilon = 10^{-6}$ para estabilidade numérica.

### Métricas de avaliação

#### IoU — Intersection over Union (Jaccard Index)

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}$$

Mede a sobreposição entre a máscara predita $P$ e a ground truth $G$. Varia de 0 (sem sobreposição) a 1 (sobreposição perfeita). É a métrica primária do experimento.

#### Dice Coefficient (F1-Score)

$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|} = \frac{2\,TP}{2\,TP + FP + FN}$$

Relacionado ao IoU por $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$. Penaliza menos os falsos positivos/negativos que o IoU, sendo tipicamente mais alto para o mesmo modelo. Métrica secundária do experimento.

> **Nota:** ambas as métricas são calculadas após binarização com `THRESHOLD=0.5`. O *best checkpoint* é salvo com base no maior **val-IoU** ao longo das épocas.

---

## Resultados

### Tabela consolidada — result.csv

Dados extraídos diretamente dos arquivos `result.csv` gerados pelo `train.py` ao final de cada run:

| scenario | seed | n_real | n_synth | best_val_iou | test_iou | test_dice | epochs_run | elapsed_s |
|:--------:|:----:|:------:|:-------:|:------------:|:--------:|:---------:|:----------:|:---------:|
| A | 42 | 200  | 0 | 0.3097 | 0.2589 | 0.3181 | 10 | 249.4 |
| A | 42 | 400  | 0 | 0.3083 | 0.3168 | 0.3583 | 10 | 255.1 |
| A | 42 | 800  | 0 | 0.3470 | 0.3241 | 0.3642 | 10 | 282.5 |
| A | 42 | 1600 | 0 | 0.3272 | 0.3493 | 0.3905 | 10 | 481.3 |

> **Campos:** `best_val_iou` = melhor IoU de validação (critério do checkpoint salvo); `test_iou`/`test_dice` = métricas no test set fixo (800 imagens); `elapsed_s` = tempo total de treino em segundos.

---

### Métricas no Test Set (800 imagens fixas)

| N° imagens treino | Train split | Val split | Test IoU | Test Dice | Tempo (s) | Best epoch |
|:-----------------:|:-----------:|:---------:|:--------:|:---------:|:---------:|:----------:|
| 200               | 180         | 20        | 0.2589   | 0.3181    | 249       | E6         |
| 400               | 360         | 40        | 0.3168   | 0.3583    | 255       | E10        |
| 800               | 720         | 80        | 0.3241   | 0.3642    | 282       | E7         |
| 1600              | 1440        | 160       | **0.3493** | **0.3905** | 481    | E7         |

---

### Evolução por Epoch

#### N=200

| Epoch | Loss treino | Loss val | IoU val | Dice val |
|:-----:|:-----------:|:--------:|:-------:|:--------:|
| 1  | 0.6746 | 0.6936 | 0.2115 | 0.2703 |
| 2  | 0.6536 | 0.7253 | 0.2904 | 0.3419 |
| 3  | 0.6382 | 0.7486 | 0.2927 | 0.3432 |
| 4  | 0.6158 | 0.7168 | 0.2902 | 0.3403 |
| 5  | 0.5974 | 0.6982 | 0.2910 | 0.3438 |
| 6  | 0.5819 | 0.5852 | **0.3097** | **0.3534** |
| 7  | 0.5674 | 0.6407 | 0.2831 | 0.3320 |
| 8  | 0.5411 | 0.5690 | 0.3044 | 0.3483 |
| 9  | 0.5214 | 0.5514 | 0.3055 | 0.3517 |
| 10 | 0.5178 | 0.5443 | 0.3065 | 0.3511 |

#### N=400

| Epoch | Loss treino | Loss val | IoU val | Dice val |
|:-----:|:-----------:|:--------:|:-------:|:--------:|
| 1  | 0.6640 | 0.7443 | 0.2455 | 0.2995 |
| 2  | 0.6223 | 0.7913 | 0.2463 | 0.3000 |
| 3  | 0.5797 | 0.6052 | 0.2460 | 0.3044 |
| 4  | 0.5461 | 0.5251 | 0.2617 | 0.3206 |
| 5  | 0.5214 | 0.5429 | 0.2846 | 0.3427 |
| 6  | 0.5005 | 0.4493 | 0.2382 | 0.2783 |
| 7  | 0.4992 | 0.5552 | 0.3018 | 0.3586 |
| 8  | 0.4750 | 0.4671 | 0.2604 | 0.2975 |
| 9  | 0.4584 | 0.4176 | 0.2671 | 0.3018 |
| 10 | 0.4375 | 0.4381 | **0.3083** | **0.3550** |

#### N=800

| Epoch | Loss treino | Loss val | IoU val | Dice val |
|:-----:|:-----------:|:--------:|:-------:|:--------:|
| 1  | 0.6443 | 0.7168 | 0.2455 | 0.3098 |
| 2  | 0.5597 | 0.5147 | 0.2810 | 0.3391 |
| 3  | 0.5175 | 0.4703 | 0.3070 | 0.3548 |
| 4  | 0.4903 | 0.4546 | 0.2933 | 0.3360 |
| 5  | 0.4673 | 0.4332 | 0.3177 | 0.3630 |
| 6  | 0.4549 | 0.3860 | 0.2850 | 0.3213 |
| 7  | 0.4316 | 0.4075 | **0.3470** | **0.3903** |
| 8  | 0.4341 | 0.3883 | 0.3349 | 0.3781 |
| 9  | 0.4075 | 0.3486 | 0.3240 | 0.3625 |
| 10 | 0.3874 | 0.3547 | 0.3317 | 0.3670 |

#### N=1600

| Epoch | Loss treino | Loss val | IoU val | Dice val |
|:-----:|:-----------:|:--------:|:-------:|:--------:|
| 1  | 0.6073 | 0.5234 | 0.2516 | 0.3081 |
| 2  | 0.5005 | 0.4610 | 0.2695 | 0.3153 |
| 3  | 0.4550 | 0.4202 | 0.2840 | 0.3187 |
| 4  | 0.4160 | 0.4066 | 0.2991 | 0.3369 |
| 5  | 0.4070 | 0.3592 | 0.2748 | 0.3069 |
| 6  | 0.3775 | 0.3772 | 0.2932 | 0.3301 |
| 7  | 0.3597 | 0.4025 | **0.3272** | **0.3738** |
| 8  | 0.3524 | 0.3291 | 0.2953 | 0.3259 |
| 9  | 0.3443 | 0.3268 | 0.3153 | 0.3551 |
| 10 | 0.3332 | 0.3161 | 0.3172 | 0.3533 |

---

## Análise

### Observações

- **Tendência clara:** IoU e Dice aumentam consistentemente com mais dados reais (200 → 1600 imagens)
- **10 epochs é subótimo:** os modelos ainda não convergiram (loss treino decrescente em todas as runs) — o experimento completo usa 50 epochs com early stopping
- **`best_val_iou` vs `test_iou`:** para N=800 o `best_val_iou`=0.347 é maior que o `test_iou`=0.324 — esperado, pois val e test têm distribuições ligeiramente diferentes por serem splits independentes
- **Variância alta com N pequeno:** com N=200/400 o best epoch oscila bastante, indicando instabilidade sem dados suficientes
- **Estimativa para experimento completo (50 epochs, CPU):**
  - N=800 → ~23 min/run · 6 runs ≈ **~2.5 h**
  - N=1600 → ~40 min/run · 6 runs ≈ **~4 h**

### Recomendações

- Usar `n_real=800` como configuração padrão para o experimento completo (conforme protocolo R2.1)
- Aumentar para 50 epochs para garantir convergência
- Rodar 3 seeds (42, 123, 456) para estimar variância e calcular mean ± std

---

## Próximos passos

1. Gerar pool sintético via `generate_synthetic.py` (Cenário B)
2. Rodar 6 runs completas: Cenário A × 3 seeds + Cenário B × 3 seeds (50 epochs cada)
3. Executar `evaluate.py` para gerar `summary.csv`
4. Preencher tabela do manuscrito (`_v7.tex`)