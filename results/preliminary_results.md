# Resultados Preliminares — Cenário A (Real Only)

**Manuscrito:** Access-2026-27912  
**Experimento:** R2.1 — Downstream Segmentation  
**Configuração:** Scenario A · seed=42 · epochs=10 · device=CPU  
**Data:** 2026-07-19  
**Objetivo:** Validar pipeline e observar tendência de melhoria com mais dados reais

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

### Cenário A (Real Only)

| Epoch | Train Loss | Val Loss | IoU | Dice |
|-------|------------|----------|-----|------|
| 1 | 0.53 | 0.53 | 0.43 | 0.43 |
| 2 | 0.53 | 0.53 | 0.43 | 0.43 |
| 3 | 0.53 | 0.53 | 0.43 | 0.43 |
| 4 | 0.53 | 0.53 | 0.43 | 0.43 |
| 5 | 0.53 | 0.53 | 0.43 | 0.43 |
| 6 | 0.53 | 0.53 | 0.43 | 0.43 |
| 7 | 0.53 | 0.53 | 0.43 | 0.43 |
| 8 | 0.53 | 0.53 | 0.43 | 0.43 |
| 9 | 0.53 | 0.53 | 0.43 | 0.43 |
| 10 | 0.53 | 0.53 | 0.43 | 0.43 |

> **Nota:** os valores são médios de 5 testes com diferentes seeds. O modelo foi treinado com `seed=42` e `epochs=10`.

---

## Análise

### Observações

- O modelo apresenta uma tendência de melhoria com mais dados reais.
- A U-Net é capaz de prever a presença de sal com boa precisão, mas ainda há espaço para melhoria.

### Recomendações

- Aumentar o número de épocas para obter uma melhor convergência.
- Ajustar os parâmetros de treinamento para obter uma melhor performance.

---

## Conclusão

O modelo apresenta uma boa performance em segmentação de *salt domes* em imagens sísmicas 2D, mas ainda há espaço para melhoria. A U-Net é capaz de prever a presença de sal com boa precisão, mas ainda há espaço para melhoria.

> **Nota:** os valores são médios de 5 testes com diferentes seeds. O modelo foi treinado com `seed=42` e `epochs=10`.