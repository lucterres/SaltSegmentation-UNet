# Métodos de Data Augmentation — Referência Experimental

> **Fonte:** Henriques, Luis, Sérgio Colcher, Ruy Milidiú, André Bulcão, e Pablo Barros.  
> *"Generating Data Augmentation samples for Semantic Segmentation of Salt Bodies in a Synthetic Seismic Image Dataset"*.  
> arXiv:2106.08269. Preprint, arXiv, 15 de junho de 2021.  
> <http://arxiv.org/abs/2106.08269>

---

## Métodos Albumentations usados na Seção 6.3

A comparação "contra outros métodos de Data Augmentation" (Seção 6.3) é feita usando métodos da biblioteca [Albumentations](https://albumentations.ai/). O texto afirma explicitamente que os autores *"compare the performance of the proposed DA method against seven distinct methods from the Albumentation library [54]"*.

Os métodos/técnicas usados foram:

1. Elastic Transform
2. Grid Distortion
3. Optical Distortion
4. CLAHE
5. Random Brightness Contrast
6. Random Gamma

Além de cada método de forma **stand-alone**, também é avaliada a composição **"+ Ours"** (cada método combinado com o método proposto).

> **Fonte:** agente Documentos, trecho da Seção 6.3 do PDF (chunk_index 4).

---

## Detalhes Experimentais para Reprodução (Seção 6.3)

### Modelo e Implementação

| Parâmetro | Valor |
|-----------|-------|
| Modelo | DeepLabV3+ com backbone `mobilenet_v3_large` |
| Implementação | [TensorFlow DeepLab oficial](https://github.com/tensorflow/models/tree/master/research/deeplab) |
| `decoder_output_stride` (MobileNetV3) | 8 |
| `atrous_rates` / `output_stride` (MobileNet V2/V3) | deixados em branco |

### Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Otimizador | Adam |
| Iterações | 40 000 |
| Mini-batch | 20 exemplos por iteração |
| Hardware (informativo) | Nvidia Tesla P100 (~1h10min por run) |

### Dataset / Entrada

| Parâmetro | Valor |
|-----------|-------|
| Patch size | 64 × 64 |
| Total de amostras | 6 000 pares |
| Treino | 2 450 |
| Validação | 1 434 |
| Teste | 2 042 |
| Métrica | IoU @ limiar 0.5 |

---

## O que exatamente foi comparado (setup do experimento 6.3)

O artigo diz que, na Seção **6.3**, eles avaliam:

- **Modelo:** DeepLabV3+ com backbone `mobilenet_v3_large`.
- **Métodos de DA comparados (Albumentations):** Elastic Transform, Grid Distortion, Optical Distortion, CLAHE, Random Brightness Contrast, Random Gamma.
- **Cenários avaliados:**
  - Cada método de DA **stand-alone** ("Alone").
  - Cada método de DA **composto com o método proposto** ("+ Ours").
- **Regra de composição:**
  > *"only the original samples are passed through the transformations. Therefore, our DA samples are only presented to the models without being passed through any transformation."*
- **Treinamento:**
  - *"All models are trained using the same settings described in section 6.2."*
  - Tempo médio: *"1 hour and 10 minutes on an Nvidia Tesla P100 GPU."*

> **Fonte:** agente Documentos, chunk_index 4.

---

## Dataset e Pré-processamento (para reproduzir o pipeline de dados)

### Origem dos dados

Os dados vêm de **dois modelos sintéticos de sal publicamente disponíveis**:

- **Pluto1.5 dataset** [20]
- **SEG/EAGE Salt model** [21]

### Extração

- *"We extract a 2D migrated image from each seismic model."*
- *"binary salt mask is extracted from its velocity model by thresholding and clipping."*

### Normalização

- *"each migrated image is normalized by its mean and standard deviation."*

### Split

- *"We split the images into training, validation, and test sets."* (percentuais não explicitados no texto).

### Geração de patches (usados em 6.2 / 6.3)

| Parâmetro | Valor |
|-----------|-------|
| Tamanho do patch | 64 × 64 |
| Grade | overlapping grid com **10% de overlap** entre patches adjacentes |
| Partições de teste | sem overlap |
| Augmentation de base | horizontal flip em cada par resultante |

### Tamanho final do dataset (segmentação semântica)

| Split | Amostras |
|-------|---------|
| Treino | 2 450 |
| Validação | 1 434 |
| Teste | 2 042 |
| **Total** | **6 000 pares** |

---

## Protocolo de Avaliação

Avaliar o DeepLabV3+ treinado em **três modos** para cada transformação:

1. **None** — baseline sem augmentation
2. **Alone** — apenas o método Albumentations
3. **+ Ours** — método Albumentations composto com o método proposto

### Regra Crítica de Composição

> *"only the original samples are passed through the transformations. Therefore, our DA samples are only presented to the models without being passed through any transformation."*

Em **"+ Ours"**: aplique o Albumentations somente aos dados originais; os samples gerados pelo método proposto entram **sem** Albumentations.

### Tamanho do "Ours"

Adição de **300 pares gerados** (máscaras de sal + patches de imagem) — configuração descrita no 6.2 e mantida no 6.3.

---

## Limitações para Reprodução Fiel

O artigo **não especifica** os parâmetros internos de cada transformação do Albumentations (ex.: `p`, limites, intensidades). Para reproduzir, escolha e documente uma das opções:

- usar os **defaults** do Albumentations, ou
- **fixar manualmente** os parâmetros, ou
- realizar **busca de hiperparâmetros** em validação.