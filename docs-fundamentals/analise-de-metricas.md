# Análise de Métricas — Impacto de Máscaras com Zero Sal

## 1. Problema: Máscara com 0% de sal quebra o cálculo do IoU

### 1.1 O que acontece matematicamente

O **IoU (Intersection over Union)** é definido como:

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$

Quando a máscara ground truth tem **zero sal** ($G = \emptyset$):

- Se o modelo prevê corretamente "nada" ($P = \emptyset$): o numerador **e** o denominador são ambos zero → **0/0 = NaN** (indefinido)
- Se o modelo prevê algum sal erroneamente ($P \neq \emptyset$): numerador = 0, denominador > 0 → **IoU = 0** (penalização severa)

O **Dice** tem o mesmo comportamento:

$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}$$

Quando $G = \emptyset$ e $P = \emptyset$: **0/0 = NaN**. Quando $P \neq \emptyset$: **Dice = 0**.

---

## 2. Composição do dataset TGS

O dataset TGS completo possui aproximadamente **47% de imagens sem sal** (cobertura 0%). Isso significa que quase metade do test set produz **NaN ou 0** no IoU por imagem, e a média global é puxada fortemente para baixo.

| Faixa de cobertura | Proporção aproximada no TGS |
|---|---|
| 0% de sal (sem sal) | ~47% |
| 1–99% de sal (casos difíceis) | ~38% |
| 100% de sal (totalmente coberto) | ~15% |

---

## 3. Por que o `subset_10_90` tem IoU ~0.84 com metade das amostras

Ao filtrar para apenas imagens com **10–90% de cobertura de sal**, eliminam-se os casos triviais (0% e 100%). O modelo é treinado e avaliado **apenas nos casos difíceis**, onde o IoU é bem definido e significativo.

Por isso o IoU **praticamente dobra** com menos amostras:

| Dataset | N amostras | Test IoU médio | Test Dice médio |
|---|:---:|:---:|:---:|
| TGS completo (distribuição real) | ~3198 | 0.4247 | 0.4598 |
| `subset_10_90` (filtrado 10–90%) | 1616 | **0.8441** | **0.9012** |

O ganho de ~+0.42 IoU **não é ganho de aprendizado** — é eliminação dos casos que produzem IoU=0 ou NaN na média.

---

## 4. Por que isso distorce a comparação Cenário A vs Cenário B

No dataset completo, os ~426 casos com 0% de sal no test set:

1. Não exercitam a capacidade discriminativa do modelo
2. Contribuem com **0 ou NaN** para a média de IoU
3. **Mascaram o efeito real dos dados sintéticos** — o sinal de melhoria dos sintéticos sísmicos existe (demonstrado no `subset_split`: +0.011 IoU), mas é diluído pelos casos triviais

### Experimento de referência — `subset_split` (seed 42)

| Cenário | Train | Test | Test IoU | Δ IoU |
|---|---|---|:---:|:---:|
| A — Real only | 1293 filtrado (10–90%) | distribuição real completa | 0.4201 | — |
| **B — Real + 955 sísmicos** | 1293 filtrado (10–90%) | distribuição real completa | **0.4308** | **+0.011** |

Este foi o **único cenário onde B > A foi observado**, justamente porque o treino filtrado eliminava os casos triviais que prejudicam o aprendizado diferencial dos sintéticos.

---

## 5. Resumo do impacto

| Situação | Efeito no IoU médio do test set |
|---|---|
| Muitas máscaras com 0% sal no test | IoU artificialmente baixo (~0.42) |
| Filtrar test para 10–90% de sal | IoU real nos casos difíceis (~0.84) |
| Sintéticos sísmicos com train filtrado | Ganho visível (+0.011) |
| Sintéticos + train completo (com 0% sal) | Ganho mascarado, aparece como perda |

---

## 6. Implicações para o design de experimentos

1. **Reportar a distribuição do test set** junto com as métricas — IoU médio sem essa informação é ambíguo.
2. **Calcular IoU separadamente por faixa de cobertura** (0%, 1–99%, 100%) para diagnóstico mais preciso.
3. **Dados sintéticos devem ter distribuição compatível com os casos difíceis** (10–90% de cobertura) para maximizar utilidade.
4. Quando o objetivo é avaliar a **qualidade de segmentação de sal**, o test set filtrado (10–90%) é mais informativo; quando o objetivo é avaliar **robustez geral**, o test set completo é necessário.

---

## 7. Referências internas

- Dados experimentais completos: `docs/relatorio-final-r21-downstream.md`
- Protocolo de experimentos: `docs-fundamentals/experimentUNet-protocol.md`
