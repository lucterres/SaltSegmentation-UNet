# Protocolo — Experimento de Segmentação Downstream (R2.1)

**Manuscrito:** Access-2026-27912  
**Comentário:** Reviewer 2, Issue 1 — Downstream segmentation experiment  
**Status:** PENDING — experimento ainda não realizado  
**Data de criação:** 2026-07-17

---

## 0 — Repositório Base Recomendado

**Repositório:** [`matin-ghorbani/Salt-Segmentation-UNet`](https://github.com/matin-ghorbani/Salt-Segmentation-UNet)  
**Licença:** MIT · **Stars:** 16 · **Linguagem:** Python 100%

### Por que este repositório

É a implementação mais próxima de um U-Net "paper-fiel" disponível para o TGS dataset:
- Vanilla U-Net (sem encoders pré-treinados, sem atenção, sem módulos especiais)
- PyTorch puro, ~200 linhas de código total, fácil de ler e modificar
- Dataloader para TGS já implementado
- Mesma arquitetura citada no `_v7.tex` (`\cite{Ronneberger2015}`)

### Arquitetura atual (código-fonte verificado)

```python
# utils/model.py — estrutura real do repositório

class Block(nn.Module):
    # Conv2d(in, out, 3) → ReLU → Conv2d(out, out, 3)
    # SEM BatchNorm, SEM padding → reduz dimensão em 4px por bloco

class Encoder(nn.Module):
    # channels: (3, 16, 32, 64) — padrão
    # ModuleList de Block + MaxPool2d(2)
    # Retorna lista de block_outputs para skip connections

class Decoder(nn.Module):
    # channels: (64, 32, 16) — padrão
    # ConvTranspose2d(ch[i], ch[i+1], 2, 2) + CenterCrop para skip + Block

class UNet(nn.Module):
    # retain_dim=True → F.interpolate para restaurar out_size
    # head: Conv2d(16, 1, 1) — logits para BCEWithLogitsLoss
```

**Input:** imagem RGB 3 canais → `transforms.Resize` para `INPUT_IMAGE_HEIGHT × INPUT_IMAGE_WIDTH`  
**Output:** mapa de segmentação 1 canal (logits), redimensionado para o tamanho original via `F.interpolate`

### Loop de treino atual (código-fonte verificado)

```python
# train.py — loop real

optimizer = torch.optim.Adam(unet.parameters(), config.LR)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(1, config.EPOCHS + 1):
    # treino
    for batch in train_loader:
        x, y = batch  → unet(x) → loss_fn → backward
    # validação (BCE loss apenas — SEM IoU/Dice)
    for batch in test_loader:
        y_pred = unet(x) → total_test_loss += loss_fn(y_pred, y)
```

**Problema identificado:** o loop atual só rastreia BCE loss — **não calcula IoU nem Dice**.

### Dataset atual (código-fonte verificado)

```python
# utils/dataset.py — dataloader real

class SegmentationDataset(Dataset):
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # 3 canais RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)           # 1 canal
        img = self.transforms(img)    # Resize + ToTensor
        mask = self.transforms(mask)  # Resize + ToTensor
        return img, mask
```

**Problema identificado:** a máscara passa pelo mesmo `transforms.Resize` que a imagem — interpolação bilinear pode criar valores não-binários. Precisa de interpolação `NEAREST` para máscaras.

---

### Modificações necessárias no repositório base

As modificações abaixo são **mínimas e cirúrgicas** — preservam a arquitetura intacta para garantir que os dois cenários (A e B) sejam comparáveis.

| # | Arquivo | Modificação | Impacto |
|---|---------|------------|---------|
| 1 | `utils/config.py` | Ajustar `INPUT_IMAGE_HEIGHT/WIDTH` para 128 (pad 101→128) ou manter 101; definir caminhos para sintéticos | Config |
| 2 | `utils/model.py` | Adicionar `padding=1` nos `Conv2d` dentro de `Block` para preservar dimensão sem `F.interpolate` | Qualidade |
| 3 | `utils/dataset.py` | Separar transforms de imagem e máscara; usar `InterpolationMode.NEAREST` para máscara; suportar path de sintéticos | Correção |
| 4 | `train.py` | Adicionar cálculo de **IoU e Dice** no loop de validação; salvar melhor modelo por IoU (não por loss); suportar argumento `--scenario A\|B`; implementar **3 seeds** | Essencial |
| 5 | `train.py` | Mudar `train_test_split` para **split estratificado por cobertura de sal** com seed fixa | Rigor |
| 6 | `train.py` (novo) | Adicionar `--augment` flag para cenário B: carrega pool sintético e concatena ao train set | Feature central |
| 7 | `evaluate.py` (novo) | Script separado: carrega modelo salvo, roda no test set, reporta IoU/Dice médio ± std | Resultado |

#### Modificação 2 — detalhe (`Block` com padding)

```python
# ANTES (reduz dimensão em 4px por bloco — requer CenterCrop no decoder)
self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

# DEPOIS (preserva dimensão — mais limpo para 101×101 e 128×128)
self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
```

#### Modificação 4 — IoU e Dice (5 linhas)

```python
def compute_metrics(pred_logits, targets, threshold=0.5):
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou   = (intersection / (union + 1e-8)).mean().item()
    dice  = (2 * intersection / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1e-8)).mean().item()
    return iou, dice
```

#### Modificação 6 — Cenário B (augmentation flag)

```python
# train.py — adição para cenário B
if args.scenario == 'B':
    synth_imgs  = sorted(glob(config.SYNTH_IMAGE_PATH + '/*.png'))
    synth_masks = sorted(glob(config.SYNTH_MASK_PATH  + '/*.png'))
    synth_dataset = SegmentationDataset(synth_imgs, synth_masks, transform)
    train_dataset = ConcatDataset([train_dataset, synth_dataset])
```

---

## 1 — Dataset e Splits

**Dataset:** TGS Salt Identification Challenge  
- Imagens: $101 \times 101$ px, 8-bit grayscale  
- Ground truth: máscara binária pixel-level (salt / no-salt)  
- Disponibilidade: público (Kaggle), ~4.000 imagens com rótulo  

**Split estratificado por cobertura de sal:**

| Split | Tamanho | Papel |
|-------|---------|-------|
| **Train-real** | 800 imagens | Treino base (ambos os cenários) |
| **Test** | 800 imagens | Avaliação final — fixo, nunca visto no treino |
| **Pool sintético** | 400 imagens geradas | Augmentation no cenário B |

> **Regra de isolamento:** as imagens sintéticas do pool são geradas a partir de **máscaras novas** amostradas do VAE treinado — nunca a partir de imagens do test set. Isso garante que o modelo B não veja informação do test set indiretamente, nem pela textura nem pela geometria.

**Estratificação:** o split train/test deve ser estratificado por faixas de cobertura de sal (e.g., 0–10%, 10–30%, 30–50%, >50%) para garantir distribuições equivalentes nos dois conjuntos.

---

## 2 — Cenários de Treinamento

| Cenário | Dados de treino | Finalidade |
|---------|----------------|-----------|
| **A — Real only** | 800 imagens reais | Baseline |
| **B — Real + Synthetic** | 800 reais + 400 sintéticos | Augmented |

Os dois modelos devem ser **idênticos em arquitetura e hiperparâmetros** — a única variável independente é o conjunto de treino.

---

## 3 — Modelo de Segmentação

**Escolha:** U-Net padrão (`\cite{Ronneberger2015}`)

**Justificativa:** arquitetura canônica para segmentação semântica binária; compatível com entradas $101\times101$ px; amplamente usada na literatura de imagens sísmicas (campeões do desafio TGS Kaggle); já citada no paper (`\cite{Ronneberger2015}`).

**Configuração:**

```
Encoder: 4 níveis
  - Conv2d 3×3, padding=1, BatchNorm2d, ReLU
  - Canais: 1 → 16 → 32 → 64 → 128
  - MaxPool2d 2×2 após cada nível

Bottleneck: Conv2d 128 → 256

Decoder: 4 níveis (mirror do encoder)
  - Bilinear upsampling ou ConvTranspose2d 2×2
  - Skip connections concatenadas do encoder
  - Conv2d 3×3, padding=1, BatchNorm2d, ReLU

Output: Conv2d 1×1 → (sem ativação — logits para BCEWithLogitsLoss)
        ou → sigmoid para inferência
```

**Hiperparâmetros de treinamento:**

| Parâmetro | Valor |
|-----------|-------|
| Loss | `BCEWithLogitsLoss` (ou Dice Loss — testar) |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Epochs | 50 (máximo) |
| Early stopping | patience = 10 (monitorar val-IoU) |
| Validation split | 10% do train-real (80 imagens), estratificado |
| Input normalization | pixel / 255.0 → [0, 1] |
| Data augmentation intra-treino | horizontal flip (p=0.5), vertical flip (p=0.5) — apenas transformações básicas, iguais nos dois cenários |

---

## 4 — Métricas de Avaliação

Calculadas no **test set real** (fixo e idêntico para ambos os cenários):

| Métrica | Fórmula | Relevância |
|---------|---------|-----------|
| **IoU** (Jaccard) | $\frac{TP}{TP+FP+FN}$ | Padrão para segmentação semântica; usada no desafio TGS Kaggle |
| **Dice coefficient** | $\frac{2\cdot TP}{2\cdot TP+FP+FN}$ | Equivalente ao F1-score; mais sensível em classes desbalanceadas |
| **Pixel accuracy** | $\frac{TP+TN}{N}$ | Complementar; pode ser enganosa com desbalanceamento |

**Análise estatística:**
- Treinar cada cenário com **3 seeds independentes** (ex.: 42, 123, 456)
- Reportar: média ± desvio padrão de IoU e Dice para cada cenário
- Teste de significância: **teste t pareado** (ou Wilcoxon signed-rank se distribuição não-normal) entre os 3 valores de IoU do cenário A e os 3 do cenário B
- Nível de significância: $\alpha = 0.05$

**Tabela alvo no manuscrito:**

| Cenário | IoU (mean ± std) | Dice (mean ± std) |
|---------|-----------------|------------------|
| A — Real only (N=800) | [TODO] | [TODO] |
| B — Real + Synthetic (N=800+400) | [TODO] | [TODO] |
| $p$-value (A vs. B) | [TODO] | [TODO] |

---

## 5 — Variante: Sub-experimento de Low-Data Regime

Motivação: o paper argumenta que o método é especialmente valioso em cenários de **escassez de dados anotados** (exploração offshore). O ganho de augmentation é tipicamente mais pronunciado quando N é pequeno.

| Sub-cenário | Train-real | Train-sintético | Total |
|-------------|-----------|----------------|-------|
| N=50 — Real only | 50 | 0 | 50 |
| N=50 — Real + Synth | 50 | 200 | 250 |
| N=100 — Real only | 100 | 0 | 100 |
| N=100 — Real + Synth | 100 | 400 | 500 |
| N=200 — Real only | 200 | 0 | 200 |
| N=200 — Real + Synth | 200 | 400 | 600 |

O test set permanece fixo (800 imagens reais).

**Gráfico alvo:** curva IoU × N (real only vs. real+synthetic) — demonstra o benefício marginal da augmentation em função do tamanho do dataset real.

---

## 6 — Geração do Pool Sintético

1. Usar o **VAE já treinado** (`_v7.tex`, Sec. III-A): amostrar 400 vetores latentes $z \sim \mathcal{N}(0, I)$, $d=100$
2. Decodificar cada $z$ para uma máscara binária $101\times101$ px
3. Aplicar o **pipeline de síntese de textura** (`_v7.tex`, Sec. III-B) a cada máscara, usando o banco de texturas do TGS
4. **Verificar:** nenhuma imagem sintética deve ser derivada de imagens do test set
5. **Filtrar:** descartar imagens incoerentes (mesmo critério do ablation study: coerência geológica)
6. Output esperado: 400 pares `(imagem_sintetica, mascara_binaria)` prontos para o dataloader

---

## 7 — Estrutura da Nova Seção no `_v7.tex`

Inserir como nova subsection dentro de `\section{Results and Discussion}` (`\label{sec:results}`), logo após a subseção de Comparative Analysis (após a Table `tab:comparison_overview`):

```latex
\subsection{Downstream Segmentation Evaluation}
\label{sec:downstream}

\subsubsection{Experimental Setup}
% - Dataset split (800 train / 800 test, estratificado)
% - U-Net config
% - Cenários A e B
% - 3 seeds, métricas IoU e Dice

\subsubsection{Results}
% Tabela: IoU ± std, Dice ± std, p-value
% Análise e discussão

\subsubsection{Low-data Regime Analysis}
% Tabela ou figura: IoU × N (real only vs real+synthetic)
% Discussão sobre utilidade em cenários de escassez
```

---

## 8 — Atualização do `response_to_reviewers.md`

Quando o experimento for concluído, adicionar em `response_to_reviewers.md` a seção:

```markdown
### Comment R2.1 — Downstream Segmentation Experiment

> *"The manuscript should include a downstream segmentation experiment..."*

**Response:**
[Descrever o experimento, resultados e onde foram inseridos no manuscrito]

**Action taken in the revised manuscript (_v7.tex):**
New subsection "Downstream Segmentation Evaluation" added: Section IV-E,
lines [TODO]–[TODO] in _v7.tex.
```

---

## 9 — Estimativa de Esforço

| Tarefa | Estimativa |
|--------|-----------|
| Implementar U-Net + dataloader TGS (PyTorch) | 2–4 h |
| Gerar 400 imagens sintéticas (VAE já treinado) | 1–2 h |
| Treinar 6 modelos (3 seeds × 2 cenários, 50 epochs) | ~2 h em GPU |
| Sub-experimento low-data (adicional 12 modelos) | +3 h |
| Escrever seção LaTeX no manuscrito | 2 h |
| Atualizar `response_to_reviewers.md` e `summary_of_changes.md` | 1 h |
| **Total estimado** | **~12–14 h** |

---

## 10 — Resultado Esperado e Interpretação

**Resultado positivo (augmentation ajuda):**
- IoU(B) > IoU(A), diferença ≥ 1–2 pontos percentuais, $p < 0.05$
- Evidência direta de que os dados sintéticos são úteis como data augmentation
- Forte resposta ao R2.1

**Resultado neutro/negativo:**
- Também publicável: indica que a qualidade da síntese não degrada o modelo
- Possível explicação: TGS tem ~4k imagens → o regime de escassez não é ativo com N=800
- Ação: reportar candidamente + mostrar ganho no sub-experimento low-data (N=50/100)
- Discussão: augmentation tem maior impacto quando dados reais são escassos, que é exatamente o cenário motivador do paper (exploração offshore)

---

## Checklist de Implementação

- [ ] Implementar U-Net (PyTorch) com as especificações da Seção 3
- [ ] Implementar dataloader TGS com split estratificado (seed fixo = 42)
- [ ] Gerar 400 imagens sintéticas com VAE + pipeline de textura
- [ ] Treinar e avaliar Cenário A (3 seeds)
- [ ] Treinar e avaliar Cenário B (3 seeds)
- [ ] Executar sub-experimento low-data (N=50, 100, 200)
- [ ] Registrar resultados: IoU, Dice, p-value
- [ ] Inserir subseção `\subsection{Downstream Segmentation Evaluation}` em `_v7.tex`
- [ ] Atualizar `response_to_reviewers.md` — seção R2.1
- [ ] Atualizar `summary_of_changes.md` — status R2.1 de **PENDING** → **DONE**
- [ ] Regenerar `latex_build/Highlighted_PDF.pdf`
