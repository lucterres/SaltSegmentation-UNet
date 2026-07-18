# Experimento Downstream — R2.1

**Manuscrito:** Access-2026-27912  
**Objetivo:** Demonstrar que treinar com dados reais + sintéticos melhora a segmentação de salt domes em dados reais de teste, em comparação com treinar apenas com dados reais.

---

## Estrutura da pasta

```
experiment-downstream/
├── README.md                    ← este arquivo
├── Salt-Segmentation-UNet/      ← repositório base (clonado, modificado)
│   ├── utils/
│   │   ├── config.py            ← configurações centrais
│   │   ├── model.py             ← U-Net (modificado: padding=1)
│   │   └── dataset.py           ← dataloader TGS (modificado: NEAREST para máscara)
│   ├── train.py                 ← loop de treino (modificado: IoU/Dice, seeds, cenários)
│   ├── evaluate.py              ← avaliação final no test set fixo
│   ├── generate_synthetic.py    ← gera pool sintético via VAE + textura
│   └── dataset/
│       ├── tgs/                 ← dados TGS (images/ + masks/)  ← BAIXAR
│       └── synthetic/           ← gerado por generate_synthetic.py
│           ├── images/
│           └── masks/
└── results/
    ├── scenario_A/              ← resultados seed 42, 123, 456
    ├── scenario_B/              ← resultados seed 42, 123, 456
    └── summary.csv              ← tabela final IoU/Dice
```

---

## Pré-requisitos

### 1 — Baixar o dataset TGS

```powershell
# Opção A: Kaggle CLI
kaggle competitions download -c tgs-salt-identification-challenge
# Descompactar em: Salt-Segmentation-UNet/dataset/tgs/

# Opção B: Download manual em https://www.kaggle.com/c/tgs-salt-identification-challenge/data
# Extrair train.zip → imagens em train/images/, máscaras em train/masks/
# Copiar para: Salt-Segmentation-UNet/dataset/tgs/images/ e dataset/tgs/masks/
```

Estrutura esperada após extração:
```
Salt-Segmentation-UNet/dataset/tgs/
├── images/   ← 4000 arquivos .png (101×101 px, grayscale)
└── masks/    ← 4000 arquivos .png (101×101 px, binário)
```

### 2 — Instalar dependências

```powershell
cd D:\IEEEEAccess\experiment-downstream\Salt-Segmentation-UNet
pip install -r requirements.txt
# Pacotes adicionais necessários:
pip install scikit-learn pandas tqdm
```

---

## Fluxo de execução

### Passo 1 — Gerar pool sintético (Cenário B)

```powershell
# A partir da raiz do workspace (onde está o VAE treinado)
cd D:\IEEEEAccess
python experiment-downstream/Salt-Segmentation-UNet/generate_synthetic.py --n 400 --out experiment-downstream/Salt-Segmentation-UNet/dataset/synthetic
```

Saída esperada:
- `dataset/synthetic/images/` — 400 imagens sintéticas .png
- `dataset/synthetic/masks/`  — 400 máscaras binárias .png

### Passo 2 — Treinar Cenário A (real only) — 3 seeds

```powershell
cd D:\IEEEEAccess\experiment-downstream\Salt-Segmentation-UNet
python train.py --scenario A --seed 42
python train.py --scenario A --seed 123
python train.py --scenario A --seed 456
```

### Passo 3 — Treinar Cenário B (real + synthetic) — 3 seeds

```powershell
python train.py --scenario B --seed 42
python train.py --scenario B --seed 123
python train.py --scenario B --seed 456
```

### Passo 4 — Avaliar no test set fixo

```powershell
python evaluate.py --results_dir ../results
```

Saída: `../results/summary.csv` com IoU e Dice para cada seed/cenário.

### Passo 5 — Low-data regime (opcional mas recomendado)

```powershell
# N=50 real only e N=50+200 synthetic
python train.py --scenario A --seed 42 --n_real 50
python train.py --scenario B --seed 42 --n_real 50 --n_synth 200
# Repetir para N=100, N=200 e seeds 123, 456
```

---

## Modificações aplicadas ao repositório base

| Arquivo | Modificação | Motivo |
|---------|------------|--------|
| `utils/model.py` | `padding=1` nos `Conv2d` do `Block` | Preserva dimensão espacial; elimina necessidade de `CenterCrop` |
| `utils/dataset.py` | Separar transforms imagem/máscara; `NEAREST` para máscara | Evita artefatos de interpolação bilinear na ground truth |
| `utils/config.py` | Caminhos para TGS e sintéticos; `ENCODER_CHANNELS=(1,16,32,64)` | TGS é grayscale (1 canal, não RGB) |
| `train.py` | IoU + Dice no loop de validação; `--scenario`, `--seed`, `--n_real`, `--n_synth`; early stopping por IoU; split estratificado | Requisitos do experimento |
| `evaluate.py` | Novo arquivo: avaliação final no test set fixo, gera `summary.csv` | Reprodutibilidade |
| `generate_synthetic.py` | Novo arquivo: interface com VAE + pipeline de textura do paper | Geração do pool sintético |

---

## Resultado esperado

Tabela alvo para o manuscrito (`_v7.tex`, subseção Downstream Segmentation Evaluation):

| Cenário | IoU (mean ± std) | Dice (mean ± std) | p-value |
|---------|-----------------|------------------|---------|
| A — Real only (N=800) | — | — | — |
| B — Real + Synthetic (N=800+400) | — | — | — |

Após executar os experimentos, preencher esta tabela com `../results/summary.csv` e atualizar:
1. `_v7.tex` — subseção `\subsection{Downstream Segmentation Evaluation}`
2. `docs/_reviewACCESS/response_to_reviewers.md` — seção R2.1
3. `docs/_reviewACCESS/summary_of_changes.md` — status R2.1 de **PENDING** → **DONE**

---

## Referências

- Protocolo completo: [`docs/_reviewACCESS/experimentUNet-protocol.md`](../../docs/_reviewACCESS/experimentUNet-protocol.md)
- Comentário do revisor: [`docs/_reviewACCESS/_Reviewer.md`](../../docs/_reviewACCESS/_Reviewer.md) — Reviewer 2, Issue 1
- Repositório base original: https://github.com/matin-ghorbani/Salt-Segmentation-UNet
