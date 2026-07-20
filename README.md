# Experimento Downstream — R2.1

**Manuscrito:** Access-2026-27912  
**Objetivo:** Demonstrar que treinar com dados reais + sintéticos melhora a segmentação de salt domes em dados reais de teste, em comparação com treinar apenas com dados reais.
1. Estimar desempenho com máximo treinamento 4000 amostras, depois 8000 com data augmentation
2. Comparar com dados sintéticos gerados por transformações geométricas e outras formas de data augmentation
3. Rodar o experimento com 800 + 800

---

## Infraestrutura de execução

| Item | Detalhe |
|------|---------|
| **Servidor** | `atn2b02n07` (Atena) |
| **GPUs** | 8 × Tesla V100-SXM2-32GB |
| **CUDA** | 12.9 (driver 575) |
| **Python** | 3.8.16 (Miniconda base) |
| **venv** | `/var/tmp/cym7/venvs/salt-unet/` (SSD NVMe local) |
| **Código** | `/u/cym7/projetos/Experiment-downstream/Salt-Segmentation-UNet/` |
| **Dataset TGS** | `/var/tmp/cym7/datasets/tgs-salt/train/` (SSD local, 3998 pares) |
| **Logs** | `/var/tmp/cym7/train_*.log` |

---

## Estrutura da pasta

```
experiment-downstream/
├── README.md                    ← este arquivo
├── setup_and_run.sh             ← script completo de setup + execução
├── Salt-Segmentation-UNet/      ← repositório base (clonado, modificado)
│   ├── utils/
│   │   ├── config.py            ← configurações centrais (TGS_PATH já configurado)
│   │   ├── model.py             ← U-Net (modificado: padding=1)
│   │   └── dataset.py           ← dataloader TGS (modificado: NEAREST para máscara)
│   ├── train.py                 ← loop de treino (IoU/Dice, seeds, cenários, early stop)
│   ├── evaluate.py              ← avaliação final no test set fixo
│   ├── generate_synthetic.py    ← gera pool sintético via VAE + textura
│   └── dataset/
│       ├── tgs/                 ← symlink/cópia de /var/tmp/cym7/datasets/tgs-salt/train/
│       └── synthetic/           ← gerado por generate_synthetic.py
│           ├── images/
│           └── masks/
└── results/
    ├── scenario_A_seed42/       ← result.csv, best_model.pth, plot.png
    ├── scenario_B_seed42/
    ├── ...
    └── summary.csv              ← tabela final IoU/Dice (gerada por evaluate.py)
```

---

## Pré-requisitos

### 1 — Conectar ao servidor (terminal SSH permanente)

```bash
# Abrir terminal SSH persistente com keepalive
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 atn2b02n07

# Ativar o ambiente Python e ir para o diretório do projeto
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/Experiment-downstream/Salt-Segmentation-UNet
```

> **Dica:** Usar sempre o terminal SSH persistente para evitar reconexões a cada comando.

### 2 — Ambiente Python (já configurado)

O venv `salt-unet` está instalado no SSD local do servidor com:

| Pacote | Versão |
|--------|--------|
| `torch` | 2.4.1+cu124 |
| `torchvision` | 0.19.1+cu124 |
| `opencv-python-headless` | 5.0.0 |
| `scikit-learn` | 1.3.2 |
| `numpy` | 1.24.4 |
| `pandas` | 2.0.3 |
| `tqdm`, `matplotlib`, `scipy`, `imutils` | ✓ |

Para recriar do zero (se necessário):
```bash
bash /u/cym7/projetos/Experiment-downstream/setup_and_run.sh
```

### 3 — Dataset TGS (já transferido)

```
/var/tmp/cym7/datasets/tgs-salt/train/
├── images/   ← 3998 arquivos .png (101×101 px, grayscale)
└── masks/    ← 4000 arquivos .png (101×101 px, binário)
```

O `config.py` já aponta para esse path:
```python
TGS_PATH = '/var/tmp/cym7/datasets/tgs-salt/train'
```

---

## Fluxo de execução no servidor

### Setup inicial (uma vez por sessão)

```bash
ssh -o ServerAliveInterval=60 atn2b02n07
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/Experiment-downstream/Salt-Segmentation-UNet
```

### Passo 1 — Treinar Cenário A (real only) — seeds 42, 123, 456

```bash
# Rodar em background com nohup (seguro contra queda de conexão)
# Os logs ficam na pasta results/ do próprio run
PROJ=/u/cym7/projetos/Experiment-downstream

nohup python -u train.py --scenario A --seed 42  --epochs 100 > $PROJ/results/scenario_A_seed42/train.log  2>&1 &
nohup python -u train.py --scenario A --seed 123 --epochs 100 > $PROJ/results/scenario_A_seed123/train.log 2>&1 &
nohup python -u train.py --scenario A --seed 456 --epochs 100 > $PROJ/results/scenario_A_seed456/train.log 2>&1 &
```

> Cada run leva ~3–4 min na V100. Rodar em paralelo usando GPUs diferentes é seguro
> (cada processo ocupa ~500 MB de VRAM na GPU 0 por default — para forçar GPUs distintas
> prefixar com `CUDA_VISIBLE_DEVICES=1`, `2`, etc.)

### Passo 2 — Gerar pool sintético e treinar Cenário B (real + synthetic)

```bash
# Gerar pool sintético (400 imagens)
python generate_synthetic.py --n 400 --out dataset/synthetic --tgs_dir /var/tmp/cym7/datasets/tgs-salt/train

# Treinar Cenário B
PROJ=/u/cym7/projetos/Experiment-downstream

nohup python -u train.py --scenario B --seed 42  --epochs 100 > $PROJ/results/scenario_B_seed42/train.log  2>&1 &
nohup python -u train.py --scenario B --seed 123 --epochs 100 > $PROJ/results/scenario_B_seed123/train.log 2>&1 &
nohup python -u train.py --scenario B --seed 456 --epochs 100 > $PROJ/results/scenario_B_seed456/train.log 2>&1 &
```

### Passo 3 — Avaliar no test set fixo

```bash
python -u evaluate.py --results_dir ../results
```

Saída: `../results/summary.csv` com IoU e Dice para cada seed/cenário.

### Passo 4 — Monitorar treinamentos em andamento

```bash
PROJ=/u/cym7/projetos/Experiment-downstream

# Acompanhar log em tempo real de um run específico
tail -f $PROJ/results/scenario_A_seed42/train.log

# Ver últimas épocas de todos os runs de uma vez
tail -3 $PROJ/results/*/train.log

# Verificar processos ativos
ps aux | grep train.py | grep -v grep

# Uso de GPU
nvidia-smi
```

### Low-data regime (opcional)

```bash
# Exploração de escala de dados: N=200, 400, 800, 1600, 2000, 3200
# Os logs ficam em results/scenario_A_seed42_nreal<N>/train.log (criado automaticamente pelo train.py)
PROJ=/u/cym7/projetos/Experiment-downstream

nohup python -u train.py --scenario A --seed 42 --n_real 200  --epochs 100 > $PROJ/results/scenario_A_seed42_nreal200/train.log  2>&1 &
nohup python -u train.py --scenario A --seed 42 --n_real 400  --epochs 100 > $PROJ/results/scenario_A_seed42_nreal400/train.log  2>&1 &
nohup python -u train.py --scenario A --seed 42 --n_real 800  --epochs 100 > $PROJ/results/scenario_A_seed42_nreal800/train.log  2>&1 &
nohup python -u train.py --scenario A --seed 42 --n_real 1600 --epochs 100 > $PROJ/results/scenario_A_seed42_nreal1600/train.log 2>&1 &
nohup python -u train.py --scenario A --seed 42 --n_real 2000 --epochs 100 > $PROJ/results/scenario_A_seed42_nreal2000/train.log 2>&1 &
```

> **Nota:** o `train.py` cria o diretório `results/<run_tag>/` automaticamente antes de salvar artefatos.
> O redirecionamento do log para esse mesmo diretório mantém logs e artefatos colocalizados:
> ```
> results/scenario_A_seed42/
> ├── train.log        ← saída completa do treinamento
> ├── best_model.pth   ← checkpoint da melhor época (val IoU)
> ├── result.csv       ← métricas finais do test set
> └── plot.png         ← curvas loss/IoU por epoch
> ```

### Rodar tudo de uma vez (script completo)

```bash
bash /u/cym7/projetos/Experiment-downstream/setup_and_run.sh
```

---

## Resultados preliminares obtidos

| N real | Device | Epochs | Test IoU | Test Dice | Elapsed |
|:------:|:------:|:------:|:--------:|:---------:|:-------:|
| 200    | CPU    | 29 *(early stop)* | 0.3102 | 0.3616 | ~667s |
| 400    | CPU    | 32 *(early stop)* | 0.3391 | 0.3820 | ~757s |
| 800    | CPU    | 50                | 0.3630 | 0.4019 | ~1461s |
| 1600   | CPU    | 50                | 0.4011 | 0.4401 | ~4724s |
| **~3200** | **GPU V100** | **57 *(early stop)*** | **0.4312** | **0.4657** | **202s** |

> Relatórios detalhados em [`results/preliminary_results_50ep.md`](results/preliminary_results_50ep.md) e [`results/preliminary_results_100ep.md`](results/preliminary_results_100ep.md)

---

## Modificações aplicadas ao repositório base

| Arquivo | Modificação | Motivo |
|---------|------------|--------|
| `utils/model.py` | `padding=1` nos `Conv2d` do `Block` | Preserva dimensão espacial; elimina necessidade de `CenterCrop` |
| `utils/dataset.py` | Separar transforms imagem/máscara; `NEAREST` para máscara | Evita artefatos de interpolação bilinear na ground truth |
| `utils/config.py` | `TGS_PATH` → `/var/tmp/cym7/datasets/tgs-salt/train`; `ENCODER_CHANNELS=(1,16,32,64)` | TGS é grayscale (1 canal); path do servidor Atena |
| `train.py` | IoU + Dice no loop de validação; `--scenario`, `--seed`, `--n_real`, `--n_synth`, `--epochs`, `--batch`, `--lr`; early stopping por IoU; split estratificado | Requisitos do experimento |
| `evaluate.py` | Avaliação final no test set fixo, gera `summary.csv` | Reprodutibilidade |
| `generate_synthetic.py` | Interface com VAE + pipeline de textura do paper | Geração do pool sintético |

---

## Resultado esperado (tabela do manuscrito)

| Cenário | IoU (mean ± std) | Dice (mean ± std) | p-value |
|---------|-----------------|------------------|---------|
| A — Real only (N~3200) | 0.4312 ± — | 0.4657 ± — | — |
| B — Real + Synthetic (N~3200+400) | — | — | — |

Após executar os experimentos, atualizar:
1. `_v7.tex` — subseção `\subsection{Downstream Segmentation Evaluation}`
2. `docs/_reviewACCESS/response_to_reviewers.md` — seção R2.1
3. `docs/_reviewACCESS/summary_of_changes.md` — status R2.1 de **PENDING** → **DONE**

---

## Referências

- Protocolo completo: [`docs/experimentUNet-protocol.md`](docs/experimentUNet-protocol.md)
- Repositório base original: https://github.com/matin-ghorbani/Salt-Segmentation-UNet
