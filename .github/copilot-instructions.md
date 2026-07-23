# GitHub Copilot Instructions — Experimento Downstream R2.1

## Contexto do projeto

**Manuscrito:** Access-2026-27912
**Objetivo:** Demonstrar que treinar com dados reais + sintéticos melhora a segmentação de salt domes em dados sísmicos reais de teste (Cenário B > Cenário A).

---

## Infraestrutura de execução

| Item | Detalhe |
|------|---------|
| Cluster | Atena (Petrobras) — nós GPU alocados via SLURM |
| Nó de referência | `atn2b02n07` (venv original criado aqui) |
| Nó atual | qualquer `atn2bXXnYY` alocado no dia |
| GPUs | 8 × Tesla V100-SXM2-32GB |
| CUDA | 12.4 (PyTorch 2.4.1+cu124) |
| Python | 3.8.16 (Miniconda base) |
| venv (SSD local) | `/var/tmp/cym7/venvs/salt-unet/` |
| venv (home backup) | `/u/cym7/venvs_backup/salt-unet/` ← **cópia persistente entre nós** |
| Código | `/u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet/` |
| Resultados | `/u/cym7/projetos/SaltSegmentation-UNet/results/` |
| Dataset TGS (SSD local) | `/var/tmp/cym7/datasets/tgs-salt/train/` (SSD local, 3998 pares) |
| Dataset TGS (home backup) | `~/datasets/tgs-salt/tgs-salt.tar` ← **arquivo tar persistente** |

> **Atenção:** `/var/tmp/` é local a cada nó e não persiste.  
> O venv de referência fica em `/u/cym7/venvs_backup/salt-unet/` (NFS home, persistente).

---

## Migração do venv para novo nó GPU (fazer 1x por nó)

```bash
# 1. Restaurar venv do backup na home para o SSD local do nó
mkdir -p /var/tmp/cym7/venvs
cp -r /u/cym7/venvs_backup/salt-unet /var/tmp/cym7/venvs/

# 2. Corrigir pyvenv.cfg
PYTHON_BIN=$(which python3)
sed -i "s|^home = .*|home = $(dirname $PYTHON_BIN)|" \
  /var/tmp/cym7/venvs/salt-unet/pyvenv.cfg

# 3. Verificar CUDA
source /var/tmp/cym7/venvs/salt-unet/bin/activate
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
```

---

## Arquivos-chave

| Arquivo | Função |
|---------|--------|
| `Salt-Segmentation-UNet/utils/config.py` | Configurações centrais (`TGS_PATH`, canais do encoder) |
| `Salt-Segmentation-UNet/utils/model.py` | U-Net com `padding=1` nos `Conv2d` |
| `Salt-Segmentation-UNet/utils/dataset.py` | DataLoader TGS; interpolação `NEAREST` para máscaras |
| `Salt-Segmentation-UNet/train.py` | Loop de treino com IoU/Dice, early stopping, `--scenario`, `--seed`, `--n_real`, `--n_synth`, `--epochs`, `--batch`, `--lr` |
| `Salt-Segmentation-UNet/evaluate.py` | Avaliação no test set fixo → gera `results/summary.csv` |
| `Salt-Segmentation-UNet/generate_synthetic.py` | Gera pool sintético via VAE + textura |
| `setup_and_run.sh` | Script completo de setup + execução end-to-end |

---

## Workflow de execução

### 1. Conectar ao nó alocado e ativar ambiente
```bash
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 <nó-alocado>
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet
```

### 2. Cenário A — Real only (seeds 42, 123, 456)
```bash
PROJ=/u/cym7/projetos/SaltSegmentation-UNet

nohup python -u train.py --scenario A --seed 42  --epochs 100 > $PROJ/results/scenario_A_seed42/train.log  2>&1 &
nohup python -u train.py --scenario A --seed 123 --epochs 100 > $PROJ/results/scenario_A_seed123/train.log 2>&1 &
nohup python -u train.py --scenario A --seed 456 --epochs 100 > $PROJ/results/scenario_A_seed456/train.log 2>&1 &
```

### 3. Cenário B — Real + Synthetic (seeds 42, 123, 456)
```bash
# Gerar pool sintético (400 imagens)
python generate_synthetic.py --n 400 --out dataset/synthetic \
  --tgs_dir /var/tmp/cym7/datasets/tgs-salt/train

PROJ=/u/cym7/projetos/SaltSegmentation-UNet

nohup python -u train.py --scenario B --seed 42  --epochs 100 > $PROJ/results/scenario_B_seed42/train.log  2>&1 &
nohup python -u train.py --scenario B --seed 123 --epochs 100 > $PROJ/results/scenario_B_seed123/train.log 2>&1 &
nohup python -u train.py --scenario B --seed 456 --epochs 100 > $PROJ/results/scenario_B_seed456/train.log 2>&1 &
```

### 4. Avaliação final
```bash
python -u evaluate.py --results_dir ../results
# Saída: ../results/summary.csv
```

### 5. Monitorar treinamentos
```bash
PROJ=/u/cym7/projetos/SaltSegmentation-UNet

tail -f $PROJ/results/scenario_A_seed42/train.log   # log em tempo real
tail -3 $PROJ/results/*/train.log                   # últimas linhas de todos
ps aux | grep train.py | grep -v grep               # processos ativos
nvidia-smi                                          # uso de GPU
```

---

## Estrutura de resultados

Cada run gera automaticamente a pasta `results/<run_tag>/`:

```
results/scenario_A_seed42/
├── train.log       ← saída completa do treinamento
├── best_model.pth  ← checkpoint da melhor época (val IoU)
├── result.csv      ← métricas finais (test set)
├── history.csv     ← métricas por época
└── plot.png        ← curvas loss/IoU por época
```

Arquivo consolidado: `results/summary.csv` (gerado por `evaluate.py`)

---

## Cenários definidos

| Cenário | Dados de treino | Objetivo |
|---------|----------------|----------|
| A | Real only (~3200 amostras) | Baseline |
| B | Real + Synthetic (~3200 + 400) | Hipótese do paper |

---

## Resultados preliminares (Cenário A)

| N real | Device | Epochs | Test IoU | Test Dice |
|:------:|:------:|:------:|:--------:|:---------:|
| 200 | CPU | 29 (early stop) | 0.3102 | 0.3616 |
| 400 | CPU | 32 (early stop) | 0.3391 | 0.3820 |
| 800 | CPU | 50 | 0.3630 | 0.4019 |
| 1600 | CPU | 50 | 0.4011 | 0.4401 |
| ~3200 | GPU V100 | 57 (early stop) | **0.4312** | **0.4657** |

---

## Convenções do código

- `TGS_PATH` em `utils/config.py` aponta para `/var/tmp/cym7/datasets/tgs-salt/train`
- `ENCODER_CHANNELS = (1, 16, 32, 64)` — TGS é grayscale (1 canal de entrada)
- Interpolação `NEAREST` para máscaras binárias (evita artefatos)
- `padding=1` no U-Net preserva dimensão espacial sem `CenterCrop`
- Split estratificado por presença de sal para reprodutibilidade
- Métricas primárias: **IoU** (critério de early stopping) e **Dice**

---

## Após executar os experimentos — atualizar

1. `_v7.tex` → `\subsection{Downstream Segmentation Evaluation}`
2. `docs/_reviewACCESS/response_to_reviewers.md` → seção R2.1
3. `docs/_reviewACCESS/summary_of_changes.md` → status R2.1: **PENDING → DONE**

---

## Referências internas

- Protocolo completo: `docs/experimentUNet-protocol.md`
- Relatórios preliminares: `results/preliminary_results_*.md`
- Repositório base original: https://github.com/matin-ghorbani/Salt-Segmentation-UNet
