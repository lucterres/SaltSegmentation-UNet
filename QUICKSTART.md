# QUICKSTART — Experimento Downstream R2.1

**Manuscrito:** Access-2026-27912 | **Última atualização:** 2026-08-06

---

## 0. Mapear Home Area

```
\\homeunix-rio.petrobras.biz\cym7 -> E:
VSCode: abrir pasta I:\0code\SaltSegment-Unet
```

```bash
export PROJ=/nethome/atena_projetos/cym7/0code/SaltSegment-Unet
```

---

## 1. Conectar ao nó GPU

### 1.1 Verificar job ativo
```bash
ssh atena03.petrobras.biz
squeue -u cym7   # coluna NODELIST mostra o nó
```

### 1.2 Solicitar nó exclusivo (recomendado)
```bash
salloc --nodes=1 -p gpu --account=pn-dscien --time=08:00:00 --gres=gpu:8 --exclusive
```

### 1.3 Conectar ao nó
```bash
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 <nó-alocado>
# ex: atn2b03n03, atn2b01n05, etc.
```

> **Nota:** sem `--exclusive` o nó pode ser compartilhado com outros usuários — GPUs podem estar parcialmente ocupadas.

---

## 2. Verificar / preparar ambiente

```bash
export PROJ=/nethome/atena_projetos/cym7/0code/SaltSegment-Unet
bash $PROJ/scripts/infra/check_node_Atena.sh   # só verifica
bash $PROJ/scripts/infra/setup_node_Atena.sh   # restaura venv + dataset se ausentes
```

---

## 3. Ativar ambiente

```bash
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd $PROJ/Salt-Segmentation-UNet
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
# Esperado: 2.4.1+cu124 | CUDA: True | GPUs: 8
```

---

## 4. Protocolo canônico — Cenário A vs B (sem data leakage)

> **Regra obrigatória:** sempre usar `--test_dir` com o test set canônico.  
> Nunca usar split interno — causa leakage quando `--train_dir` aponta para o TGS completo.

| Parâmetro canônico | Valor |
|-------------------|-------|
| `--train_dir` | `/var/tmp/cym7/datasets/tgs-salt/train` (3998 amostras) |
| `--test_dir` | `/var/tmp/cym7/datasets/subset_split/test` (800 amostras fixas) |
| `--n_synth` | 1200 |
| `--epochs` | 100 |

### 4.1 Lançar experimento completo (A + B × 6 datasets × 3 seeds)

#### N_real = 3998 (dataset completo)
```bash
nohup bash $PROJ/scripts/batch/run_batch_B_seeds123_456.sh \
  > $PROJ/results/slurm_logs/launchers/batch_nreal3998.log 2>&1 &
```

#### N_real = 1200 (low-data regime)
```bash
nohup bash $PROJ/scripts/batch/run_batch_nreal1200.sh \
  > $PROJ/results/slurm_logs/launchers/batch_nreal1200.log 2>&1 &
```

> Os launchers usam **lockfile por GPU** — sem duplicatas, cascata automática quando GPU libera.

### 4.2 Monitorar progresso
```bash
# GPUs em tempo real
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# Logs do launcher
tail -f $PROJ/results/slurm_logs/launchers/batch_nreal1200.log

# Progresso de todas as runs
bash $PROJ/scripts/monitoring/check_progress_6albu_seed42.sh

# Cancelar tudo
bash $PROJ/scripts/monitoring/kill_cenarioB_6albu_seed42.sh
```

---

## 5. Argumentos do `train.py`

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--scenario` | A/B | obrigatório | A = real only; B = real + sintéticos |
| `--seed` | int | 42 | Semente de reprodutibilidade |
| `--n_real` | int | None | Limitar amostras reais (low-data regime) |
| `--n_synth` | int | 400 | Número de sintéticos no Cenário B |
| `--epochs` | int | 100 | Máximo de épocas |
| `--batch` | int | 16 | Batch size |
| `--lr` | float | 1e-4 | Learning rate |
| `--train_dir` | path | None | Pasta `images/`+`masks/` de treino (sobrescreve `TGS_PATH`) |
| `--test_dir` | path | None | **Obrigatório** — pasta test set canônico externo |
| `--synth_dir` | path | None | Pasta sintéticos (sobrescreve symlink `dataset/synthetic`) |

**Run tag gerado:** `scenario_<A|B>_seed<S>[_nreal<N>]_<train_dir_name>[_<synth_name>_ns<N_synth>]`

---

## 6. Datasets sintéticos disponíveis (albumentations clean)

Localizados em `$PROJ/Salt-Segmentation-UNet/dataset/` — **clean** = sem IDs do test canônico.

| Dataset | Tipo | Amostras |
|---------|------|:--------:|
| `clahe1600clean` | Intensidade | 1600 |
| `elastic_transform1600clean` | Geométrico | 1600 |
| `grid_distortion1600clean` | Geométrico | 1600 |
| `optical_distortion1600clean` | Geométrico | 1600 |
| `random_brightness_contrast1600clean` | Intensidade | 1600 |
| `random_gamma1600clean` | Intensidade | 1600 |

---

## 7. Resultados de referência (test canônico 800, sem leakage)

### N_real = 3998

| Cenário | Dataset sintético | IoU médio (3 seeds) | Δ vs A |
|---------|------------------|:-------------------:|:------:|
| **A** | — | **0.4734** | — |
| **B** ✅ | `random_gamma` | **0.4870** | **+0.014** |
| **B** | `grid_distortion` | 0.4774 | +0.004 |

### N_real = 1200

| Cenário | Dataset sintético | IoU médio (3 seeds) | Δ vs A |
|---------|------------------|:-------------------:|:------:|
| **A** | — | **0.4081** | — |
| **B** ✅ | `elastic_transform` | **0.4276** | **+0.020** |
| **B** | `grid_distortion` | 0.4272 | +0.019 |

> **Todos os 6 datasets superam A com N=1200**. Com N=3998 apenas 2 superam.

---

## 8. Estrutura de resultados

```
results/
├── scenario_A_seed42_train/          ← result.csv, history.csv, best_model.pth, plot.png
├── scenario_B_seed42_train_<DS>_ns1200/
├── ...
└── slurm_logs/
    ├── launchers/   ← logs dos batch launchers
    ├── nreal3998/   ← logs individuais N=3998
    └── nreal1200/   ← logs individuais N=1200
```

**`result.csv`:** `scenario, seed, n_real, n_synth, best_val_iou, test_iou, test_dice, epochs_run, elapsed_s`

---

## 9. Scripts organizados

```
scripts/
├── infra/       check_node_Atena.sh | setup_node_Atena.sh
├── batch/       run_batch_nreal1200.sh ✅ | run_batch_B_seeds123_456.sh ✅
└── monitoring/  check_progress_6albu_seed42.sh | kill_cenarioB_6albu_seed42.sh
```

Ver `scripts/README.md` para detalhes.

---

## 10. Relatórios

| Arquivo | Conteúdo |
|---------|----------|
| `docs/relatorio-cenarioB-albumentations-3seeds.md` | Resultados completos N=3998 e N=1200, 3 seeds, 6 datasets |
| `docs/relatorio-final-r21-downstream.md` | Relatório geral R2.1 |
