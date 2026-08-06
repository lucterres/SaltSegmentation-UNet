# Scripts — Experimentos R2.1

Organização dos scripts usados nos experimentos de downstream segmentation (Access-2026-27912).

## Estrutura

```
scripts/
├── infra/          ← Setup e verificação do nó Atena
├── batch/          ← Launchers de experimentos (múltiplas runs em paralelo)
└── monitoring/     ← Acompanhamento e cancelamento de runs
```

---

## infra/

| Script | Função |
|--------|--------|
| `setup_node_Atena.sh` | Restaura venv e dataset TGS no SSD local do nó |
| `check_node_Atena.sh` | Verifica estado do nó (venv, GPU, dataset) sem modificar |

**Uso:**
```bash
export PROJ=/nethome/atena_projetos/cym7/0code/SaltSegment-Unet
bash $PROJ/scripts/infra/check_node_Atena.sh
bash $PROJ/scripts/infra/setup_node_Atena.sh
```

---

## batch/

| Script | Experimento | N_real | N_synth | Seeds |
|--------|-------------|:------:|:-------:|:-----:|
| `run_cenarioB_6albu_seed42.sh` | SLURM array — B×6 datasets | 3998 | 1200 | 42 |
| `run_cenarioB_6albu_seed42_parallel.sh` | nohup paralelo — B×6 datasets | 3998 | 1200 | 42 |
| `run_batch_B_seeds123_456.sh` | B×6 datasets, seeds 123+456 | 3998 | 1200 | 123, 456 |
| `run_batch_AxB_seeds123_456.sh` | A+B×6, seeds 123+456 (v1 com bug) | 3998 | 1200 | 123, 456 |
| `run_batch_nreal1200.sh` | A+B×6, todas seeds | 1200 | 1200 | 42, 123, 456 |

> ⚠️ `run_batch_AxB_seeds123_456.sh` tem bug de duplicata (sleep muito curto) — usar `run_batch_B_seeds123_456.sh` como referência.

**Script recomendado para novos experimentos:** `run_batch_nreal1200.sh` (lockfile por GPU, anti-duplicata, cascata automática)

**Uso:**
```bash
nohup bash $PROJ/scripts/batch/run_batch_nreal1200.sh \
  > $PROJ/results/slurm_logs/launchers/batch_nreal1200.log 2>&1 &
```

---

## monitoring/

| Script | Função |
|--------|--------|
| `check_progress_6albu_seed42.sh` | Mostra status (PENDING/RUNNING/DONE) e IoU de cada run |
| `kill_cenarioB_6albu_seed42.sh` | Mata todos os `train.py` ativos |

**Uso:**
```bash
bash $PROJ/scripts/monitoring/check_progress_6albu_seed42.sh

# Cancelar tudo:
bash $PROJ/scripts/monitoring/kill_cenarioB_6albu_seed42.sh
```

---

## Notas de infraestrutura

- **Nó:** `atn2b03n03` — 8× Tesla V100-SXM2-32GB
- **Alocação exclusiva:** `salloc --nodes=1 -p gpu --account=pn-dscien --time=08:00:00 --gres=gpu:8 --exclusive`
- **Alocação compartilhada (atual):** apenas `--gres=gpu:1` — nó compartilhado com outros usuários
- **venv:** `/var/tmp/cym7/venvs/salt-unet/` (SSD local, não persiste entre alocações)
- **Logs de runs:** `results/slurm_logs/{launchers,nreal3998,nreal1200}/`
