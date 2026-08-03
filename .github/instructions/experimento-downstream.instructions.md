---
applyTo: "Salt-Segmentation-UNet/**/*.py"
---

# Índice — Experimento Downstream R2.1

> Este arquivo foi refatorado. O conteúdo foi distribuído nas camadas corretas:

| Camada | Arquivo | Escopo |
|--------|---------|--------|
| Infra & ambiente | `infra.instructions.md` | `**/*.sh` |
| Experimentos & datasets | `experimento.instructions.md` | `Salt-Segmentation-UNet/**/*.py` |
| Relatório | `relatorio.instructions.md` | `docs/**/*.md` |
| Rodar experimento | `../prompts/run-experiment.prompt.md` | — |
| Atualizar relatório | `../prompts/update-report.prompt.md` | — |
| Análise de métricas | `../agents/researcher.agent.md` | — |
| Gerar subset | `../skills/generate-subset.md` | — |

---

## 1. Caminhos absolutos

| Recurso | Path absoluto (servidor) |
|---------|--------------------------|
| Código `train.py` | `/nethome/atena_projetos/cym7/0code/SaltSegment-Unet/Salt-Segmentation-UNet/` |
| Resultados | `/nethome/atena_projetos/cym7/0code/SaltSegment-Unet/results/` |
| Relatório final | `/nethome/atena_projetos/cym7/0code/SaltSegment-Unet/docs/relatorio-final-r21-downstream.md` |
| venv (SSD local) | `/var/tmp/cym7/venvs/salt-unet/` |
| venv (backup tar) | `/nethome/atena_projetos/cym7/envs/salt-unet-venv.tar` |
| Dataset TGS completo | `/var/tmp/cym7/datasets/tgs-salt/train/` (3998 pares) |
| Dataset TGS (tar) | `/nethome/atena_projetos/cym7/dataset/tgsSalt/tgs-salt.tar` |
| **subset_split/train_filtered** | `/var/tmp/cym7/datasets/subset_split/train_filtered/` (1293 amostras, 10–90%) |
| **subset_split/test** | `/var/tmp/cym7/datasets/subset_split/test/` (800 amostras, dist. real) ← **test canônico** |
| **subset_1_99** | `/var/tmp/cym7/datasets/subset_1_99/` (2209 amostras, 1–99%) |
| **subset_10_90** | `/var/tmp/cym7/datasets/subset_10_90/` (1616 amostras, 10–90%) |
| Sintéticos sísmicos | `$PROJ/Salt-Segmentation-UNet/dataset/geometric1600_seismic/pairs1600_seismic/` (955 pares) |
| Symlink sintéticos | `dataset/synthetic` → path acima |
| subset_split tar | `$PROJ/Salt-Segmentation-UNet/dataset/subset_split.tar` |
| subset_10_90 tar | `$PROJ/Salt-Segmentation-UNet/dataset/subset_10_90.tar` |
| subset_1_99 | gerado por script inline (ver seção 7) |

### Nó GPU

```bash
ssh atena03.petrobras.biz
salloc --nodes=1 -p gpu --account=pn-dscien --time=08:00:00
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 <nó-alocado>
```

---

## 2. Preparar ambiente e rodar treino (seed 42)

```bash
export PROJ=/nethome/atena_projetos/cym7/0code/SaltSegment-Unet
bash $PROJ/check_node_Atena.sh
# Se ausente: bash $PROJ/setup_node_Atena.sh

source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd $PROJ/Salt-Segmentation-UNet

SPLIT=/var/tmp/cym7/datasets/subset_split
```

**Cenário A — TGS completo (baseline):**
```bash
mkdir -p $PROJ/results/scenario_A_seed42
python -u train.py --scenario A --seed 42 --epochs 100 \
  2>&1 | tee $PROJ/results/scenario_A_seed42/train.log
```

**Cenário A — subset_1_99 com test canônico (melhor resultado):**
```bash
mkdir -p $PROJ/results/scenario_A_seed42_subset199
env TGS_PATH=/var/tmp/cym7/datasets/subset_1_99 \
  python -u train.py --scenario A --seed 42 --epochs 100 \
  --test_dir $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_A_seed42_subset199/train.log
```

**Cenário A — train_filtered com test canônico:**
```bash
mkdir -p $PROJ/results/scenario_A_seed42_train_filtered
python -u train.py --scenario A --seed 42 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_A_seed42_train_filtered/train.log
```

**Cenário B — train_filtered + 955 sintéticos sísmicos:**
```bash
mkdir -p $PROJ/results/scenario_B_seed42_train_filtered
python -u train.py --scenario B --seed 42 --n_synth 955 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_B_seed42_train_filtered/train.log
```

> `tee` = foreground com saída no terminal + salva no log. Para background: `nohup ... > log 2>&1 &`

---

## 3. Monitorar treinamento

```bash
tail -f $PROJ/results/scenario_A_seed42_subset199/train.log
watch -n 15 'tail -n 2 '$PROJ'/results/*/train.log'
ps aux | grep train.py | grep -v grep
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
```

---

## 4. Avaliar resultado

```bash
cat $PROJ/results/scenario_A_seed42_subset199/result.csv

for f in $PROJ/results/*/result.csv; do
  echo "--- $(basename $(dirname $f)) ---"; cat "$f"
done
```

**Colunas:** `scenario, seed, n_real, n_synth, best_val_iou, test_iou, test_dice, epochs_run, elapsed_s`
