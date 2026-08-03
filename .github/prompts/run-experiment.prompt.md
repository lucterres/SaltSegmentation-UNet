---
description: Executa um experimento de treinamento (Cenário A ou B) no cluster Atena com seed e dataset configuráveis.
---

# Rodar Experimento — R2.1

## Pré-condições

```bash
export PROJ=/nethome/atena_projetos/cym7/0code/SaltSegment-Unet
bash $PROJ/check_node_Atena.sh
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd $PROJ/Salt-Segmentation-UNet

SPLIT=/var/tmp/cym7/datasets/subset_split
```

---

## Cenário A — Real only

**TGS completo (baseline):**
```bash
mkdir -p $PROJ/results/scenario_A_seed42
python -u train.py --scenario A --seed 42 --epochs 100 \
  2>&1 | tee $PROJ/results/scenario_A_seed42/train.log
```

**subset_1_99 + test canônico (melhor resultado):**
```bash
mkdir -p $PROJ/results/scenario_A_seed42_subset199
env TGS_PATH=/var/tmp/cym7/datasets/subset_1_99 \
  python -u train.py --scenario A --seed 42 --epochs 100 \
  --test_dir $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_A_seed42_subset199/train.log
```

**train_filtered + test canônico:**
```bash
mkdir -p $PROJ/results/scenario_A_seed42_train_filtered
python -u train.py --scenario A --seed 42 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_A_seed42_train_filtered/train.log
```

---

## Cenário B — Real + Sintético

**train_filtered + 955 sintéticos sísmicos:**
```bash
mkdir -p $PROJ/results/scenario_B_seed42_train_filtered
python -u train.py --scenario B --seed 42 --n_synth 955 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_B_seed42_train_filtered/train.log
```

**subset_1_99 + 955 sintéticos sísmicos (pendente):**
```bash
mkdir -p $PROJ/results/scenario_B_seed42_subset199
python -u train.py --scenario B --seed 42 --n_synth 955 --epochs 100 \
  --train_dir /var/tmp/cym7/datasets/subset_1_99 --test_dir $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_B_seed42_subset199/train.log
```

> Para background: substituir `2>&1 | tee <log>` por `> <log> 2>&1 &` (ou `nohup ... &`).

---

## Monitorar

```bash
tail -f $PROJ/results/<run_tag>/train.log
watch -n 15 'tail -n 2 '$PROJ'/results/*/train.log'
ps aux | grep train.py | grep -v grep
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
```

---

## Avaliar

```bash
cat $PROJ/results/<run_tag>/result.csv

for f in $PROJ/results/*/result.csv; do
  echo "--- $(basename $(dirname $f)) ---"; cat "$f"
done
```
