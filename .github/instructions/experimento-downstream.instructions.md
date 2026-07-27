---
applyTo: "**"
---

# Instruções — Experimento Downstream R2.1

**Manuscrito:** Access-2026-27912 | **Última atualização:** 2026-07-27

---

## 1. Caminhos absolutos

| Recurso | Path absoluto (servidor) |
|---------|--------------------------|
| Código `train.py` | `/u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet/` |
| Resultados | `/u/cym7/projetos/SaltSegmentation-UNet/results/` |
| Relatório final | `/u/cym7/projetos/SaltSegmentation-UNet/docs/relatorio-final-r21-downstream.md` |
| venv (SSD local) | `/var/tmp/cym7/venvs/salt-unet/` |
| venv (home backup) | `/u/cym7/venvs_backup/salt-unet/` |
| Dataset TGS completo | `/var/tmp/cym7/datasets/tgs-salt/train/` (3998 pares) |
| Dataset TGS (tar) | `~/datasets/tgs-salt/tgs-salt.tar` |
| **subset_split/train_filtered** | `/var/tmp/cym7/datasets/subset_split/train_filtered/` (1293 amostras, 10–90%) |
| **subset_split/test** | `/var/tmp/cym7/datasets/subset_split/test/` (800 amostras, dist. real) ← **test canônico** |
| **subset_1_99** | `/var/tmp/cym7/datasets/subset_1_99/` (2209 amostras, 1–99%) |
| **subset_10_90** | `/var/tmp/cym7/datasets/subset_10_90/` (1616 amostras, 10–90%) |
| Sintéticos sísmicos | `dataset/geometric1600_seismic/pairs1600_seismic/` (955 pares) |
| Symlink sintéticos | `dataset/synthetic` → path acima |
| subset_split tar | `dataset/subset_split.tar` |
| subset_10_90 tar | `dataset/subset_10_90.tar` |
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
bash ~/projetos/SaltSegmentation-UNet/check_node_Atena.sh
# Se ausente: bash ~/projetos/SaltSegmentation-UNet/setup_node_Atena.sh

source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet

PROJ=/u/cym7/projetos/SaltSegmentation-UNet
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

### Resultados de referência (test canônico — 800 amostras reais)

| Dataset treino | N treino | Test IoU | Test Dice |
|:--------------:|:--------:|:--------:|:---------:|
| TGS completo | 3198 | 0.4312 | 0.4657 |
| `subset_10_90` (10–90%) | 1616 | 0.4590 | 0.4860 |
| **`subset_1_99` (1–99%)** | **2209** | **0.4791** | **0.5058** |
| `train_filtered` (10–90%) | 1293 | 0.4201 | 0.4553 |
| B + 955 sísmicos (train_filtered) | 1293+955 | 0.4308 | 0.4672 |

---

## 5. Editar relatório final

```
servidor: /u/cym7/projetos/SaltSegmentation-UNet/docs/relatorio-final-r21-downstream.md
windows:  f:\projetos\SaltSegmentation-UNet\docs\relatorio-final-r21-downstream.md
```

Seções a atualizar: `## 2`, `## 3`, `## 12`, `## 6. Conclusão final`

```markdown
### 2.X Cenário A — `<nome>` com test canônico (seed 42)
| Seed | N real | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:------:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42 | XXXX | X.XXXX | X.XXXX | X.XXXX | XX | XXX |
```

---

## 6. Achados-chave

### Ranking de datasets de treino (test canônico)

| Dataset treino | Filtro | N treino | Test IoU |
|:--------------:|:------:|:--------:|:--------:|
| **`subset_1_99`** | **1–99%** | **2209** | **0.4791 ✅** |
| `subset_10_90` | 10–90% | 1616 | 0.4590 |
| TGS completo | nenhum | 3198 | 0.4312 |
| `train_filtered` | 10–90% | 1293 | 0.4201 |

### Sintéticos (dataset completo ~3200 reais)

| Config | Test IoU | Δ vs A |
|:------:|:--------:|:------:|
| A puro | 0.4247 | — |
| B + 955 sísmicos (TGS) | 0.4204 | −0.004 |
| **B + 955 sísmicos (train_filtered)** | **0.4308** | **+0.011 ✅** |
| B + 400 originais | 0.4127 | −0.012 |
| B + 1600 geométricos | 0.4070 | −0.018 |

### ⚠️ Métricas incomparáveis

| Test set | Test IoU | Comparável? |
|:--------:|:--------:|:-----------:|
| filtrado interno subset_10_90 (~293) | 0.8340 | ❌ |
| filtrado interno subset_1_99 (~442) | 0.7662 | ❌ |
| **canônico (800 reais)** | 0.41–0.48 | **✅** |

Sempre usar `--test_dir /var/tmp/cym7/datasets/subset_split/test`

### Novos argumentos do train.py

```
--train_dir <path>   # substitui TGS_PATH
--test_dir  <path>   # pula split interno, usa test fixo externo
```

### Pendências

- [ ] `subset_1_99` seeds 123 e 456
- [ ] Cenário B com `subset_1_99` + 955 sísmicos (seeds 42, 123, 456)
- [ ] Atualizar `_v7.tex` e `response_to_reviewers.md` (R2.1)

---

## 7. Gerar novo dataset filtrando cobertura de sal

```bash
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet

python - <<'EOF'
import os, shutil, pandas as pd

MIN_PCT = 1.0    # ← alterar aqui
MAX_PCT = 99.0   # ← alterar aqui
OUT_DIR = '/var/tmp/cym7/datasets/subset_1_99'  # ← alterar aqui

STATS   = 'dataset/subset_split/split_stats.csv'
SRC_IMG = '/var/tmp/cym7/datasets/tgs-salt/train/images'
SRC_MSK = '/var/tmp/cym7/datasets/tgs-salt/train/masks'

df = pd.read_csv(STATS)
filtered = df[(df['salt_pct'] >= MIN_PCT) & (df['salt_pct'] <= MAX_PCT)]
print(f'Total: {len(df)} | Filtrado: {len(filtered)}')

os.makedirs(f'{OUT_DIR}/images', exist_ok=True)
os.makedirs(f'{OUT_DIR}/masks',  exist_ok=True)

copied = 0
for _, row in filtered.iterrows():
    stem = row['id']
    if os.path.exists(f'{SRC_IMG}/{stem}.png'):
        shutil.copy2(f'{SRC_IMG}/{stem}.png', f'{OUT_DIR}/images/{stem}.png')
        shutil.copy2(f'{SRC_MSK}/{stem}.png', f'{OUT_DIR}/masks/{stem}.png')
        copied += 1

print(f'Copiados: {copied}')
filtered.to_csv(f'{OUT_DIR}/subset_stats.csv', index=False)
EOF
```

| Subset | MIN | MAX | N |
|:------:|:---:|:---:|:-:|
| subset_1_99 | 1.0 | 99.0 | ~2209 |
| subset_5_95 | 5.0 | 95.0 | ~2000 |
| subset_10_90 | 10.0 | 90.0 | ~1616 |
| subset_20_80 | 20.0 | 80.0 | ~1200 |
