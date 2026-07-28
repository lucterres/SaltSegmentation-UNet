# Skill — Gerar Subset por Cobertura de Sal

## Descrição

Filtra o dataset TGS-Salt pelo percentual de cobertura de sal (`salt_pct`) e copia os pares `images/masks` selecionados para um novo diretório.

## Pré-condições

- `split_stats.csv` disponível em `dataset/subset_split/split_stats.csv`
- Dataset TGS original em `/var/tmp/cym7/datasets/tgs-salt/train/`
- venv ativo: `source /var/tmp/cym7/venvs/salt-unet/bin/activate`

## Parâmetros configuráveis

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `MIN_PCT` | Percentual mínimo de sal | `1.0` |
| `MAX_PCT` | Percentual máximo de sal | `99.0` |
| `OUT_DIR` | Diretório de saída | `/var/tmp/cym7/datasets/subset_1_99` |

## Subsets predefinidos

| Nome | MIN | MAX | N esperado |
|:----:|:---:|:---:|:----------:|
| `subset_1_99` | 1.0 | 99.0 | ~2209 |
| `subset_2_98` | 2.0 | 98.0 | 2080 |
| `subset_5_95` | 5.0 | 95.0 | ~2000 |
| `subset_10_90` | 10.0 | 90.0 | ~1616 |
| `subset_20_80` | 20.0 | 80.0 | ~1200 |

## Script

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

## Uso com `train.py`

```bash
python -u train.py --scenario A --seed 42 --epochs 100 \
  --train_dir /var/tmp/cym7/datasets/subset_1_99 \
  --test_dir /var/tmp/cym7/datasets/subset_split/test
```
