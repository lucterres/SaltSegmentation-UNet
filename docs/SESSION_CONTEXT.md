# Contexto de Sessão — Experimento Downstream R2.1

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
| subset_1_99 | gerado por script inline (ver QUICKSTART) |

### Nó GPU atual

```bash
# Nó de login
ssh atena03.petrobras.biz

# Solicitar nó GPU
salloc --nodes=1 -p gpu --account=pn-dscien --time=08:00:00

# Conectar ao nó alocado (substituir pelo hostname do dia)
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 <nó-alocado>
# ex: atn2b03n06, atn2b01n01, etc.
```

---

## 2. Preparar ambiente e rodar treino (seed 42)

### 2.1 Preparar ambiente

```bash
# Verificar se tudo está OK
bash ~/projetos/SaltSegmentation-UNet/check_node_Atena.sh

# Se venv ou dataset ausente:
bash ~/projetos/SaltSegmentation-UNet/setup_node_Atena.sh

# Ativar venv e entrar no diretório do código
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet
```

### 2.2 Variáveis comuns

```bash
PROJ=/u/cym7/projetos/SaltSegmentation-UNet
SPLIT=/var/tmp/cym7/datasets/subset_split
```

### 2.3 Comando de treino — seed 42 (exemplos)

**Cenário A — TGS completo (baseline):**
```bash
mkdir -p $PROJ/results/scenario_A_seed42
env python -u train.py --scenario A --seed 42 --epochs 100 \
  2>&1 | tee $PROJ/results/scenario_A_seed42/train.log
```

**Cenário A — subset_1_99 com test canônico (melhor resultado até agora):**
```bash
mkdir -p $PROJ/results/scenario_A_seed42_subset199
env TGS_PATH=/var/tmp/cym7/datasets/subset_1_99 \
  python -u train.py --scenario A --seed 42 --epochs 100 \
  --test_dir $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_A_seed42_subset199/train.log
```

**Cenário A — subset_split/train_filtered com test canônico:**
```bash
mkdir -p $PROJ/results/scenario_A_seed42_train_filtered
python -u train.py --scenario A --seed 42 --epochs 100 \
  --train_dir $SPLIT/train_filtered \
  --test_dir  $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_A_seed42_train_filtered/train.log
```

**Cenário B — subset_split + 955 sintéticos sísmicos:**
```bash
mkdir -p $PROJ/results/scenario_B_seed42_train_filtered
python -u train.py --scenario B --seed 42 --n_synth 955 --epochs 100 \
  --train_dir $SPLIT/train_filtered \
  --test_dir  $SPLIT/test \
  2>&1 | tee $PROJ/results/scenario_B_seed42_train_filtered/train.log
```

> **Nota `tee`:** roda em foreground com saída visível no terminal E salva no arquivo.  
> Para rodar em background sem bloquear o terminal, use `nohup ... > log 2>&1 &`

---

## 3. Monitorar treinamento em tempo real

```bash
# Se rodou em foreground (tee): acompanhe diretamente no terminal SSH

# Se rodou em background (nohup):
tail -f $PROJ/results/scenario_A_seed42_subset199/train.log

# Ver todas as épocas resumidas de todos os runs ativos
watch -n 15 'tail -n 2 '$PROJ'/results/*/train.log'

# Ver processos ativos
ps aux | grep train.py | grep -v grep

# Uso de GPU
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
```

---

## 4. Avaliar resultado

```bash
# Ler result.csv do run
cat $PROJ/results/scenario_A_seed42_subset199/result.csv

# Comparar todos os runs disponíveis
for f in $PROJ/results/*/result.csv; do
  echo "--- $(basename $(dirname $f)) ---"
  cat "$f"
done
```

**Colunas do result.csv:**
`scenario, seed, n_real, n_synth, best_val_iou, test_iou, test_dice, epochs_run, elapsed_s`

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

```bash
# Path no servidor (home NFS — sincronizado com Windows)
/u/cym7/projetos/SaltSegmentation-UNet/docs/relatorio-final-r21-downstream.md

# Path no Windows (VS Code)
f:\projetos\SaltSegmentation-UNet\docs\relatorio-final-r21-downstream.md
```

**Seções do relatório a atualizar após novo experimento:**

1. Acrescentar nova subseção em **`## 2. Protocolo executado`** com tabela do run
2. Atualizar **`## 3. Comparação principal`** com nova linha na tabela geral
3. Atualizar **`## 12. Comparação por dataset de treino`** se for novo dataset
4. Atualizar **`## 6. Conclusão final`** se o resultado mudar o achado principal

**Template para nova subseção:**
```markdown
### 2.X Cenário A — `<nome_dataset>` com test canônico (seed 42)

| Seed | N real | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:------:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42 | XXXX | X.XXXX | X.XXXX | X.XXXX | XX | XXX |
```

---

## 6. Achados-chave da sessão (não perder)

### 6.1 Filtrar dataset de treino melhora o resultado

Usando o **mesmo test canônico de 800 amostras reais**, a ordem de desempenho é:

| Dataset treino | Filtro | N treino | Test IoU |
|:--------------:|:------:|:--------:|:--------:|
| `subset_1_99` | 1–99% | 2209 | **0.4791** ✅ melhor |
| `subset_10_90` + test canônico | 10–90% | 1616 | 0.4590 |
| TGS completo | nenhum | 3198 | 0.4312 |
| `train_filtered` | 10–90% | 1293 | 0.4201 |

**Conclusão:** remover apenas os casos triviais extremos (0% e 100% de sal) melhora a generalização, mesmo com menos dados.

### 6.2 Sintéticos sísmicos são melhores que geométricos

| Sintéticos | N synth | Test IoU (vs A=0.4247) | Δ |
|:----------:|:-------:|:----------------------:|:-:|
| 400 originais | 400 | 0.4127 | −0.012 |
| 1600 geométricos | 1600 | 0.4070 | −0.018 |
| **955 sísmicos** | **955** | **0.4204** | **−0.004** |
| 955 sísmicos + train_filtered | 955 | **0.4308** | **+0.011 ✅** |

**Conclusão:** B > A **apenas** com train filtrado (1–99% ou 10–90%) + sintéticos sísmicos.

### 6.3 Cuidado com métricas em test sets diferentes

| Configuração | Test set | Test IoU | ⚠️ Comparável? |
|:------------:|:--------:|:--------:|:--------------:|
| `subset_10_90` split interno | filtrado (~293) | 0.8340 | ❌ não comparar com os demais |
| `subset_1_99` split interno | filtrado (~442) | 0.7662 | ❌ não comparar com os demais |
| Todos os demais | **800 canônicos** | 0.41–0.48 | ✅ comparáveis entre si |

Sempre usar `--test_dir /var/tmp/cym7/datasets/subset_split/test` para resultados comparáveis.

### 6.4 Novos argumentos do train.py (adicionados nesta sessão)

```
--train_dir <path>   # substitui TGS_PATH para treino externo
--test_dir  <path>   # usa test set externo fixo (pula split interno 80/20)
```

O `run_tag` gerado inclui automaticamente o nome da pasta de `--train_dir`:
- `--train_dir .../train_filtered` → `scenario_A_seed42_train_filtered`
- `--train_dir .../subset_1_99` → `scenario_A_seed42_subset_1_99`

### 6.5 Datasets gerados e disponíveis

| Dataset | Como foi gerado | Onde está |
|---------|----------------|-----------|
| `subset_1_99` | script inline Python — filtrar `split_stats.csv` salt_pct 1–99 | `/var/tmp/cym7/datasets/subset_1_99/` (SSD local) |
| `subset_split` | tar já disponível | `dataset/subset_split.tar` → extrair para `/var/tmp/cym7/datasets/` |
| `subset_10_90` | tar já disponível | `dataset/subset_10_90.tar` → extrair para `/var/tmp/cym7/datasets/` |

### 6.6 Pendências para completar o experimento

- [ ] Rodar `subset_1_99` com seeds 123 e 456 (para média ± std)
- [ ] Rodar Cenário B com `subset_1_99` + 955 sísmicos (seeds 42, 123, 456)
- [ ] Confirmar se B > A também em `subset_1_99` (já confirmado em `train_filtered` seed 42: +0.011)
- [ ] Atualizar seção R2.1 no `_v7.tex` e `response_to_reviewers.md`

---

## 7. Gerar novo dataset filtrando por cobertura de sal

O arquivo `dataset/subset_split/split_stats.csv` contém todas as 3998 amostras TGS com a coluna `salt_pct` (cobertura em %). Use o script inline abaixo para criar qualquer subset filtrado.

### 7.1 Script inline — executar no servidor

```bash
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet

python - <<'EOF'
import os, shutil, pandas as pd

# ── CONFIGURAR AQUI ──────────────────────────────────────────
MIN_PCT = 1.0    # cobertura mínima de sal (%)
MAX_PCT = 99.0   # cobertura máxima de sal (%)
OUT_DIR = '/var/tmp/cym7/datasets/subset_1_99'  # nome do diretório de saída
# ─────────────────────────────────────────────────────────────

STATS   = 'dataset/subset_split/split_stats.csv'
SRC_IMG = '/var/tmp/cym7/datasets/tgs-salt/train/images'
SRC_MSK = '/var/tmp/cym7/datasets/tgs-salt/train/masks'

df = pd.read_csv(STATS)
filtered = df[(df['salt_pct'] >= MIN_PCT) & (df['salt_pct'] <= MAX_PCT)]
print(f'Total: {len(df)} | Filtrado ({MIN_PCT}–{MAX_PCT}%): {len(filtered)}')

os.makedirs(f'{OUT_DIR}/images', exist_ok=True)
os.makedirs(f'{OUT_DIR}/masks',  exist_ok=True)

copied, missing = 0, 0
for _, row in filtered.iterrows():
    stem = row['id']
    si = f'{SRC_IMG}/{stem}.png'
    sm = f'{SRC_MSK}/{stem}.png'
    if os.path.exists(si) and os.path.exists(sm):
        shutil.copy2(si, f'{OUT_DIR}/images/{stem}.png')
        shutil.copy2(sm, f'{OUT_DIR}/masks/{stem}.png')
        copied += 1
    else:
        missing += 1

print(f'Copiados: {copied} | Não encontrados: {missing}')
print(f'Min: {filtered["salt_pct"].min():.2f}% | Max: {filtered["salt_pct"].max():.2f}% | Média: {filtered["salt_pct"].mean():.2f}%')
filtered.to_csv(f'{OUT_DIR}/subset_stats.csv', index=False)
print(f'Dataset salvo em: {OUT_DIR}')
EOF
```

### 7.2 Parâmetros para subsets comuns

| Subset desejado | MIN_PCT | MAX_PCT | N esperado |
|:---------------:|:-------:|:-------:|:----------:|
| `subset_1_99` — remove extremos absolutos | 1.0 | 99.0 | ~2209 |
| `subset_5_95` — margem maior | 5.0 | 95.0 | ~2000 |
| `subset_10_90` — apenas casos intermediários | 10.0 | 90.0 | ~1616 |
| `subset_20_80` — casos mais balanceados | 20.0 | 80.0 | ~1200 |
| `subset_0_50` — apenas com pouco sal | 0.0 | 50.0 | ~2200 |

### 7.3 Usar o dataset gerado no treino

```bash
# Com split interno (80/20 do próprio subset):
env TGS_PATH=/var/tmp/cym7/datasets/subset_1_99 \
  python -u train.py --scenario A --seed 42 --epochs 100 \
  2>&1 | tee $PROJ/results/scenario_A_seed42_subset199_internaltest/train.log

# Com test canônico fixo (800 amostras reais — RECOMENDADO para comparação):
env TGS_PATH=/var/tmp/cym7/datasets/subset_1_99 \
  python -u train.py --scenario A --seed 42 --epochs 100 \
  --test_dir /var/tmp/cym7/datasets/subset_split/test \
  2>&1 | tee $PROJ/results/scenario_A_seed42_subset199/train.log
```

> ⚠️ **Sempre usar `--test_dir` com o test canônico** para resultados comparáveis entre subsets.
