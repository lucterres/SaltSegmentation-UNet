# QUICKSTART — Experimento Downstream R2.1

**Manuscrito:** Access-2026-27912 | **Data:** 2026-07-23

---

## 0. Mapear Home Area 

    \\homeunix-rio.petrobras.biz\cym7 -> E:
    no VSCode abrir pasta do codigo fonte - E:\projetos\SaltSegmentation-UNet

---

## 1. Conectar ao nó GPU alocado
### 1.1 ssh no nó de login
```bash
ssh atena03.petrobras.biz
```

### 1.2 solicitar no com gpu: 
```bash
salloc --nodes=1 -p gpu --account=pn-dscien --time=08:00:00
```

### 1.3 conectar 
```bash
# Abrir terminal SSH persistente (substituir pelo nó alocado no dia)
# Para descobrir o nó após reconexão: ssh atena03 → squeue -u cym7 → ver NODELIST
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 atn2b03n07
# ex: atn2b03n01, atn2b04n02, etc.
```

---

## 2. Verificar / preparar o ambiente

```bash
bash ~/projetos/SaltSegmentation-UNet/check_node_Atena.sh
```

O script verifica (sem modificar nada):
- ✅ Dataset local (`/var/tmp/cym7/datasets/tgs-salt/train/images/` — esperado: 3998 imagens)
- ✅ Venv (`/var/tmp/cym7/venvs/salt-unet/`) — Python, PyTorch e CUDA
- ✅ GPUs disponíveis via `nvidia-smi`

**Se algum item aparecer como AUSENTE**, execute o setup completo:

```bash
bash ~/projetos/SaltSegmentation-UNet/setup_node_Atena.sh
```

O `setup_node_Atena.sh` restaura automaticamente:
1. Dataset — extrai de `\\dfs.petrobras.biz\cientifico\cenpes\atena_projetos\cym7\dataset\tgsSalt\tgs-salt.tar` → `/var/tmp/cym7/datasets/`
2. Venv — extrai de `\\dfs.petrobras.biz\cientifico\cenpes\pdiep\res_ee\projetos30\cym7\envs\salt-unet-venv.tar` → `/var/tmp/cym7/venvs/` e corrige `pyvenv.cfg`

Importações Python (import torch, import numpy, etc.) carregam dezenas de .so do venv → no NFS isso pode levar 10–30s extras por processo
---

## 3. Ativar o ambiente e ir para o projeto

```bash
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet

# Confirmar GPU
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
# Esperado: 2.4.1+cu124 | CUDA: True | GPUs: 8
```

---

## 4. Comandos de treinamento

> Sempre definir `PROJ` antes de rodar. Os diretórios de resultado são criados automaticamente pelo `train.py`.

```bash
PROJ=/u/cym7/projetos/SaltSegmentation-UNet
```

### Argumentos do `train.py`

| Argumento | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--scenario` | A/B | obrigatório | A = real only; B = real + sintéticos |
| `--seed` | int | 42 | Semente para reprodutibilidade |
| `--n_real` | int | None | Limitar amostras reais (low-data regime) |
| `--n_synth` | int | 400 | Número de sintéticos no Cenário B |
| `--epochs` | int | 50 | Máximo de épocas (early stop por val IoU) |
| `--batch` | int | 16 | Batch size |
| `--lr` | float | 1e-4 | Learning rate |
| `--train_dir` | path | None | **Novo** — pasta `images/`+`masks/` para treino externo (sobrescreve `TGS_PATH`) |
| `--test_dir` | path | None | **Novo** — pasta `images/`+`masks/` para test set fixo externo (pula split interno) |

> **Nota:** quando `--train_dir` é fornecido, o `run_tag` recebe o sufixo do nome da pasta (ex: `scenario_A_seed42_train_filtered`).  
> Quando `TGS_PATH` é definido via variável de ambiente (ex: `env TGS_PATH=.../subset_1_99`), o sufixo é extraído automaticamente do nome do diretório (ex: `scenario_A_seed42_subset_1_99`).  
> Quando `--test_dir` é fornecido, o split interno 80/20 é ignorado — o test set externo é usado diretamente.

---

### Cenário A — Real only (seeds 42, 123, 456 — dataset completo ~3200)

```bash
mkdir -p $PROJ/results/scenario_A_seed{42,123,456}

nohup python -u train.py --scenario A --seed 42  --epochs 100 > $PROJ/results/scenario_A_seed42/train.log  2>&1 &
nohup python -u train.py --scenario A --seed 123 --epochs 100 > $PROJ/results/scenario_A_seed123/train.log 2>&1 &
nohup python -u train.py --scenario A --seed 456 --epochs 100 > $PROJ/results/scenario_A_seed456/train.log 2>&1 &
```

### Cenário A — Low-data regime (seed 42, N amostras)

```bash
# Substituir <N> por: 200, 400, 800, 1200, 2000
N=800
mkdir -p $PROJ/results/scenario_A_seed42_nreal${N}
nohup python -u train.py --scenario A --seed 42 --n_real $N --epochs 100 \
  > $PROJ/results/scenario_A_seed42_nreal${N}/train.log 2>&1 & echo "PID: $!"
```

### Cenário B — Real + Synthetic (seeds 42, 123, 456 — dataset completo)

```bash
# Verificar/ajustar symlink para o pool sintético desejado:
# 400 sintéticos originais:
ln -sfn /var/tmp/cym7/datasets/tgs-salt/tgs-salt/synthetic400 dataset/synthetic
# 955 sintéticos sísmicos (melhor resultado):
ln -sfn /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet/dataset/geometric1600_seismic/pairs1600_seismic dataset/synthetic

ls dataset/synthetic/  # deve mostrar: images  masks

mkdir -p $PROJ/results/scenario_B_seed{42,123,456}

nohup python -u train.py --scenario B --seed 42  --n_synth 955 --epochs 100 > $PROJ/results/scenario_B_seed42/train.log  2>&1 &
nohup python -u train.py --scenario B --seed 123 --n_synth 955 --epochs 100 > $PROJ/results/scenario_B_seed123/train.log 2>&1 &
nohup python -u train.py --scenario B --seed 456 --n_synth 955 --epochs 100 > $PROJ/results/scenario_B_seed456/train.log 2>&1 &
```

### Cenário A/B — com dataset externo `subset_split` (protocolo canônico — melhor resultado)

```bash
# Pré-requisito: extrair subset_split no SSD local
tar -xf /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet/dataset/subset_split.tar \
    -C /var/tmp/cym7/datasets/

SPLIT=/var/tmp/cym7/datasets/subset_split

# Cenário A — 3 seeds
mkdir -p $PROJ/results/scenario_A_seed{42,123,456}_train_filtered
nohup python -u train.py --scenario A --seed 42  --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  > $PROJ/results/scenario_A_seed42_train_filtered/train.log  2>&1 &
nohup python -u train.py --scenario A --seed 123 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  > $PROJ/results/scenario_A_seed123_train_filtered/train.log 2>&1 &
nohup python -u train.py --scenario A --seed 456 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  > $PROJ/results/scenario_A_seed456_train_filtered/train.log 2>&1 &

# Cenário B — 3 seeds (+ 955 sintéticos sísmicos)
mkdir -p $PROJ/results/scenario_B_seed{42,123,456}_train_filtered
nohup python -u train.py --scenario B --seed 42  --n_synth 955 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  > $PROJ/results/scenario_B_seed42_train_filtered/train.log  2>&1 &
nohup python -u train.py --scenario B --seed 123 --n_synth 955 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  > $PROJ/results/scenario_B_seed123_train_filtered/train.log 2>&1 &
nohup python -u train.py --scenario B --seed 456 --n_synth 955 --epochs 100 \
  --train_dir $SPLIT/train_filtered --test_dir $SPLIT/test \
  > $PROJ/results/scenario_B_seed456_train_filtered/train.log 2>&1 &
```

### Cenário A/B — com `subset_10_90` (train e test filtrados 10–90%)

```bash
# Pré-requisito: extrair subset_10_90 no SSD local
tar -xf /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet/dataset/subset_10_90.tar \
    -C /var/tmp/cym7/datasets/

mkdir -p $PROJ/results/scenario_{A,B}_seed42_subset1090

nohup env TGS_PATH=/var/tmp/cym7/datasets/subset_10_90 \
  python -u train.py --scenario A --seed 42 --epochs 100 \
  > $PROJ/results/scenario_A_seed42_subset1090/train.log 2>&1 &

nohup env TGS_PATH=/var/tmp/cym7/datasets/subset_10_90 \
  python -u train.py --scenario B --seed 42 --n_synth 955 --epochs 100 \
  > $PROJ/results/scenario_B_seed42_subset1090/train.log 2>&1 &
```

### Cenário B — Low-data regime (800 reais + sintéticos)

```bash
mkdir -p $PROJ/results/scenario_B_seed{42,123,456}_nreal800

nohup python -u train.py --scenario B --seed 42  --n_real 800 --n_synth 400 --epochs 100 > $PROJ/results/scenario_B_seed42_nreal800/train.log  2>&1 &
nohup python -u train.py --scenario B --seed 123 --n_real 800 --n_synth 400 --epochs 100 > $PROJ/results/scenario_B_seed123_nreal800/train.log 2>&1 &
nohup python -u train.py --scenario B --seed 456 --n_real 800 --n_synth 400 --epochs 100 > $PROJ/results/scenario_B_seed456_nreal800/train.log 2>&1 &
```

---

## 5. Monitorar treinamentos

```bash
PROJ=/u/cym7/projetos/SaltSegmentation-UNet

# Acompanhar um run em tempo real
tail -f $PROJ/results/scenario_A_seed42/train.log

# Ver últimas linhas de todos os runs ativos
watch -n 15 'tail -n 2 '$PROJ'/results/*/train.log'

# Processos ativos
ps aux | grep train.py | grep -v grep

# Uso de GPU
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

---

## 6. Avaliação final

```bash
cd /u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet
python -u evaluate.py --results_dir ../results
# Saída: ../results/summary.csv
```

---

## 7. Resultados obtidos até 2026-07-23

### Cenário A — Escala de dados (seed 42, GPU V100)

| N real | Test IoU | Test Dice | Épocas | Tempo |
|:------:|:--------:|:---------:|:------:|:-----:|
| 200 | 0.2587 | 0.3178 | 10 | 4.5s |
| 400 | 0.3198 | 0.3623 | 10 | 6.6s |
| 800 | 0.3771 | 0.4168 | 75 | 79s |
| 1200 | 0.3862 | 0.4252 | 72 | 104s |
| 2000 | 0.4067 | 0.4423 | 46 | 110s |
| ~3200 | **0.4312** | **0.4657** | 57 | 202s |

### Cenário A vs B — N=800, 3 seeds (GPU V100)

| Cenário | N real | N synth | Seed | Test IoU | Test Dice |
|:-------:|:------:|:-------:|:----:|:--------:|:---------:|
| A | 800 | 0 | 42 | 0.3771 | 0.4168 |
| A | 800 | 0 | 123 | 0.3874 | 0.4249 |
| A | 800 | 0 | 456 | 0.3812 | 0.4219 |
| **A média** | | | | **0.382** | **0.421** |
| B | 800 | 400 | 42 | 0.3650 | 0.4057 |
| B | 800 | 400 | 123 | 0.3646 | 0.4037 |
| B | 800 | 400 | 456 | 0.3789 | 0.4191 |
| **B média** | | | | **0.369** | **0.410** |

### Cenário A — Dataset completo (~3200), 3 seeds (GPU V100)

| Seed | Test IoU | Test Dice | Épocas |
|:----:|:--------:|:---------:|:------:|
| 42 | 0.4312 | 0.4657 | 57 |
| 123 | 0.4190 | 0.4544 | 52 |
| 456 | 0.4240 | 0.4593 | 54 |
| **média** | **0.425 ± 0.006** | **0.460 ± 0.006** | |

> **Pendente:** Cenário B com dataset completo (~3200 + 400) — comparação principal do paper.

---

## 8. Paths de referência

| Recurso | Path |
|---------|------|
| Código | `/u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet/` |
| Resultados | `/u/cym7/projetos/SaltSegmentation-UNet/results/` |
| venv (SSD local) | `/var/tmp/cym7/venvs/salt-unet/` |
| venv (backup home) | `/u/cym7/venvs_backup/salt-unet/` |
| Dataset TGS (SSD local) | `/var/tmp/cym7/datasets/tgs-salt/train/` |
| Dataset TGS (tar backup) | `\\dfs.petrobras.biz\cientifico\cenpes\atena_projetos\cym7\dataset\tgsSalt\tgs-salt.tar` |
| venv (tar backup) | `\\dfs.petrobras.biz\cientifico\cenpes\pdiep\res_ee\projetos30\cym7\envs\salt-unet-venv.tar` |
| Dados sintéticos | `/var/tmp/cym7/datasets/tgs-salt/tgs-salt/synthetic400/` |
| Symlink sintéticos | `dataset/synthetic/` → path acima |
