---
applyTo: "**/*.sh"
---

# Instruções — Infraestrutura & Ambiente

**Manuscrito:** Access-2026-27912 | **Última atualização:** 2026-07-27

---

## Caminhos absolutos no servidor

| Recurso | Path absoluto |
|---------|--------------|
| Código `train.py` | `/u/cym7/projetos/SaltSegmentation-UNet/Salt-Segmentation-UNet/` |
| Resultados | `/u/cym7/projetos/SaltSegmentation-UNet/results/` |
| Relatório final | `/u/cym7/projetos/SaltSegmentation-UNet/docs/relatorio-final-r21-downstream.md` |
| venv (SSD local) | `/var/tmp/cym7/venvs/salt-unet/` |
| venv (home backup) | `/u/cym7/venvs_backup/salt-unet/` ← **cópia persistente entre nós** |
| Dataset TGS (SSD local) | `/var/tmp/cym7/datasets/tgs-salt/train/` (3998 pares) |
| Dataset TGS (tar) | `~/datasets/tgs-salt/tgs-salt.tar` |
| **subset_split/train_filtered** | `/var/tmp/cym7/datasets/subset_split/train_filtered/` (1293 amostras, 10–90%) |
| **subset_split/test** | `/var/tmp/cym7/datasets/subset_split/test/` (800 amostras, dist. real) ← **test canônico** |
| **subset_1_99** | `/var/tmp/cym7/datasets/subset_1_99/` (2209 amostras, 1–99%) |
| **subset_10_90** | `/var/tmp/cym7/datasets/subset_10_90/` (1616 amostras, 10–90%) |
| Sintéticos sísmicos | `dataset/geometric1600_seismic/pairs1600_seismic/` (955 pares) |
| Symlink sintéticos | `dataset/synthetic` → path acima |
| subset_split tar | `dataset/subset_split.tar` |
| subset_10_90 tar | `dataset/subset_10_90.tar` |

> **Atenção:** `/var/tmp/` é local a cada nó SLURM e **não persiste** entre alocações.  
> O venv de referência fica em `/u/cym7/venvs_backup/salt-unet/` (NFS home, persistente).

---

## Hardware do cluster Atena (Petrobras)

| Item | Detalhe |
|------|---------|
| Cluster | Atena — nós GPU alocados via SLURM |
| Nó de referência | `atn2b02n07` (venv original criado aqui) |
| GPUs | 8 × Tesla V100-SXM2-32GB |
| CUDA | 12.4 (PyTorch 2.4.1+cu124) |
| Python | 3.8.16 (Miniconda base) |

---

## Alocar nó GPU

```bash
ssh atena03.petrobras.biz
salloc --nodes=1 -p gpu --account=pn-dscien --time=08:00:00
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=10 <nó-alocado>
```

---

## Verificar / configurar ambiente no nó

```bash
bash ~/projetos/SaltSegmentation-UNet/check_node_Atena.sh
# Se ausente ou desatualizado:
bash ~/projetos/SaltSegmentation-UNet/setup_node_Atena.sh
```

---

## Migração do venv para novo nó (fazer 1× por nó)

```bash
# 1. Restaurar do backup para o SSD local
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
