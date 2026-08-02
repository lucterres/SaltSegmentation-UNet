#!/bin/bash
# setup_node_Atena.sh — Prepara o nó de computação para os experimentos
#
# Executar após conectar a qualquer nó Atena:
#   bash ~/projetos/SaltSegmentation-UNet/setup_node_Atena.sh
#
# O que faz:
#   1. Extrai o dataset TGS do tar (DFS → SSD local), se necessário
#   2. Restaura o venv do tar (DFS → SSD local), se necessário
#   3. Imprime os comandos prontos para rodar os experimentos
#
# Para apenas verificar o estado do nó (sem modificar nada):
#   bash ~/projetos/SaltSegmentation-UNet/check_node_Atena.sh

set -e

PROJ="/u/cym7/projetos/SaltSegmentation-UNet"
VENV="/var/tmp/cym7/venvs/salt-unet"
VENV_BACKUP_TAR="//dfs.petrobras.biz/cientifico/cenpes/pdiep/res_ee/projetos30/cym7/envs/salt-unet-venv.tar"
LOCAL_TGS="/var/tmp/cym7/datasets/tgs-salt"
TGS_BACKUP="//dfs.petrobras.biz/cientifico/cenpes/atena_projetos/cym7/dataset/tgsSalt/tgs-salt.tar"
REQUIREMENTS="$PROJ/Salt-Segmentation-UNet/requirements.txt"

echo "============================================"
echo " Setup do nó: $(hostname)"
echo " Data       : $(date)"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Dataset — copia do backup persistente em DFS para SSD local se ainda não existir
# ---------------------------------------------------------------------------
echo ""
echo "[1/2] Dataset local..."

if [ -d "$LOCAL_TGS/train/images" ]; then
    N=$(ls "$LOCAL_TGS/train/images" | wc -l)
    echo "  OK  Dataset já presente: $N imagens"
else
    echo "  Extraindo $TGS_BACKUP → $(dirname $LOCAL_TGS) ..."
    mkdir -p "$(dirname "$LOCAL_TGS")"
    tar -xf "$TGS_BACKUP" -C "$(dirname "$LOCAL_TGS")"
    N=$(ls "$LOCAL_TGS/train/images" 2>/dev/null | wc -l || echo "?")
    echo "  OK  Extração concluída: $N imagens"
fi

# ---------------------------------------------------------------------------
# 2. Venv — restaura do backup tar em DFS se não existir no SSD local
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Venv..."

if [ ! -f "$VENV/bin/python" ]; then
    echo "  Restaurando venv de $VENV_BACKUP_TAR → $(dirname $VENV) ..."
    mkdir -p "$(dirname "$VENV")"
    tar -xf "$VENV_BACKUP_TAR" -C "$(dirname "$VENV")"
    # Corrigir pyvenv.cfg para o Python do nó atual
    PYTHON_BIN=$(which python3)
    sed -i "s|^home = .*|home = $(dirname $PYTHON_BIN)|" "$VENV/pyvenv.cfg"
    echo "  OK  Venv restaurado e reconfigurado."
else
    PY_VER=$("$VENV/bin/python" --version 2>&1)
    TORCH_VER=$("$VENV/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "torch não instalado")
    CUDA_OK=$("$VENV/bin/python" -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'SEM CUDA')" 2>/dev/null || echo "erro")
    echo "  OK  Python  : $PY_VER"
    echo "      PyTorch : $TORCH_VER"
    echo "      GPU     : $CUDA_OK"
fi

# ---------------------------------------------------------------------------
# Pronto — imprime comandos para rodar os experimentos
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo " Nó pronto! Comandos para iniciar:"
echo "============================================"
echo ""
echo "source $VENV/bin/activate"
echo "cd $PROJ/Salt-Segmentation-UNet"
echo ""
echo "# Cenário A (real only):"
echo "nohup python -u train.py --scenario A --seed 42  --epochs 100 > $PROJ/results/scenario_A_seed42/train.log  2>&1 &"
echo "nohup python -u train.py --scenario A --seed 123 --epochs 100 > $PROJ/results/scenario_A_seed123/train.log 2>&1 &"
echo "nohup python -u train.py --scenario A --seed 456 --epochs 100 > $PROJ/results/scenario_A_seed456/train.log 2>&1 &"
echo ""
echo "# Cenário B (real + sintético):"
echo "nohup python -u train.py --scenario B --seed 42  --epochs 100 > $PROJ/results/scenario_B_seed42/train.log  2>&1 &"
echo "nohup python -u train.py --scenario B --seed 123 --epochs 100 > $PROJ/results/scenario_B_seed123/train.log 2>&1 &"
echo "nohup python -u train.py --scenario B --seed 456 --epochs 100 > $PROJ/results/scenario_B_seed456/train.log 2>&1 &"
echo ""
