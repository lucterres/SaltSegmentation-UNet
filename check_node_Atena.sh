#!/bin/bash
# check_node_Atena.sh — Verifica o estado do nó (sem modificar nada)
#
# Uso:
#   bash ~/projetos/SaltSegmentation-UNet/check_node_Atena.sh

PROJ="/u/cym7/projetos/SaltSegmentation-UNet"
VENV="$PROJ/venv"
LOCAL_TGS="/var/tmp/cym7/datasets/tgs-salt"

echo "============================================"
echo " Verificação do nó: $(hostname)"
echo " Data             : $(date)"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Dataset local
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Dataset local..."
if [ -d "$LOCAL_TGS/train/images" ]; then
    N=$(ls "$LOCAL_TGS/train/images" | wc -l)
    echo "  OK  $N imagens em $LOCAL_TGS/train/images"
else
    echo "  AUSENTE: $LOCAL_TGS/train/images"
    echo "  Execute setup_node_Atena.sh para extrair o dataset."
fi

# ---------------------------------------------------------------------------
# 2. Venv
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Venv..."
if [ -f "$VENV/bin/python" ]; then
    PY_VER=$("$VENV/bin/python" --version 2>&1)
    TORCH_VER=$("$VENV/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "torch não instalado")
    CUDA_OK=$("$VENV/bin/python" -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'SEM CUDA')" 2>/dev/null || echo "erro")
    echo "  OK  Python  : $PY_VER"
    echo "      PyTorch : $TORCH_VER"
    echo "      GPU     : $CUDA_OK"
else
    echo "  AUSENTE: venv não encontrado em $VENV"
    echo "  Execute setup_node_Atena.sh para criar o venv."
fi

# ---------------------------------------------------------------------------
# 3. GPUs disponíveis
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] GPUs (nvidia-smi)..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
        awk -F',' '{printf "  GPU %s: %s | total %s | livre %s\n", $1, $2, $3, $4}'
else
    echo "  nvidia-smi não disponível."
fi

echo ""
echo "============================================"
echo " Fim da verificação"
echo "============================================"
