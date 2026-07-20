#!/bin/bash
# setup_and_run.sh — Configura ambiente e executa todos os experimentos no servidor remoto
#
# Adaptado para: atn2b02n07 (8× Tesla V100-SXM2-32GB, CUDA 12.9, driver 575)
#
# Usage (rodar no servidor remoto):
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
#
# Pré-requisitos no servidor:
#   - Dataset TGS já em: /var/tmp/cym7/datasets/tgs-salt/
#   - Python 3.8 (Miniconda base) disponível em:
#     /cenpes/projetos30/cym7/Miniconda3/bin/python
#
# O venv será criado em: /var/tmp/cym7/venvs/salt-unet/  (SSD NVMe local)
# Resultados em:         $SCRIPT_DIR/results/

set -e  # exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/Salt-Segmentation-UNet"
RESULTS_DIR="$SCRIPT_DIR/results"

# Dataset já transferido para o SSD local
TGS_SRC="/var/tmp/cym7/datasets/tgs-salt"
TGS_DIR="$WORK_DIR/dataset/tgs"
SYNTH_DIR="$WORK_DIR/dataset/synthetic"

# venv no SSD local (rápido, sem NFS)
VENV_DIR="/var/tmp/cym7/venvs/salt-unet"
PYTHON_BASE="/cenpes/projetos30/cym7/Miniconda3/bin/python"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

echo "============================================"
echo " R2.1 Downstream Segmentation Experiment"
echo " Servidor  : $(hostname)"
echo " Work dir  : $WORK_DIR"
echo " Results   : $RESULTS_DIR"
echo " venv      : $VENV_DIR"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Criar venv e instalar dependências
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Configurando ambiente Python"

if [ -f "$PYTHON" ]; then
    echo "  → venv já existe em $VENV_DIR"
else
    echo "  → Criando venv com $($PYTHON_BASE --version)..."
    "$PYTHON_BASE" -m venv "$VENV_DIR"
    "$PIP" install --upgrade pip --quiet
    echo "  → venv criado"
fi

# Verificar se torch+CUDA já está instalado corretamente
TORCH_OK=$("$PYTHON" -c "import torch; print('ok' if '+cu' in torch.__version__ else 'rocm')" 2>/dev/null || echo "missing")

if [ "$TORCH_OK" = "ok" ]; then
    echo "  → PyTorch+CUDA já instalado: $("$PYTHON" -c 'import torch; print(torch.__version__)')"
else
    echo "  → Instalando PyTorch 2.4.1+cu124 (JFrog interno)..."
    # JFrog PyPI interno já configurado em ~/.pip/pip.conf
    "$PIP" install --quiet \
        "torch==2.4.1+cu124" \
        "torchvision==0.19.1+cu124"
    echo "  → PyTorch instalado"
fi

# Instalar demais dependências
echo "  → Instalando dependências do requirements.txt..."
"$PIP" install --quiet \
    "opencv-python-headless>=4.7.0" \
    "imutils>=0.5.4" \
    "scikit-learn>=1.2.0" \
    "pandas>=2.0.0" \
    "scipy>=1.10.0" \
    "numpy>=1.24.0" \
    "tqdm>=4.65.0" \
    "matplotlib>=3.7.0"

# Verificar CUDA
echo ""
"$PYTHON" -c "
import torch
print(f'  torch   : {torch.__version__}')
print(f'  cuda    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  gpu[0]  : {torch.cuda.get_device_name(0)}')
    print(f'  gpus    : {torch.cuda.device_count()}')
"
echo "  → Ambiente pronto"

# ---------------------------------------------------------------------------
# 2. Preparar dataset TGS
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Verificando dataset TGS"

N_IMAGES=$(ls -1 "$TGS_DIR/images/"*.png 2>/dev/null | wc -l)
N_MASKS=$(ls -1 "$TGS_DIR/masks/"*.png 2>/dev/null | wc -l)

if [ "$N_IMAGES" -ge 3000 ] && [ "$N_MASKS" -ge 3000 ]; then
    echo "  → Dataset já presente: $N_IMAGES imagens, $N_MASKS máscaras"
else
    echo "  → Copiando dataset de $TGS_SRC ..."
    mkdir -p "$TGS_DIR"

    # O dataset TGS tem imagens e máscaras na estrutura:
    #   tgs-salt/train/images/*.png  e  tgs-salt/train/masks/*.png
    if [ -d "$TGS_SRC/train/images" ]; then
        cp -r "$TGS_SRC/train/images" "$TGS_DIR/images"
        cp -r "$TGS_SRC/train/masks"  "$TGS_DIR/masks"
    elif [ -d "$TGS_SRC/images" ]; then
        cp -r "$TGS_SRC/images" "$TGS_DIR/images"
        cp -r "$TGS_SRC/masks"  "$TGS_DIR/masks"
    else
        echo "  ERRO: estrutura inesperada em $TGS_SRC"
        ls "$TGS_SRC"
        exit 1
    fi
    echo "  → Dataset pronto: $(ls -1 $TGS_DIR/images/*.png | wc -l) imagens"
fi

# Atualizar paths no config.py
sed -i "s|TGS_PATH\s*=.*|TGS_PATH = '$TGS_DIR'|g" "$WORK_DIR/utils/config.py"
sed -i "s|SYNTH_PATH\s*=.*|SYNTH_PATH = '$SYNTH_DIR'|g" "$WORK_DIR/utils/config.py"
echo "  → config.py: TGS_PATH=$TGS_DIR"

# ---------------------------------------------------------------------------
# 3. Gerar pool sintético (Cenário B) — 400 imagens
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Gerando pool sintético para o Cenário B"

N_SYNTH=$(ls -1 "$SYNTH_DIR/images/"*.png 2>/dev/null | wc -l)
if [ "$N_SYNTH" -ge 400 ]; then
    echo "  → Pool sintético já presente ($N_SYNTH imagens)"
else
    echo "  → Rodando generate_synthetic.py (n=400)..."
    cd "$WORK_DIR"
    "$PYTHON" generate_synthetic.py \
        --n 400 \
        --out "$SYNTH_DIR" \
        --tgs_dir "$TGS_DIR"
    echo "  → Pool sintético pronto"
    cd "$SCRIPT_DIR"
fi

# ---------------------------------------------------------------------------
# 4. Executar todos os treinamentos
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Executando experimentos"
mkdir -p "$RESULTS_DIR"

cd "$WORK_DIR"

SEEDS=(42 123 456)
SCENARIOS=(A B)
COUNT=0

for SCENARIO in "${SCENARIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        RUN_TAG="scenario_${SCENARIO}_seed${SEED}"
        LOG="$RESULTS_DIR/${RUN_TAG}_log.txt"

        echo ""
        echo "  [$COUNT/6] Cenário $SCENARIO | Seed $SEED → $RUN_TAG"

        if [ -f "$RESULTS_DIR/$RUN_TAG/result.csv" ]; then
            echo "  → Já concluído, pulando"
            continue
        fi

        START=$(date +%s)
        "$PYTHON" -u train.py \
            --scenario "$SCENARIO" \
            --seed "$SEED" \
            2>&1 | tee "$LOG"
        END=$(date +%s)
        echo "  → Concluído em $((END - START))s"
    done
done

# ---------------------------------------------------------------------------
# 5. Consolidar resultados
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Consolidando resultados"
cd "$WORK_DIR"
"$PYTHON" -u evaluate.py --results_dir "$RESULTS_DIR"

echo ""
echo "============================================"
echo " CONCLUÍDO — resultados em: $RESULTS_DIR/summary.csv"
echo "============================================"
cat "$RESULTS_DIR/summary.csv" 2>/dev/null || echo "(summary.csv ainda não disponível)"
