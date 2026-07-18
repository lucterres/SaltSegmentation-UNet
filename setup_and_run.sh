#!/bin/bash
# setup_and_run.sh — Configure environment and run all experiment runs on remote GPU
#
# Usage:
#   1. Copy this script and the experiment code to the remote machine:
#        scp -r experiment-downstream/ user@remote-host:~/R2.1-experiment/
#        scp setup_and_run.sh user@remote-host:~/R2.1-experiment/
#
#   2. On the remote machine:
#        chmod +x setup_and_run.sh
#        ./setup_and_run.sh
#
# Requirements on remote machine:
#   - conda (miniforge/miniconda)
#   - CUDA driver >= 525 (for cu126)
#   - kaggle API token at ~/.kaggle/kaggle.json (for dataset download)
#     OR dataset already at $TGS_DIR
#
# Results will be in: ~/R2.1-experiment/results/

set -e  # exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/Salt-Segmentation-UNet"
RESULTS_DIR="$SCRIPT_DIR/results"
ENV_NAME="r21_exp"
TGS_DIR="$WORK_DIR/dataset/tgs"
SYNTH_DIR="$WORK_DIR/dataset/synthetic"

echo "============================================"
echo " R2.1 Downstream Segmentation Experiment"
echo " Work dir : $WORK_DIR"
echo " Results  : $RESULTS_DIR"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Setting up conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "  → Environment already exists, skipping creation"
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi

# Install PyTorch (CUDA 12.6 — compatible with A100/H100 drivers)
conda run -n "$ENV_NAME" pip install \
    torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
conda run -n "$ENV_NAME" pip install \
    opencv-python imutils scikit-learn pandas scipy tqdm matplotlib kaggle

echo "  → Environment ready"

# Verify CUDA
conda run -n "$ENV_NAME" python -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  cuda:  {torch.cuda.is_available()}')
print(f'  gpu:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')
"

# ---------------------------------------------------------------------------
# 2. Download TGS dataset
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Checking TGS dataset"

if [ -d "$TGS_DIR/images" ] && [ "$(ls -1 $TGS_DIR/images/*.png 2>/dev/null | wc -l)" -gt 3000 ]; then
    echo "  → Dataset already present ($(ls -1 $TGS_DIR/images/*.png | wc -l) images)"
else
    echo "  → Downloading TGS Salt Identification Challenge dataset..."
    mkdir -p "$TGS_DIR"
    conda run -n "$ENV_NAME" kaggle competitions download \
        -c tgs-salt-identification-challenge \
        -p "$TGS_DIR"
    echo "  → Extracting..."
    cd "$TGS_DIR"
    unzip -q tgs-salt-identification-challenge.zip
    # Move train/images and train/masks to TGS_DIR root
    if [ -d "train/images" ]; then
        mv train/images ./images
        mv train/masks  ./masks
        rm -rf train test *.zip *.csv 2>/dev/null || true
    fi
    cd "$SCRIPT_DIR"
    echo "  → Dataset ready: $(ls -1 $TGS_DIR/images/*.png | wc -l) images"
fi

# Update config.py with the correct TGS path
sed -i "s|TGS_PATH.*=.*|TGS_PATH = '$TGS_DIR'|g" "$WORK_DIR/utils/config.py"
echo "  → config.py updated: TGS_PATH = $TGS_DIR"

# ---------------------------------------------------------------------------
# 3. Generate synthetic pool (400 images) — Scenario B
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Generating synthetic pool for Scenario B"

if [ -d "$SYNTH_DIR/images" ] && [ "$(ls -1 $SYNTH_DIR/images/*.png 2>/dev/null | wc -l)" -ge 400 ]; then
    echo "  → Synthetic pool already present ($(ls -1 $SYNTH_DIR/images/*.png | wc -l) images)"
else
    echo "  → Running generate_synthetic.py (n=400)..."
    echo "  → NOTE: Point --vae_ckpt to your VAE checkpoint if available"
    echo "  →       Without checkpoint, random masks will be used (for testing)"
    conda run -n "$ENV_NAME" python "$WORK_DIR/generate_synthetic.py" \
        --n 400 \
        --out "$SYNTH_DIR" \
        --tgs_dir "$TGS_DIR"
    echo "  → Synthetic pool ready"
fi

# ---------------------------------------------------------------------------
# 4. Run all training scenarios
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Running experiments"
mkdir -p "$RESULTS_DIR"

cd "$WORK_DIR"

SEEDS=(42 123 456)
SCENARIOS=(A B)
TOTAL=${#SEEDS[@]}
COUNT=0

for SCENARIO in "${SCENARIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        RUN_TAG="scenario_${SCENARIO}_seed${SEED}"
        echo ""
        echo "  [$COUNT/6] Scenario $SCENARIO | Seed $SEED → $RUN_TAG"

        if [ -f "$RESULTS_DIR/$RUN_TAG/result.csv" ]; then
            echo "  → Already done, skipping"
            continue
        fi

        START=$(date +%s)
        conda run -n "$ENV_NAME" python -u train.py \
            --scenario "$SCENARIO" \
            --seed "$SEED" \
            2>&1 | tee "$RESULTS_DIR/${RUN_TAG}_log.txt"
        END=$(date +%s)
        echo "  → Done in $((END - START))s"
    done
done

# ---------------------------------------------------------------------------
# 5. Consolidate results
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Consolidating results"
conda run -n "$ENV_NAME" python -u evaluate.py
echo ""
echo "============================================"
echo " DONE — results at: $RESULTS_DIR/summary.csv"
echo "============================================"
cat "$RESULTS_DIR/summary.csv" 2>/dev/null || echo "(summary.csv not yet available)"
