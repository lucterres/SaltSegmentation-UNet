#!/bin/bash
# =============================================================================
# run_cenarioB_6albu_seed42_parallel.sh
# =============================================================================
# Lança 6 runs em paralelo, cada uma em uma GPU dedicada (GPUs 1–6 livres).
# GPU 0 está em uso (17%) — as 6 runs usam GPUs 1, 2, 3, 4, 5, 6.
#
# Uso (no nó atn2b03n03, já com o nó alocado):
#   bash $PROJ/run_cenarioB_6albu_seed42_parallel.sh
#
# Acompanhar:
#   bash $PROJ/check_progress_6albu_seed42.sh
#   tail -f $PROJ/results/slurm_logs/cenB_seed42_<nome>.log
# =============================================================================

set -e

PROJ="/nethome/atena_projetos/cym7/0code/SaltSegment-Unet"
VENV="/var/tmp/cym7/venvs/salt-unet"
TRAIN_DIR="/var/tmp/cym7/datasets/tgs-salt/train"
TEST_DIR="/var/tmp/cym7/datasets/subset_split/test"
SYNTH_BASE="$PROJ/Salt-Segmentation-UNet/dataset"
LOG_DIR="$PROJ/results/slurm_logs"

mkdir -p "$LOG_DIR"

# Datasets sintéticos clean (sem leakage do test canônico)
SYNTH_DATASETS=(
    "clahe1600clean"
    "elastic_transform1600clean"
    "grid_distortion1600clean"
    "optical_distortion1600clean"
    "random_brightness_contrast1600clean"
    "random_gamma1600clean"
)

# GPUs livres (evita GPU 0 que está em uso)
GPUS=(1 2 3 4 5 6)

echo "============================================================"
echo " Lançando 6 runs em paralelo — Cenário B | seed=42"
echo " N_real=3198 (TGS completo) | N_synth=1200"
echo " Node : $(hostname)"
echo " Data : $(date)"
echo "============================================================"
echo ""

# Verifica ambiente
source "$VENV/bin/activate"
cd "$PROJ/Salt-Segmentation-UNet"

for i in "${!SYNTH_DATASETS[@]}"; do
    SYNTH_NAME="${SYNTH_DATASETS[$i]}"
    SYNTH_DIR="$SYNTH_BASE/$SYNTH_NAME"
    GPU_ID="${GPUS[$i]}"
    LOG="$LOG_DIR/cenB_seed42_${SYNTH_NAME}.log"
    EXPECTED_RUN="$PROJ/results/scenario_B_seed42_train_${SYNTH_NAME}_ns1200"

    # Verifica se já existe resultado
    if [ -f "$EXPECTED_RUN/result.csv" ]; then
        echo "  [SKIP] GPU $GPU_ID — $SYNTH_NAME (já concluído)"
        continue
    fi

    # Verifica dataset sintético
    if [ ! -d "$SYNTH_DIR/images" ]; then
        echo "  [ERROR] Sintético ausente: $SYNTH_DIR — pulando"
        continue
    fi

    N_SYNTH=$(ls "$SYNTH_DIR/images" | wc -l)
    echo "  [START] GPU $GPU_ID — $SYNTH_NAME ($N_SYNTH amostras)"
    echo "          Log: $LOG"

    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u train.py \
        --scenario  B \
        --seed      42 \
        --n_synth   1200 \
        --epochs    100 \
        --train_dir "$TRAIN_DIR" \
        --test_dir  "$TEST_DIR" \
        --synth_dir "$SYNTH_DIR" \
        > "$LOG" 2>&1 &

    echo "          PID: $!"
    echo ""
done

echo "============================================================"
echo " Todos os jobs lançados. PIDs ativos:"
jobs -l
echo ""
echo " Acompanhar progresso:"
echo "   bash $PROJ/check_progress_6albu_seed42.sh"
echo ""
echo " Ver log em tempo real (ex: clahe):"
echo "   tail -f $LOG_DIR/cenB_seed42_clahe1600clean.log"
echo ""
echo " Ver todas as GPUs:"
echo "   watch -n 10 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv'"
echo "============================================================"
