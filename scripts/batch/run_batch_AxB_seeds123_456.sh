#!/bin/bash
# Lança até MAX_JOBS runs, uma por GPU livre

PROJ="/nethome/atena_projetos/cym7/0code/SaltSegment-Unet"
VENV="/var/tmp/cym7/venvs/salt-unet"
TRAIN_DIR="/var/tmp/cym7/datasets/tgs-salt/train"
TEST_DIR="/var/tmp/cym7/datasets/subset_split/test"
SYNTH_BASE="$PROJ/Salt-Segmentation-UNet/dataset"
LOG_DIR="$PROJ/results/slurm_logs"
N_SYNTH=1200

SYNTH_DATASETS=(
    "clahe1600clean"
    "elastic_transform1600clean"
    "grid_distortion1600clean"
    "optical_distortion1600clean"
    "random_brightness_contrast1600clean"
    "random_gamma1600clean"
)
SEEDS=(42 123 456)

# Monta fila completa: A s42 s123 s456 + B×6datasets×3seeds
QUEUE=()

# Cenário A — seeds 123 e 456 (42 já existe)
for SEED in 123 456; do
    TAG="scenario_A_seed${SEED}_train"
    QUEUE+=("A|${SEED}|none|${TAG}")
done

# Cenário B — seeds 123 e 456 × 6 datasets (seed 42 já existe)
for SEED in 123 456; do
    for DS in "${SYNTH_DATASETS[@]}"; do
        TAG="scenario_B_seed${SEED}_train_${DS}_ns${N_SYNTH}"
        QUEUE+=("B|${SEED}|${DS}|${TAG}")
    done
done

mkdir -p "$LOG_DIR"
source "$VENV/bin/activate"
cd "$PROJ/Salt-Segmentation-UNet"

echo "Fila total: ${#QUEUE[@]} runs"
echo ""

for ITEM in "${QUEUE[@]}"; do
    IFS='|' read -r SCENARIO SEED DS TAG <<< "$ITEM"
    RUN_DIR="$PROJ/results/$TAG"
    LOG="$LOG_DIR/${TAG}.log"

    # Pula se já concluído
    if [ -f "$RUN_DIR/result.csv" ]; then
        echo "[SKIP] $TAG"
        continue
    fi

    # Aguarda GPU livre (utilização < 10% e memória < 2000 MiB)
    while true; do
        GPU_FREE=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
            awk -F', ' '$2 < 10 && $3 < 2000 {print $1; exit}')
        if [ -n "$GPU_FREE" ]; then
            break
        fi
        echo "  [WAIT] Aguardando GPU livre para $TAG ..."
        sleep 30
    done

    # Monta comando
    if [ "$SCENARIO" = "A" ]; then
        CMD="python -u train.py --scenario A --seed $SEED --epochs 100 --train_dir $TRAIN_DIR --test_dir $TEST_DIR"
    else
        CMD="python -u train.py --scenario B --seed $SEED --n_synth $N_SYNTH --epochs 100 --train_dir $TRAIN_DIR --test_dir $TEST_DIR --synth_dir $SYNTH_BASE/$DS"
    fi

    echo "[START] GPU $GPU_FREE — $TAG"
    CUDA_VISIBLE_DEVICES=$GPU_FREE nohup $CMD > "$LOG" 2>&1 &
    echo "  PID: $!"
    sleep 2  # pequena pausa para GPU registrar ocupação
done

echo ""
echo "[DONE] Todos os jobs lançados."
