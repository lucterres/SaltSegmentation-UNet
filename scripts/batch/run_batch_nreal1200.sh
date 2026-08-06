#!/bin/bash
# Cenário A e B — N_real=1200, N_synth=1200, test canônico 800
# 3 seeds × (1 A + 6 B) = 21 runs, 1 por GPU, lockfile

PROJ="/nethome/atena_projetos/cym7/0code/SaltSegment-Unet"
VENV="/var/tmp/cym7/venvs/salt-unet"
TRAIN_DIR="/var/tmp/cym7/datasets/tgs-salt/train"
TEST_DIR="/var/tmp/cym7/datasets/subset_split/test"
SYNTH_BASE="$PROJ/Salt-Segmentation-UNet/dataset"
LOG_DIR="$PROJ/results/slurm_logs"
LOCK_DIR="/tmp/gpu_locks_nreal1200_$$"
N_REAL=1200
N_SYNTH=1200

DATASETS=(clahe1600clean elastic_transform1600clean grid_distortion1600clean optical_distortion1600clean random_brightness_contrast1600clean random_gamma1600clean)

# Fila: A×3seeds + B×6datasets×3seeds = 21 runs
QUEUE=()
for SEED in 42 123 456; do
    QUEUE+=("A|${SEED}|none|scenario_A_seed${SEED}_nreal${N_REAL}_train")
done
for SEED in 42 123 456; do
    for DS in "${DATASETS[@]}"; do
        QUEUE+=("B|${SEED}|${DS}|scenario_B_seed${SEED}_nreal${N_REAL}_train_${DS}_ns${N_SYNTH}")
    done
done

mkdir -p "$LOG_DIR" "$LOCK_DIR"
source "$VENV/bin/activate"
cd "$PROJ/Salt-Segmentation-UNet"

echo "Fila: ${#QUEUE[@]} runs | N_real=$N_REAL | N_synth=$N_SYNTH"

acquire_gpu() {
    for GPU in 1 2 3 4 5 6 7; do
        LOCK="$LOCK_DIR/gpu_${GPU}.lock"
        if ( set -o noclobber; echo "$$" > "$LOCK" ) 2>/dev/null; then
            UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU | tr -d ' ')
            MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
            if [ "$UTIL" -lt 10 ] && [ "$MEM" -lt 2000 ]; then
                echo "$GPU"; return 0
            fi
            rm -f "$LOCK"
        fi
    done
    return 1
}

release_gpu() { rm -f "$LOCK_DIR/gpu_${1}.lock"; }

for ITEM in "${QUEUE[@]}"; do
    IFS='|' read -r SCENARIO SEED DS TAG <<< "$ITEM"
    RUN_DIR="$PROJ/results/$TAG"
    LOG="$LOG_DIR/${TAG}.log"

    if [ -f "$RUN_DIR/result.csv" ]; then
        echo "[SKIP] $TAG"; continue
    fi

    while true; do
        GPU=$(acquire_gpu); [ -n "$GPU" ] && break; sleep 15
    done

    echo "[START] GPU $GPU — $TAG"

    if [ "$SCENARIO" = "A" ]; then
        CMD="python -u train.py --scenario A --seed $SEED --n_real $N_REAL --epochs 100 --train_dir $TRAIN_DIR --test_dir $TEST_DIR"
    else
        CMD="python -u train.py --scenario B --seed $SEED --n_real $N_REAL --n_synth $N_SYNTH --epochs 100 --train_dir $TRAIN_DIR --test_dir $TEST_DIR --synth_dir $SYNTH_BASE/$DS"
    fi

    (
        CUDA_VISIBLE_DEVICES=$GPU $CMD > "$LOG" 2>&1
        release_gpu $GPU
        echo "[DONE] GPU $GPU — $TAG"
    ) &

    echo "  PID: $!"
    sleep 5
done

echo "Todos os jobs lançados. Aguardando..."
wait
echo "=== TUDO CONCLUÍDO ==="
rm -rf "$LOCK_DIR"
