#!/bin/bash
# Lança Cenário B seeds 123 e 456 × 6 datasets, 1 por GPU, sem duplicatas
# Usa lockfile por GPU para garantir exclusividade

PROJ="/nethome/atena_projetos/cym7/0code/SaltSegment-Unet"
VENV="/var/tmp/cym7/venvs/salt-unet"
TRAIN_DIR="/var/tmp/cym7/datasets/tgs-salt/train"
TEST_DIR="/var/tmp/cym7/datasets/subset_split/test"
SYNTH_BASE="$PROJ/Salt-Segmentation-UNet/dataset"
LOG_DIR="$PROJ/results/slurm_logs"
LOCK_DIR="/tmp/gpu_locks_$$"
N_SYNTH=1200

DATASETS=(clahe1600clean elastic_transform1600clean grid_distortion1600clean optical_distortion1600clean random_brightness_contrast1600clean random_gamma1600clean)

# Fila: B seed123 × 6 + B seed456 × 6
QUEUE=()
for SEED in 123 456; do
    for DS in "${DATASETS[@]}"; do
        QUEUE+=("B|${SEED}|${DS}|scenario_B_seed${SEED}_train_${DS}_ns${N_SYNTH}")
    done
done

mkdir -p "$LOG_DIR" "$LOCK_DIR"
source "$VENV/bin/activate"
cd "$PROJ/Salt-Segmentation-UNet"

echo "Fila: ${#QUEUE[@]} runs"

acquire_gpu() {
    # Retorna ID da primeira GPU livre (util<10%, mem<2000), obtendo lock exclusivo
    for GPU in 1 2 3 4 5 6 7; do
        LOCK="$LOCK_DIR/gpu_${GPU}.lock"
        # Tenta criar lockfile atomicamente
        if ( set -o noclobber; echo "$$" > "$LOCK" ) 2>/dev/null; then
            UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU | tr -d ' ')
            MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
            if [ "$UTIL" -lt 10 ] && [ "$MEM" -lt 2000 ]; then
                echo "$GPU"
                return 0
            fi
            rm -f "$LOCK"
        fi
    done
    return 1
}

release_gpu() {
    rm -f "$LOCK_DIR/gpu_${1}.lock"
}

for ITEM in "${QUEUE[@]}"; do
    IFS='|' read -r SCENARIO SEED DS TAG <<< "$ITEM"
    RUN_DIR="$PROJ/results/$TAG"
    LOG="$LOG_DIR/${TAG}.log"

    if [ -f "$RUN_DIR/result.csv" ]; then
        echo "[SKIP] $TAG"
        continue
    fi

    # Aguarda GPU livre
    while true; do
        GPU=$(acquire_gpu)
        [ -n "$GPU" ] && break
        sleep 15
    done

    echo "[START] GPU $GPU — $TAG"
    (
        CUDA_VISIBLE_DEVICES=$GPU python -u train.py \
            --scenario B --seed $SEED --n_synth $N_SYNTH \
            --epochs 100 \
            --train_dir "$TRAIN_DIR" \
            --test_dir  "$TEST_DIR" \
            --synth_dir "$SYNTH_BASE/$DS" \
            > "$LOG" 2>&1
        release_gpu $GPU
        echo "[DONE] GPU $GPU — $TAG"
    ) &

    echo "  PID: $!"
    sleep 5  # espera GPU aparecer no nvidia-smi antes do próximo acquire
done

echo "Todos os jobs lançados. Aguardando conclusão..."
wait
echo "=== TUDO CONCLUÍDO ==="
rm -rf "$LOCK_DIR"
