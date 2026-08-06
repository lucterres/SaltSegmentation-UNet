#!/bin/bash
# =============================================================================
# check_progress_6albu_seed42.sh
# =============================================================================
# Verifica o progresso das 6 runs do Cenário B (6 datasets albumentations,
# seed=42, N=3198+1200 sintéticos, test canônico 800).
#
# Uso (local ou no nó Atena):
#   bash check_progress_6albu_seed42.sh
# =============================================================================

PROJ="/nethome/atena_projetos/cym7/0code/SaltSegment-Unet"
RESULTS="$PROJ/results"

RUNS=(
    "scenario_B_seed42_train_clahe1600clean_ns1200"
    "scenario_B_seed42_train_elastic_transform1600clean_ns1200"
    "scenario_B_seed42_train_grid_distortion1600clean_ns1200"
    "scenario_B_seed42_train_optical_distortion1600clean_ns1200"
    "scenario_B_seed42_train_random_brightness_contrast1600clean_ns1200"
    "scenario_B_seed42_train_random_gamma1600clean_ns1200"
)

echo ""
echo "============================================================"
echo " Progresso — Cenário B | 6 Albumentations | seed=42"
echo " $(date)"
echo "============================================================"
printf "%-3s  %-52s  %-8s  %-10s  %-10s  %s\n" \
    "ID" "Run" "Status" "Val IoU" "Test IoU" "Época atual"
echo "----  $(printf '%.0s-' {1..52})  --------  ----------  ----------  ------------"

for i in "${!RUNS[@]}"; do
    RUN="${RUNS[$i]}"
    RUN_DIR="$RESULTS/$RUN"

    # Status
    if [ -f "$RUN_DIR/result.csv" ]; then
        STATUS="DONE ✓"
        TEST_IOU=$(tail -1 "$RUN_DIR/result.csv" | awk -F',' '{
            # Encontra coluna test_iou
            for(i=1;i<=NF;i++) if ($i~/test_iou/) col=i
        }END{print $col}' 2>/dev/null || echo "?")
        # Lê diretamente pelo header
        TEST_IOU=$(python3 -c "
import csv
with open('$RUN_DIR/result.csv') as f:
    r = list(csv.DictReader(f))
    print(r[-1].get('test_iou','?') if r else '?')
" 2>/dev/null || echo "?")
        VAL_IOU=$(python3 -c "
import csv
with open('$RUN_DIR/result.csv') as f:
    r = list(csv.DictReader(f))
    print(r[-1].get('best_val_iou','?') if r else '?')
" 2>/dev/null || echo "?")
        EPOCH="$(wc -l < "$RUN_DIR/history.csv" 2>/dev/null || echo "?") ep"
    elif [ -f "$RUN_DIR/history.csv" ]; then
        STATUS="RUNNING"
        TEST_IOU="-"
        # Última val IoU do history
        VAL_IOU=$(tail -1 "$RUN_DIR/history.csv" | awk -F',' 'NR==1{for(i=1;i<=NF;i++) if($i~/val_iou/) col=i} NR>1{print $col}' 2>/dev/null)
        if [ -z "$VAL_IOU" ]; then
            VAL_IOU=$(python3 -c "
import csv
with open('$RUN_DIR/history.csv') as f:
    rows = list(csv.DictReader(f))
    print(rows[-1].get('val_iou','?') if rows else '?')
" 2>/dev/null || echo "?")
        fi
        LINES=$(wc -l < "$RUN_DIR/history.csv" 2>/dev/null || echo 1)
        EPOCH="$((LINES - 1)) ep"
    elif [ -d "$RUN_DIR" ]; then
        STATUS="STARTED"
        VAL_IOU="-"
        TEST_IOU="-"
        EPOCH="-"
    else
        STATUS="PENDING"
        VAL_IOU="-"
        TEST_IOU="-"
        EPOCH="-"
    fi

    printf "%-3s  %-52s  %-8s  %-10s  %-10s  %s\n" \
        "$i" "$RUN" "$STATUS" "$VAL_IOU" "$TEST_IOU" "$EPOCH"
done

echo ""

# ---------------------------------------------------------------------------
# Resumo de jobs SLURM ativos (se disponível)
# ---------------------------------------------------------------------------
if command -v squeue &>/dev/null; then
    echo "--- Jobs SLURM ativos (cenB_6albu_s42) ---"
    squeue -u "$USER" --name=cenB_6albu_s42 \
        -o "%-8i %-12j %-6T %-8M %-5D %R" 2>/dev/null || echo "  (squeue não disponível)"
    echo ""
fi

# ---------------------------------------------------------------------------
# Resumo de IoUs das runs concluídas
# ---------------------------------------------------------------------------
DONE_COUNT=0
SUM_TEST_IOU=0

for RUN in "${RUNS[@]}"; do
    RUN_DIR="$RESULTS/$RUN"
    if [ -f "$RUN_DIR/result.csv" ]; then
        DONE_COUNT=$((DONE_COUNT + 1))
        IOU=$(python3 -c "
import csv
with open('$RUN_DIR/result.csv') as f:
    r = list(csv.DictReader(f))
    v = r[-1].get('test_iou','0') if r else '0'
    print(v)
" 2>/dev/null || echo "0")
        SUM_TEST_IOU=$(python3 -c "print(round($SUM_TEST_IOU + float('$IOU'), 4))" 2>/dev/null || echo "$SUM_TEST_IOU")
    fi
done

echo "--- Resumo ---"
echo "  Concluídas : $DONE_COUNT / ${#RUNS[@]}"
if [ "$DONE_COUNT" -gt 0 ]; then
    AVG=$(python3 -c "print(round($SUM_TEST_IOU / $DONE_COUNT, 4))" 2>/dev/null || echo "?")
    echo "  Test IoU médio (concluídas): $AVG"
fi
echo ""
