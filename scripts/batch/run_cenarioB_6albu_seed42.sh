#!/bin/bash
# =============================================================================
# run_cenarioB_6albu_seed42.sh
# =============================================================================
# Cenário B — real (N=3198, TGS completo) + 1200 sintéticos albumentations
# 6 datasets albumentations clean (sem data leakage), seed=42
# Test canônico: /var/tmp/cym7/datasets/subset_split/test (800 amostras)
# Uma GPU por run via SLURM array (ARRAY_ID 0–5 → 6 jobs)
#
# Uso (no nó de login do Atena):
#   sbatch run_cenarioB_6albu_seed42.sh
#
# Ou para submeter job por job manualmente:
#   for i in $(seq 0 5); do
#     SLURM_ARRAY_TASK_ID=$i bash run_cenarioB_6albu_seed42.sh
#   done
# =============================================================================

#SBATCH --job-name=cenB_6albu_s42
#SBATCH --account=pn-dscien
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/nethome/atena_projetos/cym7/0code/SaltSegment-Unet/results/slurm_logs/cenB_6albu_s42_%A_%a.out
#SBATCH --error=/nethome/atena_projetos/cym7/0code/SaltSegment-Unet/results/slurm_logs/cenB_6albu_s42_%A_%a.err

set -e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ="/nethome/atena_projetos/cym7/0code/SaltSegment-Unet"
VENV="/var/tmp/cym7/venvs/salt-unet"
TRAIN_DIR="/var/tmp/cym7/datasets/tgs-salt/train"    # N=3198 (TGS completo)
TEST_DIR="/var/tmp/cym7/datasets/subset_split/test"  # test canônico 800

# Datasets sintéticos (albumentations, clean = sem leakage do test set canônico)
SYNTH_DATASETS=(
    "clahe1600clean"
    "elastic_transform1600clean"
    "grid_distortion1600clean"
    "optical_distortion1600clean"
    "random_brightness_contrast1600clean"
    "random_gamma1600clean"
)

SYNTH_BASE="$PROJ/Salt-Segmentation-UNet/dataset"

# ---------------------------------------------------------------------------
# Seleciona dataset sintético pelo SLURM_ARRAY_TASK_ID
# ---------------------------------------------------------------------------
SYNTH_NAME="${SYNTH_DATASETS[$SLURM_ARRAY_TASK_ID]}"
SYNTH_DIR="$SYNTH_BASE/$SYNTH_NAME"

echo "============================================================"
echo " Job array ID : $SLURM_ARRAY_TASK_ID"
echo " Dataset synth: $SYNTH_NAME"
echo " Node         : $(hostname)"
echo " GPU          : $(echo $CUDA_VISIBLE_DEVICES)"
echo " Data         : $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Verifica e restaura ambiente no nó (idempotente)
# ---------------------------------------------------------------------------
bash "$PROJ/setup_node_Atena.sh"

# Verifica datasets sintéticos no SSD local ou usa path do NFS
if [ ! -d "$SYNTH_DIR/images" ]; then
    echo "[WARN] Sintético não encontrado em SSD local: $SYNTH_DIR"
    echo "       Verificando NFS path direto..."
    if [ ! -d "$SYNTH_DIR/images" ]; then
        echo "[ERROR] Dataset sintético ausente: $SYNTH_DIR"
        exit 1
    fi
fi

echo "[INFO] Sintético OK: $(ls $SYNTH_DIR/images | wc -l) imagens"

# ---------------------------------------------------------------------------
# Ativa venv e entra no diretório do projeto
# ---------------------------------------------------------------------------
source "$VENV/bin/activate"
cd "$PROJ/Salt-Segmentation-UNet"

# Verifica GPU
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ---------------------------------------------------------------------------
# Cria diretório de logs SLURM
# ---------------------------------------------------------------------------
mkdir -p "$PROJ/results/slurm_logs"

# ---------------------------------------------------------------------------
# Executa treinamento
# ---------------------------------------------------------------------------
# run_tag gerado pelo train.py:
#   scenario_B_seed42_train_<SYNTH_NAME>_ns1200
#   ex: scenario_B_seed42_train_clahe1600clean_ns1200
#
# Diretório de saída:
#   $PROJ/results/scenario_B_seed42_train_<SYNTH_NAME>_ns1200/
# ---------------------------------------------------------------------------

EXPECTED_RUN="$PROJ/results/scenario_B_seed42_train_${SYNTH_NAME}_ns1200"
if [ -f "$EXPECTED_RUN/result.csv" ]; then
    echo "[SKIP] Resultado já existe: $EXPECTED_RUN/result.csv"
    echo "       Delete o diretório para re-executar."
    exit 0
fi

echo ""
echo "[INFO] Iniciando treino — Cenário B | dataset=$SYNTH_NAME | seed=42 | n_real=3198 | n_synth=1200"
echo ""

python -u train.py \
    --scenario B \
    --seed 42 \
    --n_synth 1200 \
    --epochs 100 \
    --train_dir "$TRAIN_DIR" \
    --test_dir  "$TEST_DIR" \
    --synth_dir "$SYNTH_DIR"

echo ""
echo "[DONE] Job $SLURM_ARRAY_TASK_ID ($SYNTH_NAME) concluído em $(date)"
