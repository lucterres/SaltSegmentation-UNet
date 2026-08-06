#!/bin/bash
# =============================================================================
# kill_cenarioB_6albu_seed42.sh
# =============================================================================
# Cancela todos os processos train.py do Cenário B seed=42 rodando no nó.
# Uso: bash $PROJ/kill_cenarioB_6albu_seed42.sh
# =============================================================================

echo "============================================================"
echo " Cancelando runs — Cenário B | 6 Albumentations | seed=42"
echo " Node : $(hostname)"
echo " Data : $(date)"
echo "============================================================"
echo ""

# Encontra PIDs de train.py com seed 42 e cenarioB
PIDS=$(pgrep -f "train.py.*--seed 42" 2>/dev/null || true)

if [ -z "$PIDS" ]; then
    echo "  Nenhum processo train.py --seed 42 encontrado."
else
    echo "  PIDs encontrados:"
    ps -o pid,stat,etime,args -p $PIDS 2>/dev/null || echo "  (detalhes indisponíveis)"
    echo ""
    echo "  Enviando SIGTERM..."
    kill $PIDS 2>/dev/null && echo "  OK — processos terminados." || echo "  [WARN] Alguns PIDs já haviam encerrado."
fi

# Garante que nenhum processo python com synth_dir ainda esteja ativo
PIDS2=$(pgrep -f "train_dir.*tgs-salt" 2>/dev/null || true)
if [ -n "$PIDS2" ]; then
    echo ""
    echo "  PIDs adicionais com train_dir tgs-salt:"
    ps -o pid,stat,etime,args -p $PIDS2 2>/dev/null
    kill $PIDS2 2>/dev/null && echo "  OK — terminados." || true
fi

echo ""
echo "  Verificação final (deve estar vazio):"
pgrep -a -f "train.py.*seed" 2>/dev/null || echo "  Nenhum processo train.py ativo."
echo ""
echo "============================================================"
echo " GPUs após cancelamento:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv
echo "============================================================"
