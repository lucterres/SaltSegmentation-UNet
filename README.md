# Experimento Downstream — R2.1

**Manuscrito:** Access-2026-27912  
**Última atualização:** 2026-08-06  
**Objetivo:** Demonstrar que treinar com dados reais + sintéticos (Albumentations) melhora a segmentação de salt domes em dados sísmicos reais de teste.

---

## Resultados principais (test canônico 800 amostras, sem data leakage)

### N_real = 3998 | N_synth = 1200 | 3 seeds (42, 123, 456)

| Cenário | Dataset sintético | IoU médio | Δ vs A |
|---------|-----------------|:---------:|:------:|
| **A** (baseline) | — | 0.4734 | — |
| **B** ✅ | `random_gamma1600clean` | **0.4870** | **+0.014** |
| **B** | `grid_distortion1600clean` | 0.4774 | +0.004 |
| **B** | outros 4 datasets | < 0.4734 | negativo |

### N_real = 1200 | N_synth = 1200 | 3 seeds (low-data regime)

| Cenário | Dataset sintético | IoU médio | Δ vs A |
|---------|-----------------|:---------:|:------:|
| **A** (baseline) | — | 0.4081 | — |
| **B** ✅ | `elastic_transform1600clean` | **0.4276** | **+0.020** |
| **B** | todos os 6 datasets | > 0.4081 | positivo ✅ |

> **Conclusão R2.1:** com escassez de dados (N=1200), **todos** os 6 métodos de augmentation sintética superam o baseline. Com N=3998, apenas `random_gamma` e `grid_distortion` superam, mas a melhoria é consistente (seed 456 + `random_gamma` → IoU = **0.5015**).

---

## Infraestrutura

| Item | Detalhe |
|------|---------|
| **Cluster** | Atena (Petrobras) — SLURM |
| **Nó atual** | `atn2b03n03` (qualquer `atn2bXXnYY` alocado) |
| **GPUs** | 8 × Tesla V100-SXM2-32GB |
| **CUDA** | 12.4 (PyTorch 2.4.1+cu124) |
| **Python** | 3.8.16 |
| **venv (SSD local)** | `/var/tmp/cym7/venvs/salt-unet/` |
| **venv (backup tar)** | `/nethome/atena_projetos/cym7/envs/salt-unet-venv.tar` |
| **Código** | `/nethome/atena_projetos/cym7/0code/SaltSegment-Unet/` |
| **Dataset TGS** | `/var/tmp/cym7/datasets/tgs-salt/train/` (3998 pares) |
| **Test canônico** | `/var/tmp/cym7/datasets/subset_split/test/` (800 amostras fixas) |

---

## Estrutura do repositório

```
SaltSegmentation-UNet/
├── QUICKSTART.md                    ← guia rápido de execução
├── scripts/
│   ├── README.md
│   ├── infra/       ← setup_node_Atena.sh | check_node_Atena.sh
│   ├── batch/       ← launchers com lockfile (anti-duplicata)
│   └── monitoring/  ← check_progress | kill
├── Salt-Segmentation-UNet/
│   ├── train.py                     ← loop principal (--scenario, --seed, --train_dir, --test_dir, --synth_dir)
│   ├── utils/config.py              ← TGS_PATH, ENCODER_CHANNELS=(1,16,32,64)
│   ├── utils/model.py               ← U-Net (padding=1)
│   ├── utils/dataset.py             ← DataLoader (interpolação NEAREST p/ máscaras)
│   └── DataAugmentation/
│       └── generate_albumentations_da.py  ← gera 1600 pares por método
├── results/
│   ├── scenario_A_seed*_train/      ← Cenário A (N=3998, test canônico)
│   ├── scenario_B_seed*_train_*_ns1200/  ← Cenário B (N=3998+1200 sintéticos)
│   ├── scenario_A_seed*_nreal1200_train/ ← Cenário A (N=1200)
│   ├── scenario_B_seed*_nreal1200_*/     ← Cenário B (N=1200+1200)
│   └── slurm_logs/{launchers,nreal3998,nreal1200}/
└── docs/
    ├── relatorio-cenarioB-albumentations-3seeds.md  ← relatório completo 2026-08-06
    └── relatorio-final-r21-downstream.md
```

---

## Protocolo canônico (sem data leakage)

```bash
export PROJ=/nethome/atena_projetos/cym7/0code/SaltSegment-Unet
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd $PROJ/Salt-Segmentation-UNet

# Exemplo: Cenário B, seed 42, N=3998+1200, random_gamma
CUDA_VISIBLE_DEVICES=5 nohup python -u train.py \
  --scenario B --seed 42 --n_synth 1200 --epochs 100 \
  --train_dir /var/tmp/cym7/datasets/tgs-salt/train \
  --test_dir  /var/tmp/cym7/datasets/subset_split/test \
  --synth_dir $PROJ/Salt-Segmentation-UNet/dataset/random_gamma1600clean \
  > $PROJ/results/slurm_logs/launchers/meu_run.log 2>&1 &

# Batch completo (A + B × 6 datasets × 3 seeds, N=1200):
nohup bash $PROJ/scripts/batch/run_batch_nreal1200.sh \
  > $PROJ/results/slurm_logs/launchers/batch_nreal1200.log 2>&1 &
```

---

## Execução local (Windows — sem GPU dedicada)

```powershell
$env:TGS_PATH = "D:\dataset\tgs-salt\train"
cd "D:\0Code\_phdSeismic\Segmentation-Unet-Experiment\Salt-Segmentation-UNet"
conda run -n unet-salt python -u train.py --scenario A --seed 42 --n_real 200 --epochs 10
```
