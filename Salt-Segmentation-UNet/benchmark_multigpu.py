"""benchmark_multigpu.py — Compara throughput (it/s) com 1 GPU vs 2 GPUs (DataParallel).

Executa N épocas curtas e mede iterações por segundo no loop de treino.
Não salva checkpoints nem faz avaliação — objetivo é só medir throughput.

Usage
-----
conda run -n cv2 python -u benchmark_multigpu.py
"""

import os
import time

import numpy as np
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from utils import config
from utils.dataset import SegmentationDataset, get_transforms
from utils.model import UNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BENCH_EPOCHS   = 3        # épocas por rodada
BENCH_STEPS    = 50       # max batches por época (None = todos)
BATCH_SIZE     = 32       # batch maior aproveita melhor multi-GPU
NUM_WORKERS    = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def coverage_class(mask_path: str) -> int:
    import cv2 as cv
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    cov = (mask > 127).mean()
    if cov == 0:       return 0
    elif cov < 0.10:   return 1
    elif cov < 0.30:   return 2
    elif cov < 0.50:   return 3
    else:              return 4


def build_loader(batch_size: int, num_workers: int) -> DataLoader:
    all_imgs  = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    all_masks = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    img_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in all_imgs}
    msk_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in all_masks}
    common   = sorted(img_stem.keys() & msk_stem.keys())
    all_imgs  = [img_stem[s] for s in common]
    all_masks = [msk_stem[s] for s in common]

    strata = [coverage_class(m) for m in all_masks]
    tr_imgs, _, tr_masks, _ = train_test_split(
        all_imgs, all_masks,
        test_size=0.20, random_state=0, stratify=strata,
    )

    img_tf, mask_tf = get_transforms(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
    ds = SegmentationDataset(tr_imgs, tr_masks, img_tf, mask_tf, augment=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers, drop_last=True,
    )
    print(f'  Dataset: {len(ds)} imagens | batches/epoch: {len(loader)}')
    return loader


def run_benchmark(model: nn.Module, loader: DataLoader,
                  device: str, n_epochs: int, max_steps) -> dict:
    """Executa loop de treino e retorna métricas de throughput."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn   = nn.BCEWithLogitsLoss()
    model.train()

    epoch_its  = []  # it/s por época

    for ep in range(1, n_epochs + 1):
        t_ep  = time.perf_counter()
        steps = 0
        imgs_seen = 0

        for imgs, masks in loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = loss_fn(out, masks)
            loss.backward()
            optimizer.step()

            steps     += 1
            imgs_seen += imgs.size(0)
            if max_steps and steps >= max_steps:
                break

        elapsed = time.perf_counter() - t_ep
        its = steps / elapsed
        epoch_its.append(its)
        print(f'    epoch {ep:02d}: {steps} steps | {elapsed:.1f}s | {its:.2f} it/s '
              f'| {imgs_seen/elapsed:.1f} img/s')

    return {
        'mean_it_s':  float(np.mean(epoch_its)),
        'std_it_s':   float(np.std(epoch_its)),
        'epochs':     n_epochs,
        'batch_size': loader.batch_size,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_gpus = torch.cuda.device_count()
    print(f'\n{"="*60}')
    print(f'  BENCHMARK MULTI-GPU  —  GPUs disponíveis: {n_gpus}')
    for i in range(n_gpus):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Batch size: {BATCH_SIZE}  |  Épocas: {BENCH_EPOCHS}  |  Max steps: {BENCH_STEPS}')
    print(f'{"="*60}\n')

    if n_gpus < 1:
        print('[ERRO] Nenhuma GPU encontrada. Abortando.')
        return

    loader = build_loader(BATCH_SIZE, NUM_WORKERS)
    results = {}

    # ------------------------------------------------------------------
    # Rodada 1: 1 GPU (cuda:0)
    # ------------------------------------------------------------------
    print('\n--- Rodada 1: 1 GPU (cuda:0) ---')
    model_1gpu = UNet()
    res_1 = run_benchmark(model_1gpu, loader, device='cuda:0',
                          n_epochs=BENCH_EPOCHS, max_steps=BENCH_STEPS)
    results['1 GPU'] = res_1
    del model_1gpu
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Rodada 2: 2 GPUs (DataParallel)
    # ------------------------------------------------------------------
    if n_gpus >= 2:
        print('\n--- Rodada 2: 2 GPUs (DataParallel) ---')
        model_2gpu = nn.DataParallel(UNet(), device_ids=[0, 1])
        res_2 = run_benchmark(model_2gpu, loader, device='cuda:0',
                              n_epochs=BENCH_EPOCHS, max_steps=BENCH_STEPS)
        results['2 GPUs (DataParallel)'] = res_2
        del model_2gpu
        torch.cuda.empty_cache()
    else:
        print(f'\n[AVISO] Apenas {n_gpus} GPU disponível — pulando rodada com 2 GPUs.')

    # ------------------------------------------------------------------
    # Resumo comparativo
    # ------------------------------------------------------------------
    print(f'\n{"="*60}')
    print('  RESUMO COMPARATIVO')
    print(f'{"="*60}')
    print(f'  {"Configuração":<30} {"it/s (média)":>12} {"±std":>8} {"speedup":>9}')
    print(f'  {"-"*60}')

    base_its = None
    for label, r in results.items():
        if base_its is None:
            base_its = r['mean_it_s']
            speedup_str = '   1.00x'
        else:
            speedup = r['mean_it_s'] / base_its
            speedup_str = f'  {speedup:.2f}x'
        print(f'  {label:<30} {r["mean_it_s"]:>12.2f} {r["std_it_s"]:>8.2f} {speedup_str}')

    print(f'{"="*60}\n')

    # Salva CSV com resultados
    import csv
    out_path = os.path.join('..', 'results', 'benchmark_multigpu.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'mean_it_s', 'std_it_s',
                                                'speedup', 'batch_size', 'epochs',
                                                'max_steps', 'n_gpus_available'])
        writer.writeheader()
        base_its = list(results.values())[0]['mean_it_s']
        for label, r in results.items():
            writer.writerow({
                'config':           label,
                'mean_it_s':        round(r['mean_it_s'],  3),
                'std_it_s':         round(r['std_it_s'],   3),
                'speedup':          round(r['mean_it_s'] / base_its, 4),
                'batch_size':       r['batch_size'],
                'epochs':           r['epochs'],
                'max_steps':        BENCH_STEPS,
                'n_gpus_available': n_gpus,
            })
    print(f'[INFO] Resultado salvo em: {out_path}')


if __name__ == '__main__':
    main()
