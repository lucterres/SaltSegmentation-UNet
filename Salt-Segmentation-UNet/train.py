"""train.py — Downstream segmentation experiment (R2.1 response).

Usage examples
--------------
# Scenario A — real data only, seed 42
conda run -n cv2 python -u train.py --scenario A --seed 42

# Scenario B — real + synthetic, seed 42
conda run -n cv2 python -u train.py --scenario B --seed 42

# Low-data regime: 50 real images only
python train.py --scenario A --seed 42 --n_real 50

# Low-data regime: 50 real + 200 synthetic
python train.py --scenario B --seed 42 --n_real 50 --n_synth 200
"""

import argparse
import os
import random
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import config
from utils.dataset import SegmentationDataset, get_transforms
from utils.model import UNet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(logits, targets, threshold=config.THRESHOLD):
    """Compute mean IoU and Dice over a batch. Returns (iou, dice) as floats."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou  = (intersection / (union + 1e-8)).mean().item()
    dice = (2 * intersection / (
        preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-8
    )).mean().item()
    return iou, dice


def coverage_class(mask_path: str) -> int:
    """Salt-coverage bucket (0–4) for stratified splitting."""
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='U-Net TGS Salt — downstream experiment')
    p.add_argument('--scenario', choices=['A', 'B'], required=True,
                   help='A = real only | B = real + synthetic')
    p.add_argument('--seed',    type=int,   default=42)
    p.add_argument('--n_real',  type=int,   default=None,
                   help='Limit real training images (low-data regime).')
    p.add_argument('--n_synth', type=int,   default=400,
                   help='Number of synthetic images for scenario B.')
    p.add_argument('--epochs',  type=int,   default=config.EPOCHS)
    p.add_argument('--batch',   type=int,   default=config.BATCH_SIZE)
    p.add_argument('--lr',      type=float, default=config.LR)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Output directory for this run
    run_tag = f'scenario_{args.scenario}_seed{args.seed}'
    if args.n_real is not None:
        run_tag += f'_nreal{args.n_real}'
    out_dir = os.path.join('..', 'results', run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f'[INFO] Run: {run_tag}  device={config.DEVICE}')

    # -----------------------------------------------------------------------
    # 1. Fixed test split (seed=0, shared across all runs)
    # -----------------------------------------------------------------------
    all_imgs  = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    all_masks = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    # Keep only matched pairs (some datasets have orphan masks without images)
    img_stem  = {os.path.splitext(os.path.basename(p))[0]: p for p in all_imgs}
    msk_stem  = {os.path.splitext(os.path.basename(p))[0]: p for p in all_masks}
    common    = sorted(img_stem.keys() & msk_stem.keys())
    all_imgs  = [img_stem[s] for s in common]
    all_masks = [msk_stem[s] for s in common]
    print(f'[INFO] Valid pairs: {len(all_imgs)} (orphans discarded)')

    strata = [coverage_class(m) for m in all_masks]

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        all_imgs, all_masks,
        test_size=config.TEST_SPLIT,
        random_state=0,      # FIXED — never change; ensures same test set for all runs
        stratify=strata,
    )

    os.makedirs(os.path.dirname(config.TEST_PATHS), exist_ok=True)
    if not os.path.exists(config.TEST_PATHS):
        with open(config.TEST_PATHS, 'w') as f:
            f.write('\n'.join(test_imgs))
    print(f'[INFO] Train pool: {len(train_imgs)} | Test: {len(test_imgs)}')

    # -----------------------------------------------------------------------
    # 2. Optional low-data regime
    # -----------------------------------------------------------------------
    if args.n_real is not None and args.n_real < len(train_imgs):
        tr_strata = [coverage_class(m) for m in train_masks]
        train_imgs, _, train_masks, _ = train_test_split(
            train_imgs, train_masks,
            train_size=args.n_real,
            random_state=args.seed,
            stratify=tr_strata,
        )
        print(f'[INFO] Low-data: {args.n_real} real images')

    # -----------------------------------------------------------------------
    # 3. Validation split (10% of training, stratified, per-seed)
    # -----------------------------------------------------------------------
    val_strata = [coverage_class(m) for m in train_masks]
    tr_imgs, val_imgs, tr_masks, val_masks = train_test_split(
        train_imgs, train_masks,
        test_size=config.VAL_SPLIT,
        random_state=args.seed,
        stratify=val_strata,
    )
    print(f'[INFO] After val split → train={len(tr_imgs)}, val={len(val_imgs)}')

    # -----------------------------------------------------------------------
    # 4. Datasets & dataloaders
    # -----------------------------------------------------------------------
    img_tf, mask_tf = get_transforms(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)

    train_dataset = SegmentationDataset(tr_imgs, tr_masks, img_tf, mask_tf, augment=True)

    if args.scenario == 'B':
        synth_imgs  = sorted(list(paths.list_images(config.SYNTH_IMAGE_PATH)))
        synth_masks = sorted(list(paths.list_images(config.SYNTH_MASK_PATH)))
        if args.n_synth < len(synth_imgs):
            synth_imgs  = synth_imgs[:args.n_synth]
            synth_masks = synth_masks[:args.n_synth]
        print(f'[INFO] Scenario B: adding {len(synth_imgs)} synthetic images')
        synth_dataset = SegmentationDataset(synth_imgs, synth_masks, img_tf, mask_tf)
        train_dataset = ConcatDataset([train_dataset, synth_dataset])

    val_dataset  = SegmentationDataset(val_imgs,  val_masks,  img_tf, mask_tf)
    test_dataset = SegmentationDataset(test_imgs, test_masks, img_tf, mask_tf)

    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, args.batch, shuffle=True,
                              pin_memory=config.PIN_MEMORY, num_workers=nw)
    val_loader   = DataLoader(val_dataset,  args.batch, shuffle=False,
                              pin_memory=config.PIN_MEMORY, num_workers=nw)
    test_loader  = DataLoader(test_dataset, args.batch, shuffle=False,
                              pin_memory=config.PIN_MEMORY, num_workers=nw)

    # -----------------------------------------------------------------------
    # 5. Model / optimizer / loss
    # -----------------------------------------------------------------------
    model     = UNet().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    loss_fn   = nn.BCEWithLogitsLoss()

    # -----------------------------------------------------------------------
    # 6. Training loop
    # -----------------------------------------------------------------------
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}
    best_iou = 0.0
    patience  = 0
    best_ckpt = os.path.join(out_dir, 'best_model.pth')

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # -- train --
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f'E{epoch}/{args.epochs} train', leave=False, ascii=True, ncols=80):
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
            loss = loss_fn(model(imgs), masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_dataset)

        # -- validate --
        model.eval()
        val_loss = iou_acc = dice_acc = nb = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
                logits = model(imgs)
                val_loss += loss_fn(logits, masks).item() * imgs.size(0)
                iou, dice = compute_metrics(logits, masks)
                iou_acc += iou; dice_acc += dice; nb += 1
        val_loss /= len(val_dataset)
        val_iou   = iou_acc  / nb
        val_dice  = dice_acc / nb

        scheduler.step(val_iou)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        flag = ''
        if val_iou > best_iou:
            best_iou = val_iou; patience = 0
            torch.save(model.state_dict(), best_ckpt)
            flag = '  ✓ best'
        else:
            patience += 1

        print(f'  E{epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} '
              f'IoU={val_iou:.4f} Dice={val_dice:.4f}{flag}')

        if patience >= config.EARLY_STOP_PATIENCE:
            print(f'  [Early stop] no improvement for {config.EARLY_STOP_PATIENCE} epochs')
            break

    elapsed = time.time() - t0
    print(f'[INFO] Training done in {elapsed:.0f}s')

    # -----------------------------------------------------------------------
    # 7. Final test evaluation (best checkpoint)
    # -----------------------------------------------------------------------
    model.load_state_dict(torch.load(best_ckpt, map_location=config.DEVICE, weights_only=True))
    model.eval()
    iou_acc = dice_acc = nb = 0.0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
            iou, dice = compute_metrics(model(imgs), masks)
            iou_acc += iou; dice_acc += dice; nb += 1
    test_iou  = iou_acc  / nb
    test_dice = dice_acc / nb

    print(f'\n[RESULT] scenario={args.scenario} seed={args.seed}')
    print(f'         Test IoU  = {test_iou:.4f}')
    print(f'         Test Dice = {test_dice:.4f}')

    # -----------------------------------------------------------------------
    # 8. Save artefacts
    # -----------------------------------------------------------------------
    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    pd.DataFrame([{
        'scenario': args.scenario, 'seed': args.seed,
        'n_real': args.n_real or (len(tr_imgs) + len(val_imgs)),
        'n_synth': args.n_synth if args.scenario == 'B' else 0,
        'best_val_iou': round(best_iou, 4),
        'test_iou':  round(test_iou,  4),
        'test_dice': round(test_dice, 4),
        'epochs_run': len(history['epoch']),
        'elapsed_s': round(elapsed, 1),
    }]).to_csv(os.path.join(out_dir, 'result.csv'), index=False)

    # Learning curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history['epoch'], history['train_loss'], label='train')
    ax[0].plot(history['epoch'], history['val_loss'],   label='val')
    ax[0].set_title('Loss'); ax[0].legend()
    ax[1].plot(history['epoch'], history['val_iou'],  label='IoU')
    ax[1].plot(history['epoch'], history['val_dice'], label='Dice')
    ax[1].set_title('Val Metrics'); ax[1].legend()
    fig.suptitle(f'Scenario {args.scenario} | Seed {args.seed}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curves.png'), dpi=100)
    plt.close()
    print(f'[INFO] Artefacts → {out_dir}')


if __name__ == '__main__':
    main()
