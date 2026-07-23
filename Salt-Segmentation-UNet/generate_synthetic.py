"""generate_synthetic.py — Generate synthetic TGS images via the paper's VAE + texture pipeline.

This script is meant to be run from the WORKSPACE ROOT (d:/IEEEEAccess), where
the VAE model checkpoint and texture synthesis code live.

Usage
-----
cd D:\\IEEEEAccess
python SaltSegmentation-UNet/Salt-Segmentation-UNet/generate_synthetic.py \
    --n 400 \
    --out SaltSegmentation-UNet/Salt-Segmentation-UNet/dataset/synthetic \
    --vae_ckpt <path_to_vae_checkpoint.pth> \
    --tgs_dir  <path_to_tgs_dataset_root>

Outputs
-------
<out>/images/synth_0000.png  ...  synth_0399.png   (101×101 grayscale)
<out>/masks/synth_0000.png   ...  synth_0399.png   (101×101 binary)
"""

import argparse
import os
import sys

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Generate synthetic TGS images')
    p.add_argument('--n',        type=int, default=400,
                   help='Number of synthetic images to generate')
    p.add_argument('--out',      type=str,
                   default='SaltSegmentation-UNet/Salt-Segmentation-UNet/dataset/synthetic',
                   help='Output directory (images/ and masks/ created inside)')
    p.add_argument('--vae_ckpt', type=str, default=None,
                   help='Path to the VAE model checkpoint (.pth). '
                        'If None, will search for a default checkpoint.')
    p.add_argument('--tgs_dir',  type=str,
                   default='SaltSegmentation-UNet/Salt-Segmentation-UNet/dataset/tgs',
                   help='Root directory of the TGS dataset (for texture patches)')
    p.add_argument('--latent_dim', type=int, default=100,
                   help='VAE latent dimension (must match training, default=100)')
    p.add_argument('--seed',     type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# VAE decoder (mirror of Sec. III-A in the paper)
# ---------------------------------------------------------------------------

class VAEDecoder(torch.nn.Module):
    """MLP decoder: latent_dim → 256 → 512 → 1024 → 4096 → reshape 64×64 → sigmoid."""
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),  torch.nn.ReLU(),
            torch.nn.Linear(256, 512),         torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),        torch.nn.ReLU(),
            torch.nn.Linear(1024, 4096),       torch.nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.net(z)                      # (B, 4096)
        return out.view(-1, 1, 64, 64)        # (B, 1, 64, 64)


def load_vae_decoder(ckpt_path: str, latent_dim: int, device: str) -> VAEDecoder:
    """Load the decoder weights from a full VAE checkpoint or a decoder-only file."""
    decoder = VAEDecoder(latent_dim).to(device)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print('[WARN] VAE checkpoint not found — using random weights (for testing only).')
        print('       Set --vae_ckpt to a real checkpoint to generate realistic masks.')
        return decoder
    state = torch.load(ckpt_path, map_location=device)
    # Handle full-VAE checkpoints where decoder weights are under 'decoder.*'
    if any(k.startswith('decoder.') for k in state.keys()):
        state = {k[len('decoder.'):]: v for k, v in state.items()
                 if k.startswith('decoder.')}
    decoder.load_state_dict(state, strict=False)
    print(f'[INFO] VAE decoder loaded from {ckpt_path}')
    return decoder


# ---------------------------------------------------------------------------
# Simple texture synthesis placeholder
# ---------------------------------------------------------------------------

def synthesize_texture(mask_64: np.ndarray, tgs_dir: str,
                       rng: np.random.Generator) -> np.ndarray:
    """Apply zone-specific texture synthesis (simplified version for experiment).

    In the full pipeline (Sec. III-B), this calls the non-parametric texture
    synthesis algorithm with the patch database.  Here we use a fast approximation:
    sample a random real TGS image and transplant textures by zone.

    Returns a 101×101 uint8 grayscale image.
    """
    import glob
    real_imgs = glob.glob(os.path.join(tgs_dir, 'images', '*.png'))
    if not real_imgs:
        # Fallback: return noisy mask (for testing without dataset)
        noise = rng.integers(80, 180, (101, 101), dtype=np.uint8)
        salt_mask = cv.resize((mask_64 > 0.5).astype(np.uint8) * 255,
                              (101, 101), interpolation=cv.INTER_NEAREST)
        synth = noise.copy()
        synth[salt_mask > 0] = rng.integers(200, 255,
                                            synth[salt_mask > 0].shape,
                                            dtype=np.uint8)
        return synth

    # Sample a real image as texture donor
    donor_path = rng.choice(real_imgs)
    donor = cv.imread(donor_path, cv.IMREAD_GRAYSCALE)   # 101×101

    # Upscale VAE mask from 64×64 → 101×101
    mask_101 = cv.resize((mask_64 > 0.5).astype(np.uint8) * 255,
                         (101, 101), interpolation=cv.INTER_NEAREST)

    # Zone-specific blending (approximation of paper's Sec. III-B)
    synth = donor.copy()

    # Salt zone: brighten texture
    salt_zone = mask_101 > 0
    if salt_zone.any():
        salt_tex = donor[salt_zone]
        salt_tex = np.clip(salt_tex.astype(np.int32) + rng.integers(10, 40), 0, 255)
        synth[salt_zone] = salt_tex.astype(np.uint8)

    # Boundary zone: slight blur to simulate transition
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv.dilate(mask_101, kernel, iterations=1) - mask_101
    if boundary.any():
        blurred = cv.GaussianBlur(synth, (3, 3), 0)
        synth[boundary > 0] = blurred[boundary > 0]

    return synth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_img_dir  = os.path.join(args.out, 'images')
    out_mask_dir = os.path.join(args.out, 'masks')
    os.makedirs(out_img_dir,  exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    decoder = load_vae_decoder(args.vae_ckpt, args.latent_dim, device)
    decoder.eval()

    print(f'[INFO] Generating {args.n} synthetic images → {args.out}')
    generated = 0
    attempts  = 0
    max_attempts = args.n * 3   # allow up to 3× attempts for filtering

    with torch.no_grad():
        pbar = tqdm(total=args.n, desc='Generating')
        while generated < args.n and attempts < max_attempts:
            attempts += 1
            # Sample latent vector
            z = torch.randn(1, args.latent_dim, device=device)
            mask_tensor = decoder(z)                     # (1,1,64,64) in [0,1]
            mask_np = mask_tensor.squeeze().cpu().numpy()  # (64,64)

            # Filter: skip masks with < 5% or > 70% salt coverage
            coverage = (mask_np > 0.5).mean()
            if coverage < 0.05 or coverage > 0.70:
                continue

            # Synthesize seismic texture
            synth_img = synthesize_texture(mask_np, args.tgs_dir, rng)

            # Binary mask at 101×101
            mask_101 = cv.resize((mask_np > 0.5).astype(np.uint8) * 255,
                                 (101, 101), interpolation=cv.INTER_NEAREST)

            # Save
            fname = f'synth_{generated:04d}.png'
            cv.imwrite(os.path.join(out_img_dir,  fname), synth_img)
            cv.imwrite(os.path.join(out_mask_dir, fname), mask_101)
            generated += 1
            pbar.update(1)
        pbar.close()

    if generated < args.n:
        print(f'[WARN] Only generated {generated}/{args.n} images '
              f'(coverage filter rejected too many). '
              f'Consider widening coverage bounds.')
    else:
        print(f'[INFO] Done. {generated} image/mask pairs saved to {args.out}')


if __name__ == '__main__':
    main()
