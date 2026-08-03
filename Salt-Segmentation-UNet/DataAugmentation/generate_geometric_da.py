"""
generate_geometric_da.py
=========================
Gera 1 600 pares (imagem + máscara) usando transformações geométricas
fisicamente coerentes para dados sísmicos — o método "Ours" do artigo.

Este script é o equivalente geométrico ao generate_albumentations_da.py,
mantendo paridade de quantidade (1 600 amostras) e estrutura de saída
para comparação justa na Seção 6.3.

Transformações aplicadas (conjunto "Ours")
------------------------------------------
✅  Flip horizontal          (p=0.5) — simetria lateral aceitável
✅  Rotação leve             ±10°    — pequenas inclinações estruturais
✅  Translação suave         ±5 px   — shift espacial realista
✅  Escala discreta          0.90×–1.10× — variação sutil de escala
✅  Ruído Gaussiano          σ ~ U[5,20] — somente na imagem

❌  Flip vertical            — altera polaridade física
❌  Deformação elástica fort — apaga falhas e horizontes
❌  Shear / rotação > ±10°   — distorção não-física

Saída
-----
    dataset/geometric1600/images/   ← 1600 PNGs de imagem
    dataset/geometric1600/masks/    ← 1600 PNGs de máscara
    dataset/geometric1600/log.csv

Uso
---
    python DataAugmentation/generate_geometric_da.py
    python DataAugmentation/generate_geometric_da.py --src /path/to/tgs/train
    python DataAugmentation/generate_geometric_da.py --n 1600

Seed: 42 (reprodutível)
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"

DEFAULT_SRC = os.environ.get("TGS_PATH", "/var/tmp/cym7/datasets/tgs-salt/train")

SEED      = 42
N_SAMPLES = 1600
OUT_NAME  = "geometric1600"

# Limites das transformações
ROT_MAX_DEG   = 10.0    # graus
TRANS_MAX_PX  = 5       # pixels (±)
SCALE_MIN     = 0.90
SCALE_MAX     = 1.10
NOISE_SIG_MIN = 5.0
NOISE_SIG_MAX = 20.0
FLIP_H_PROB   = 0.5


# ---------------------------------------------------------------------------
# Helpers de carregamento / salvamento
# ---------------------------------------------------------------------------

def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler: {path}")
    return img


def load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((101, 101), dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((101, 101), dtype=np.uint8)
    return mask


def save_png(arr: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), arr)


def collect_pairs(src_dir: Path):
    img_dir  = src_dir / "images"
    mask_dir = src_dir / "masks"
    if not img_dir.exists():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {img_dir}")
    pairs = []
    for img_path in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / f"{img_path.stem}.png"
        pairs.append((img_path, mask_path))
    if not pairs:
        raise RuntimeError(f"Nenhuma imagem em {img_dir}")
    return pairs


# ---------------------------------------------------------------------------
# Núcleo das transformações geométricas
# ---------------------------------------------------------------------------

def apply_geometric(
    img: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Aplica combinação aleatória das transformações físicas.
    Retorna (img_aug, mask_aug, params_dict).
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    params: dict = {}

    # ── Flip horizontal ──────────────────────────────────────────────────
    if rng.random() < FLIP_H_PROB:
        img  = cv2.flip(img,  1)
        mask = cv2.flip(mask, 1)
        params["flip_h"] = True

    # ── Rotação + Translação + Escala (affine única) ──────────────────────
    angle  = rng.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
    tx     = rng.randint(-TRANS_MAX_PX, TRANS_MAX_PX)
    ty     = rng.randint(-TRANS_MAX_PX, TRANS_MAX_PX)
    scale  = rng.uniform(SCALE_MIN, SCALE_MAX)

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    img  = cv2.warpAffine(img,  M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REFLECT_101)

    params.update({"rot": round(angle, 1), "tx": tx, "ty": ty,
                   "scale": round(scale, 3)})

    # ── Ruído Gaussiano (somente imagem) ─────────────────────────────────
    if rng.random() < 0.7:          # 70% das amostras recebem ruído
        sigma = rng.uniform(NOISE_SIG_MIN, NOISE_SIG_MAX)
        noise = np_rng.normal(0, sigma, img.shape).astype(np.float32)
        img_f = img.astype(np.float32) + noise
        img   = np.clip(img_f, 0, 255).astype(np.uint8)
        params["noise_sigma"] = round(sigma, 1)

    return img, mask, params


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------

def generate(src_dir: Path, n_samples: int = N_SAMPLES) -> None:
    print(f"\n{'='*60}")
    print(f"  Método  : geometric (Ours)")
    print(f"  Amostras: {n_samples}")
    print(f"  Seed    : {SEED}")
    print(f"  Fonte   : {src_dir}")
    print(f"{'='*60}")

    out_dir  = DATASET_DIR / OUT_NAME
    img_out  = out_dir / "images"
    mask_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(src_dir)
    print(f"  Total de pares disponíveis: {len(pairs)}")

    rng    = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    selected = [rng.choice(pairs) for _ in range(n_samples)]

    log_rows = [["index", "output_image", "output_mask",
                 "source_image", "source_mask", "augmentations"]]

    for i, (img_path, mask_path) in enumerate(tqdm(selected, desc="geometric")):
        img  = load_gray(img_path)
        mask = load_mask(mask_path)

        img_aug, mask_aug, params = apply_geometric(img, mask, rng, np_rng)

        out_stem   = f"geom_{i:04d}"
        img_fname  = f"{out_stem}.png"
        mask_fname = f"{out_stem}.png"

        save_png(img_aug,  img_out  / img_fname)
        save_png(mask_aug, mask_out / mask_fname)

        aug_str = "_".join(
            [("flip_h" if params.get("flip_h") else None)] +
            [f"rot_{params['rot']}"] +
            [f"tx{params['tx']}_ty{params['ty']}"] +
            [f"scale_{params['scale']}"] +
            ([f"noise_{params['noise_sigma']}"] if "noise_sigma" in params else [])
        )
        aug_str = "_".join(x for x in aug_str.split("_") if x and x != "None")

        log_rows.append([
            i,
            f"images/{img_fname}",
            f"masks/{mask_fname}",
            img_path.name,
            mask_path.name if mask_path.exists() else "empty",
            aug_str,
        ])

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

    print(f"  ✓ Gerado em: {out_dir}")
    print(f"  ✓ Log      : {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera 1600 amostras geométricas físicas (método 'Ours')."
    )
    parser.add_argument("--src", default=DEFAULT_SRC,
                        help=f"Diretório raiz TGS (default: {DEFAULT_SRC}).")
    parser.add_argument("--n", type=int, default=N_SAMPLES,
                        help=f"Número de amostras (default: {N_SAMPLES}).")
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src)
    if not src_dir.exists():
        print(f"ERRO: diretório fonte não existe: {src_dir}", file=sys.stderr)
        print("  Defina TGS_PATH ou use --src <path>", file=sys.stderr)
        sys.exit(1)
    generate(src_dir, args.n)
    print("\n✅ Concluído.")


if __name__ == "__main__":
    main()
