"""
generate_albumentations_da.py
==============================
Gera 1 600 pares (imagem + máscara) para cada um dos 6 métodos Albumentations
comparados na Seção 6.3 do artigo:

    Henriques et al., arXiv:2106.08269

Métodos gerados
---------------
1. elastic_transform
2. grid_distortion
3. optical_distortion
4. clahe
5. random_brightness_contrast
6. random_gamma

Saída
-----
Para cada método <name>, é criado:
    dataset/<name>1600/images/   ← 1600 PNGs de imagem  (grayscale, 101×101)
    dataset/<name>1600/masks/    ← 1600 PNGs de máscara (binária, 101×101)
    dataset/<name>1600/log.csv   ← índice, arquivo-fonte, parâmetros aplicados

Uso
---
    # Gera todos os 6 métodos (default)
    python DataAugmentation/generate_albumentations_da.py

    # Gera apenas um método específico
    python DataAugmentation/generate_albumentations_da.py --method clahe

    # Especifica diretório do TGS (sobrescreve TGS_PATH env var)
    python DataAugmentation/generate_albumentations_da.py --src /path/to/tgs/train

Parâmetros fixados (reprodutíveis)
-----------------------------------
Todos os transforms usam p=1.0 para garantir que TODA amostra seja transformada.
Parâmetros específicos são documentados abaixo e no log.csv gerado.
Seed global: 42

Nota sobre máscaras
--------------------
- Transforms geométricos (ElasticTransform, GridDistortion, OpticalDistortion)
  são aplicados de forma acoplada (mesma seed) a imagem e máscara.
- Transforms de intensidade (CLAHE, RandomBrightnessContrast, RandomGamma)
  são aplicados SOMENTE à imagem; a máscara é copiada sem alteração.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuração de paths (relativa ao diretório Salt-Segmentation-UNet)
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent          # DataAugmentation/
PROJECT_DIR = SCRIPT_DIR.parent                        # Salt-Segmentation-UNet/
DATASET_DIR = PROJECT_DIR / "dataset"

# Fonte padrão: TGS_PATH env var → fallback para path remoto
DEFAULT_SRC = os.environ.get("TGS_PATH", "/var/tmp/cym7/datasets/tgs-salt/train")

# ---------------------------------------------------------------------------
# Parâmetros fixados para cada método (reprodutíveis — seed=42)
# ---------------------------------------------------------------------------
SEED = 42

TRANSFORMS = {
    "elastic_transform": {
        "description": "ElasticTransform: alpha=80, sigma=9, p=1.0",
        "geometric": True,
        "build": lambda: A.ElasticTransform(
            alpha=80,
            sigma=9,
            p=1.0,
        ),
    },
    "grid_distortion": {
        "description": "GridDistortion: num_steps=5, distort_limit=0.3, p=1.0",
        "geometric": True,
        "build": lambda: A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=1.0,
        ),
    },
    "optical_distortion": {
        "description": "OpticalDistortion: distort_limit=0.4, shift_limit=0.08, p=1.0",
        "geometric": True,
        "build": lambda: A.OpticalDistortion(
            distort_limit=0.4,
            shift_limit=0.08,
            p=1.0,
        ),
    },
    "clahe": {
        "description": "CLAHE: clip_limit=4.0, tile_grid_size=(4,4), p=1.0 — imagem only",
        "geometric": False,
        "build": lambda: A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(4, 4),
            p=1.0,
        ),
    },
    "random_brightness_contrast": {
        "description": "RandomBrightnessContrast: brightness_limit=0.3, contrast_limit=0.3, p=1.0 — imagem only",
        "geometric": False,
        "build": lambda: A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=1.0,
        ),
    },
    "random_gamma": {
        "description": "RandomGamma: gamma_limit=(60,140), p=1.0 — imagem only",
        "geometric": False,
        "build": lambda: A.RandomGamma(
            gamma_limit=(60, 140),
            p=1.0,
        ),
    },
}

N_SAMPLES = 1600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_gray(path: Path) -> np.ndarray:
    """Carrega PNG em escala de cinza como uint8 (H×W)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler: {path}")
    return img


def load_mask(path: Path) -> np.ndarray:
    """Carrega máscara como uint8 binária (0 ou 255)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        # Máscara vazia (imagem sem sal — aceito no TGS)
        return np.zeros((101, 101), dtype=np.uint8)
    return mask


def save_png(arr: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), arr)


def collect_pairs(src_dir: Path):
    """Retorna lista de (img_path, mask_path) do dataset TGS."""
    img_dir  = src_dir / "images"
    mask_dir = src_dir / "masks"

    if not img_dir.exists():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {img_dir}")

    pairs = []
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        pairs.append((img_path, mask_path))

    if not pairs:
        raise RuntimeError(f"Nenhuma imagem PNG encontrada em {img_dir}")

    return pairs


def apply_geometric(transform, img: np.ndarray, mask: np.ndarray):
    """Aplica transform geométrico acoplado: mesma deformação em img e máscara."""
    result = transform(image=img, mask=mask)
    return result["image"], result["mask"]


def apply_intensity(transform, img: np.ndarray, mask: np.ndarray):
    """Aplica transform de intensidade somente na imagem; máscara inalterada."""
    result = transform(image=img)
    return result["image"], mask


# ---------------------------------------------------------------------------
# Geração principal
# ---------------------------------------------------------------------------

def generate(method_name: str, src_dir: Path, n_samples: int = N_SAMPLES) -> None:
    cfg = TRANSFORMS[method_name]
    print(f"\n{'='*60}")
    print(f"  Método  : {method_name}")
    print(f"  Config  : {cfg['description']}")
    print(f"  Amostras: {n_samples}")
    print(f"  Fonte   : {src_dir}")
    print(f"{'='*60}")

    out_dir   = DATASET_DIR / f"{method_name}1600"
    img_out   = out_dir / "images"
    mask_out  = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(src_dir)
    print(f"  Total de pares disponíveis: {len(pairs)}")

    # Reprodutibilidade
    rng = random.Random(SEED)
    np.random.seed(SEED)

    # Amostragem com reposição se necessário
    selected = [rng.choice(pairs) for _ in range(n_samples)]

    transform = cfg["build"]()
    is_geometric = cfg["geometric"]
    apply_fn = apply_geometric if is_geometric else apply_intensity

    log_rows = [["index", "output_image", "output_mask",
                 "source_image", "source_mask", "method", "params"]]

    for i, (img_path, mask_path) in enumerate(tqdm(selected, desc=method_name)):
        img  = load_image_gray(img_path)
        mask = load_mask(mask_path)

        # Albumentations espera uint8; para transforms de intensidade,
        # a imagem pode precisar de 3 canais — usamos expand_dims antes
        if not is_geometric:
            img_rgb = np.stack([img, img, img], axis=-1)  # H×W×3 uint8
            result_img, result_mask = apply_intensity(transform, img_rgb, mask)
            result_img = result_img[:, :, 0]  # volta para grayscale
        else:
            result_img, result_mask = apply_fn(transform, img, mask)

        out_stem   = f"albu_{method_name}_{i:04d}"
        img_fname  = f"{out_stem}.png"
        mask_fname = f"{out_stem}.png"

        save_png(result_img,  img_out  / img_fname)
        save_png(result_mask, mask_out / mask_fname)

        log_rows.append([
            i,
            f"images/{img_fname}",
            f"masks/{mask_fname}",
            img_path.name,
            mask_path.name if mask_path.exists() else "empty",
            method_name,
            cfg["description"],
        ])

    # Grava log
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
        description="Gera 1600 amostras Albumentations por método (Seção 6.3)."
    )
    parser.add_argument(
        "--method",
        choices=list(TRANSFORMS.keys()) + ["all"],
        default="all",
        help="Método a gerar (default: all — gera todos os 6).",
    )
    parser.add_argument(
        "--src",
        default=DEFAULT_SRC,
        help=f"Diretório raiz do TGS (default: {DEFAULT_SRC}).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_SAMPLES,
        help=f"Número de amostras por método (default: {N_SAMPLES}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src)

    if not src_dir.exists():
        print(f"ERRO: diretório fonte não existe: {src_dir}", file=sys.stderr)
        print("  Defina TGS_PATH ou use --src <path>", file=sys.stderr)
        sys.exit(1)

    methods = list(TRANSFORMS.keys()) if args.method == "all" else [args.method]

    for method in methods:
        generate(method, src_dir, args.n)

    print("\n✅ Concluído. Datasets gerados em Salt-Segmentation-UNet/dataset/")


if __name__ == "__main__":
    main()
