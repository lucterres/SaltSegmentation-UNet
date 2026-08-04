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
    dataset/<name><suffix>/images/   ← PNGs de imagem  (grayscale, 101×101)
    dataset/<name><suffix>/masks/    ← PNGs de máscara (binária, 101×101)
    dataset/<name><suffix>/log.csv   ← índice, arquivo-fonte, parâmetros aplicados

Uso
---
    # Gera todos os 6 métodos (default)
    python DataAugmentation/generate_albumentations_da.py

    # Gera apenas um método específico
    python DataAugmentation/generate_albumentations_da.py --method clahe

    # Especifica diretório do TGS (sobrescreve TGS_PATH env var)
    python DataAugmentation/generate_albumentations_da.py --src /path/to/tgs/train

    # MODO LIMPO: exclui IDs do test set canônico para evitar data leakage
    python DataAugmentation/generate_albumentations_da.py \
        --src D:/dataset/tgs-salt/train \
        --exclude_csv dataset/subset_split/split_stats.csv \
        --exclude_split test \
        --out_suffix 1600clean

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
        "description": "OpticalDistortion: distort_limit=0.4, p=1.0",
        "geometric": True,
        "build": lambda: A.OpticalDistortion(
            distort_limit=0.4,
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

def load_exclusion_ids(exclude_csv: Path, exclude_split: str) -> set:
    """Carrega IDs a excluir de um split_stats.csv."""
    if not exclude_csv.exists():
        raise FileNotFoundError(f"CSV de exclusão não encontrado: {exclude_csv}")
    ids = set()
    with open(exclude_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "") == exclude_split:
                ids.add(row["id"].strip())
    print(f"  [exclusão] {len(ids)} IDs do split '{exclude_split}' serão excluídos.")
    return ids


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
        return np.zeros((101, 101), dtype=np.uint8)
    return mask


def save_png(arr: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), arr)


def collect_pairs(src_dir: Path, exclude_ids: set = None):
    """Retorna lista de (img_path, mask_path) do dataset TGS, excluindo IDs opcionais."""
    img_dir  = src_dir / "images"
    mask_dir = src_dir / "masks"

    if not img_dir.exists():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {img_dir}")

    exclude_ids = exclude_ids or set()
    pairs = []
    skipped = 0
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        if stem in exclude_ids:
            skipped += 1
            continue
        mask_path = mask_dir / f"{stem}.png"
        pairs.append((img_path, mask_path))

    if skipped:
        print(f"  [exclusão] {skipped} imagens removidas do pool de treino.")

    if not pairs:
        raise RuntimeError(f"Nenhuma imagem PNG encontrada em {img_dir}")

    return pairs


def apply_geometric(transform, img: np.ndarray, mask: np.ndarray):
    result = transform(image=img, mask=mask)
    return result["image"], result["mask"]


def apply_intensity(transform, img: np.ndarray, mask: np.ndarray):
    result = transform(image=img)
    return result["image"], mask


# ---------------------------------------------------------------------------
# Geração principal
# ---------------------------------------------------------------------------

def generate(method_name: str, src_dir: Path, n_samples: int = N_SAMPLES,
             exclude_ids: set = None, out_suffix: str = None) -> None:
    cfg = TRANSFORMS[method_name]
    suffix = out_suffix if out_suffix is not None else str(n_samples)
    print(f"\n{'='*60}")
    print(f"  Método  : {method_name}")
    print(f"  Config  : {cfg['description']}")
    print(f"  Amostras: {n_samples}")
    print(f"  Fonte   : {src_dir}")
    print(f"  Saída   : {method_name}{suffix}")
    if exclude_ids:
        print(f"  Excluídos: {len(exclude_ids)} IDs do test set canônico")
    print(f"{'='*60}")

    out_dir   = DATASET_DIR / f"{method_name}{suffix}"
    img_out   = out_dir / "images"
    mask_out  = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(src_dir, exclude_ids)
    print(f"  Total de pares disponíveis (após exclusão): {len(pairs)}")

    rng = random.Random(SEED)
    np.random.seed(SEED)

    selected = [rng.choice(pairs) for _ in range(n_samples)]

    transform = cfg["build"]()
    is_geometric = cfg["geometric"]
    apply_fn = apply_geometric if is_geometric else apply_intensity

    log_rows = [["index", "output_image", "output_mask",
                 "source_image", "source_mask", "method", "params"]]

    for i, (img_path, mask_path) in enumerate(tqdm(selected, desc=method_name)):
        img  = load_image_gray(img_path)
        mask = load_mask(mask_path)

        if not is_geometric:
            img_rgb = np.stack([img, img, img], axis=-1)
            result_img, result_mask = apply_intensity(transform, img_rgb, mask)
            result_img = result_img[:, :, 0]
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
        description="Gera amostras Albumentations por método DA."
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
    parser.add_argument(
        "--exclude_csv",
        type=str,
        default=None,
        help="Caminho para split_stats.csv com IDs a excluir do pool de treino "
             "(ex: dataset/subset_split/split_stats.csv). "
             "Evita data leakage com o test set canônico.",
    )
    parser.add_argument(
        "--exclude_split",
        type=str,
        default="test",
        help="Valor do campo 'split' no CSV de exclusão (default: 'test').",
    )
    parser.add_argument(
        "--out_suffix",
        type=str,
        default=None,
        help="Sufixo do diretório de saída (default: número de amostras, ex: '1600'). "
             "Use '1600clean' para indicar pool sem leakage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src)

    if not src_dir.exists():
        print(f"ERRO: diretório fonte não existe: {src_dir}", file=sys.stderr)
        print("  Defina TGS_PATH ou use --src <path>", file=sys.stderr)
        sys.exit(1)

    # Carregar IDs a excluir (opcional)
    exclude_ids = set()
    if args.exclude_csv:
        exclude_ids = load_exclusion_ids(
            Path(args.exclude_csv), args.exclude_split
        )

    methods = list(TRANSFORMS.keys()) if args.method == "all" else [args.method]

    for method in methods:
        generate(method, src_dir, args.n,
                 exclude_ids=exclude_ids,
                 out_suffix=args.out_suffix)

    suffix = args.out_suffix or str(args.n)
    print(f"\n✅ Concluído. Datasets gerados em Salt-Segmentation-UNet/dataset/*{suffix}/")


if __name__ == "__main__":
    main()
