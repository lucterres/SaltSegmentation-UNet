"""create_train_pool.py
======================
Cria o diretório dataset/train_pool/ com as amostras do TGS que NÃO pertencem
ao test set canônico (split_stats.csv, split != 'test').

Resultado: 3198 pares imagem/máscara livres de data leakage com o test set.

Uso
---
    python create_train_pool.py
    python create_train_pool.py --tgs_dir D:/dataset/tgs-salt/train
    python create_train_pool.py --dry_run   # apenas conta, não copia
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths default
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_TGS = r"D:\dataset\tgs-salt\train"
DEFAULT_CSV = DATASET_DIR / "subset_split" / "split_stats.csv"
DEFAULT_OUT = DATASET_DIR / "train_pool"


def parse_args():
    p = argparse.ArgumentParser(
        description="Cria train_pool: TGS completo menos o test set canônico."
    )
    p.add_argument("--tgs_dir", default=DEFAULT_TGS,
                   help=f"Diretório TGS com images/ e masks/ (default: {DEFAULT_TGS})")
    p.add_argument("--split_csv", default=str(DEFAULT_CSV),
                   help=f"split_stats.csv (default: {DEFAULT_CSV})")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT),
                   help=f"Diretório de saída (default: {DEFAULT_OUT})")
    p.add_argument("--dry_run", action="store_true",
                   help="Apenas conta amostras, não copia arquivos.")
    return p.parse_args()


def main():
    args = parse_args()
    tgs_dir  = Path(args.tgs_dir)
    csv_path = Path(args.split_csv)
    out_dir  = Path(args.out_dir)

    if not tgs_dir.exists():
        print(f"ERRO: TGS dir não existe: {tgs_dir}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.exists():
        print(f"ERRO: split_stats.csv não existe: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    train_df = df[df["split"] != "test"]
    test_df  = df[df["split"] == "test"]

    print(f"Total TGS  : {len(df)}")
    print(f"Test set   : {len(test_df)} (excluídos)")
    print(f"Train pool : {len(train_df)} (a copiar)")

    if args.dry_run:
        print("\n[dry_run] Nenhum arquivo copiado.")
        return

    img_out  = out_dir / "images"
    mask_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for _, row in train_df.iterrows():
        stem = row["id"]
        src_img  = tgs_dir / "images" / f"{stem}.png"
        src_mask = tgs_dir / "masks"  / f"{stem}.png"
        dst_img  = img_out  / f"{stem}.png"
        dst_mask = mask_out / f"{stem}.png"

        if not src_img.exists():
            skipped += 1
            continue

        shutil.copy2(src_img,  dst_img)
        if src_mask.exists():
            shutil.copy2(src_mask, dst_mask)
        else:
            # Máscara vazia (sem sal)
            import cv2
            import numpy as np
            cv2.imwrite(str(dst_mask), np.zeros((101, 101), dtype=np.uint8))

        copied += 1
        if copied % 500 == 0:
            print(f"  Copiados: {copied}/{len(train_df)}...")

    print(f"\n✅ train_pool criado em: {out_dir}")
    print(f"   Copiados : {copied}")
    print(f"   Pulados  : {skipped}")
    print(f"\nUso no treino:")
    print(f"  python train.py --scenario A --seed 42 --epochs 100 \\")
    print(f"    --train_dir {out_dir} --test_dir <subset_split/test>")


if __name__ == "__main__":
    main()
