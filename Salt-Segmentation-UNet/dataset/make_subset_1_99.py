"""make_subset_1_99.py — Gera dataset filtrado com cobertura de sal entre 1% e 99%.

Lê split_stats.csv (gerado pelo script de split), filtra amostras com
coverage entre 1% e 99%, e copia as imagens e máscaras correspondentes
para um novo diretório subset_1_99/.

Uso:
    python dataset/make_subset_1_99.py \
        --src_images /var/tmp/cym7/datasets/tgs-salt/train/images \
        --src_masks  /var/tmp/cym7/datasets/tgs-salt/train/masks \
        --stats      dataset/subset_split/split_stats.csv \
        --out        /var/tmp/cym7/datasets/subset_1_99 \
        --min_pct    1.0 \
        --max_pct    99.0
"""

import argparse
import os
import shutil

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src_images', required=True,
                   help='Pasta com imagens originais (.png)')
    p.add_argument('--src_masks',  required=True,
                   help='Pasta com máscaras originais (.png)')
    p.add_argument('--stats',      required=True,
                   help='Path para split_stats.csv com coluna salt_coverage_pct')
    p.add_argument('--out',        required=True,
                   help='Diretório de saída (será criado)')
    p.add_argument('--min_pct',    type=float, default=1.0,
                   help='Cobertura mínima de sal em %% (default: 1.0)')
    p.add_argument('--max_pct',    type=float, default=99.0,
                   help='Cobertura máxima de sal em %% (default: 99.0)')
    return p.parse_args()


def main():
    args = parse_args()

    # Carregar stats
    df = pd.read_csv(args.stats)
    print(f'[INFO] Total de amostras no stats: {len(df)}')

    # Detectar coluna de coverage (pode ser salt_coverage_pct ou coverage_pct)
    cov_col = None
    for candidate in ['salt_coverage_pct', 'coverage_pct', 'coverage', 'salt_pct']:
        if candidate in df.columns:
            cov_col = candidate
            break
    if cov_col is None:
        print(f'[ERRO] Colunas disponíveis: {list(df.columns)}')
        raise ValueError('Coluna de cobertura não encontrada. Verifique o CSV.')

    print(f'[INFO] Coluna de cobertura: {cov_col}')
    print(f'[INFO] Filtro: {args.min_pct}% ≤ {cov_col} ≤ {args.max_pct}%')

    # Filtrar
    mask = (df[cov_col] >= args.min_pct) & (df[cov_col] <= args.max_pct)
    filtered = df[mask].copy()
    print(f'[INFO] Amostras após filtro: {len(filtered)} / {len(df)}')

    # Detectar coluna de id
    id_col = None
    for candidate in ['id', 'image_id', 'sample_id', 'name', 'filename']:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        print(f'[ERRO] Colunas disponíveis: {list(df.columns)}')
        raise ValueError('Coluna de ID não encontrada. Verifique o CSV.')

    print(f'[INFO] Coluna de ID: {id_col}')

    # Criar estrutura de saída
    out_images = os.path.join(args.out, 'images')
    out_masks  = os.path.join(args.out, 'masks')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks,  exist_ok=True)

    # Copiar arquivos
    copied = 0
    missing = 0
    for _, row in filtered.iterrows():
        img_id = str(row[id_col])
        # Remover extensão se já incluída
        stem = os.path.splitext(img_id)[0]

        src_img  = os.path.join(args.src_images, f'{stem}.png')
        src_mask = os.path.join(args.src_masks,  f'{stem}.png')
        dst_img  = os.path.join(out_images, f'{stem}.png')
        dst_mask = os.path.join(out_masks,  f'{stem}.png')

        if os.path.exists(src_img) and os.path.exists(src_mask):
            shutil.copy2(src_img,  dst_img)
            shutil.copy2(src_mask, dst_mask)
            copied += 1
        else:
            missing += 1
            if missing <= 5:
                print(f'  [WARN] Não encontrado: {stem}')

    print(f'[INFO] Copiados: {copied} | Não encontrados: {missing}')
    print(f'[INFO] Dataset salvo em: {args.out}')

    # Salvar CSV do subset
    out_csv = os.path.join(args.out, 'subset_1_99.csv')
    filtered.to_csv(out_csv, index=False)
    print(f'[INFO] CSV do subset: {out_csv}')

    # Estatísticas
    print(f'\n[STATS] Distribuição de cobertura:')
    print(f'  Min:    {filtered[cov_col].min():.2f}%')
    print(f'  Max:    {filtered[cov_col].max():.2f}%')
    print(f'  Média:  {filtered[cov_col].mean():.2f}%')
    print(f'  Mediana:{filtered[cov_col].median():.2f}%')


if __name__ == '__main__':
    main()
