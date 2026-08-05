"""
Consolida todos os arquivos result.csv encontrados recursivamente em results/
Adiciona coluna 'source_path' com o caminho relativo ao diretório raiz do projeto.
Separador de lista: ;
Saída: results/consolidated_results.csv
"""

import os
import glob
import csv
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "consolidated_results.csv")

pattern = os.path.join(RESULTS_DIR, "**", "result.csv")
files = sorted(glob.glob(pattern, recursive=True))

print(f"Encontrados {len(files)} arquivos result.csv")

rows = []
header = None

for fpath in files:
    rel_path = os.path.relpath(fpath, ROOT).replace("\\", "/")
    try:
        with open(fpath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            file_header = reader.fieldnames
            if file_header is None:
                print(f"  [SKIP] sem cabeçalho: {rel_path}")
                continue
            if header is None:
                header = list(file_header)
            for row in reader:
                # Preenche colunas faltantes com vazio
                full_row = {col: row.get(col, "") for col in header}
                full_row["source_path"] = rel_path
                rows.append(full_row)
    except Exception as e:
        print(f"  [ERRO] {rel_path}: {e}")

if not rows:
    print("Nenhum dado encontrado.")
    sys.exit(1)

final_header = header + ["source_path"]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=final_header, delimiter=";", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"\nConsolidado em: {os.path.relpath(OUTPUT_FILE, ROOT)}")
print(f"Total de linhas: {len(rows)}")
