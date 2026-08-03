# DataAugmentation — Scripts de Geração de Amostras

Scripts para gerar os datasets de comparação da Seção 6.3 do artigo  
(Henriques et al., arXiv:2106.08269).

---

## Scripts disponíveis

| Script | Método | Saída |
|--------|--------|-------|
| `generate_albumentations_da.py` | 6 métodos Albumentations (paper Seção 6.3) | `dataset/<method>1600/` |
| `generate_geometric_da.py` | Método geométrico físico ("Ours") | `dataset/geometric1600/` |

---

## Datasets gerados

```
dataset/
├── elastic_transform1600/   images/ + masks/ + log.csv
├── grid_distortion1600/     images/ + masks/ + log.csv
├── optical_distortion1600/  images/ + masks/ + log.csv
├── clahe1600/               images/ + masks/ + log.csv
├── random_brightness_contrast1600/  images/ + masks/ + log.csv
├── random_gamma1600/        images/ + masks/ + log.csv
└── geometric1600/           images/ + masks/ + log.csv   ← "Ours"
```

Cada dataset: **1 600 pares** PNG (imagem grayscale 101×101 + máscara binária).

---

## Parâmetros fixados (reprodutíveis, seed=42)

| Método | Parâmetros |
|--------|-----------|
| ElasticTransform | `alpha=80, sigma=9, p=1.0` |
| GridDistortion | `num_steps=5, distort_limit=0.3, p=1.0` |
| OpticalDistortion | `distort_limit=0.4, shift_limit=0.08, p=1.0` |
| CLAHE | `clip_limit=4.0, tile_grid_size=(4,4), p=1.0` |
| RandomBrightnessContrast | `brightness_limit=0.3, contrast_limit=0.3, p=1.0` |
| RandomGamma | `gamma_limit=(60,140), p=1.0` |
| **Geometric ("Ours")** | `flip_h=0.5, rot=±10°, trans=±5px, scale=0.9–1.1, noise σ~U[5,20]` |

> Transforms de intensidade (CLAHE, RBC, Gamma) são aplicados **somente à imagem**;  
> a máscara é copiada sem alteração.  
> Transforms geométricos são aplicados de forma **acoplada** (imagem + máscara).

---

## Como executar

```bash
# A partir do diretório Salt-Segmentation-UNet/
# (ou defina TGS_PATH apontando para o dataset TGS)

# Gera todos os 6 métodos Albumentations
python DataAugmentation/generate_albumentations_da.py

# Gera apenas um método
python DataAugmentation/generate_albumentations_da.py --method clahe

# Gera método geométrico ("Ours")
python DataAugmentation/generate_geometric_da.py

# Especificar fonte manualmente
python DataAugmentation/generate_albumentations_da.py --src /var/tmp/cym7/datasets/tgs-salt/train
python DataAugmentation/generate_geometric_da.py      --src /var/tmp/cym7/datasets/tgs-salt/train
```

---

## Dependências

```
albumentations>=1.3.0
opencv-python>=4.5
numpy>=1.21
tqdm
```

Instalar:
```bash
pip install albumentations opencv-python numpy tqdm
```
