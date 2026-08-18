Leio o resultado do seed 456:

Sent `cat $PROJ/results/scenario_B_seed456_nreal1200_train_pairs1600_seismic_ns1200...` to terminal

Tenho os 3 resultados. Análise completa:

---

## Resultados — Cenário B | N=1200 + 1200 sísmicos | `pairs1600_seismic` | 3 seeds

| Seed | Best val IoU | Test IoU | Test Dice | Épocas | Tempo (s) |
|:----:|:------------:|:--------:|:---------:|:------:|:---------:|
| 42   | 0.4032 | 0.4241 | 0.4623 | 41 | 112 |
| 123  | 0.4004 | 0.4025 | 0.4411 | 24 | 66 |
| 456  | 0.4047 | **0.4233** | **0.4596** | 46 | 125 |
| **Média** | **0.4028** | **0.4166** | **0.4543** | 37 | 101 |

### Comparação com Cenário A (N=1200, baseline)

| Cenário | Pool sintético | IoU médio | Δ vs A |
|---------|---------------|:---------:|:------:|
| **A** | — | **0.4081** | — |
| **B** `pairs1600_seismic` | sísmico 1600 | **0.4166** | **+0.0085 ✅** |
| B `elastic_transform` *(albu)* | geométrico albu | 0.4276 | +0.0195 |
| B `grid_distortion` *(albu)* | geométrico albu | 0.4272 | +0.0191 |

**Conclusão:** O pool sísmico (`pairs1600_seismic`) supera o baseline A (+0.0085), mas fica **abaixo dos melhores métodos Albumentations** com N=1200. Com N=3998 o pool sísmico era o padrão anterior — com Albumentations o ganho é maior em regime de escassez.



Mapa completo das pastas:

N3998 — N_real = 3998 (TGS completo, test canônico 800)
Tipo	Pastas
Cenário A	scenario_A_seed{42,123,456}_train/
Cenário B — 6 Albumentations × 3 seeds	scenario_B_seed{42,123,456}_train_{dataset}_ns1200/
21 runs — os albumentations com N=3998 estão aqui.

results (raiz) — N_real = 1200 (low-data regime, test canônico 800)
Tipo	Pastas
Cenário A	scenario_A_seed{42,123,456}_nreal1200_train/
Cenário B — 6 Albumentations × 3 seeds	scenario_B_seed{42,123,456}_nreal1200_train_{dataset}_ns1200/
Cenário B — Pool sísmico pairs1600_seismic ns=1200	scenario_B_seed{42,123,456}_nreal1200_train_pairs1600_seismic_ns1200/
Cenário B — Pool sísmico pairs1600_seismic ns=955	scenario_B_seed{42,123,456}_nreal1200_train_pairs1600_seismic_ns955/
27 runs — albumentations + sísmico com N=1200 estão aqui.