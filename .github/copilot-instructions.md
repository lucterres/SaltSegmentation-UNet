# GitHub Copilot Instructions — Experimento Downstream R2.1

## Contexto do projeto

**Manuscrito:** Access-2026-27912  
**Objetivo:** Demonstrar que treinar com dados reais + sintéticos melhora a segmentação de salt domes em dados sísmicos reais de teste (Cenário B > Cenário A).

---

## Cenários definidos

| Cenário | Dados de treino | Objetivo |
|---------|----------------|----------|
| A | Real only | Baseline |
| B | Real + Sintético | Hipótese do paper |

---

## Arquivos-chave

| Arquivo | Função |
|---------|--------|
| `Salt-Segmentation-UNet/utils/config.py` | Configurações centrais (`TGS_PATH`, canais do encoder) |
| `Salt-Segmentation-UNet/utils/model.py` | U-Net com `padding=1` nos `Conv2d` |
| `Salt-Segmentation-UNet/utils/dataset.py` | DataLoader TGS; interpolação `NEAREST` para máscaras |
| `Salt-Segmentation-UNet/train.py` | Loop de treino: `--scenario`, `--seed`, `--n_real`, `--n_synth`, `--epochs`, `--batch`, `--lr`, `--train_dir`, `--test_dir` |
| `Salt-Segmentation-UNet/evaluate.py` | Avaliação no test set fixo → gera `results/summary.csv` |
| `Salt-Segmentation-UNet/generate_synthetic.py` | Gera pool sintético via VAE + textura |
| `docs/relatorio-final-r21-downstream.md` | Relatório completo dos experimentos |

---

## Convenções de código

- `TGS_PATH` em `utils/config.py` → `/var/tmp/cym7/datasets/tgs-salt/train`
- `ENCODER_CHANNELS = (1, 16, 32, 64)` — TGS é grayscale (1 canal de entrada)
- Interpolação `NEAREST` para máscaras binárias (evita artefatos)
- `padding=1` no U-Net preserva dimensão espacial sem `CenterCrop`
- Split estratificado por presença de sal para reprodutibilidade
- Métricas primárias: **IoU** (critério de early stopping) e **Dice**
- `--train_dir <path>` sobrescreve `TGS_PATH`; `--test_dir <path>` usa test set externo fixo

---

## Estrutura de resultados

```
results/<run_tag>/
├── train.log       ← saída completa do treinamento
├── best_model.pth  ← checkpoint da melhor época (val IoU)
├── result.csv      ← métricas finais (test set)
├── history.csv     ← métricas por época
└── plot.png        ← curvas loss/IoU por época
```

Arquivo consolidado: `results/summary.csv` (gerado por `evaluate.py`)

---

## Outras camadas de instruções

| Camada | Arquivo | Escopo |
|--------|---------|--------|
| Infra & ambiente | `.github/instructions/infra.instructions.md` | `**/*.sh` |
| Experimentos & datasets | `.github/instructions/experimento.instructions.md` | `Salt-Segmentation-UNet/**/*.py` |
| Relatório | `.github/instructions/relatorio.instructions.md` | `docs/**/*.md` |
| Rodar experimento | `.github/prompts/run-experiment.prompt.md` | — |
| Atualizar relatório | `.github/prompts/update-report.prompt.md` | — |
| Análise de métricas | `.github/agents/researcher.agent.md` | — |
| Gerar subset | `.github/skills/generate-subset.md` | — |

---

## Após executar os experimentos — atualizar

1. `_v7.tex` → `\subsection{Downstream Segmentation Evaluation}`
2. `docs/_reviewACCESS/response_to_reviewers.md` → seção R2.1
3. `docs/_reviewACCESS/summary_of_changes.md` → status R2.1: **PENDING → DONE**

---

## Referências internas

- Protocolo completo: `docs/experimentUNet-protocol.md`
- Repositório base: https://github.com/matin-ghorbani/Salt-Segmentation-UNet
