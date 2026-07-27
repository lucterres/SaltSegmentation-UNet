---
applyTo: "docs/**/*.md"
---

# Instruções — Relatório Final

**Manuscrito:** Access-2026-27912 | **Última atualização:** 2026-07-27

---

## Arquivo do relatório

```
servidor: /u/cym7/projetos/SaltSegmentation-UNet/docs/relatorio-final-r21-downstream.md
windows:  f:\projetos\SaltSegmentation-UNet\docs\relatorio-final-r21-downstream.md
```

---

## Seções a atualizar após cada experimento

| Seção | Conteúdo |
|-------|----------|
| `## 2` | Resultados dos Cenários A e B por dataset |
| `## 3` | Comparativo entre seeds (42, 123, 456) |
| `## 12` | Tabela consolidada com todos os runs |
| `## 6. Conclusão final` | Narrative final do manuscrito |

---

## Template padrão de tabela por cenário

```markdown
### 2.X Cenário A — `<nome-dataset>` com test canônico (seed 42)

| Seed | N real | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
|:----:|:------:|:--------:|:---------:|:------------:|:------:|:---------:|
| 42   | XXXX   | X.XXXX   | X.XXXX    | X.XXXX       | XX     | XXX       |
| 123  | XXXX   | X.XXXX   | X.XXXX    | X.XXXX       | XX     | XXX       |
| 456  | XXXX   | X.XXXX   | X.XXXX    | X.XXXX       | XX     | XXX       |
```

---

## Colunas do `result.csv`

`scenario, seed, n_real, n_synth, best_val_iou, test_iou, test_dice, epochs_run, elapsed_s`

---

## Convenções de escrita

- Usar **negrito** para melhor resultado de cada grupo
- Marcar com ✅ o dataset/cenário vencedor
- Não comparar métricas de test sets diferentes (ver `experimento.instructions.md`)
- Métrica principal: **IoU** — secundária: **Dice**
