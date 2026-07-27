---
description: Atualiza o relatório final após novos experimentos, preenchendo as seções 2, 3, 12 e Conclusão com os resultados do result.csv.
---

# Atualizar Relatório Final — R2.1

## Arquivo a editar

```
f:\projetos\SaltSegmentation-UNet\docs\relatorio-final-r21-downstream.md
```

## Passos

1. **Ler os resultados do run** — copiar os valores de `result.csv`:
   ```
   scenario, seed, n_real, n_synth, best_val_iou, test_iou, test_dice, epochs_run, elapsed_s
   ```

2. **Preencher a tabela na seção `## 2`** usando o template:

   ```markdown
   ### 2.X Cenário <A|B> — `<nome-dataset>` com test canônico

   | Seed | N real | Test IoU | Test Dice | Best val IoU | Épocas | Tempo (s) |
   |:----:|:------:|:--------:|:---------:|:------------:|:------:|:---------:|
   | 42   | XXXX   | X.XXXX   | X.XXXX    | X.XXXX       | XX     | XXX       |
   ```

3. **Atualizar `## 3`** — adicionar linha à tabela comparativa de seeds.

4. **Atualizar `## 12`** — tabela consolidada com todos os runs concluídos.

5. **Atualizar `## 6. Conclusão final`** — se o novo resultado alterar o ranking ou a narrativa.

## Regras de formatação

- **Negrito** no melhor resultado de cada grupo
- ✅ no dataset/cenário vencedor
- Nunca comparar métricas de test sets distintos (ver `experimento.instructions.md`)
- Métricas devem ter 4 casas decimais

## Após atualizar o relatório

- [ ] `_v7.tex` → `\subsection{Downstream Segmentation Evaluation}`
- [ ] `docs/_reviewACCESS/response_to_reviewers.md` → seção R2.1
- [ ] `docs/_reviewACCESS/summary_of_changes.md` → R2.1: **PENDING → DONE**
