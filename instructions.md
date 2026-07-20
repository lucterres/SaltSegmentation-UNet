Terminal do VS Code conectado ao servidor via Remote-SSH você tem acesso direto, sem buffering, vê o output em tempo real e pode abrir os arquivos remotos diretamente.

**Para continuar de lá, os próximos passos são:**

### 1 — Abrir terminal no servidor
No VS Code conectado ao `cdi-gpu`:
```
Ctrl+` (backtick) → abre terminal já no servidor
```

### 2 — Verificar config e ambiente
```bash
# Confirmar config
grep -E 'TGS_PATH|BATCH_SIZE|DEVICE' ~/projetos/R2.1-downstream/Salt-Segmentation-UNet/utils/config.py

# Ver ambientes conda disponíveis
conda env list
```

### 3 — Instalar dependências (se necessário) e rodar
```bash
cd ~/projetos/R2.1-downstream/Salt-Segmentation-UNet

# Instalar deps no ambiente correto
conda activate <nome_do_env>
pip install torch torchvision imutils scikit-learn pandas scipy tqdm matplotlib

# Rodar Cenário A seed 42
python -u train.py --scenario A --seed 42
```

### 4 — Quando terminar, copiar results de volta
```powershell
# Aqui no Windows:
scp -r cdi-gpu:~/projetos/R2.1-downstream/results/ D:\IEEEEAccess\experiment-downstream\results\
```

---

Me chame quando quiser continuar daqui — seja para interpretar os resultados, atualizar o _v7.tex com a nova seção, ou atualizar o response_to_reviewers.md com o R2.1. 🚀