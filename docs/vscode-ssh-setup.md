# Setup VSCode + SSH Remoto + GitHub Copilot

**Servidor:** `atn2b02n07` (Atena) — 8 × V100-SXM2-32GB  
**Usuário:** `cym7`  
**Plataforma local:** Windows + VSCode

---

## 1 — Pré-requisitos locais

| Item | Versão mínima | Download |
|------|---------------|---------- |
| Visual Studio Code | 1.85+ | https://code.visualstudio.com |
| Git for Windows | 2.40+ | https://git-scm.com |
| OpenSSH client | embutido no Windows 10/11 | ativado em *Configurações → Apps → Recursos opcionais* |

Verificar se o OpenSSH está disponível:

```powershell
ssh -V
# OpenSSH_for_Windows_9.x, ...
```

---

## 2 — Extensões VSCode obrigatórias

Instalar pelo marketplace (`Ctrl+Shift+X`) ou pelo terminal:

```powershell
# Remote Development (inclui Remote-SSH, Remote-Containers, WSL)
code --install-extension ms-vscode-remote.vscode-remote-extensionpack

# GitHub Copilot
code --install-extension GitHub.copilot

# GitHub Copilot Chat
code --install-extension GitHub.copilot-chat
```

> **Alternativa rápida:** abrir VSCode → `Ctrl+P` → colar cada linha:
> ```
> ext install ms-vscode-remote.remote-ssh
> ext install GitHub.copilot
> ext install GitHub.copilot-chat
> ```

---

## 3 — Configurar chave SSH (sem senha toda vez)

### 3.1 — Gerar par de chaves (se ainda não existir)

```powershell
# No PowerShell local
ssh-keygen -t ed25519 -C "cym7@atena" -f "$env:USERPROFILE\.ssh\id_ed25519_atena"
# Deixar passphrase em branco para uso sem prompt
```

### 3.2 — Copiar chave pública para o servidor

```powershell
# Substitua <senha> pela sua senha do servidor (apenas nesta etapa)
type "$env:USERPROFILE\.ssh\id_ed25519_atena.pub" | ssh cym7@atn2b02n07 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### 3.3 — Arquivo `~/.ssh/config` local

Criar/editar `C:\Users\cym7\.ssh\config`:

```ssh-config
Host atena
    HostName atn2b02n07
    User cym7
    IdentityFile ~/.ssh/id_ed25519_atena
    ServerAliveInterval 60
    ServerAliveCountMax 10
    ForwardAgent yes
```

Testar a conexão:

```powershell
ssh atena
# Deve conectar sem pedir senha
```

---

## 4 — Conectar via Remote-SSH no VSCode

1. Abrir a paleta: `Ctrl+Shift+P`
2. Digitar `Remote-SSH: Connect to Host…`
3. Selecionar **`atena`** (definido no `~/.ssh/config`)
4. Uma nova janela do VSCode abre já dentro do servidor
5. Abrir a pasta do projeto:
   ```
   /u/cym7/projetos/Experiment-downstream
   ```

> **Dica:** usar `File → Add Folder to Workspace…` para adicionar a pasta do dataset
> `/var/tmp/cym7/datasets/tgs-salt/train` ao mesmo workspace, se necessário.

---

## 5 — Terminal SSH integrado no VSCode

Com a janela remota aberta, o terminal padrão já roda **no servidor**:

```
Ctrl+`   ← abrir terminal integrado
```

Ativar o ambiente Python a cada nova sessão:

```bash
source /var/tmp/cym7/venvs/salt-unet/bin/activate
cd /u/cym7/projetos/Experiment-downstream/Salt-Segmentation-UNet
```

Para não repetir isso manualmente, adicionar ao `~/.bashrc` do servidor:

```bash
# ~/.bashrc — trecho a adicionar no servidor
if [ -d /var/tmp/cym7/venvs/salt-unet ]; then
    source /var/tmp/cym7/venvs/salt-unet/bin/activate
fi
```

---

## 6 — Ativar GitHub Copilot na sessão remota

### 6.1 — Autenticar (primeira vez)

1. Na janela remota do VSCode: `Ctrl+Shift+P` → `GitHub Copilot: Sign In`
2. Um código de dispositivo é gerado; abrir https://github.com/login/device **no navegador local**
3. Inserir o código e autorizar
4. O ícone do Copilot na barra de status fica ativo (sem `X`)

### 6.2 — Verificar se o Copilot está funcionando

- Abrir qualquer arquivo `.py` no servidor
- Digitar uma função parcial — o Copilot deve sugerir completações em cinza
- Abrir o chat: `Ctrl+Alt+I` (ou painel lateral Copilot Chat)

### 6.3 — Extensões instaladas **no servidor remoto**

O Remote-SSH instala extensões *separadamente* no servidor. Verificar/instalar:

```
Ctrl+Shift+X → clique em "Install in SSH: atena"
```

Extensões necessárias no lado remoto:

| Extensão | ID |
|----------|----|
| GitHub Copilot | `GitHub.copilot` |
| GitHub Copilot Chat | `GitHub.copilot-chat` |
| Python | `ms-python.python` |
| Pylance | `ms-python.vscode-pylance` |

---

## 7 — Selecionar o interpretador Python correto

1. Abrir qualquer arquivo `.py` no projeto
2. Clicar no seletor de interpretador (canto inferior direito) **ou** `Ctrl+Shift+P` → `Python: Select Interpreter`
3. Escolher:
   ```
   /var/tmp/cym7/venvs/salt-unet/bin/python
   ```
4. O IntelliSense e o Pylance passam a reconhecer todos os pacotes do venv

---

## 8 — Workspace recomendado (`.code-workspace`)

Salvar o arquivo abaixo como `experiment.code-workspace` na pasta local  
`U:\projetos\Experiment-downstream\` para abrir tudo com um clique:

```jsonc
{
    "folders": [
        {
            "name": "Experiment-downstream (remoto)",
            "uri": "vscode-remote://ssh-remote+atena/u/cym7/projetos/Experiment-downstream"
        }
    ],
    "settings": {
        "python.defaultInterpreterPath": "/var/tmp/cym7/venvs/salt-unet/bin/python",
        "terminal.integrated.defaultProfile.linux": "bash",
        "editor.formatOnSave": true,
        "github.copilot.enable": {
            "*": true,
            "python": true,
            "markdown": true
        }
    }
}
```

Abrir: `File → Open Workspace from File…` → selecionar `experiment.code-workspace`

---

## 9 — Solução de problemas comuns

| Sintoma | Causa provável | Solução |
|---------|---------------|---------|
| `Permission denied (publickey)` | Chave não copiada ou `authorized_keys` com permissão errada | Repetir §3.2; no servidor: `chmod 700 ~/.ssh` |
| Copilot com ícone `X` vermelho | Extensão não instalada no lado remoto | `Ctrl+Shift+X` → `Install in SSH: atena` |
| Copilot não sugere nada | Sem conexão com api.github.com | Verificar proxy: `echo $https_proxy` no terminal remoto |
| Terminal fecha ao desconectar | Processo não rodando em background | Usar `nohup … &` ou `tmux` |
| Interpretador Python não encontrado | Path errado no workspace settings | Verificar §7 e confirmar que o venv existe com `ls /var/tmp/cym7/venvs/` |

### Manter sessão viva com `tmux` (recomendado para treinamentos longos)

```bash
# Criar sessão nomeada
tmux new -s train

# Dentro do tmux: rodar o treinamento normalmente
source /var/tmp/cym7/venvs/salt-unet/bin/activate
nohup python -u train.py --scenario A --seed 42 --epochs 100 &

# Desanexar sem matar (VSCode pode fechar à vontade)
Ctrl+B  D

# Reanexar depois de reconectar
tmux attach -t train
```

---

## 10 — Referências rápidas

| Atalho VSCode (janela remota) | Ação |
|-------------------------------|------|
| `Ctrl+Shift+P` | Paleta de comandos |
| `` Ctrl+` `` | Terminal integrado (no servidor) |
| `Ctrl+Alt+I` | Abrir Copilot Chat |
| `Ctrl+Shift+X` | Extensões |
| `Ctrl+Shift+G` | Source Control (Git) |
| `F5` | Rodar/debugar script Python atual |

| Comando SSH útil | Descrição |
|------------------|-----------|
| `nvidia-smi` | Status das GPUs |
| `ps aux \| grep train.py` | Ver treinamentos em andamento |
| `tail -f results/*/train.log` | Acompanhar todos os logs |
| `tmux ls` | Listar sessões tmux ativas |

---

*Documento gerado em 2026-07-20 — Experimento Downstream R2.1 (Access-2026-27912)*
