# Azure Deployment Guide — DataPrepAgent

Deploy DataPrepAgent to Azure App Service in ~10 minutes.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| [Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli) ≥ 2.50 | `az --version` to verify |
| Azure account | Free trial works; B1 App Service plan ~$13/month |
| `openssl` | Pre-installed on macOS/Linux; Git Bash on Windows |
| Environment variables set | See `.env.example` |

---

## Step 1 — Log in to Azure

```bash
az login
```

A browser window opens. Sign in with your Azure account. Verify with:

```bash
az account show
```

---

## Step 2 — Set environment variables

Copy `.env.example` to `.env` and fill in your Azure AI Foundry values:

```bash
cp .env.example .env
# Edit .env with your values, then:
set -a && source .env && set +a    # bash/zsh
# OR export each variable manually:
export AZURE_AI_PROJECT_ENDPOINT="https://..."
export AZURE_AI_PROJECT_KEY="..."
export AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4o"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="..."
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://..."
```

---

## Step 3 — Run the deployment script

```bash
bash infra/deploy.sh
```

The script will:
1. Create a resource group (`dataprepagent-rg` in `eastus`)
2. Create a Linux App Service plan (B1 SKU)
3. Create a Python 3.11 web app with a unique name
4. Push all environment variables as app settings
5. Upload and deploy your code with `az webapp up`

Total time: ~5–8 minutes.

---

## Step 4 — Verify

The script prints the app URL at the end:

```
App URL: https://dataprepagent-<hex>.azurewebsites.net
```

Open that URL in your browser. You should see the DataPrepAgent landing page. The first cold start may take 30–60 seconds while pip installs dependencies.

---

## Troubleshooting

### "Command not found: openssl"
On Windows, install [Git for Windows](https://git-scm.com/download/win) which bundles openssl, or run the script inside WSL.

### "az login" opens no browser
Use device code flow:
```bash
az login --use-device-code
```

### App shows 503 / Application Error
Check logs:
```bash
az webapp log tail --resource-group dataprepagent-rg --name <APP_NAME>
```
Common causes: missing env vars, pip install failure. Verify all 5 `AZURE_*` variables are set.

### "Quota exceeded" on B1 plan
Use a different region:
```bash
LOCATION=westeurope bash infra/deploy.sh
```

### Re-deploying after code changes
```bash
az webapp up --resource-group dataprepagent-rg --name <APP_NAME> --runtime "PYTHON:3.11"
```

### Tear down all resources
```bash
az group delete --name dataprepagent-rg --yes --no-wait
```

---

## Cost estimate

| Resource | SKU | Est. cost |
|---|---|---|
| App Service Plan | B1 Linux | ~$13/month |
| App Service | — | included |

Free tier (F1) is available but lacks always-on and has 60 CPU-minutes/day limit — not suitable for a demo.
