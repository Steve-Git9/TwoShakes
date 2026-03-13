#!/bin/bash
set -e

# Variables
RESOURCE_GROUP="dataprepagent-rg"
LOCATION="eastus"
APP_NAME="dataprepagent-$(openssl rand -hex 4)"  # unique name
PLAN_NAME="dataprepagent-plan"

echo "=== DataPrepAgent Azure Deployment ==="

# Login check
az account show > /dev/null 2>&1 || { echo "Run 'az login' first"; exit 1; }

# Create resource group
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service plan (B1 for hackathon — F1 free tier doesn't support always-on)
echo "Creating App Service plan..."
az appservice plan create \
    --name $PLAN_NAME \
    --resource-group $RESOURCE_GROUP \
    --sku B1 \
    --is-linux

# Create web app
echo "Creating web app..."
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $PLAN_NAME \
    --name $APP_NAME \
    --runtime "PYTHON:3.11" \
    --startup-file "startup.sh"

# Configure environment variables
echo "Setting environment variables..."
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings \
    AZURE_AI_PROJECT_ENDPOINT="$AZURE_AI_PROJECT_ENDPOINT" \
    AZURE_AI_PROJECT_KEY="$AZURE_AI_PROJECT_KEY" \
    AZURE_AI_MODEL_DEPLOYMENT_NAME="$AZURE_AI_MODEL_DEPLOYMENT_NAME" \
    AZURE_DOCUMENT_INTELLIGENCE_KEY="$AZURE_DOCUMENT_INTELLIGENCE_KEY" \
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="$AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"

# Deploy from local
echo "Deploying code..."
az webapp up \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --runtime "PYTHON:3.11"

echo "=== Deployment complete ==="
echo "App URL: https://$APP_NAME.azurewebsites.net"
