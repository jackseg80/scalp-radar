#!/bin/bash
set -e

DEPLOY_DIR="${DEPLOY_DIR:-/opt/scalp-radar}"
cd "$DEPLOY_DIR"

echo "========================================"
echo "  SCALP RADAR - Deploy"
echo "========================================"

# Créer les dossiers persistants si absents
mkdir -p data logs

# Graceful shutdown (SIGTERM → lifespan sauvegarde l'état)
echo "[*] Arrêt propre des containers..."
docker compose down --timeout 30 || true

# Pull
echo "[*] Mise à jour du code..."
git pull origin main

# Build
echo "[*] Build des images..."
docker compose build

# Start
echo "[*] Lancement..."
docker compose up -d

# Health check avec rollback
echo "[*] Vérification health..."
sleep 5
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "[OK] Deploy réussi"
    echo "[OK] Backend  : http://localhost:8000"
    echo "[OK] Frontend : http://localhost"
    echo "[OK] Health   : http://localhost:8000/health"
else
    echo "[ERREUR] Health check échoué — rollback"
    docker compose down
    docker compose up -d --no-build
    exit 1
fi
