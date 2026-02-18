#!/bin/bash
set -e

DEPLOY_DIR="${DEPLOY_DIR:-$HOME/scalp-radar}"
cd "$DEPLOY_DIR"

# Parse flags
CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean|-c) CLEAN=true ;;
    esac
done

echo "========================================"
echo "  SCALP RADAR - Deploy"
if [ "$CLEAN" = true ]; then
    echo "  Mode: CLEAN (fresh start)"
else
    echo "  Mode: NORMAL (state preserved)"
fi
echo "========================================"

# Créer les dossiers persistants si absents
mkdir -p data logs

# Reset config files to git version — prod overrides go in .env (gitignored)
echo "[*] Reset config files..."
git checkout -- config/

# Pull
echo "[*] Mise à jour du code..."
git pull origin main

# Build
echo "[*] Build des images..."
docker compose build

if [ "$CLEAN" = true ]; then
    # Kill brutal (pas de sauvegarde graceful au shutdown)
    echo "[*] Kill backend (pas de sauvegarde état)..."
    docker compose kill backend || true
    docker compose down --timeout 5 || true

    # Suppression des fichiers state (PAS la DB)
    echo "[*] 🧹 State files cleaned (fresh start)"
    rm -f data/simulator_state.json data/executor_state.json
else
    # Graceful shutdown (SIGTERM → lifespan sauvegarde l'état)
    echo "[*] Arrêt propre des containers..."
    docker compose down --timeout 30 || true
    echo "[*] 📦 State files preserved"
fi

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
