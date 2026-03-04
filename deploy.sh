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
echo "  Mode: $([ "$CLEAN" = true ] && echo "CLEAN (fresh start)" || echo "NORMAL (state preserved)")"
echo "========================================"

# 1. Préparation
mkdir -p data/backups logs

# 2. Sauvegarde de sécurité (DB)
if [ -f "data/scalp_radar.db" ]; then
    echo "[*] Sauvegarde de la base de données..."
    cp data/scalp_radar.db data/backups/scalp_radar_$(date +%Y%m%d_%H%M%S).db
    # Garder seulement les 5 dernières sauvegardes pour économiser l'espace
    ls -t data/backups/*.db 2>/dev/null | tail -n +6 | xargs -r rm -f || true
fi

# 3. Mise à jour du code
echo "[*] Reset config files..."
git checkout -- config/
echo "[*] Mise à jour du code..."
git pull origin main

# 4. Build (avec --pull pour les derniers patchs de sécurité)
echo "[*] Build des images..."
docker compose build --pull

# 5. Arrêt des containers
if [ "$CLEAN" = true ]; then
    echo "[*] Kill backend (mode CLEAN)..."
    docker compose kill backend || true
    docker compose down --timeout 5 || true
    echo "[*] 🧹 Nettoyage des fichiers d'état JSON"
    rm -f data/simulator_state.json data/executor_state.json
else
    echo "[*] Arrêt propre (timeout 30s)..."
    docker compose down --timeout 30 || true
fi

# 6. Lancement
echo "[*] Lancement..."
docker compose up -d

# 7. Health check avec boucle de retry et diagnostic
echo "[*] Vérification health..."
MAX_RETRIES=10
COUNT=0
HEALTH_URL="http://localhost:8000/health"

while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "[OK] Deploy réussi"
        echo "[OK] Backend  : http://localhost:8000"
        echo "[OK] Frontend : http://localhost"
        
        # Nettoyage des images Docker inutiles pour libérer de l'espace
        echo "[*] Nettoyage des images orphelines..."
        docker image prune -f
        exit 0
    fi
    COUNT=$((COUNT+1))
    echo "Attente de l'application... ($COUNT/$MAX_RETRIES)"
    sleep 3
done

# ECHEC : Diagnostic et Rollback
echo "========================================"
echo "[ERREUR] Health check échoué !"
echo "--- DERNIERS LOGS BACKEND ---"
docker compose logs --tail=50 backend
echo "-----------------------------"
echo "[*] Tentative de Rollback vers les images précédentes..."
docker compose down
docker compose up -d --no-build
echo "[AVERTISSEMENT] Rollback effectué."
echo "========================================"
exit 1
