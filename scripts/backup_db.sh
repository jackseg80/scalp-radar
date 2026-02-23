#!/bin/bash
# Backup de la base de données SQLite — Scalp Radar
# Usage : bash scripts/backup_db.sh
# Compatible Linux (prod + cron) et Windows Git Bash / WSL

set -euo pipefail

DB_PATH="${DB_PATH:-data/scalp_radar.db}"
BACKUP_DIR="${BACKUP_DIR:-data/backups}"
RETENTION_DAYS=7
DATE=$(date +%Y%m%d)
BACKUP_FILE="${BACKUP_DIR}/scalp_radar_${DATE}.db"

_log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Créer le dossier backups si absent
mkdir -p "$BACKUP_DIR"

# Vérifier que la DB source existe
if [ ! -f "$DB_PATH" ]; then
    _log "ERROR: DB introuvable : $DB_PATH"
    exit 1
fi

# Copier (écrase si backup du jour déjà existant)
cp "$DB_PATH" "$BACKUP_FILE"
SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
_log "INFO: Backup OK → $BACKUP_FILE ($SIZE)"

# Supprimer les backups de plus de RETENTION_DAYS jours
DELETED=0
while IFS= read -r -d '' f; do
    rm -f "$f"
    _log "INFO: Ancien backup supprimé : $f"
    DELETED=$((DELETED + 1))
done < <(find "$BACKUP_DIR" -name "scalp_radar_*.db" -mtime +$RETENTION_DAYS -print0 2>/dev/null)

_log "INFO: Cleanup : $DELETED ancien(s) backup(s) supprimé(s)"
_log "INFO: Backups conservés :"
ls -lh "$BACKUP_DIR"/scalp_radar_*.db 2>/dev/null || _log "  (aucun)"
