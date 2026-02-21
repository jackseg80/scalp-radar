# Sprint Auto-Update Candles + Hotfix Nettoyage Timeframes

## Date : 21 février 2026

## Sprint Auto-Update Candles

### Objectifs
1. Tâche automatique quotidienne côté serveur — fetch Binance (max historique) + Bitget (90 jours) pour tous les assets
2. Endpoint API `POST /api/data/backfill` pour déclenchement manuel
3. Bouton frontend dans l'OverviewPage pour voir l'état des données et déclencher un backfill
4. Progression en temps réel via WebSocket

### Fichiers créés
- `backend/core/candle_updater.py` — CandleUpdater class (boucle 03:00 UTC, backfill, progression WS)
- `frontend/src/components/CandleStatus.jsx` + `.css` — tableau + bouton + barre progression
- `tests/test_candle_updater.py` — 8 tests

### Fichiers modifiés
- `backend/core/database.py` — +`get_candle_stats()`
- `backend/api/data_routes.py` — +`GET /api/data/candle-status` + `POST /api/data/backfill`
- `backend/api/server.py` — CandleUpdater dans lifespan
- `frontend/src/components/OverviewPage.jsx` — +CollapsibleCard CandleStatus

### Tests : 8 nouveaux → 1578 passants

---

## Hotfix Nettoyage Timeframes

### Objectifs
- Supprimer 1m de tous les assets (inutilisé)
- Ajouter 4h et 1d partout (analyse multi-TF)
- Top 6 (BTC, ETH, SOL, DOGE, LINK, XRP) gardent 5m+15m

### Fichiers modifiés
- `config/assets.yaml` — timeframes mis à jour pour 21 assets
- `backend/core/candle_updater.py` — utilise `asset.timeframes` au lieu de `["1h"]` hardcodé
- `backend/core/config.py` — +`backfill_enabled` dans SecretsConfig
- `tests/test_config_assets.py` — +4 tests timeframes

### Impact
- DataEngine WS : 84 flux → 75 flux
- `BACKFILL_ENABLED=false` désactive le cron, garde l'endpoint POST manuel

### Tests : 4 nouveaux → 1582 passants, 0 régression
