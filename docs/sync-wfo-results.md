# Sync WFO Results — Documentation Technique

## Vue d'ensemble

Système de synchronisation unidirectionnelle des résultats d'optimisation WFO depuis l'environnement de développement local (Windows) vers le serveur de production (Linux, 192.168.1.200).

**Principe** : Le local reste maître des backtests/WFO. Le serveur reçoit les résultats passivement via POST API pour consultation depuis le dashboard.

---

## Architecture

### Flux automatique (optimize.py)

```
optimize.py
    └─> report.py::save_report()
        ├─> JSON local (backup)
        ├─> DB locale (SQLite)
        └─> push_to_server() [best-effort, ne crashe JAMAIS]
            └─> POST http://192.168.1.200:8000/api/optimization/results
                └─> serveur : save_result_from_payload_sync()
```

### Flux manuel (historique existant)

```
uv run python -m scripts.sync_to_server [--dry-run]
    ├─> Lit toutes les rows de la DB locale
    └─> POST chaque résultat vers le serveur (INSERT OR IGNORE)
```

---

## Configuration

### Local (Windows — `.env`)

```env
# Sync WFO vers le serveur
SYNC_ENABLED=true
SYNC_SERVER_URL=http://192.168.1.200:8000
SYNC_API_KEY=<secret-partagé>
```

### Serveur (Linux — `.env`)

```env
# Clé pour recevoir les résultats WFO
SYNC_API_KEY=<même-secret-partagé>
```

**Génération de la clé** (à faire une seule fois) :

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copier la même clé dans les deux `.env` (local + serveur).

---

## Schéma DB : colonne `source`

Ajoutée à la table `optimization_results` :

```sql
ALTER TABLE optimization_results ADD COLUMN source TEXT DEFAULT 'local'
```

**Valeurs** :
- `'local'` : résultat poussé depuis l'environnement de dev
- `'server'` : résultat généré directement sur le serveur (si jamais optimize.py tourne là-bas)

**Migration** : idempotente, appliquée automatiquement au boot via `database.py::_migrate_optimization_source()`.

---

## Endpoint POST

**URL** : `POST /api/optimization/results`

**Auth** : Header `X-API-Key: <secret>`

**Payload** : JSON avec tous les champs de `optimization_results` (voir exemple ci-dessous).

**Codes retour** :
- **201 Created** : résultat inséré avec succès
- **200 OK** : résultat déjà existant (doublon UNIQUE constraint `strategy_name, asset, timeframe, created_at`)
- **401 Unauthorized** : clé API manquante, invalide, ou `sync_api_key` vide côté serveur
- **422 Unprocessable Entity** : champs obligatoires manquants

**Champs obligatoires** (NOT NULL en DB) :
- `strategy_name`, `asset`, `timeframe`, `created_at`
- `grade`, `total_score`, `n_windows`, `best_params`

**Exemple payload** :

```json
{
  "strategy_name": "vwap_rsi",
  "asset": "BTC/USDT",
  "timeframe": "5m",
  "created_at": "2026-02-13T12:35:00",
  "duration_seconds": 120.5,
  "grade": "A",
  "total_score": 87.0,
  "oos_sharpe": 1.8,
  "consistency": 0.85,
  "oos_is_ratio": 0.92,
  "dsr": 0.95,
  "param_stability": 0.88,
  "monte_carlo_pvalue": 0.02,
  "mc_underpowered": 0,
  "n_windows": 20,
  "n_distinct_combos": 600,
  "best_params": "{\"rsi_period\": 14}",
  "wfo_windows": "{\"windows\": [...]}",
  "monte_carlo_summary": "{\"p_value\": 0.02, \"significant\": true}",
  "validation_summary": "{\"bitget_sharpe\": 1.5, \"transfer_ratio\": 0.85}",
  "warnings": "[\"Test warning\"]",
  "source": "local"
}
```

---

## Transaction sûre (`save_result_from_payload_sync`)

**Problème à éviter** : si on fait `UPDATE is_latest=0` AVANT `INSERT OR IGNORE`, un doublon perd le flag sans insérer la nouvelle row.

**Solution** : INSERT d'abord, UPDATE seulement si inséré.

```python
BEGIN TRANSACTION
  1. INSERT OR IGNORE INTO optimization_results (...) VALUES (...)
  2. IF rowcount == 0:
       COMMIT  # Doublon, ne rien toucher
       RETURN "exists"
  3. ELSE:
       new_id = cursor.lastrowid
       UPDATE optimization_results SET is_latest=0
       WHERE strategy=? AND asset=? AND timeframe=? AND is_latest=1 AND id != new_id
       COMMIT
       RETURN "created"
```

Ainsi, si le résultat existe déjà (UNIQUE constraint match), `rowcount=0` et le flag `is_latest` de l'ancien reste intact.

---

## Fonctions principales

### `optimization_db.py` (4 nouvelles fonctions)

| Fonction | Rôle |
|----------|------|
| `save_result_from_payload_sync(db_path, payload)` | Insère un payload JSON brut depuis POST (transaction sûre) |
| `build_push_payload(report, wfo_windows, duration, timeframe)` | Construit le payload JSON depuis un FinalReport |
| `build_payload_from_db_row(row)` | Construit le payload depuis une row DB (pour sync_to_server.py) |
| `push_to_server(report, wfo_windows, duration, timeframe)` | Pousse vers le serveur (best-effort, ne crashe jamais) |

### `report.py` (modification)

```python
def save_report(...):
    # 1. JSON (existant)
    filepath = _save_json(report, output_dir)
    # 2. DB locale (existant)
    save_result_sync(db_path, report, wfo_windows, duration, timeframe)
    # 3. Push serveur (NOUVEAU — best-effort)
    push_to_server(report, wfo_windows, duration, timeframe)
    return filepath
```

---

## Script `sync_to_server.py`

**Usage** :

```bash
# Dry-run : affiche ce qui serait envoyé sans envoyer
uv run python -m scripts.sync_to_server --dry-run

# Envoi réel
uv run python -m scripts.sync_to_server
```

**Comportement** :

1. Lit toutes les rows de `optimization_results` en DB locale
2. Pour chaque row : POST vers le serveur avec `build_payload_from_db_row(row)`
3. Log le résultat (created / already_exists / error)
4. Continue même si un POST échoue (resilience)
5. Récap final : X créés, Y déjà existants, Z erreurs

**Idempotent** : relanceable à volonté (INSERT OR IGNORE côté serveur).

---

## Sécurité

### Authentification

- **Clé API partagée** (local ↔ serveur), passée dans header `X-API-Key`
- Le serveur refuse les POST si `sync_api_key` vide dans sa config (= sync désactivée)
- Pas de JWT/OAuth : LAN privé (192.168.1.x), clé simple suffit

### Portée

- L'endpoint POST fait **uniquement INSERT** (jamais DELETE/UPDATE des résultats existants)
- Pas d'exposition publique : CORS limité à localhost:5173 (frontend dev)
- Production : Firewall + Reverse Proxy doivent limiter l'accès

### Risques couverts

| Risque | Mitigation |
|--------|------------|
| Clé API leak dans les logs | Ne jamais logger la clé, seulement "auth OK/KO" |
| Serveur injoignable | try/except + log WARNING, run local continue |
| Transaction race condition | INSERT OR IGNORE + rowcount check avant UPDATE |
| Payload trop gros | ~5KB max (30 fenêtres WFO), aucun problème |
| DB locked pendant migration | WAL mode + migration au boot (pas concurrent) |

---

## Tests

**555 tests** (533 existants + 22 nouveaux) :

### `test_optimization_db.py` (+7 tests)

- `test_save_result_from_payload_sync_created` : insertion OK
- `test_save_result_from_payload_sync_duplicate` : doublon → "exists", `is_latest` intact
- `test_save_result_from_payload_sync_updates_is_latest` : nouveau run met à jour l'ancien
- `test_save_result_from_payload_sync_json_as_dict` : payload avec dicts/lists (pas que strings)
- `test_save_result_sync_with_source` : param `source` respecté
- `test_build_push_payload_structure` : tous champs requis, NaN sanitizés
- `test_build_payload_from_db_row` : conversion row DB → payload

### `test_optimization_routes.py` (+6 tests)

- `test_post_result_created` : 201 + "created"
- `test_post_result_duplicate` : 200 + "already_exists"
- `test_post_result_no_api_key` : 401
- `test_post_result_wrong_api_key` : 401 "invalide"
- `test_post_result_no_server_key` : 401 "non configuré"
- `test_post_result_invalid_payload` : 422 "champs manquants"

### `test_push_to_server.py` (+5 tests)

- `test_push_disabled` : `sync_enabled=false` → pas d'appel HTTP
- `test_push_success` : 201 → log info, pas d'exception
- `test_push_server_down` : `httpx.ConnectError` → log warning, pas d'exception
- `test_push_timeout` : `httpx.TimeoutException` → log warning, pas d'exception
- `test_push_empty_server_url` : URL vide → pas d'appel HTTP

### `test_sync_to_server.py` (+4 tests)

- `test_load_all_results` : charge toutes les rows
- `test_sync_dry_run` : --dry-run n'envoie rien
- `test_sync_sends_all_results` : N POST = N rows
- `test_sync_handles_errors` : erreur sur 1 résultat ne bloque pas les suivants

---

## Maintenance

### Ajouter un nouveau champ à `optimization_results`

1. Ajouter la colonne en DB (schema + migration ALTER TABLE)
2. Mettre à jour `save_result_sync()` : ajouter le champ dans l'INSERT
3. Mettre à jour `build_push_payload()` : ajouter le champ dans le dict retourné
4. Mettre à jour `save_result_from_payload_sync()` : ajouter le champ dans l'INSERT
5. Si NOT NULL : ajouter à `_REQUIRED_FIELDS` dans `optimization_routes.py`
6. Mettre à jour les fixtures de tests (`temp_db` dans `test_optimization_db.py`)
7. Lancer les tests : `uv run python -m pytest`

### Changer l'URL du serveur

Modifier `SYNC_SERVER_URL` dans le `.env` local, relancer optimize.py (pas besoin de redémarrer le serveur).

### Désactiver temporairement le push

```env
SYNC_ENABLED=false
```

Ou commenter la ligne dans `.env`. Les résultats restent en DB locale.

---

## Dépannage

### "Sync non configuré sur ce serveur" (401)

Le serveur n'a pas de `SYNC_API_KEY` dans son `.env`. Ajouter la clé et redémarrer le serveur.

### "Clé API invalide" (401)

Les clés local/serveur ne correspondent pas. Vérifier que le secret est identique dans les 2 `.env`.

### "Push serveur échoué (réseau)" dans les logs

Le serveur est injoignable (down, firewall, mauvaise URL). Vérifier :
- `SYNC_SERVER_URL` correcte (http://192.168.1.200:8000, pas de slash final)
- Le serveur tourne (`curl http://192.168.1.200:8000/health`)
- Firewall/VPN ne bloque pas

Le run local continue normalement, lancer `sync_to_server.py` plus tard pour rattraper.

### Doublons refusés alors qu'ils semblent différents

UNIQUE constraint sur `(strategy_name, asset, timeframe, created_at)`. Si 2 runs ont exactement le même timestamp ISO 8601, le 2ème sera refusé. Peu probable (<1s de résolution), mais possible si relance immédiate. Pas un problème : le résultat est déjà en DB.

---

## Hors scope

- **Sync bidirectionnelle** (serveur → local) : jamais implémenté, le local reste maître
- **DELETE/UPDATE via API** : pas d'endpoint pour modifier/supprimer via le réseau (sécurité)
- **Retry automatique avec queue** : best-effort suffit, `sync_to_server.py` rattrape
- **Compression payload** : ~5KB, inutile
- **HTTPS** : LAN privé 192.168.1.x, pas de TLS
- **Authentification JWT/OAuth** : overkill pour un LAN, clé simple suffit
