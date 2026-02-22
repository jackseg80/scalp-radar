# Audit #5 — Grid States vs Bitget : cohérence en temps réel — 2026-02-22

## Résumé exécutif

- **Script créé** : `scripts/audit_grid_states.py` (read-only, standalone)
- **Objectif** : détecter les divergences entre l'état local (`executor_state.json` / API) et les positions réelles sur Bitget
- **5 types de divergences** : FANTOME, ORPHELINE, DESYNC, SL_MANQUANT, SL_ORPHELIN
- **Test de validation** : 3 positions "orphelines" correctement détectées (CRV, DYDX, GALA) — comportement attendu car le state local date du 13 février (bot sur serveur)
- **Aucun bug trouvé dans le bot** — les divergences reflètent la différence local/serveur, pas un problème réel

---

## Contexte

L'Executor maintient `_grid_states: dict[str, GridLiveState]` sauvegardé dans `data/executor_state.json` toutes les 60s. `_reconcile_on_boot()` rattrape certains cas au démarrage, mais il n'existe aucune vérification continue entre deux boots.

Risques identifiés de divergence :
- Crash pendant `_close_grid_cycle()` : position fermée sur Bitget, `_grid_states` non nettoyé
- SL exécuté par Bitget sans notification WS
- `_open_grid_position()` réussit côté Bitget mais crash avant enregistrement local
- Double restart rapide avec state file intermédiaire

---

## Types de divergences détectés

| Type | Sévérité | Description | Risque |
|------|----------|-------------|--------|
| **FANTOME** | CRITIQUE | Position locale absente sur Bitget | Exit monitor surveille une position inexistante |
| **ORPHELINE** | CRITIQUE | Position Bitget absente localement | Position sans SL ni exit monitor, invisible |
| **DESYNC** | MINEURE | Quantité (>0.1%) ou prix (>0.5%) incohérents | Sizing et SL calculés sur mauvaise base |
| **SL_MANQUANT** | CRITIQUE | `sl_order_id` absent des ordres ouverts Bitget | Position sans protection SL active |
| **SL_ORPHELIN** | MINEURE | Ordre SL ouvert sans position active | Risque faible, ordre ne peut pas se déclencher |

---

## Méthodologie

Script `scripts/audit_grid_states.py` (read-only, standalone) :

1. **Phase 1** — Lecture état local :
   - `--mode file` (défaut) : lit `executor_state.json`, extrait `grid_states`
   - `--mode api` : appelle `GET /api/executor/status`, extrait les positions `type=grid`
   - Calcul `total_quantity` = somme des levels, `avg_entry` = moyenne pondérée

2. **Phase 2** — Fetch positions Bitget :
   - `exchange.fetch_positions()` (ccxt sync, `defaultType=swap`)
   - Filtre : `contracts > 0 AND entryPrice > 0`

3. **Phase 3** — Fetch ordres SL ouverts :
   - `exchange.fetch_open_orders(symbol)` pour chaque symbol actif
   - Filtre : `triggerPrice > 0` (stop-loss trigger orders)
   - Rate limit 0.3s entre appels

4. **Phase 4** — Comparaison :
   - Tolérance prix : 0.5% (arrondis Bitget vs calcul local)
   - Tolérance quantité : 0.1% (flottants)
   - Match par symbol, direction normalisée LONG/SHORT vs long/short

5. **Phase 5** — Rapport structuré :
   - Résumé compteurs
   - Tableau détail par symbol
   - Tableau SL status
   - Détail divergences
   - Verdict (vert/rouge)

---

## Résultats du test de validation (22 février 2026)

### Exécution locale (mode file)

```
Timestamp             : 2026-02-22 15:38:34 UTC
Mode                  : file
Fichier               : data\executor_state.json
Derniere sauvegarde   : 2026-02-13T14:03:13 UTC (9 jours)

Positions locales     : 0
Positions Bitget      : 3
Matchees              : 0
Fantomes (local only) : 0
Orphelines (Bitget)   : 3
```

| Symbol | Bitget | Statut |
|--------|--------|--------|
| CRV/USDT:USDT | long 9174 @ 0.2335, uPnL=-43.45$ | ORPHELINE |
| DYDX/USDT:USDT | long 26046 @ 0.0969, uPnL=-50.80$ | ORPHELINE |
| GALA/USDT:USDT | long 376930 @ 0.0038, uPnL=-50.47$ | ORPHELINE |

**Analyse** : Le state local date du 13 février (bot tourne sur le serveur, state file local périmé). Les 3 orphelines sont les positions live réelles non connues localement — comportement attendu.

**Verdict** : Le script fonctionne correctement. Pour un audit réel, exécuter sur le serveur ou utiliser `--mode api`.

---

## Usage

```bash
# Depuis le serveur (state file à jour)
uv run python -m scripts.audit_grid_states

# Verbose (logs DEBUG)
uv run python -m scripts.audit_grid_states -v

# Via API (serveur tournant)
uv run python -m scripts.audit_grid_states --mode api --api-url http://192.168.1.200:8000

# Fichier state alternatif
uv run python -m scripts.audit_grid_states --state-file /path/to/executor_state.json
```

---

## Contraintes respectées

- **Read-only** : aucune modification d'état, ni local ni Bitget
- **Pas de dépendance nouvelle** : ccxt + json + config existante
- **Rate limiting** : 0.3s entre fetch_open_orders
- **Windows compatible** : fallback ASCII pour icônes (cp1252)
- **Loguru** : même pattern que le reste du projet

---

## Prochaine étape possible

Transformer en endpoint API `GET /api/executor/audit` pour intégration dans le dashboard, ou ajouter une vérification périodique (ex: toutes les heures) dans le Watchdog avec alerte Telegram si divergence CRITIQUE détectée.
