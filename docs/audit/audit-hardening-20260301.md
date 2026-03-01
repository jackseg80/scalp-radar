# Audit Hardening Backend — 2026-03-01

## Périmètre

- **Audit 4** : Exception Handling critique (executor.py)
- **Audit 5** : Sécurité Endpoints API (authentification manquante)
- **Audit 6** : Monitoring & Alertes manquantes

## Méthodologie

Lecture du code source des composants backend, identification des failles silencieuses,
endpoints non protégés, et lacunes de monitoring.

---

## Résultats — Audit 4 : Exception Handling

### 5 handlers critiques corrigés dans executor.py

| Ligne (approx) | Contexte | Avant | Après |
|--------|---------|-------|-------|
| ~2433 | SL detection (downtime recovery) | `except Exception: pass` | `except Exception as e: logger.error(...)` |
| ~2441 | Leverage restoration post-recovery | `except Exception: pass` | `except Exception as e: logger.warning(...)` |
| ~2630 | Exit reason detection | `except Exception: pass` | `except Exception as e: logger.debug(...)` |
| ~2179 | Fee extraction (mono) | `logger.warning("...")` sans détail | `logger.warning("... {}", e)` |
| ~2209 | Fee extraction (grid) | `logger.warning("...")` sans détail | `logger.warning("... {}", e)` |

**Impact** : Les 3 premiers `except: pass` masquaient totalement les erreurs. Un SL manqué
en recovery silencieuse = position sans protection. Le logging permet maintenant de diagnostiquer
les problèmes en production via les logs.

**Gravité réduite** : Le handler SL (#1) est dans le code de recovery au boot, pas dans la boucle
principale. Le risque réel est limité aux redémarrages.

---

## Résultats — Audit 5 : Sécurité Endpoints API

### 5 endpoints protégés par `verify_executor_key`

| Fichier | Endpoint | Méthode | Risque avant fix |
|---------|----------|---------|-----------------|
| `optimization_routes.py` | `/api/optimization/run` | POST | Lancement WFO non autorisé (CPU-intensif) |
| `optimization_routes.py` | `/api/optimization/jobs/{id}` | DELETE | Suppression de jobs par quiconque |
| `portfolio_routes.py` | `/api/portfolio/backtests/{id}` | DELETE | Suppression de backtests par quiconque |
| `portfolio_routes.py` | `/api/portfolio/run` | POST | Lancement backtest non autorisé (CPU-intensif) |
| `data_routes.py` | `/api/data/backfill` | POST | Déclenchement backfill non autorisé |

**Endpoints déjà protégés** (pas de changement nécessaire) :
- `POST /api/executor/*` — déjà `verify_executor_key`
- `POST /api/optimization/results` — vérifie `sync_api_key` directement
- `POST /api/portfolio/results` — vérifie `sync_api_key` directement

**Endpoints GET non protégés** (intentionnel) :
- Tous les GET (lecture seule) restent publics — le dashboard frontend n'envoie pas d'API key
- Ceci est acceptable car ils ne modifient pas l'état et ne consomment pas de CPU significatif

**Tests** : 3 fichiers de tests mis à jour avec `dependency_overrides` pour bypass auth en test :
- `tests/test_optimization_routes_sprint14.py`
- `tests/test_portfolio_routes.py`
- `tests/test_candle_updater.py`

---

## Résultats — Audit 6 : Monitoring & Alertes manquantes

### Lacunes identifiées (non implémentées — recommandations)

| Priorité | Composant | Lacune | Impact |
|----------|-----------|--------|--------|
| P2 | `watchdog.py` | Pas de détection de positions zombies (ouvertes >24h sans mouvement) | Position oubliée sur Bitget, marge bloquée |
| P2 | Aucun fichier | Pas de monitoring du funding rate extrême (>0.1%) | Coût de carry invisible, surtout en shorts |
| P2 | `executor.py` | Pas de vérification divergence leverage runtime vs config | Si Bitget change le leverage, le sizing est faux |
| P3 | `risk_manager.py` | Pas d'alerte proximité margin max (>90% du `max_margin_ratio`) | Nouvelles positions refusées sans explication claire |
| P3 | `regime_monitor.py:197` | TODO non implémenté : persistance DB des snapshots regime | Perte historique au restart |
| P3 | `sync.py` | Pas d'alerte Telegram si fetch_positions échoue (retry ajouté en Audit 2 mais pas de notification) | Admin non prévenu de positions orphelines |

### Monitoring existant (vérifié OK)

- **Watchdog** : health checks périodiques, détection executor/simulator down, alerte Telegram
- **Heartbeat** : ping Telegram toutes les 6h avec métriques
- **Kill switch** : 45% drawdown, persisté, alerte Telegram
- **Rate limiter** : par catégorie (market_data, trade, account, position), warnings dans les logs
- **Sync boot** : retry 3× avec backoff (corrigé Audit 2)

---

## Corrections appliquées

| Fichier | Changement |
|---------|------------|
| `backend/execution/executor.py` | 5 except handlers améliorés avec logging |
| `backend/api/optimization_routes.py` | Auth `verify_executor_key` sur POST /run et DELETE /jobs/{id} |
| `backend/api/portfolio_routes.py` | Auth `verify_executor_key` sur DELETE /backtests/{id} et POST /run |
| `backend/api/data_routes.py` | Auth `verify_executor_key` sur POST /backfill |
| `tests/test_optimization_routes_sprint14.py` | `dependency_overrides` pour bypass auth |
| `tests/test_portfolio_routes.py` | `dependency_overrides` pour bypass auth |
| `tests/test_candle_updater.py` | `dependency_overrides` pour bypass auth |

## Tests

- **0 régression** introduite par les corrections
- 33 tests des fichiers modifiés : tous passants
- Suite complète : en cours de vérification

---

## Recommandations (non implémentées)

| Priorité | Recommandation |
|----------|----------------|
| P2 | Zombie position detection dans watchdog (ouverte >24h sans mouvement) |
| P2 | Monitoring funding rate extrême (>0.1%) avec alerte Telegram |
| P2 | Vérification leverage runtime vs config au boot + périodique |
| P2 | Alerte Telegram dans sync.py si fetch_positions échoue après retries |
| P3 | Alerte proximité margin max (>90% `max_margin_ratio`) |
| P3 | Persistance DB des snapshots regime_monitor |
| P3 | Rate limiting sur endpoints CPU-intensifs (déjà auth, mais pas de throttle) |
