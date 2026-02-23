# Hotfix DATA_STALE — Guard prix stale dans l'exit monitor

**Date :** 23 février 2026
**Priorité :** P0 (corrige comportement live incorrect)
**Tests :** +4 (1753 total)

---

## Contexte

Le WebSocket Bitget de certains symbols (XRP, BCH, BNB, AAVE, OP) tombe périodiquement (~5 min) à cause du rate-limiting. Pendant ce silence, le buffer DataEngine (`_buffers[symbol][tf][-1]`) conserve sa dernière candle reçue, dont le prix devient stale (5-15 min de retard).

L'exit monitor de l'Executor (`_check_grid_exit`) lisait ce prix stale sans vérification → les conditions TP/SL n'étaient pas évaluées correctement pendant les silences WS.

---

## Fixes

### P0 — executor.py : Guard prix stale avec fallback REST

**Fichier :** `backend/execution/executor.py` — méthode `_check_grid_exit()` (bloc ~ligne 831)

**Logique :**
1. Lire la candle la plus récente du buffer (1m → 5m → 15m → 1h)
2. Calculer l'âge : `now_utc - candle.timestamp`
3. Si âge ≤ 120s → utiliser le prix normalement
4. Si âge > 120s (stale) :
   - Log WARNING
   - Appeler `self._exchange.fetch_ticker(futures_sym)` via REST
   - Si ticker retourne un prix valide → utiliser ce prix pour le check TP/SL
   - Si fetch_ticker échoue (exception) → log ERROR + **return** (skip l'exit)
   - Si `_exchange is None` → log ERROR + **return** (skip l'exit)

**Principe :** ne rien faire plutôt qu'agir sur un prix faux.

### P1 — data_engine.py : Seuil restart stale 600s → 300s

**Fichier :** `backend/core/data_engine.py` — heartbeat loop (stale per-symbol auto-recovery)

Avant : relance la task watch_ d'un symbol seulement après 10 min de silence
Après : relance après **5 min** (aligné avec le seuil de détection stale à 300s)

### P2 — data_engine.py : Batching WS moins agressif

**Fichier :** `backend/core/data_engine.py` — constantes de classe

| Constante | Avant | Après |
|-----------|-------|-------|
| `_SUBSCRIBE_BATCH_SIZE` | 5 | 3 |
| `_SUBSCRIBE_BATCH_DELAY` | 2.0s | 3.0s |

Impact : au boot, 21 assets → 7 batchs × 3s = ~21s (vs ~8s avant). Réduit la pression rate-limit Bitget (code 30006) pendant la phase de souscription initiale.

---

## Tests ajoutés

**Fichier :** `tests/test_executor_autonomous.py` (classe `TestExitAutonomous`)

| Test | Scénario |
|------|----------|
| `test_stale_price_fallback_to_rest` | Candle stale 200s → fetch_ticker retourne 106 > TP 105 → exit tp_global avec exit_price=106 |
| `test_stale_price_rest_fails_skip_exit` | Candle stale → fetch_ticker lève Exception → `_close_grid_cycle` pas appelé |
| `test_stale_price_no_exchange_skip_exit` | Candle stale + `_exchange=None` → `_close_grid_cycle` pas appelé |
| `test_fresh_price_no_ticker_call` | Candle fraîche 30s → `fetch_ticker` pas appelé (régression) |

**Corrections tests existants :** `test_exit_intracandle_tp` et `test_exit_no_false_tp_with_correct_sma` — ajout de `candle_mock.timestamp = datetime.now(...) - timedelta(seconds=30)` pour que les candles fraîches passent le nouveau guard.

---

## Fichiers modifiés

- `backend/execution/executor.py` — guard stale + fallback REST
- `backend/core/data_engine.py` — seuil 300s + batch 3/3s
- `tests/test_executor_autonomous.py` — 4 nouveaux tests + 2 tests corrigés
- `docs/ROADMAP.md` — état actuel mis à jour (1753 tests, 23 fév 2026)
