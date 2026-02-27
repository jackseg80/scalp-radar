# Sprint 56 — Protection Partial Fill Close Grid

**Date** : 27 février 2026
**Durée** : 1 session
**Priorité** : P0 — Bug critique vécu en production

---

## Problème

Lors d'un `tp_close` grid_multi_tf ETH SHORT :
- 3 entries : 0.03 + 0.03 + 0.01 = `0.06999...` (floating point Python)
- Market close envoyé pour `0.06999...` → Bitget arrondit et fill à `0.06`
- Bot compare `filled` vs `amount` sans tolérance → croit tout fermé
- `grid_state` vidé → **0.01 résiduel sans SL, sans exit monitor**
- Position orpheline pendant des heures

## Root Causes

1. **Floating point** : `state.total_quantity` = `0.03+0.03+0.01` = `0.06999...` jamais arrondi avant l'envoi à l'exchange
2. **Absence de vérification fill** : `_close_grid_cycle()` ne lit pas `close_order["filled"]`
3. **Absence de filet post-close** : pas de `fetch_positions` pour valider que la position est bien fermée

## Solution implementée

### Niveau 0 — Fix floating point à la source
`_close_grid_cycle()` arrondit désormais `total_quantity` via `_round_quantity()` avant l'envoi du market close order. Corrige le root cause principal.

### Niveau 1 — Vérification fill après market close
Nouveau helper `_handle_partial_close_fill(futures_sym, close_side, requested_qty, filled_qty, strategy_name)` :
- Compare `residual_raw = abs(requested - filled)` avec `min_qty` (via `self._markets[sym]["limits"]["amount"]["min"]`)
- Si `residual_raw < min_qty` → ignoré (floating point négligeable)
- Sinon → `_round_quantity(residual_raw)` puis 2ème market order `reduceOnly=True`
- Si 2ème order échoue → log CRITICAL + notify `AnomalyType.PARTIAL_FILL`
- Toujours : `notify_anomaly(PARTIAL_FILL)` si résidu détecté

### Niveau 2 — Filet de sécurité post-close
Nouveau helper `_verify_no_residual_position(futures_sym, close_side, strategy_name)` :
- `await asyncio.sleep(1.5)` pour laisser Bitget propager le fill
- `fetch_positions([futures_sym])` via `_fetch_positions_safe()`
- Si `contracts > 0` → market close immédiat + `notify_anomaly(PARTIAL_FILL)`
- Erreur réseau → log warning, pas de crash (graceful degradation)

### Chemins protégés
- `_close_grid_cycle()` : niveau 0 + niveau 1 (après market close) + niveau 2 (après cleanup)
- `_handle_grid_sl_executed()` : niveau 2 uniquement (SL Bitget, pas de filled qty disponible)

### AnomalyType.PARTIAL_FILL
Ajouté dans `notifier.py` avec cooldown 60s (critique, chaque occurrence compte).

## Fichiers modifiés

- `backend/alerts/notifier.py` : `PARTIAL_FILL` dans enum + message + cooldown
- `backend/execution/executor.py` :
  - Import `AnomalyType`
  - Section "Partial fill protection" : 3 helpers (`_get_min_quantity`, `_handle_partial_close_fill`, `_verify_no_residual_position`)
  - `_close_grid_cycle()` : `close_qty = _round_quantity(total_quantity)` + appels aux 2 niveaux
  - `_handle_grid_sl_executed()` : appel niveau 2

## Tests (16 nouveaux — `tests/test_executor_partial_fill.py`)

| Classe | Tests |
|--------|-------|
| `TestPartialCloseFillRetry` | retry envoyé si résidu ; retry échoue → anomaly quand même |
| `TestFullFillNoRetry` | full fill → pas de retry |
| `TestFloatingPointTolerance` | résidu < min_qty ignoré ; résidu == min_qty → retry ; résidu > min_qty → retry |
| `TestPostClosePositionCheck` | pas de résidu → OK ; résidu détecté → cleanup ; fetch_positions échoue gracefully ; 0 contracts → OK |
| `TestCloseGridCyclePartialFill` | intégration close cycle partial fill ; intégration full fill |
| `TestHandleGridSlPartialFill` | SL avec résidu → cleanup ; SL propre → pas de cleanup |
| `TestGetMinQuantity` | retourne min depuis markets ; 0 si symbol inconnu |

**Résultat** : 16/16 passent, 2058 tests total, 2054 passants, 0 régression.

## Notes techniques

- Le mock `amount_to_precision` doit utiliser `round()` (pas `int()`) pour simuler ccxt correctement — `int(0.06999 * 100) = 6` vs `round(0.06999, 2) * 100 = 7`
- `_round_quantity()` enforces `max(min_amount, rounded)` — utiliser `residual_raw < min_qty` AVANT d'appeler `_round_quantity()` pour éviter les faux positifs sur résidu nul
