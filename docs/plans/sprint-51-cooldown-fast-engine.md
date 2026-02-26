# Sprint 51 — Cooldown Fast Engine + Hotfix Fees Polling

## Contexte

Le cooldown anti-churn (`cooldown_candles`) empêche les ré-entrées rapides après fermeture d'une grid. Il est actif en live (Executor) et paper (GridStrategyRunner) avec `cooldown_candles=3` par défaut (3h sur 1h TF).

**Problème** : Le fast backtest engine (`fast_multi_backtest.py`) utilisé par WFO n'implémente le cooldown que pour `grid_momentum`. Les 8 autres stratégies grid (grid_atr, grid_boltrend, grid_multi_tf, grid_range_atr, grid_trend, envelope_dca, envelope_dca_short, grid_funding) émettent juste un warning. Conséquence : les paramètres WFO sont sélectionnés sur un backtest trop optimiste (ré-entrées instantanées), créant un biais systématique vs la réalité live.

**Hotfix #2** : Deux méthodes polling dans `executor.py` (`_check_position_still_open`, `_check_grid_still_open`) ne font pas d'extraction de fees réelles avant de calculer le P&L — elles utilisent `_calculate_pnl()` estimé au lieu de `_calculate_real_pnl()` avec fees Bitget. Le TODO Hotfix 34 est incomplet.

## Fichiers

| Fichier | Action |
|---------|--------|
| `backend/optimization/fast_multi_backtest.py` | Modifier — ajouter cooldown à `_simulate_grid_common` + `_simulate_grid_boltrend` + `_simulate_grid_range` |
| `backend/execution/executor.py` | Modifier — compléter extraction fees dans polling methods |
| `tests/test_cooldown_fast_engine.py` | Créer — ~15 tests (regression + parity + cooldown) |
| `tests/test_executor_real_pnl.py` | Modifier — +3 tests polling fees |
| `COMMANDS.md` | Inchangé |
| `docs/ROADMAP.md` | Mettre à jour |

---

## Partie A — Cooldown dans le Fast Engine

### Architecture actuelle

Le fast engine a 4 boucles de simulation distinctes :

1. **`_simulate_grid_common()`** (ligne 137) — Boucle chaude unifiée pour grid_atr, envelope_dca, grid_multi_tf, grid_trend. Signature : `(entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction, ...)`.
2. **`_simulate_grid_boltrend()`** (ligne 883) — State machine breakout Bollinger, propre boucle.
3. **`_simulate_grid_range()`** (ligne 478) — Bidirectional LONG+SHORT, propre boucle.
4. **`_simulate_grid_momentum()`** (ligne 1153) — **Cooldown déjà implémenté** via `last_exit_candle_idx`.

**Référence existante** (grid_momentum ligne 1207+1334) :
```python
last_exit_candle_idx = -999  # init
# ...
# À la sortie :
last_exit_candle_idx = i
# ...
# Avant entrée :
if cooldown_candles > 0 and (i - last_exit_candle_idx) < cooldown_candles:
    continue
```

**⚠️ Note** : grid_momentum utilise `continue` car sa boucle est structurée pour que les exits soient traités **avant** ce point. Pour les 3 autres simulateurs, on utilise un **flag `can_open_new`** car `continue` sauterait des sections critiques (exits, trailing, HWM).

### Étape A1 — `_simulate_grid_common()` : ajouter param `cooldown_candles`

**Signature** : ajouter `cooldown_candles: int = 0` à la fin.

**Init** (après ligne 184) :
```python
last_exit_candle_idx = -999  # cooldown tracking
```

**Record exit** (ligne 348, dans le bloc `if exit_reason is not None`, après `positions = []`) :
```python
positions = []
hwm = 0.0
first_entry_idx = -1
last_exit_candle_idx = i  # NOUVEAU — pour cooldown
```

**Idem** pour le force-close direction flip (ligne 224, bloc dynamique) :
```python
positions = []
hwm = 0.0
first_entry_idx = -1
last_exit_candle_idx = i  # NOUVEAU — pour cooldown
```

**Flag cooldown** (en début de boucle candle, après les variables de candle) :
```python
# Cooldown flag — bloque uniquement l'ouverture, pas les exits/trailing
can_open_new = not (
    cooldown_candles > 0
    and not positions
    and (i - last_exit_candle_idx) < cooldown_candles
)
```

**Guard avant entrée** (ligne 375, dans `# 5. Ouvrir de nouvelles positions`) :
```python
# 5. Ouvrir de nouvelles positions si niveaux touchés
if can_open_new and len(positions) < num_levels:
    # ... reste du code inchangé (PAS de continue)
```

**IMPORTANT** : On utilise un **flag `can_open_new`** et non `continue`, car `continue` sauterait tout le reste de la boucle candle (exits, trailing stop, HWM updates). Le flag ne bloque que la section d'ouverture.

### Étape A2 — Propager `cooldown_candles` depuis les wrappers

**`_simulate_grid_atr()`** (ligne 1514) :
```python
return _simulate_grid_common(
    entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction,
    max_hold_candles=params.get("max_hold_candles", 0),
    min_profit_pct=params.get("min_profit_pct", 0.0),
    cooldown_candles=params.get("cooldown_candles", 0),  # NOUVEAU
)
```

**`_simulate_envelope_dca()`** (ligne 1495) :
```python
return _simulate_grid_common(
    entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction,
    cooldown_candles=params.get("cooldown_candles", 0),  # NOUVEAU
)
```

**`_simulate_grid_multi_tf()`** (ligne 1536) :
```python
return _simulate_grid_common(
    entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct,
    direction=1,
    directions=dir_arr,
    cooldown_candles=params.get("cooldown_candles", 0),  # NOUVEAU
)
```

**Note** : grid_trend utilise aussi `_simulate_grid_common` via les wrappers avec `trail_mult` — il passera par le même chemin.

### Étape A3 — `_simulate_grid_boltrend()` : cooldown natif

grid_boltrend a sa propre boucle (state machine breakout). Ajouter le cooldown identiquement :

**Param** : extraire `cooldown_candles = params.get("cooldown_candles", 0)`.

**Init** (après ligne 944) :
```python
last_exit_candle_idx = -999
```

**Record exit** (ligne 1024-1027, dans le bloc exit) :
```python
positions = []
entry_levels = []
direction = 0
breakout_candle_idx = -1
last_exit_candle_idx = i  # NOUVEAU — pour cooldown
```

**Flag cooldown** (en début de boucle candle, après les variables de candle) :
```python
# Cooldown flag — bloque uniquement le breakout detection, pas les exits
can_open_new = not (
    cooldown_candles > 0
    and direction == 0
    and not positions
    and (i - last_exit_candle_idx) < cooldown_candles
)
```

**Guard avant breakout detection** (ligne 1074-1075) :
```python
# === 5. Breakout detection (si grid inactif) ===
if can_open_new and direction == 0 and not positions:
    # ... reste du breakout detection (PAS de continue)
```

**IMPORTANT** : Même pattern flag que `_simulate_grid_common` — pas de `continue` qui sauterait exits et mises à jour.

### Étape A4 — `_simulate_grid_range()` : cooldown per-side

grid_range_atr est bidirectionnel (LONG+SHORT simultanés avec TP/SL individuels). Le cooldown doit être **per-direction** : après avoir fermé toutes les positions LONG, cooldown avant d'en ouvrir de nouvelles LONG (mais SHORT reste possible).

C'est plus complexe. grid_range_atr ferme les positions **individuellement** (pas toutes d'un coup), donc le concept "grid fermée → cooldown" est différent. Ici, chaque position a son propre TP/SL.

**Approche simplifiée** : cooldown global quand **toutes** les positions sont fermées (aucune position ouverte → attendre cooldown avant de rouvrir). C'est cohérent avec le comportement live du GridStrategyRunner qui bloque toutes les entrées en cooldown, pas par direction.

**Param** : extraire `cooldown_candles = params.get("cooldown_candles", 0)`.

**Init** (après ligne 515) :
```python
last_exit_candle_idx = -999
```

**Record exit** : Dans la section #1 (Check TP/SL individuel, après close positions), ajouter après chaque close :
```python
# Quand la dernière position est fermée, noter l'index pour cooldown
if not remaining_positions:
    last_exit_candle_idx = i
```

Il faut identifier le point exact. grid_range ferme les positions une par une et reconstruit la liste. Cherchons ce code.

La section de fermeture (lignes 537+) ferme individuellement puis reconstruit la liste :
```python
positions = [p for idx, p in enumerate(positions) if idx not in closed_indices]
```
→ Juste après, checker :
```python
if not positions:
    last_exit_candle_idx = i
```

**Flag cooldown** (en début de boucle candle) :
```python
can_open_new = not (
    cooldown_candles > 0
    and not positions
    and (i - last_exit_candle_idx) < cooldown_candles
)
```

**Guard avant entrée** : Dans la section ouverture, wrapper le bloc d'entrée :
```python
if can_open_new:
    # --- 2. Check si nouveaux niveaux touchés ---
    # ... reste du code d'entrée (PAS de continue)
```

**Important** : On ne bloque que si `not positions` — si des positions existent déjà, le cooldown n'est pas actif (on est encore dans un cycle de trading). Pattern flag identique aux autres simulateurs.

### Étape A5 — Supprimer le warning obsolète

Ligne 1422-1430, remplacer le warning par la propagation :
```python
# Cooldown propagé aux simulateurs (Sprint 51)
# grid_momentum le gère nativement, les autres via cooldown_candles param
```

### Étape A6 — Tests `tests/test_cooldown_fast_engine.py`

**15 tests** couvrant la régression, la parité et le cooldown :

**TestCooldownGridCommon (5 tests) :**
1. `test_cooldown_blocks_reentry_grid_atr` — Avec cooldown=3, vérifier que le nombre de trades diminue vs cooldown=0
2. `test_cooldown_zero_no_effect` — cooldown=0 → même nombre de trades qu'avant (backward compat)
3. `test_cooldown_exits_then_waits` — Données synthétiques : trigger exit → vérifier que les N candles suivantes ne déclenchent pas de nouvelle entrée
4. `test_cooldown_grid_multi_tf_directions` — Cooldown respecté même quand direction change (force-close → cooldown)
5. `test_cooldown_envelope_dca` — Vérifier que envelope_dca reçoit et applique le cooldown

**TestCooldownGridBoltrend (3 tests) :**
6. `test_cooldown_blocks_breakout` — Après close, pas de nouveau breakout pendant cooldown
7. `test_cooldown_zero_allows_immediate` — cooldown=0 → breakout immédiat après close
8. `test_cooldown_counts_candles` — Vérifier que le cooldown compte bien en candles (pas en timestamps)

**TestCooldownGridRange (2 tests) :**
9. `test_cooldown_blocks_after_all_closed` — Cooldown actif quand toutes positions fermées
10. `test_cooldown_no_block_while_positions_open` — Positions encore ouvertes → pas de cooldown sur les nouvelles

**TestRegression (2 tests) :**
11. `test_cooldown_zero_bit_for_bit` — Run grid_atr avec cooldown=0 vs sans param cooldown → résultats **identiques** (return, trades, DD). Garantit que le code ajouté ne change rien quand cooldown désactivé.
12. `test_cooldown_zero_boltrend_bit_for_bit` — Idem pour grid_boltrend.

**TestParity (3 tests) :**
13. `test_warning_removed` — Plus de UserWarning émis quand cooldown > 0
14. `test_cooldown_reduces_trades_subset` — Run identique grid_atr avec cooldown=0 et cooldown=3 : vérifier trades_3 <= trades_0 **ET** que les timestamps des trades avec cooldown sont un sous-ensemble des timestamps sans cooldown (pas juste un comptage)
15. `test_cooldown_does_not_skip_exits` — Données synthétiques : position ouverte + cooldown actif → vérifier que le SL/TP se déclenche quand même (prouve que le flag ne bloque pas les exits)

---

## Partie B — Hotfix Fees Polling (Executor)

### Étape B1 — `_check_position_still_open()` (ligne 2114)

Avant l'appel à `_handle_exchange_close`, extraire les fees via `_fetch_fill_price` :

```python
if not has_open:
    logger.info(
        "Executor: position {} fermée côté exchange (détectée par polling)",
        symbol,
    )
    exit_price = await self._fetch_exit_price(symbol)
    exit_reason = await self._determine_exit_reason(symbol)

    # Hotfix 34 completion : extraire fees réelles
    exit_fee: float | None = None
    try:
        if pos.sl_order_id or pos.tp_order_id:
            order_id = pos.tp_order_id or pos.sl_order_id or ""
            _, exit_fee = await self._fetch_fill_price(order_id, symbol, exit_price)
    except Exception:
        logger.warning("Executor: fee extraction failed for %s, using estimate", symbol)
        exit_fee = None  # fallback estimé

    await self._handle_exchange_close(symbol, exit_price, exit_reason, exit_fee)
```

**Note** : On essaie de récupérer la fee via l'order_id du TP ou SL (le plus probable trigger quand l'exchange ferme). Si aucun order_id disponible, `exit_fee` reste None et le handler utilise `_calculate_pnl()` estimé (fallback existant).

### Étape B2 — `_check_grid_still_open()` (ligne 2139)

Même pattern :

```python
if not has_open:
    logger.info(
        "Executor: grid {} fermée côté exchange (détectée par polling)", symbol,
    )
    exit_price = await self._fetch_exit_price(symbol)

    # Hotfix 34 completion : extraire fees réelles
    exit_fee: float | None = None
    try:
        if state.sl_order_id:
            _, exit_fee = await self._fetch_fill_price(
                state.sl_order_id, symbol, exit_price,
            )
    except Exception:
        logger.warning("Executor: fee extraction failed for grid %s, using estimate", symbol)
        exit_fee = None  # fallback estimé

    await self._handle_grid_sl_executed(symbol, state, exit_price, exit_fee)
```

### Étape B3 — Tests `tests/test_executor_real_pnl.py`

Ajouter 3 tests à la classe existante :

1. `test_polling_position_extracts_fee` — Mock `_fetch_fill_price` retourne (price, 0.12) → vérifier que `_handle_exchange_close` reçoit `exit_fee=0.12`
2. `test_polling_grid_extracts_fee` — Mock `_fetch_fill_price` retourne (price, 0.08) → vérifier que `_handle_grid_sl_executed` reçoit `exit_fee=0.08`
3. `test_polling_fallback_no_fee` — Mock `_fetch_fill_price` retourne (price, None) → vérifier que handler reçoit `exit_fee=None` (fallback estimé)

---

## Ordre d'implémentation

1. `_simulate_grid_common()` — ajouter `cooldown_candles` param + tracking
2. Wrappers (`_simulate_grid_atr`, `_simulate_envelope_dca`, `_simulate_grid_multi_tf`) — propager le param
3. `_simulate_grid_boltrend()` — ajouter cooldown natif
4. `_simulate_grid_range()` — ajouter cooldown global
5. Supprimer le warning obsolète
6. Tests `test_cooldown_fast_engine.py` — 15 tests
7. `executor.py` — compléter les 2 méthodes polling
8. Tests `test_executor_real_pnl.py` — 3 tests
9. Run tests, 0 régression
10. ROADMAP.md, commit

## Vérification

```bash
# Tests Sprint 51
uv run pytest tests/test_cooldown_fast_engine.py -x -q
uv run pytest tests/test_executor_real_pnl.py -x -q

# Parity check : WFO grid_atr avec cooldown activé
uv run pytest tests/test_cooldown_churning.py -x -q

# 0 régression
uv run pytest tests/ -x -q

# Vérif fonctionnelle : optimizer grid_atr avec cooldown=3
# Comparer n_trades avant/après (devrait être <= avant)
```
