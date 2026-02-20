# Plan : Sprint Time-Based Stop Loss (`max_hold_candles`)

## Contexte

Les stratégies `grid_atr` et `grid_boltrend` sont du mean-reversion. Sans rebond, les positions restent ouvertes indéfiniment jusqu'au SL global ou kill switch, accumulant du funding (toutes les 8h) et immobilisant du capital. `max_hold_candles` applique le principe de "demi-vie du mean-reversion" : si la position est en perte après X candles, l'hypothèse est invalidée → on coupe.

**Décision param_grids** : A/B test séparé — le paramètre est dans le code et la config, **pas** dans param_grids.yaml pour l'instant.

---

## Fichiers à modifier (5) + 1 à créer

| Fichier | Action |
|---------|--------|
| `backend/core/config.py` | +champ dans `GridATRConfig` et `GridBolTrendConfig` |
| `backend/strategies/grid_atr.py` | `should_close_all()` + `get_params()` |
| `backend/strategies/grid_boltrend.py` | `should_close_all()` + `get_params()` |
| `backend/optimization/fast_multi_backtest.py` | `_simulate_grid_common()` + `_simulate_grid_atr()` wrapper + `_simulate_grid_boltrend()` |
| `tests/test_time_stop.py` | NOUVEAU — ~27 tests |

---

## Étape 1 — Config (`backend/core/config.py`)

### `GridATRConfig` (ligne 247)
Après `weight: float = Field(default=0.20, ge=0, le=1)` :
```python
max_hold_candles: int = Field(default=0, ge=0)
```

### `GridBolTrendConfig` (ligne 374)
Idem, même position.

---

## Étape 2 — Stratégies (layer paper/live)

### Helpers partagés

Constante à définir en haut du fichier ou localement dans chaque méthode :
```python
_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
```

Calcul candles écoulées depuis la première entrée :
```python
if self._config.max_hold_candles > 0 and grid_state.positions:
    first_entry = min(p.entry_time for p in grid_state.positions)
    tf_secs = _TF_SECONDS.get(self._config.timeframe, 3600)
    delta = ctx.timestamp - first_entry
    candles_held = int(delta.total_seconds() / tf_secs)
    if candles_held >= self._config.max_hold_candles:
        close = indicators.get("close", float("nan"))
        direction = grid_state.positions[0].direction
        avg_e = grid_state.avg_entry_price
        total_qty = grid_state.total_quantity
        if direction == Direction.LONG:
            unrealized = (close - avg_e) * total_qty
        else:
            unrealized = (avg_e - close) * total_qty
        if unrealized < 0:
            return "time_stop"
```

### `grid_atr.py` — `should_close_all()` (ligne 158, avant `return None`)

Ordre actuel dans le code : **TP check (L143-147) → SL check (L149-156) → return None (L158)**

Insérer le bloc time_stop **après le check SL** (après ligne 156), juste avant `return None` (ligne 158). Le time_stop ne fire que si ni TP ni SL n'ont triggé — cohérent avec la priorité TP > SL > time_stop.

Référence : pattern identique à `grid_funding.py:132-138` pour le comptage depuis `min(p.entry_time for p in grid_state.positions)`.

### `grid_boltrend.py` — `should_close_all()` (entre SL et TP inverse)

Ordre actuel : SL check (ligne 212) → TP inverse/signal_exit (ligne 222) → return None

Insérer le bloc time_stop **après le check SL** (après ligne 219), **avant le check TP inverse** (avant ligne 221).

### `get_params()` dans les deux stratégies

Ajouter `"max_hold_candles": self._config.max_hold_candles` dans le dict retourné.

---

## Étape 3 — Fast engine `_simulate_grid_common()` (`fast_multi_backtest.py:131`)

### Signature
Ajouter `max_hold_candles: int = 0` aux paramètres.

### State tracking
Déclarer `first_entry_idx: int = -1` avec les variables d'état (après `hwm = 0.0`).

### Ouverture (section "Ouvrir de nouvelles positions", ligne ~345)
Après `positions.append(...)` :
```python
if len(positions) == 1:   # première position du cycle
    first_entry_idx = i
```

### Reset à la fermeture (section EXIT, après `positions = []` et avant `continue`)
```python
first_entry_idx = -1
```

### Check time_stop (dans la section EXIT, après SL/TP, avant `if exit_reason is not None`)
```python
if exit_reason is None and max_hold_candles > 0 and first_entry_idx >= 0:
    candles_held = i - first_entry_idx
    if candles_held >= max_hold_candles:
        if direction == 1:
            unrealized = (cache.closes[i] - avg_entry) * total_qty
        else:
            unrealized = (avg_entry - cache.closes[i]) * total_qty
        if unrealized < 0:
            exit_reason = "time_stop"
            exit_price = cache.closes[i]
```

Utiliser les variables `avg_entry` et `total_qty` déjà calculées en début du bloc `if positions:`.

**Fees** : `time_stop` = taker_fee + slippage (comme `sl_global`, fermeture marché forcée). Le bloc de dispatch fees existant déjà (`if exit_reason == "tp_global": fee=maker / else: fee=taker`) gère ça automatiquement.

### Wrapper `_simulate_grid_atr()` (ligne 1128)

Passer `max_hold_candles` à `_simulate_grid_common` :
```python
return _simulate_grid_common(
    entry_prices, sma_arr, cache, bt_config, num_levels, sl_pct, direction,
    max_hold_candles=params.get("max_hold_candles", 0),
)
```

**Note** : `_simulate_grid_multi_tf` et `_simulate_grid_trend` passent aussi par `_simulate_grid_common` mais avec `max_hold_candles=0` implicitement (défaut), donc aucune modification requise.

---

## Étape 4 — Fast engine `_simulate_grid_boltrend()` (ligne 815)

### State tracking
Déclarer `breakout_candle_idx: int = -1` avec `positions = []` (ligne 873).

### Set au breakout (section "Breakout detection", ligne ~1018)
Après `positions.append((0, ep0, qty, entry_fee))` :
```python
breakout_candle_idx = i
```

### Reset à la fermeture (après `positions = []`, avant `continue`)
```python
breakout_candle_idx = -1
```

### Check time_stop (après SL, avant TP inverse)

Le fast engine boltrend calcule `sl_hit` et `tp_hit` séparément puis résout les conflits en un bloc (lignes 908-924). Insérer le check time_stop **après la résolution** des conflits SL/TP (après ligne 924) mais **avant** `if exit_reason is not None:` (ligne 926). Le time_stop ne fire que si `exit_reason is None`.

```python
max_hold = params.get("max_hold_candles", 0)
if exit_reason is None and max_hold > 0 and breakout_candle_idx >= 0:
    candles_held = i - breakout_candle_idx
    if candles_held >= max_hold:
        total_qty_b = sum(p[2] for p in positions)
        avg_entry_b = avg_entry  # déjà calculé ligne 887
        if direction == 1:
            unrealized_b = (close_i - avg_entry_b) * total_qty_b
        else:
            unrealized_b = (avg_entry_b - close_i) * total_qty_b
        if unrealized_b < 0:
            exit_reason = "time_stop"
            exit_price = close_i
```

**Fees** : `time_stop` → `fee = taker_fee`, `slip = slippage_pct` (comme `sl_global` dans boltrend).

---

## Étape 5 — Tests (`tests/test_time_stop.py`)

### Section 1 — grid_atr strategy layer (~8 tests)
1. `max_hold_candles=0` → time_stop jamais déclenché (backward compat)
2. 48 candles + PnL < 0 → `"time_stop"`
3. 48 candles + PnL > 0 → `None` (pas de time_stop en profit)
4. 47 candles + PnL < 0 → `None` (pas encore le seuil)
5. SL ET time_stop applicables → SL prioritaire (`"sl_global"`)
6. PnL = 0 exactement → `None` (pas strictement < 0)
7. TP atteint ET time_stop → TP prioritaire (`"tp_global"`)
8. Timeframe 4h : 12 candles × 4h = 48h → valide

**Helper** : `_make_grid_state_with_time(entry_time, n_hours_ago)` qui crée des positions avec `entry_time = ctx.timestamp - timedelta(hours=n_hours_ago)`.

### Section 2 — grid_boltrend strategy layer (~5 tests)
9. `max_hold_candles=0` → backward compat
10. 48 candles + PnL < 0 → `"time_stop"`
11. 48 candles + PnL > 0 → `None`
12. SL prioritaire sur time_stop
13. TP inverse (signal_exit) prioritaire sur time_stop

### Section 3 — Fast engine grid_common (~6 tests)
14. `max_hold_candles=0` → résultat identique à avant (pas de régression)
15. Position ouverte 50 candles en perte → trade fermé par time_stop
16. Position ouverte 50 candles en profit → pas de time_stop
17. first_entry_idx reset correctement après fermeture (nouveau cycle = nouveau compteur)
18. Fees time_stop = taker + slippage (vérifier via net_pnl)
19. `_simulate_envelope_dca` fonctionne toujours sans max_hold (défaut 0)

### Section 4 — Fast engine grid_boltrend (~4 tests)
20. `max_hold_candles=0` → résultat identique
21. breakout_candle_idx utilisé correctement (non un first_entry_idx séparé)
22. Position en perte depuis 60 candles → time_stop
23. SL atteint avant time_stop → SL prioritaire

### Section 5 — Config (~3 tests)
24. `GridATRConfig(max_hold_candles=48)` → valide
25. `GridATRConfig(max_hold_candles=-1)` → `ValidationError` (ge=0)
26. `GridBolTrendConfig(max_hold_candles=0)` → valide (backward compat, défaut)

### Section 6 — Parité fast engine vs strategy layer (~1 test)

27. Données synthétiques : position ouverte 50 candles en perte → vérifier que fast engine ET `should_close_all()` produisent `"time_stop"` sur la même candle

---

## Couches non modifiées

- `GridStrategyRunner` : `should_close_all()` retourne déjà `exit_reason` traité génériquement → `"time_stop"` déclenche `close_all_positions()` normalement
- `Executor` live : idem, `_monitor_exit_conditions()` → `_close_grid_cycle()` normalement
- `PortfolioEngine` / `MultiPositionEngine` : zéro modif
- `_simulate_grid_range()` : pas de `should_close_all` global (positions indépendantes)
- `_simulate_grid_trend()` : désactivé (forward test -28%), `max_hold_candles=0` implicite
- `param_grids.yaml` : pas modifié (A/B test séparé)

---

## Vérification

1. `uv run pytest tests/test_time_stop.py -v` → 27 tests verts
2. `uv run pytest tests/ -x -q` → zéro régression (>= 1452 tests)
3. Test manuel : `GridATRConfig(max_hold_candles=48)` → vérifier `get_params()` retourne bien `max_hold_candles=48`
4. Test backtest manuel : `uv run python scripts/run_backtest.py --strategy grid_atr --symbol BTC/USDT --params max_hold_candles=48` → vérifier que des trades `time_stop` apparaissent dans le rapport
