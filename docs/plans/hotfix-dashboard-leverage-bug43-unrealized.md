# Plan : Améliorations Dashboard — Leverage, Bug 4/3, Bug unrealized_pnl

## Contexte

3 changements demandés sur le dashboard :
1. Afficher le leverage partout
2. Investiguer et fixer le bug "4/3" (niveaux > max) sur FET/DYDX
3. Fixer le unrealized_pnl à +0.00$ dans le Simulator sidebar

---

## 1. AFFICHER LE LEVERAGE

### Analyse
- `get_grid_state()` expose déjà `leverage` par grid (simulator.py:2104)
- `get_status()` de GridStrategyRunner ne l'expose PAS encore → à ajouter
- `ExecutorPanel` affiche déjà le leverage par position live (ligne 175-179) mais pas en résumé global

### Changements

**Backend** — `backend/backtesting/simulator.py`
- `GridStrategyRunner.get_status()` (~L1342) : ajouter `"leverage": self._leverage`

**Frontend** — 3 fichiers

**a) `frontend/src/components/ActivePositions.jsx`** — GridList
- Après le badge `g.strategy` (ligne 116), ajouter badge `x{g.leverage}`

**b) `frontend/src/components/ExecutorPanel.jsx`** — Sidebar
- Après la ligne "Solde Bitget" (~L68), ajouter `StatusRow` "Leverage" si `executor.leverage` existe
- NOTE : l'executor n'expose pas de leverage global directement. Utiliser le leverage de la première position, ou ne rien afficher si aucune position.
- Alternative : utiliser `wsData.grid_state?.grid_positions` pour extraire le leverage du premier grid visible → plus fiable

**c) `frontend/src/components/Scanner.jsx`** — Sous-titre stratégie
- Quand `strategyFilter` est actif, afficher sous le `<h2>Scanner</h2>` :
  `"grid_atr — 6x — 10 assets"`
- Sources : `strategyFilter` (nom), `wsData.strategies[strategyFilter]?.leverage` (leverage), `watchedSymbols?.size` (nb assets)

---

## 2. BUG : Grid "4/3"

### Diagnostic — ROOT CAUSE IDENTIFIÉE

**Ce n'est PAS un bug d'affichage. C'est un vrai bug de logique.**

Le per_asset `num_levels` dans `strategies.yaml` est **ignoré** par la stratégie. Voici le problème :

```yaml
grid_atr:
  num_levels: 3         # ← top-level = 3
  per_asset:
    FET/USDT:
      num_levels: 4     # ← per_asset = 4, MAIS IGNORÉ PAR LA STRATÉGIE
```

- `max_positions` (grid_atr.py:35) → `self._config.num_levels` = **3** (top-level)
- `compute_grid()` (grid_atr.py:98) → `range(self._config.num_levels)` = **range(3)**
- `on_candle()` guard (simulator.py:981) → `len(positions) < self._strategy.max_positions` = **< 3**

Le per_asset `num_levels` n'est jamais lu par la stratégie ni par le runner pour le garde-fou. C'est **dead config** — le WFO optimise num_levels par asset mais le Simulator l'ignore.

Le "4/3" s'explique par des positions restaurées (`restore_state()`) depuis un état antérieur quand le top-level était à 4 (ou un config temporaire).

### Fix

Résoudre `num_levels` depuis per_asset dans le **runner** (cohérent avec `_get_sl_percent()`), ET patcher temporairement la config de la stratégie avant `compute_grid()` pour que la stratégie génère le bon nombre de niveaux.

Les configs sont des Pydantic `BaseModel` non-frozen → mutation OK. L'event loop est single-threaded → pas de race condition.

**`backend/backtesting/simulator.py` — GridStrategyRunner**

a) Ajouter méthode `_get_num_levels(symbol)` :
```python
def _get_num_levels(self, symbol: str) -> int:
    """Résout num_levels pour un symbol (avec override per_asset)."""
    config = self._strategy._config
    default = self._strategy.max_positions
    per_asset = getattr(config, "per_asset", {})
    if isinstance(per_asset, dict):
        overrides = per_asset.get(symbol, {})
        if isinstance(overrides, dict) and "num_levels" in overrides:
            return int(overrides["num_levels"])
    return default
```

b) Modifier le guard d'ouverture (~L981) :
```python
effective_max = self._get_num_levels(symbol)
if len(positions) < effective_max:
```

c) **CRITIQUE** — Patcher temporairement `num_levels` avant `compute_grid()` (~L982) :
```python
original_num_levels = self._strategy._config.num_levels
self._strategy._config.num_levels = effective_max
try:
    levels = self._strategy.compute_grid(ctx, grid_state)
finally:
    self._strategy._config.num_levels = original_num_levels
```
Cela garantit que `compute_grid()` génère le bon nombre de niveaux (4 pour FET, 2 pour BTC, etc.)
Fonctionne pour TOUTES les stratégies grid (grid_atr, grid_boltrend, grid_multi_tf, etc.) car elles utilisent toutes `self._config.num_levels` dans `compute_grid()`.

d) Modifier le sizing qui utilise `num_levels` (~L1012) :
```python
num_levels = effective_max  # au lieu de self._strategy.max_positions
```

e) Modifier `get_grid_state()` (~L2086) pour `levels_max` :
```python
"levels_max": runner._get_num_levels(symbol),
```

f) Modifier la ligne de log (~L1094) :
```python
"levels_max": self._get_num_levels(symbol),
```

---

## 3. BUG : unrealized_pnl +0.00$ dans SessionStats

### Diagnostic — ROOT CAUSE IDENTIFIÉE

`GridStrategyRunner.get_status()` calcule le P&L non réalisé à partir de `self._last_prices` (L1302). Mais `_last_prices` est UNIQUEMENT mis à jour dans `on_candle()` quand `timeframe == self._strategy_tf` (1h). Les candles 1m sont filtrées à la ligne 796-797.

Résultat : entre deux bougies 1h, `_last_prices` a le dernier close 1h, qui est souvent très proche du prix d'entrée → unrealized_pnl ≈ 0.

Pendant ce temps, `get_grid_state()` utilise `_get_current_price()` (DataEngine, candles 1m fraîches) → les ActivePositions affichent le vrai P&L.

### Fix

Déplacer la mise à jour de `_last_prices` AVANT le filtre timeframe, pour que les candles 1m rafraîchissent aussi le prix :

**`backend/backtesting/simulator.py` — `GridStrategyRunner.on_candle()`**

```python
async def on_candle(self, symbol, timeframe, candle):
    if self._kill_switch_triggered:
        return

    # Filtre per_asset (pas de changement)
    if self._per_asset_keys and symbol not in self._per_asset_keys:
        return

    # ← NOUVEAU : Rafraîchir le prix courant AVANT le filtre timeframe
    # Permet un P&L temps réel dans get_status() (toutes les 1m au lieu de 1h)
    self._last_prices[symbol] = candle.close

    # Filtre timeframe (pas de changement)
    if timeframe != self._strategy_tf:
        return

    # ... reste de la méthode inchangé
    # SUPPRIMER la ligne 829 : self._last_prices[symbol] = candle.close (doublon)
```

---

## Fichiers modifiés (résumé)

| Fichier | Changement |
|---------|-----------|
| `backend/backtesting/simulator.py` | +`leverage` dans get_status(), `_get_num_levels()`, fix guard on_candle, fix `_last_prices` timing |
| `frontend/src/components/ActivePositions.jsx` | Badge leverage sur chaque grid row |
| `frontend/src/components/ExecutorPanel.jsx` | Ligne leverage en sidebar |
| `frontend/src/components/Scanner.jsx` | Sous-titre "stratégie — Nx — M assets" |

---

## Vérification

1. **Tests existants** : `uv run pytest tests/ -x -q` — vérifier que rien ne casse
2. **Dashboard** : Lancer `dev.bat`, vérifier :
   - Scanner : sous-titre avec leverage et nb assets quand stratégie filtrée
   - ActivePositions : badge leverage visible sur chaque grid
   - SessionStats : unrealized_pnl non-nul si positions ouvertes
   - Scanner : colonne Grid montre le bon `levels_open/levels_max` per_asset
3. **Test spécifique** : ajouter test pour `_get_num_levels()` qui vérifie la résolution per_asset
