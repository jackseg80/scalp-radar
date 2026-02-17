# grid_trend — Grid Trend (Trend Following DCA)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `grid_trend` |
| Catégorie | Grid/DCA |
| Timeframe | 1h |
| Sprint d'origine | Sprint 23 |
| Grade actuel | F (grille réduite), non évalué sur grille complète |
| Statut | **Désactivé** (échec forward test) |
| Fichier source | `backend/strategies/grid_trend.py` |
| Config class | `GridTrendConfig` (`backend/core/config.py:302`) |

## Description

Trend following DCA. Utilise un croisement EMA (fast/slow) comme filtre directionnel et l'ADX pour confirmer la force du trend. Ouvre des positions DCA sur les pullbacks vers l'EMA fast, espacées par des multiples d'ATR. Le TP est un trailing stop basé sur l'ATR (pas un retour à la moyenne). Force la fermeture de toutes les positions si la direction EMA flippe.

**Régime ciblé** : Marchés en tendance (ADX > seuil). Zone neutre quand ADX < seuil (pas de nouveaux niveaux).

## Logique d'entrée

Méthode : `compute_grid(ctx, grid_state) -> list[GridLevel]`

### Pré-condition : force du trend

```
ADX = ADX(high, low, close, adx_period)
Si ADX < adx_threshold → retourne [] (zone neutre, pas de nouveaux niveaux)
```

### Direction

```
EMA_fast = EMA(close, ema_fast)
EMA_slow = EMA(close, ema_slow)

EMA_fast > EMA_slow → LONG
EMA_fast < EMA_slow → SHORT
```

### Niveaux d'entrée (pullbacks)

```
ATR = ATR(high, low, close, atr_period)

Pour i = 0 à num_levels - 1 :
  offset = pull_start + i × pull_step

  LONG  : entry_price = EMA_fast - ATR × offset
  SHORT : entry_price = EMA_fast + ATR × offset
```

**Direction lock** : si des positions dans l'autre sens sont ouvertes, aucun nouveau niveau n'est généré. La fermeture se fait via `should_close_all()`.

Chaque niveau a `size_fraction = 1.0 / num_levels`.

## Logique de sortie

Méthode : `should_close_all(ctx, grid_state) -> str | None`

**Direction flip** (force close) :
- LONG et `EMA_fast < EMA_slow` → `"direction_flip"`
- SHORT et `EMA_fast > EMA_slow` → `"direction_flip"`

**SL global** :
- LONG : `close <= avg_entry_price × (1 - sl_percent / 100)`
- SHORT : `close >= avg_entry_price × (1 + sl_percent / 100)`

**Trailing stop ATR** : géré par le fast engine / `GridStrategyRunner` (externe à `should_close_all`). `get_tp_price()` retourne `NaN` — il n'y a pas de TP fixe.

```
Trailing stop LONG : HWM - ATR × trail_mult
Trailing stop SHORT : LWM + ATR × trail_mult
```

Le HWM (High Water Mark) est mis à jour à chaque nouveau high pour les LONG, et le LWM (Low Water Mark) à chaque nouveau low pour les SHORT.

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `ema_fast` | int | 20 | 5-50 | [20, 30, 40] | Période EMA rapide |
| `ema_slow` | int | 50 | 20-200 | [50, 100] | Période EMA lente |
| `adx_period` | int | 14 | 7-30 | [14] | Période ADX |
| `adx_threshold` | float | 20.0 | 10-40 | [15, 20, 25] | Seuil ADX (zone neutre si <) |
| `atr_period` | int | 14 | 5-30 | [14] | Période ATR |
| `pull_start` | float | 1.0 | > 0 | [0.5, 1.0, 1.5] | Offset ATR du 1er pullback |
| `pull_step` | float | 0.5 | > 0 | [0.5, 1.0] | Incrément entre pullbacks |
| `num_levels` | int | 3 | 1-6 | [2, 3] | Nombre de niveaux DCA |
| `trail_mult` | float | 2.0 | > 0 | [1.5, 2.0, 2.5, 3.0] | Multiplicateur ATR trailing stop |
| `sl_percent` | float | 15.0 | > 0 | [10, 15, 20] | Stop loss global (%) |
| `sides` | list | ["long", "short"] | — | — | Côtés autorisés |
| `leverage` | int | 6 | 1-20 | — | Levier (fixe) |
| `timeframe` | str | "1h" | — | — | Timeframe (fixe) |

**Config WFO** : IS = 360j, OOS = 90j, step = 90j, **2 592 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| EMA (close, ema_fast) | 1h | Filtre directionnel + base des pullbacks |
| EMA (close, ema_slow) | 1h | Filtre directionnel |
| ADX (high, low, close) | 1h | Force du trend (zone neutre si < seuil) |
| ATR (high, low, close) | 1h | Espacement des niveaux + trailing stop |

`compute_live_indicators()` calcule EMA fast/slow + ADX depuis le buffer de candles 1h pour le mode live/portfolio.

## Résultats

### WFO

- Grille réduite (12 combos) : Grade F (OOS Sharpe -4.10, consistance 5.6%) — attendu avec paramètres sous-optimaux
- Grille complète (2 592 combos) : freeze à la fenêtre 4/18 (ProcessPoolExecutor Windows)

### Portfolio Backtest

| Configuration | Période | Return | Max DD | Runners profitables |
|---------------|---------|--------|--------|---------------------|
| 6 assets | 730j | +77% | -20.2% | 5/6 |
| **6 assets** | **Forward 365j** | **-28%** | — | **1/5** |

**Per-asset overrides** (6 assets) : AR, CRV, ICP, JUP, SOL, XTZ — tous avec overrides sur 10 paramètres.

### Points forts

- Complémentaire en théorie avec grid_atr (trend following vs mean reversion)
- Force close au flip EMA protège contre les retournements
- Zone neutre ADX évite les faux signaux en range

### Points faibles

- **Échec en forward test** : 1/5 runners profitables sur 365j de bear market
- Le trend following DCA nécessite des trends prolongés pour être rentable
- WFO freeze sur grande grille (ProcessPoolExecutor Windows)

## Remarques

- **Non déployé** : le forward test est le critère définitif. grid_trend +77% sur 730j mais -28% en forward.
- **Trailing stop HWM SHORT** : guard `if hwm > 0 else lows[i]` pour éviter blocage à 0.0
- **Cohérence masks** : `_build_entry_prices()` et `_simulate_grid_trend()` doivent utiliser la même logique pour `long_mask` / `short_mask`
- **`sma_arr` déplacé** : dans `_build_entry_prices()`, le calcul `sma_arr` est dans chaque branche stratégie (pas avant le `if`), sinon `KeyError` pour grid_trend qui n'a pas `ma_period`
- **IndicatorCache** : nécessite `ema_by_period` et `adx_by_period` (dicts) — tests doivent les passer au constructeur
- **Différence avec grid_atr** : grid_atr = mean reversion (TP = retour SMA), grid_trend = trend following (TP = trailing stop, force close au flip)
