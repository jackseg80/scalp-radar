# grid_momentum — Grid Momentum (Breakout DCA + Trailing Stop)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `grid_momentum` |
| Catégorie | Grid/DCA |
| Timeframe | 1h |
| Sprint d'origine | Sprint grid_momentum (24 fév 2026) |
| Grade actuel | — (WFO pas encore lancé) |
| Statut | **Désactivé** (WFO à lancer) |
| Fichier source | `backend/strategies/grid_momentum.py` |
| Config class | `GridMomentumConfig` (`backend/core/config.py:394`) |

## Description

Stratégie breakout/trend-following à profil de payoff **convexe** : petites pertes répétées sur les faux breakouts, gros gains sur les vrais trends. Intentionnellement décorrélée de grid_atr (profil concave, mean-reversion).

Un breakout Donchian (cassure du plus haut/bas des N dernières bougies) avec confirmation de volume déclenche l'entrée. Les niveaux DCA suivants sont positionnés en pullback sous le prix de breakout. La sortie est gérée par un **trailing stop ATR** (HWM) ou un **flip de direction** Donchian.

**Régime ciblé** : Marchés en breakout/trend directionnel. Zone neutre quand le marché reste dans les bandes Donchian (pas de signal). Filtre ADX optionnel pour exiger un trend établi.

**Complémentarité** : décorrélé de grid_atr (mean-reversion range) et grid_multi_tf (mean-reversion piloté ST). grid_momentum profite des breakouts que grid_atr subit comme des SL.

## Logique d'entrée

Méthode : `compute_grid(ctx, grid_state) -> list[GridLevel]`

### Détection du breakout

```
Donchian_high = max(high[i-N : i])   ← anti-lookahead : bougie courante exclue
Donchian_low  = min(low[i-N  : i])

Volume_SMA = SMA(volume, vol_sma_period)
ADX = ADX(high, low, close, adx_period)   ← période fixe (non optimisée)

Breakout LONG  : close > Donchian_high
             ET volume > Volume_SMA × vol_multiplier
             ET (adx_threshold = 0 OU ADX > adx_threshold)
             ET "long" ∈ sides

Breakout SHORT : close < Donchian_low
             ET mêmes filtres volume/ADX
             ET "short" ∈ sides
```

**Cooldown** : si la dernière fermeture remonte à moins de `cooldown_candles` bougies → ignorer le breakout.

### Niveaux d'entrée

```
ATR = ATR(high, low, close, atr_period)
anchor = close[i]   ← prix du breakout

Level 0 (immédiat) : anchor
Level k (k ≥ 1)   :
  LONG  : anchor - ATR × (pullback_start + (k-1) × pullback_step)
  SHORT : anchor + ATR × (pullback_start + (k-1) × pullback_step)
```

Le Level 0 est rempli immédiatement à la bougie du breakout. Les niveaux suivants s'activent si le prix revient en pullback vers la grille (DCA progressif).

Chaque niveau a `size_fraction = 1.0 / num_levels`.

## Logique de sortie

Méthode : `should_close_all(ctx, grid_state) -> str | None`

**Trailing stop ATR** (HWM) :
```
LONG  : HWM = max(HWM, high[i])
        trail_price = HWM - ATR × trailing_atr_mult
        close < trail_price → "trail_stop"

SHORT : LWM = min(LWM, low[i])
        trail_price = LWM + ATR × trailing_atr_mult
        close > trail_price → "trail_stop"
```

**Direction flip** (sortie de protection — SANS filtre volume/ADX) :
- LONG ouvert et `close < Donchian_low` → `"direction_flip"`
- SHORT ouvert et `close > Donchian_high` → `"direction_flip"`

Le filtre volume/ADX est pour l'ENTRÉE uniquement. Le flip de direction ferme sans condition dès que le breakout inverse est détecté.

**SL global** : géré par `get_sl_price()` + `check_global_tp_sl()` (heuristique OHLC) — pas dans `should_close_all()`.

```
SL LONG  : avg_entry × (1 - sl_percent / 100)
SL SHORT : avg_entry × (1 + sl_percent / 100)
```

**TP fixe** : aucun. `get_tp_price()` retourne `float("nan")`. La sortie en profit est exclusivement gérée par le trailing stop ATR.

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `donchian_period` | int | 30 | 10-100 | [20, 30, 48, 72] | Période canal Donchian (anti-lookahead) |
| `vol_sma_period` | int | 20 | 5-50 | — (fixe) | Période SMA du volume (non optimisé) |
| `vol_multiplier` | float | 1.5 | > 0 | [1.2, 1.5, 2.0, 2.5] | Multiplicateur volume pour confirmation |
| `adx_period` | int | 14 | 5-30 | — (fixe) | Période ADX (non optimisé) |
| `adx_threshold` | float | 0.0 | ≥ 0 | [0, 20] | Seuil ADX (0 = désactivé) |
| `atr_period` | int | 14 | 5-30 | [10, 14, 20] | Période ATR (espacement + trailing) |
| `pullback_start` | float | 1.0 | > 0 | [0.5, 1.0, 1.5] | Offset ATR du 1er niveau de pullback |
| `pullback_step` | float | 0.5 | > 0 | [0.5, 1.0] | Incrément ATR entre niveaux |
| `num_levels` | int | 3 | 1-6 | [2, 3, 4] | Nombre de niveaux DCA |
| `trailing_atr_mult` | float | 3.0 | > 0 | [2.0, 3.0, 4.0] | Multiplicateur ATR pour le trailing stop |
| `sl_percent` | float | 15.0 | > 0 | [15, 20] | Stop loss global (%) |
| `cooldown_candles` | int | 3 | 0-10 | [0, 3] | Bougies de cooldown entre trades |
| `sides` | list | ["long", "short"] | — | — | Côtés autorisés |
| `leverage` | int | 6 | 1-20 | — | Levier (fixe) |
| `timeframe` | str | "1h" | — | — | Timeframe (fixe) |

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **~20 736 combinaisons** (4×4×3×2×3×2×3×3×2×2).

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| Donchian high (rolling max, excl. courante) | 1h | Signal breakout LONG + exit direction flip SHORT |
| Donchian low (rolling min, excl. courante) | 1h | Signal breakout SHORT + exit direction flip LONG |
| ATR (high, low, close, atr_period) | 1h | Espacement des niveaux DCA + trailing stop |
| Volume SMA (volume, vol_sma_period=20) | 1h | Confirmation du breakout |
| ADX (high, low, close, adx_period=14) | 1h | Filtre force du trend (optionnel si threshold=0) |

`compute_live_indicators()` calcule Donchian high/low + ATR + Volume SMA + ADX depuis le buffer de candles 1h pour le mode live/portfolio.

Le **HWM** (High Water Mark) est tracké séparément :
- **Fast engine** : variable locale dans `_simulate_grid_momentum()`
- **Live** : `GridStrategyRunner._hwm: dict[str, float]`, injecté via `indicators["hwm"]` avant `should_close_all()`

## Résultats

### WFO

*Non lancé — WFO à programmer après validation des tests.*

### Portfolio Backtest

*Non lancé.*

### Points forts (théoriques)

- Profil convexe : pertes bornées sur faux breakouts, gains non bornés sur vrais trends
- Décorrélé de grid_atr (concave) → diversification de profil de payoff en portefeuille
- Direction flip sans filtre = sortie rapide sur retournement, sans attendre le SL global
- ADX optionnel : peut fonctionner pur volume (adx_threshold=0) ou avec filtre trend

### Points faibles (anticipés)

- Marchés range → drawdown par faux breakouts répétés
- Cooldown peut manquer un vrai breakout après une fausse sortie
- Trailing stop trop serré → sortie prématurée sur pullback normal du trend

## Remarques

- **HWM live** : `GridStrategyRunner._hwm[symbol]` initialisé au `high` de la bougie du breakout (1ère position). Mis à jour à chaque `on_candle()`. Reset à `float("nan")` quand toutes les positions fermées. Si HWM absent (NaN), trailing stop désactivé (fallback SL uniquement).
- **Anti-lookahead Donchian** : `_rolling_max(highs, N)[i] = max(highs[i-N:i])` — bougie courante exclue. Cohérent avec la convention du cache d'indicateurs.
- **SL dans get_sl_price(), pas should_close_all()** : le SL global est géré par `check_global_tp_sl()` via heuristique OHLC (plus précis que check sur `close` uniquement). `should_close_all()` ne gère que trailing stop et direction flip.
- **Fast engine** : fonction dédiée `_simulate_grid_momentum()` (state machine 3 états : INACTIVE → ACTIVE → EXIT). Incompatible avec `_simulate_grid_common()` car les entry prices sont fixés dynamiquement au breakout (pas pré-calculés).
- **STRATEGIES_NEED_EXTRA_DATA** : grid_momentum n'est PAS dans cette liste (pas de funding rates ni OI requis).
- **Différence avec grid_trend** : grid_trend = EMA cross (trend établi) + pullbacks depuis EMA, grid_momentum = Donchian breakout (cassure de niveau clé) + pullbacks depuis prix de breakout. Trailing stop identique en structure.
- **Différence avec grid_boltrend** : grid_boltrend = Bollinger breakout + TP inversé retour SMA (mean-reversion), grid_momentum = Donchian breakout + trailing stop (trend following pur, pas de retour à la moyenne).
