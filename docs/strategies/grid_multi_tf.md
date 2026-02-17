# grid_multi_tf — Grid Multi-TF (Supertrend 4h + Grid ATR 1h)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `grid_multi_tf` |
| Catégorie | Grid/DCA |
| Timeframe | 1h (exécution) + 4h (filtre directionnel) |
| Sprint d'origine | Sprint 21a |
| Grade actuel | Non évalué (WFO terminé, validation Bitget en cours) |
| Statut | **Désactivé** |
| Fichier source | `backend/strategies/grid_multi_tf.py` |
| Config class | `GridMultiTFConfig` (`backend/core/config.py:249`) |

## Description

Combine un filtre directionnel Supertrend 4h avec l'exécution Grid ATR 1h. Le Supertrend 4h détermine la direction (UP → LONG, DOWN → SHORT). L'exécution des niveaux DCA est identique à grid_atr. Force la fermeture si le Supertrend 4h flippe.

L'objectif est de résoudre le défaut principal de grid_atr : ouverture de positions LONG en bear market. Le filtre Supertrend 4h empêche d'aller LONG quand la tendance 4h est baissière.

**Régime ciblé** : Tous régimes grâce au filtre directionnel (LONG en bull, SHORT en bear).

## Logique d'entrée

Méthode : `compute_grid(ctx, grid_state) -> list[GridLevel]`

### Étape 1 : Resampling 1h → 4h (anti-lookahead)

```
bucket_4h = timestamp // 14400  (frontières UTC 00h/04h/08h/12h/16h/20h)
OHLC 4h = agrégation par bucket
Supertrend 4h = SuperTrend(OHLC_4h, st_atr_period, st_atr_multiplier)
```

**Anti-lookahead critique** : chaque candle 1h utilise la direction du bucket 4h **précédent** (pas le courant). Sans cela, le backtest "voit le futur".

### Étape 2 : Direction

```
st_direction == 1  → LONG
st_direction == -1 → SHORT
```

### Étape 3 : Niveaux d'entrée (identique à grid_atr)

```
SMA = SMA(close, ma_period)
ATR = ATR(high, low, close, atr_period)

Pour i = 0 à num_levels - 1 :
  multiplier = atr_multiplier_start + i × atr_multiplier_step

  LONG  : entry_price = SMA - ATR × multiplier
  SHORT : entry_price = SMA + ATR × multiplier
```

## Logique de sortie

Méthode : `should_close_all(ctx, grid_state) -> str | None`

**Direction flip** (force close) :
- LONG et `st_direction == -1` → `"direction_flip"`
- SHORT et `st_direction == 1` → `"direction_flip"`

**TP global** (retour à la SMA) :
- LONG : `close >= SMA`
- SHORT : `close <= SMA`

**SL global** :
- LONG : `close <= avg_entry_price × (1 - sl_percent / 100)`
- SHORT : `close >= avg_entry_price × (1 + sl_percent / 100)`

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `st_atr_period` | int | 10 | 2-50 | [10, 14] | Période ATR du Supertrend 4h |
| `st_atr_multiplier` | float | 3.0 | > 0 | [2.0, 3.0] | Multiplicateur ATR du Supertrend 4h |
| `ma_period` | int | 14 | 2-50 | [7, 10] | Période SMA 1h |
| `atr_period` | int | 14 | 2-50 | [10, 14] | Période ATR 1h |
| `atr_multiplier_start` | float | 2.0 | > 0 | [1.5, 2.0] | Multiplicateur ATR 1er niveau |
| `atr_multiplier_step` | float | 1.0 | > 0 | [0.5, 1.0] | Incrément entre niveaux |
| `num_levels` | int | 3 | 1-6 | [2, 3] | Nombre de niveaux DCA |
| `sl_percent` | float | 20.0 | > 0 | [15.0, 20.0, 25.0] | Stop loss global (%) |
| `sides` | list | ["long", "short"] | — | — | Côtés autorisés |
| `leverage` | int | 6 | 1-20 | — | Levier (fixe) |
| `timeframe` | str | "1h" | — | — | Timeframe (fixe) |

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **384 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| SMA (close) | 1h | Base des enveloppes + TP |
| ATR (high, low, close) | 1h | Espacement adaptatif des niveaux |
| SuperTrend (ATR) | 4h (resample depuis 1h) | Filtre directionnel |

`compute_live_indicators()` calcule le Supertrend 4h depuis le buffer de candles 1h (nécessite `st_atr_period × 4 + 8` candles minimum).

## Résultats

### WFO

384 combinaisons testées, WFO terminé avec succès. Résultats non publiés en détail — validation Bitget en cours.

### Per-asset overrides en production

20 assets ont des overrides complets sur les 8 paramètres optimisables. Les overrides touchent `st_atr_period`, `st_atr_multiplier`, `ma_period`, `atr_period`, `atr_multiplier_start`, `atr_multiplier_step`, `num_levels`, `sl_percent`.

### Points forts

- Corrige le défaut principal de grid_atr en bear market (SHORT autorisé via Supertrend 4h)
- Resampling anti-lookahead garantit l'intégrité du backtest
- Formules d'exécution identiques à grid_atr (code factorisé)

### Points faibles

- Complexité ajoutée (resampling, directions dynamiques) vs filtre Darwinien simple (Sprint 27)
- Nécessite un buffer de candles 1h suffisant pour le calcul Supertrend 4h
- Pas encore validé en live/paper

## Remarques

- **Bugfix 21a-bis** : `compute_indicators()` ne calculait pas le Supertrend 4h → 0 trades en validation Bitget
- **Bugfix 21a-ter** : portfolio backtest 0 trades car `GridStrategyRunner.on_candle()` n'appelle pas `compute_indicators()`. Fix : `compute_live_indicators()` dans BaseGridStrategy
- **Différence avec grid_atr** : grid_multi_tf ajoute le filtre Supertrend 4h pour la direction, au lieu de n'utiliser que LONG
- **Différence avec grid_trend** : grid_multi_tf utilise le Supertrend 4h (indicator) pour la direction, grid_trend utilise le croisement EMA (signal)
- **`_simulate_grid_common()` avec `directions` array** : les callers existants (grid_atr, envelope_dca) passent `None` = aucun changement
