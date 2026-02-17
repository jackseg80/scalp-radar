# grid_funding — Grid Funding (DCA sur Funding Rate Négatif)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `grid_funding` |
| Catégorie | Grid/DCA |
| Timeframe | 1h |
| Sprint d'origine | Sprint 22 |
| Grade actuel | Non publié en détail (WFO terminé) |
| Statut | **Désactivé** |
| Fichier source | `backend/strategies/grid_funding.py` |
| Config class | `GridFundingConfig` (`backend/core/config.py:277`) |

## Description

DCA sur funding rate négatif. LONG-only. L'edge est indépendant du prix : quand le funding rate est négatif, les shorts paient les longs. Plus le funding est négatif, plus de niveaux DCA sont ouverts. Le TP est atteint quand le funding redevient positif ou que le prix repasse au-dessus de la SMA.

**Régime ciblé** : Marchés où le funding rate est négatif de manière soutenue (forte demande short, crowded short trade).

## Logique d'entrée

Méthode : `compute_grid(ctx, grid_state) -> list[GridLevel]`

Le signal d'entrée est basé sur le funding rate, **pas sur le prix** :

```
funding_rate = ctx.extra_data["funding_rate"]  (décimal, déjà divisé par 100)

Pour i = 0 à num_levels - 1 :
  threshold = -(funding_threshold_start + i × funding_threshold_step)

  Si funding_rate <= threshold :
    entry_price = close  (prix courant)
    direction = LONG
```

Plus le funding est négatif, plus de niveaux se remplissent. Le premier niveau s'active à `-0.0005`, le deuxième à `-0.001`, etc. (avec les défauts).

Chaque niveau a `size_fraction = 1.0 / num_levels`.

## Logique de sortie

Méthode : `should_close_all(ctx, grid_state) -> str | None`

**SL global** (toujours actif, même pendant `min_hold`) :
- `close <= avg_entry_price × (1 - sl_percent / 100)`

**Min hold** : bloque le TP (pas le SL) pendant `min_hold_candles` bougies (défaut 8 = 1 période funding de 8h).

**TP modes** (configurable via `tp_mode`) :

| Mode | Condition de fermeture |
|------|----------------------|
| `"funding_positive"` | `funding_rate > 0` |
| `"sma_cross"` | `close >= SMA(close, ma_period)` |
| `"funding_or_sma"` (défaut) | `funding_rate > 0` **OU** `close >= SMA` |

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `funding_threshold_start` | float | 0.0005 | > 0 | [0.0003, 0.0005, 0.0008, 0.001] | Seuil FR du 1er niveau |
| `funding_threshold_step` | float | 0.0005 | > 0 | [0.0003, 0.0005, 0.001] | Incrément entre niveaux |
| `num_levels` | int | 3 | 1-6 | [2, 3] | Nombre de niveaux DCA |
| `tp_mode` | str | "funding_or_sma" | enum | ["funding_positive", "sma_cross", "funding_or_sma"] | Mode de TP |
| `ma_period` | int | 14 | 2-50 | [7, 14, 21] | Période SMA (pour TP sma_cross) |
| `sl_percent` | float | 15.0 | > 0 | [10.0, 15.0, 20.0, 25.0] | Stop loss global (%) |
| `min_hold_candles` | int | 8 | >= 0 | [4, 8, 16] | Bougies minimum avant TP |
| `sides` | list | ["long"] | — | — | LONG-only |
| `leverage` | int | 6 | 1-20 | — | Levier (fixe) |
| `timeframe` | str | "1h" | — | — | Timeframe (fixe) |

**Config WFO** : IS = 360j, OOS = 90j, step = 90j, **2 592 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| SMA (close) | 1h | TP mode sma_cross |
| Funding rate | externe (extra_data) | Signal d'entrée + TP mode funding_positive |

Les données de funding rate sont chargées via `extra_data_builder.py` (table `funding_rates` en DB, intervalle 8h resample en 1h).

## Résultats

### WFO

2 592 combinaisons testées sur IS = 360j, OOS = 90j. WFO terminé avec succès. Edge structurel indépendant du prix (shorts paient les longs).

### Points forts

- Edge **indépendant du prix** : le profit vient des paiements de funding, pas de la direction du marché
- `min_hold_candles` évite de sortir trop tôt (au moins 1 cycle de funding 8h)
- 3 modes de TP configurables pour s'adapter à différentes conditions

### Points faibles

- Événements rares (funding rate extrême ne dure pas longtemps)
- LONG-only — pas de couverture en bear
- Nécessite des données de funding rate historiques en DB pour le backtest

## Remarques

- **Convention /100 cascade** : `extra_data_builder.py` divise par 100, `grid_funding.py` ne divise plus (sinon double division)
- **Settlement mask vectorisé** : `hours = ((candle_ts_ms / 3600000) % 24).astype(int)`, `mask = hours % 8 == 0`
- **Signe funding unifié** : `payment = -fr × notional × direction`. Positif = on reçoit, négatif = on paie
- **`STRATEGIES_NEED_EXTRA_DATA`** : étendu à toutes les 6 stratégies grid (pas seulement grid_funding)
- **Bugfix 22-bis** : `extra_data_by_timestamp` manquant dans `run_multi_backtest_single` pour `report.py` et `overfitting.py`
- **Différence avec la stratégie `funding` (scalp 5m)** : grid_funding est DCA multi-niveaux avec funding rate comme signal, `funding` est mono-position avec funding rate extrême comme signal
