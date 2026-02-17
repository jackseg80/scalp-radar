# grid_atr — Grid ATR (Mean Reversion Adaptative)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `grid_atr` |
| Catégorie | Grid/DCA |
| Timeframe | 1h |
| Sprint d'origine | Sprint 19 |
| Grade actuel | A/B (14 Grade A, 7 Grade B sur 21 assets) |
| Statut | **Paper trading actif** (Top 10 assets) |
| Fichier source | `backend/strategies/grid_atr.py` |
| Config class | `GridATRConfig` (`backend/core/config.py:227`) |

## Description

Mean reversion adaptative à la volatilité. Place N enveloppes d'entrée autour d'une SMA, espacées par des multiples d'ATR. Quand le prix s'éloigne de la SMA et touche un niveau, une position DCA est ouverte. Le TP est atteint quand le prix revient à la SMA.

L'ATR rend les enveloppes **auto-adaptatives** : elles s'élargissent en haute volatilité et se resserrent en basse volatilité. C'est l'avantage principal par rapport à envelope_dca (enveloppes % fixes).

**Régime ciblé** : Range et trend modéré. Souffre en bear market soutenu sans recovery.

## Logique d'entrée

Méthode : `compute_grid(ctx, grid_state) -> list[GridLevel]`

Pour chaque bougie 1h, on calcule les N niveaux d'entrée :

```
SMA = SMA(close, ma_period)
ATR = ATR(high, low, close, atr_period)

Pour i = 0 à num_levels - 1 :
  multiplier = atr_multiplier_start + i × atr_multiplier_step

  LONG  : entry_price = SMA - ATR × multiplier
  SHORT : entry_price = SMA + ATR × multiplier
```

**Règle du côté unique** : si des positions LONG sont ouvertes, seuls les niveaux LONG sont générés (et inversement pour SHORT). Un seul côté actif à la fois.

Chaque niveau a `size_fraction = 1.0 / num_levels` (allocation égale).

## Logique de sortie

Méthode : `should_close_all(ctx, grid_state) -> str | None`

**TP global** (retour à la SMA) :
- LONG : `close >= SMA` → fermeture de toutes les positions
- SHORT : `close <= SMA` → fermeture de toutes les positions

**SL global** (% depuis le prix moyen) :
- LONG : `close <= avg_entry_price × (1 - sl_percent / 100)`
- SHORT : `close >= avg_entry_price × (1 + sl_percent / 100)`

Le SL est **global** : si le prix moyen pondéré des positions ouvertes subit une perte de `sl_percent`, tout est fermé.

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `ma_period` | int | 14 | 2-50 | [7, 10, 14, 20] | Période de la SMA |
| `atr_period` | int | 14 | 2-50 | [10, 14, 20] | Période de l'ATR |
| `atr_multiplier_start` | float | 2.0 | > 0 | [1.0, 1.5, 2.0, 2.5, 3.0] | Multiplicateur ATR du 1er niveau |
| `atr_multiplier_step` | float | 1.0 | > 0 | [0.5, 1.0, 1.5] | Incrément entre niveaux |
| `num_levels` | int | 3 | 1-6 | [2, 3, 4] | Nombre de niveaux DCA |
| `sl_percent` | float | 20.0 | > 0 | [15.0, 20.0, 25.0, 30.0] | Stop loss global (%) |
| `sides` | list | ["long"] | — | — | Côtés autorisés |
| `leverage` | int | 6 | 1-20 | — | Levier (fixe) |
| `timeframe` | str | "1h" | — | — | Timeframe (fixe) |

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **3 240 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| SMA (close) | 1h | Base des enveloppes + TP dynamique |
| ATR (high, low, close) | 1h | Espacement adaptatif des niveaux |

## Résultats WFO

### Par asset (21 assets testés)

- **14 Grade A** (score 85-100) : Sharpe OOS 4.5-12+, consistance 75-100%
- **7 Grade B** (score 71-84) : Sharpe OOS 3.5-8, consistance 60-80%
- **0 Grade D/F** — edge structurel démontré sur tous les assets

### Portfolio Backtest

| Configuration | Période | Return | Max DD | Peak Margin | Runners profitables |
|---------------|---------|--------|--------|-------------|---------------------|
| 21 assets | 730j | +181% | -31.6% | 23.5% | 20/21 |
| **Top 10** | **730j** | **+221%** | **-29.8%** | **25.0%** | **10/10** |
| **Top 10** | **Forward 365j** | **+82.4%** | **-25.7%** | — | **9/10** |

**Top 10 sélectionnés** : BTC, CRV, DOGE, DYDX, ENJ, FET, GALA, ICP, NEAR, AVAX.

### Per-asset overrides en production

| Asset | ma_period | atr_period | atr_mult_start | atr_mult_step | num_levels | sl_percent |
|-------|-----------|------------|----------------|---------------|------------|------------|
| AVAX/USDT | 7 | 10 | 1.5 | 0.5 | 3 | 15.0 |
| BTC/USDT | 7 | 20 | 2.0 | 1.5 | 2 | 15.0 |
| CRV/USDT | 7 | 10 | 1.0 | 1.5 | 4 | 30.0 |
| DOGE/USDT | 10 | 10 | 1.5 | 0.5 | 4 | 20.0 |
| DYDX/USDT | 7 | 14 | 1.5 | 1.5 | 2 | 15.0 |
| ENJ/USDT | 7 | 14 | 1.5 | 1.0 | 4 | 20.0 |
| FET/USDT | 7 | 14 | 1.0 | 1.5 | 4 | 30.0 |
| GALA/USDT | 7 | 14 | 2.5 | 1.0 | 4 | 15.0 |
| ICP/USDT | 7 | 10 | 1.0 | 0.5 | 2 | 30.0 |
| NEAR/USDT | 7 | 10 | 2.0 | 0.5 | 4 | 30.0 |

### Points forts

- **0 Grade D/F** sur 21 assets — meilleur ratio que toutes les autres stratégies
- ATR adaptatif > enveloppes % fixes (vs envelope_dca qui a 2 Grade D)
- Forward test positif (+82.4% sur 365j)
- Sizing equal allocation simple et robuste

### Points faibles

- 7/21 assets ont un Sharpe négatif sur les 90 derniers jours Bitget (bear soutenu nov 2025 - fév 2026)
- Le mean-reversion souffre en bear sans recovery (les prix ne reviennent pas à la SMA)
- LONG-only par défaut — pas de couverture en bear

## Remarques

- **Remplace envelope_dca** comme stratégie principale depuis le Sprint 19
- **Filtre Darwinien** (Sprint 27) ajouté pour bloquer les nouvelles grilles si le WFO Sharpe < 0 dans le régime actuel
- **Funding costs** (Sprint 26) : les coûts de funding 8h sont inclus dans le backtest
- Le fast engine numpy (`_simulate_grid_common()`) est partagé avec toutes les stratégies Grid/DCA
- Données sinusoïdales (`100 + 8×sin(2π×i/48)`) utilisées dans les tests pour générer un ATR réaliste
