# grid_atr — Grid ATR (Mean Reversion Adaptative)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `grid_atr` |
| Catégorie | Grid/DCA |
| Timeframe | 1h |
| Sprint d'origine | Sprint 19 |
| Grade actuel | A/B (6 Grade A, 7 Grade B sur 21 assets — 13 éligibles) |
| Statut | **LIVE** (13 assets, leverage 4-7x) |
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
effective_atr = max(ATR, close × min_grid_spacing_pct / 100)

Pour i = 0 à num_levels - 1 :
  multiplier = atr_multiplier_start + i × atr_multiplier_step

  LONG  : entry_price = SMA - effective_atr × multiplier
  SHORT : entry_price = SMA + effective_atr × multiplier
```

**Règle du côté unique** : si des positions LONG sont ouvertes, seuls les niveaux LONG sont générés (et inversement pour SHORT). Un seul côté actif à la fois.

Chaque niveau a `size_fraction = 1.0 / num_levels` (allocation égale).

## Logique de sortie

Méthode : `should_close_all(ctx, grid_state) -> str | None`

**TP global** (retour à la SMA + profit minimum) :
- LONG : `close >= SMA ET close >= avg_entry × (1 + min_profit_pct / 100)` → fermeture de toutes les positions
- SHORT : `close <= SMA ET close <= avg_entry × (1 - min_profit_pct / 100)` → fermeture de toutes les positions

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
| `sl_percent` | float | 20.0 | > 0 | [15.0, 20.0, 25.0] | Stop loss global (%) |
| `min_grid_spacing_pct` | float | 0.0 | ≥ 0 | [0, 0.8, 1.2, 1.8] | Plancher espacement grille (% du prix) |
| `min_profit_pct` | float | 0.0 | ≥ 0 | [0, 0.2, 0.4] | Profit minimum au TP (%) |
| `sides` | list | ["long"] | — | — | Côtés autorisés |
| `leverage` | int | 6 | 1-20 | — | Levier (fixe) |
| `timeframe` | str | "1h" | — | — | Timeframe (fixe) |

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **12 960 combinaisons** (V2 avec min_grid_spacing_pct + min_profit_pct).

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| SMA (close) | 1h | Base des enveloppes + TP dynamique |
| ATR (high, low, close) | 1h | Espacement adaptatif des niveaux |

## V2 — Paramètres adaptatifs (Sprint 47)

### Problème résolu

En basse volatilité, l'ATR s'écrase → les grilles deviennent microscopiques → les cycles de TP ne couvrent plus les fees.
Exemple réel : 7 assets ouverts en 30s, tous fermés en perte (-2.32$) car profit brut < fees.

### min_grid_spacing_pct (plancher de grille)

Empêche les grilles de devenir trop petites :

```
effective_atr = max(ATR, prix × min_grid_spacing_pct / 100)
```

- Si 0.0 : comportement classique (désactivé)
- Si 1.8 : l'espacement ne descend jamais sous 1.8% du prix
- Utilisé par 17/21 assets (valeur médiane : 1.2%)

### min_profit_pct (profit minimum au TP)

Le TP ne se déclenche que si le profit minimum est garanti :

```
TP = close >= SMA ET close >= avg_entry × (1 + min_profit_pct / 100)
```

- Si 0.0 : TP classique (désactivé)
- En pratique quasi inutile (19/21 assets à 0.0) car le spacing résout le problème en amont

### Résultats WFO V2

| Métrique | V1 | V2  |
|----------|-----|-----|
| Grade A | 0 | 6 |
| Grade B | 14 | 7 |
| Total A/B | 14 | 13 |
| Portfolio return | +208% | +262% |
| Max DD | -9.2% | -6.6% |
| CVaR 30j | 26.9% | 24.3% |
| Verdict robustness | VIABLE | VIABLE |

---

## Résultats WFO

### Par asset (21 assets testés — V2)

- **6 Grade A** (score 90-100) : edge fort, prêt pour le live
- **7 Grade B** (score 80-89) : bon, live avec surveillance
- **13 assets éligibles** au total (CRV retiré de la rotation V2)

### Portfolio Backtest (V1 — référence historique)

| Configuration | Période | Return | Max DD | Peak Margin | Runners profitables |
|---------------|---------|--------|--------|-------------|---------------------|
| 21 assets | 730j | +181% | -31.6% | 23.5% | 20/21 |
| **Top 10** | **730j** | **+221%** | **-29.8%** | **25.0%** | **10/10** |
| **Top 10** | **Forward 365j** | **+82.4%** | **-25.7%** | — | **9/10** |

**Top 10 V1** : BTC, CRV, DOGE, DYDX, ENJ, FET, GALA, ICP, NEAR, AVAX.

### Per-asset overrides en production

| Asset | ma_period | atr_period | atr_mult_start | atr_mult_step | num_levels | sl_percent |
|-------|-----------|------------|----------------|---------------|------------|------------|
| AVAX/USDT | 7 | 10 | 1.5 | 0.5 | 3 | 15.0 |
| BTC/USDT | 7 | 20 | 2.0 | 1.5 | 2 | 15.0 |
| CRV/USDT | 7 | 10 | 1.0 | 1.5 | 4 | 30.0 |
| DOGE/USDT | 10 | 10 | 1.5 | 0.5 | 4 | 20.0 |
| DYDX/USDT | 7 | 14 | 1.5 | 1.5 | 2 | 15.0 |
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
