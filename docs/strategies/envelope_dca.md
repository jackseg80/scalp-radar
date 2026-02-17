# envelope_dca — Envelope DCA (Mean Reversion Multi-Niveaux)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `envelope_dca` |
| Catégorie | Grid/DCA |
| Timeframe | 1h |
| Sprint d'origine | Sprint 10 (création), Sprint 11 (paper), Sprint 12 (executor) |
| Grade actuel | A/B/D (3 Grade A, 18 Grade B, 2 Grade D sur 23 assets) |
| Statut | **Désactivé** (remplacé par grid_atr) |
| Fichier source | `backend/strategies/envelope_dca.py` |
| Config class | `EnvelopeDCAConfig` (`backend/core/config.py:185`) |

## Description

Mean reversion multi-niveaux à enveloppes fixes. Place N enveloppes autour d'une SMA, espacées par des pourcentages fixes. Les enveloppes sont **asymétriques** pour compenser la non-linéarité des log-returns. Quand le prix s'éloigne de la SMA et touche un niveau, une position DCA est ouverte. Le TP est atteint quand le prix revient à la SMA.

Première stratégie DCA viable du projet. A démontré que l'edge en crypto vient de la structure DCA multi-niveaux, pas des indicateurs mono-position. Remplacée par grid_atr (enveloppes ATR adaptatives > % fixes).

**Régime ciblé** : Range et trend modéré (identique à grid_atr).

## Logique d'entrée

Méthode : `compute_grid(ctx, grid_state) -> list[GridLevel]`

```
SMA = SMA(close, ma_period)

Pour i = 0 à num_levels - 1 :
  lower_offset = envelope_start + i × envelope_step
  upper_offset = 1 / (1 - lower_offset) - 1    ← formule asymétrique

  LONG  : entry_price = SMA × (1 - lower_offset)
  SHORT : entry_price = SMA × (1 + upper_offset)
```

**Enveloppes asymétriques** : la formule `upper = 1/(1-lower) - 1` garantit que l'aller-retour (baisse puis hausse) est cohérent en log-return. Exemple : si `lower = 5%`, `upper = 5.26%` (pas 5%).

**Règle du côté unique** : si des positions LONG sont ouvertes, seuls les niveaux LONG sont générés.

Chaque niveau a `size_fraction = 1.0 / num_levels`.

## Logique de sortie

Méthode : `should_close_all(ctx, grid_state) -> str | None`

**TP global** (retour à la SMA) :
- LONG : `close >= SMA`
- SHORT : `close <= SMA`

**SL global** (% depuis le prix moyen) :
- LONG : `close <= avg_entry_price × (1 - sl_percent / 100)`
- SHORT : `close >= avg_entry_price × (1 + sl_percent / 100)`

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `ma_period` | int | 5 | 2-50 | [5, 7, 8, 10] | Période de la SMA |
| `num_levels` | int | 4 | 1-6 | [2, 3, 4] | Nombre de niveaux DCA |
| `envelope_start` | float | 0.05 | > 0 | [0.05, 0.07, 0.10, 0.12, 0.15] | Offset du 1er niveau (5%) |
| `envelope_step` | float | 0.05 | > 0 | [0.02, 0.03, 0.05] | Incrément entre niveaux |
| `sl_percent` | float | 25.0 | > 0 | [15.0, 20.0, 25.0, 30.0] | Stop loss global (%) |
| `sides` | list | ["long"] | — | — | Côtés autorisés |
| `leverage` | int | 6 | 1-20 | — | Levier (fixe) |
| `timeframe` | str | "1h" | — | — | Timeframe (fixe) |

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **720 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| SMA (close) | 1h | Base des enveloppes + TP dynamique |

Stratégie minimaliste — un seul indicateur technique.

## Résultats WFO

### Première série (5 assets, avant fix grading)

| Asset | Grade | OOS Sharpe | Consistance | OOS/IS Ratio |
|-------|-------|------------|-------------|--------------|
| BTC | B | 14.27 | 100% | 1.84 |
| ETH | C | 4.22 | 85% | 0.54 |
| SOL | C | 8.23 | 85% | 0.82 |
| DOGE | D | 3.58 | 92% | 0.38 |
| LINK | F | 1.46 | 75% | 0.17 |

### Après fix grading (Sprint 15c/15d, 23 assets)

| Asset | Grade | Score | Sharpe | Consistance |
|-------|-------|-------|--------|-------------|
| ETH | A | 88 | 5.43 | 68% |
| DOGE | A | 85 | 6.90 | 97% |
| SOL | A | 85 | 9.02 | 92% |
| 18 assets | B | 71-81 | 4.98-11.43 | 62-89% |
| BNB | D | 50 | 3.47 | 46% |
| BTC | D | 47 | 3.20 | 40% |

### Per-asset overrides en production

20 assets avec overrides sur `ma_period`, `num_levels`, `envelope_start`, `envelope_step`, `sl_percent`.

### Points forts

- Première preuve que le DCA multi-niveaux fonctionne en crypto
- Logique très simple (1 seul indicateur : SMA)
- 21/23 assets Grade A ou B

### Points faibles

- 2 Grade D (BNB, BTC) — enveloppes % fixes pas assez adaptatives
- Remplacée par grid_atr (ATR adaptatif > % fixes)

## Remarques

- **P&L overflow historique** : GridStrategyRunner ne déduisait pas la marge du capital → compounding exponentiel → capital à 9.4 quintillions$. Fix : margin accounting + `_realized_pnl` tracking
- **Double-counting fees** : `close_all_positions()` inclut déjà `entry_fee` dans `net_pnl`. Ne pas déduire aussi à l'ouverture
- **Kill switch grid** : utiliser `_realized_pnl`, pas `capital - initial_capital` (marge verrouillée = fausse "perte")
- **MagicMock piège** : `hasattr(mock, "anything")` retourne toujours True → utiliser `isinstance(getattr(obj, "x", None), (int, float))`
- **Warm-up compound overflow** : batch historique → capital 10k à 83M$. Fix : sizing fixe `_initial_capital` pendant warm-up
- **Différence avec grid_atr** : envelope_dca utilise des % fixes, grid_atr utilise des multiples d'ATR (adaptatif à la volatilité)
