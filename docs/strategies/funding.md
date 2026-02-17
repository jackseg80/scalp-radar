# funding — Funding Rate Arbitrage (Scalp)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `funding` |
| Catégorie | Scalp mono-position |
| Timeframe | 15m |
| Sprint d'origine | Sprint 3 |
| Grade actuel | Non évalué (non backtestable) |
| Statut | **Désactivé** (paper only) |
| Fichier source | `backend/strategies/funding.py` |
| Config class | `FundingConfig` (`backend/core/config.py:113`) |

## Description

Scalp lent sur taux de financement extrêmes. Quand le funding rate est extrêmement négatif, les shorts paient les longs → signal LONG. Quand il est extrêmement positif → signal SHORT. Inclut un délai de confirmation pour éviter les faux signaux transitoires.

Non backtestable sur historiques car les données de funding rate ne sont pas disponibles en DB pour la validation WFO standard. Validation en paper trading uniquement.

**Régime ciblé** : Toujours rapporté comme RANGING (pas de détection dynamique). Pertinent quand le funding rate est en déséquilibre.

## Logique d'entrée

Méthode : `evaluate(ctx) -> StrategySignal | None`

### Pré-condition

- `funding_rate` doit être disponible dans `ctx.extra_data`

### Signal

- **LONG** : `funding_rate < extreme_negative_threshold` (défaut -0.03%)
- **SHORT** : `funding_rate > extreme_positive_threshold` (défaut +0.03%)

### Délai de confirmation

La première détection démarre un timer interne. L'entrée ne se fait qu'après `entry_delay_minutes` (défaut 5 min) si le signal est toujours actif. Si le signal disparaît avant le délai, le timer est remis à zéro.

### Score

`min(1.0, intensity × 0.7 + 0.3)` où `intensity = |funding_rate| / max(|thresholds|)`

## Logique de sortie

**TP** : `entry × (1 ± tp_percent / 100)` — défaut 0.4%

**SL** : `entry × (1 ∓ sl_percent / 100)` — défaut 0.2%

**Sortie anticipée** (`check_exit()`) :
- `|funding_rate| < 0.01` (revenu à neutre) → `"signal_exit"`

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `extreme_positive_threshold` | float | 0.03 | — | [0.01, 0.02, 0.03, 0.05] | Seuil FR positif (%) |
| `extreme_negative_threshold` | float | -0.03 | — | — | Seuil FR négatif (%) |
| `entry_delay_minutes` | int | 5 | >= 0 | [0, 3, 5, 10] | Délai avant entrée (min) |
| `tp_percent` | float | 0.4 | > 0 | [0.2, 0.3, 0.4, 0.6] | Take profit (%) |
| `sl_percent` | float | 0.2 | > 0 | [0.1, 0.2, 0.3] | Stop loss (%) |
| `timeframe` | str | "15m" | — | — | Timeframe (note : 15m, pas 5m) |

**Config WFO** : IS = 120j, OOS = 30j, step = 30j, **192 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| Funding rate | externe (extra_data) | Signal unique d'entrée/sortie |
| Close price | 15m | Calcul TP/SL |

Aucun indicateur technique classique — le signal vient entièrement du funding rate.

## Résultats

Non backtestable (pas de données funding rate historiques en DB au moment de la validation). Pas de grade WFO.

### Points forts

- Edge théorique solide (arbitrage de structure du marché)
- Délai de confirmation réduit les faux positifs

### Points faibles

- Non backtestable → pas de validation statistique
- Mono-position (pas de DCA)
- TP/SL très serrés (0.4% / 0.2%)

## Remarques

- **Différence avec grid_funding** : `funding` est mono-position scalp avec FR extrême comme signal. `grid_funding` est DCA multi-niveaux avec FR négatif progressif comme signal
- `live_eligible: false` — pas autorisé en live
- Le timeframe est 15m (pas 5m comme les autres scalps) pour laisser le temps au funding rate de se stabiliser
