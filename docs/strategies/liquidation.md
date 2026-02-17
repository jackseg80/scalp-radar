# liquidation — Liquidation Zone Hunting

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `liquidation` |
| Catégorie | Scalp mono-position |
| Timeframe | 5m |
| Sprint d'origine | Sprint 3 |
| Grade actuel | Non évalué (non backtestable) |
| Statut | **Désactivé** (paper only) |
| Fichier source | `backend/strategies/liquidation.py` |
| Config class | `LiquidationConfig` (`backend/core/config.py:63`) |

## Description

Estime les zones de liquidation en calculant les niveaux de prix auxquels les positions avec le levier moyen du marché seraient liquidées. Quand le prix approche ces zones avec un open interest élevé, anticipe une cascade de liquidations et prend position dans le sens de la cascade.

Non backtestable car les données d'open interest historiques ne sont pas disponibles en DB pour la validation WFO.

**Régime ciblé** : Toujours rapporté comme RANGING (pas de détection dynamique). Pertinent en forte volatilité avec positions fortement levierisées.

## Logique d'entrée

Méthode : `evaluate(ctx) -> StrategySignal | None`

### Pré-condition

- `open_interest` et `oi_change_pct` doivent être dans `ctx.extra_data`
- `oi_change_pct >= oi_change_threshold` (défaut 5.0%) — l'OI doit avoir augmenté significativement (leviers chargés)

### Calcul des zones de liquidation

```
liq_long_zone  = close × (1 - 1 / leverage_estimate)   ← ex: 15x → ~6.67% sous le prix
liq_short_zone = close × (1 + 1 / leverage_estimate)   ← ex: 15x → ~6.67% au-dessus
```

### Signal

- Proximité : `dist_to_zone < zone_buffer_percent / 100` (défaut 1.5%)
- **LONG** : prix approche la zone de liquidation des shorts (short squeeze anticipé)
- **SHORT** : prix approche la zone de liquidation des longs (cascade de liquidation)

### Score

`oi_score × 0.5 + proximity_score × 0.5`

## Logique de sortie

**TP** : `entry × (1 ± tp_percent / 100)` — défaut 0.8%

**SL** : `entry × (1 ∓ sl_percent / 100)` — défaut 0.4%

**Sortie anticipée** (`check_exit()`) :
- `oi_change_pct < -3.0` (OI chute brutalement, cascade terminée) → `"signal_exit"`

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `oi_change_threshold` | float | 5.0 | > 0 | [3.0, 5.0, 7.0, 10.0] | Seuil hausse OI (%) |
| `leverage_estimate` | int | 15 | >= 1 | [10, 15, 20, 25] | Levier moyen estimé du marché |
| `zone_buffer_percent` | float | 1.5 | > 0 | [0.5, 1.0, 1.5, 2.0] | Distance buffer aux zones (%) |
| `tp_percent` | float | 0.8 | > 0 | [0.4, 0.6, 0.8] | Take profit (%) |
| `sl_percent` | float | 0.4 | > 0 | [0.2, 0.3, 0.4, 0.5] | Stop loss (%) |
| `timeframe` | str | "5m" | — | — | Timeframe |

**Config WFO** : IS = 120j, OOS = 30j, step = 30j, **768 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| Open Interest | externe (extra_data) | Détection de leviers chargés |
| OI change % | externe (extra_data) | Seuil d'activation |
| Close price | 5m | Calcul zones de liquidation + TP/SL |

Aucun indicateur technique classique.

## Résultats

Non backtestable (pas de données OI historiques en DB). Pas de grade WFO.

### Points forts

- Concept intéressant basé sur la microstructure du marché
- Sortie anticipée sur chute d'OI (détection fin de cascade)

### Points faibles

- Non backtestable → pas de validation statistique
- Le `leverage_estimate` est une approximation grossière (15x par défaut)
- Les zones de liquidation réelles dépendent de la distribution des positions, pas d'un levier moyen
- Mono-position (pas de DCA)

## Remarques

- `live_eligible: false` — pas autorisé en live
- Les données d'OI sont récupérées via `extra_data` mais pas stockées en DB pour les backtests historiques
- Le `leverage_estimate` de 15x est une hypothèse raisonnable pour les futures crypto, mais varie selon les assets et les conditions de marché
