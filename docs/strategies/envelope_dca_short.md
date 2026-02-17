# envelope_dca_short — Envelope DCA Short (Miroir SHORT)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `envelope_dca_short` |
| Catégorie | Grid/DCA |
| Timeframe | 1h |
| Sprint d'origine | Sprint 15 |
| Grade actuel | Non évalué (WFO en attente) |
| Statut | **Désactivé** |
| Fichier source | `backend/strategies/envelope_dca_short.py` |
| Config class | `EnvelopeDCAShortConfig` (`backend/core/config.py:206`) |

## Description

Miroir SHORT d'envelope_dca. Sous-classe minimale (26 lignes) qui hérite toute la logique de `EnvelopeDCAStrategy` et ne change que les défauts : `sides = ["short"]`, SL et niveaux ajustés.

L'architecture en sous-classe a été préférée à un paramétrage de l'existant pour :
- Éviter toute régression sur la stratégie LONG en production
- Supporter deux grilles WFO indépendantes
- Fast engine avec `direction=-1`

**Régime ciblé** : Range et trend modéré (côté SHORT).

## Logique d'entrée

Identique à envelope_dca (héritée). Seul le défaut `sides = ["short"]` change, ce qui fait que seuls les niveaux SHORT sont générés :

```
SMA = SMA(close, ma_period)

Pour i = 0 à num_levels - 1 :
  lower_offset = envelope_start + i × envelope_step
  upper_offset = 1 / (1 - lower_offset) - 1

  SHORT : entry_price = SMA × (1 + upper_offset)
```

## Logique de sortie

Identique à envelope_dca (héritée) :

- **TP** : `close <= SMA` (retour à la SMA par le bas)
- **SL** : `close >= avg_entry_price × (1 + sl_percent / 100)`

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `ma_period` | int | **7** | 2-50 | [5, 7, 10] | Période de la SMA |
| `num_levels` | int | **2** | 1-6 | [2, 3, 4] | Nombre de niveaux DCA |
| `envelope_start` | float | 0.05 | > 0 | [0.05, 0.07, 0.10] | Offset du 1er niveau (5%) |
| `envelope_step` | float | **0.02** | > 0 | [0.02, 0.03, 0.05] | Incrément entre niveaux |
| `sl_percent` | float | **20.0** | > 0 | [15.0, 20.0, 25.0, 30.0] | Stop loss global (%) |
| `sides` | list | **["short"]** | — | — | SHORT-only |
| `leverage` | int | 6 | 1-20 | — | Levier (fixe) |
| `timeframe` | str | "1h" | — | — | Timeframe (fixe) |

**En gras** : différences avec envelope_dca.

**Config WFO** : IS = 180j, OOS = 60j, step = 60j, **324 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| SMA (close) | 1h | Base des enveloppes + TP dynamique |

## Résultats

WFO non lancé. Pas de résultats disponibles.

### Points forts

- Sous-classe minimale — 0 risque de régression sur le LONG
- Couverture SHORT complémentaire à envelope_dca

### Points faibles

- Pas de validation WFO
- Supersédée par grid_multi_tf (qui supporte LONG et SHORT nativement via Supertrend 4h)

## Remarques

- **OHLC heuristic** : inversée pour SHORT (bougie rouge = favorable)
- **Découverte dynamique** : le frontend détecte automatiquement les nouvelles stratégies via l'API `/api/optimization/strategies`
- **Défauts ajustés** : `num_levels=2` (vs 4 LONG), `envelope_step=0.02` (vs 0.05 LONG), `sl_percent=20.0` (vs 25.0 LONG) — les enveloppes SHORT sont plus serrées car les rallyes bear market sont plus courts
- **Per-asset** : aucun override (vide)
