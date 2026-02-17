# orderflow — Orderflow (Non Implémenté)

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `orderflow` |
| Catégorie | — |
| Timeframe | 1m (config) |
| Sprint d'origine | — |
| Grade actuel | — |
| Statut | **Non implémenté** (config placeholder) |
| Fichier source | Aucun (`backend/strategies/orderflow.py` n'existe pas) |
| Config class | `OrderflowConfig` (`backend/core/config.py:82`) |

## Description

Stratégie d'orderflow basée sur les déséquilibres du carnet d'ordres, les grosses transactions et l'absorption. Prévue comme signal de confirmation (pas autonome).

**Seule la config existe** — aucune classe `OrderflowStrategy` n'a été implémentée. Le fichier `backend/strategies/orderflow.py` n'existe pas.

## Config existante

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `enabled` | bool | false | Toujours désactivé |
| `timeframe` | str | "1m" | Timeframe cible |
| `imbalance_threshold` | float | 2.0 | Seuil de déséquilibre bid/ask |
| `large_order_multiplier` | float | 5.0 | Multiplicateur pour détecter les grosses transactions |
| `absorption_threshold` | float | 0.7 | Seuil d'absorption (0-1) |
| `confirmation_only` | bool | true | Signal de confirmation uniquement |
| `weight` | float | 0.20 | Poids dans le score composite |

**Pas de `per_asset`**, pas de `get_params_for_symbol()`, pas de `live_eligible`.

## Remarques

- Initialement prévue comme signal de confirmation pour les autres stratégies (pas autonome)
- Nécessiterait un flux de données orderbook (WebSocket L2/L3) non implémenté
- La config `confirmation_only: true` indique l'intention originale
- Non prioritaire étant donné que les stratégies mono-position n'ont pas d'edge en crypto
