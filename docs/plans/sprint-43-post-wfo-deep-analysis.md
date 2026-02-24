# Sprint 43 — Post-WFO Deep Analysis

**Date :** Février 2026
**Status :** Terminé

## Problème

Le grade WFO (A/B/C) est insuffisant pour décider de l'activation d'une stratégie. Un Grade B peut masquer :
- Un SL×leverage > 100% (margin call immédiat)
- Un Sharpe négatif en régime RANGE (perd de l'argent en consolidation)
- Un DSR=0 (risk de data mining)
- Un CI95 entièrement négatif (pas de edge statistique sur Bitget)

## Solution

Script `scripts/analyze_wfo_deep.py` qui lit la DB et produit un verdict **VIABLE / BORDERLINE / ELIMINATED** par asset Grade A/B.

## Red flags implémentés

| Sévérité | Check | Seuil |
|----------|-------|-------|
| CRITICAL | SL×leverage | > 100% du capital |
| WARNING  | SL×leverage | > 80% du capital |
| CRITICAL | Sharpe RANGE | < 0 |
| CRITICAL | Sharpe régime dominant | < 0 |
| CRITICAL | CI95 entièrement négatif | ci_high < 0 |
| CRITICAL | Sharpe régime quelconque | < -5 |
| WARNING  | DSR | = 0 (data mining risk) |
| WARNING  | Bitget trades | < 10 |
| WARNING  | OOS/IS ratio | > 5 |

## Verdict logic

```
CRITICAL présent → ELIMINATED
WARNING seul     → BORDERLINE
Aucun flag       → VIABLE
```

## Validation grid_boltrend (6 assets Grade B)

| Asset | Verdict | Raison principale |
|-------|---------|-------------------|
| BTC/USDT | VIABLE | Aucun flag critique |
| ETH/USDT | BORDERLINE | WARNING DSR=0 |
| DOGE/USDT | ELIMINATED | Sharpe RANGE=-1.02 (CRITICAL) |
| SOL/USDT | ELIMINATED | CI95 négatif |
| XRP/USDT | ELIMINATED | Sharpe RANGE négatif |
| BNB/USDT | ELIMINATED | SL×leverage > 80% |

## Fichiers modifiés

- `scripts/analyze_wfo_deep.py` (CRÉÉ, 280 lignes, lecture seule)
- `docs/WORKFLOW_WFO.md` — Étape 1b (Deep Analysis) insérée, Étape 1b → 1c (Apply)
- `docs/STRATEGIES.md` — Étapes 11 (Deep Analysis) + 12 (Apply) ajoutées
- `COMMANDS.md` — Section Deep Analysis dans §19
- `docs/ROADMAP.md` — Sprint 43 ajouté

## Tests

0 tests ajoutés (script lecture seule, pas de logique métier modifiable).
Tests existants : 1840 passants (échec pré-existant `test_assets_count` non lié à ce sprint).

## Usage

```bash
# Analyser une stratégie
uv run python -m scripts.analyze_wfo_deep --strategy grid_boltrend

# Analyser toutes les stratégies avec résultats Grade A/B
uv run python -m scripts.analyze_wfo_deep --all
```
