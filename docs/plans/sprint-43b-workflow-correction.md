# Sprint 43b — Workflow Correction (Deep Analysis = diagnostic)

**Date :** Février 2026
**Status :** Terminé

## Problème

Sprint 43 avait instauré le Deep Analysis comme étape 1b **obligatoire** entre le WFO et le --apply.
L'idée : filtrer les assets AT RISK avant de les appliquer.

**Preuve que c'est faux :**
- grid_boltrend : BCH (SL×6=120%) et DYDX (DSR=0) → classés AT RISK individuellement
- Portfolio backtest avec les 6 Grade B (incluant BCH et DYDX) : +552.2%, DD -15.3%
- BCH contribue +1018$, DYDX +778$ au portfolio
- Les red flags individuels sont compensés par la diversification

## Correction

### Workflow corrigé

```
WFO → Apply (TOUS Grade A/B) → Portfolio backtest (LE vrai filtre) → Deep Analysis si portfolio échoue
```

### Changements scripts/analyze_wfo_deep.py

- `ELIMINATED` renommé `AT RISK` (moins définitif)
- `print_workflow_advice()` : message "portfolio backtest avec TOUS les assets" (pas de --exclude)
- Docstring : clarification outil diagnostique, pas filtre
- `print_summary_table()` : NOTE diagnostique en sortie

### Changements docs/WORKFLOW_WFO.md

- **Ancien** : Étape 1b (Deep Analysis filtre) → Étape 1c (Apply VIABLE+BORDERLINE seulement)
- **Nouveau** : Étape 1b (Apply TOUS Grade A/B) → Étape 2b (Deep Analysis diagnostic) après Étape 2
- Étape 2 renommée "LE vrai filtre"

### Changements docs/STRATEGIES.md

Checklist "Comment ajouter une nouvelle stratégie" :
- Étape 11 : Apply (TOUS) + Étape 12 : Portfolio backtest + Étape 13 : Deep Analysis si échoue

### Changements COMMANDS.md

§19 Deep Analysis : description mise à jour ("outil DIAGNOSTIQUE — pas un filtre"), workflow corrigé

### Changements docs/ROADMAP.md

- Sprint 43b ajouté
- grid_boltrend sorti de la section "stratégies abandonnées" (re-validé Sprint 43)
- "Insight clé" mis à jour : 3 stratégies viables (grid_atr, grid_multi_tf, **grid_boltrend**)
- ÉTAT ACTUEL : + Sprint 43b

## Résultats grid_boltrend (référence)

| Métrique | Valeur |
|----------|--------|
| WFO | 6 Grade B / 19 assets |
| Assets | BTC, ETH, DOGE, LINK, DYDX, BCH |
| Leverage | 6x |
| Return (2008j) | +552.2% |
| Max DD | -15.3% |
| Alpha vs BTC | +92.1% |
| Kill switch | 0 |
| Régimes | Tous positifs sauf CRASH (-0.37$/j) |
| Stress test | 6x confirmé (0 KS) |
| Statut | En attente corrélation avec grid_atr |

## Tests

0 tests (corrections terminologie + docs uniquement, pas de logique métier modifiée).
