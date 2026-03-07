# Scalp-Radar — Guide complet Backtest & WFO (Version Anti-Biais)

## Architecture du système

Le pipeline complet va de la donnée brute à la certification de production :
```
Données historiques (Binance 3+ ans)
  → WFO (Walk-Forward Optimization) par asset
  → Sanity Check (analyze_wfo_deep) : FILTRE DE COHÉRENCE
  → Application des paramètres (Grade A/B + Validés)
  → Portfolio backtest (Validation de diversification)
  → Robustesse Statistique (Bootstrap & Stress)
  → Paper trading → Live trading
```

## Philosophie Anti-Biais

Pour garantir la fiabilité des revenus live, ce workflow impose des barrières strictes :
1. **Anti-Data Snooping** : Utilisation du DSR (Deflated Sharpe) et de l'Embargo 7j pour éviter de "prédire" les données OOS.
2. **Anti-Selection Bias** : Interdiction de retirer un actif simplement parce qu'il perd de l'argent en backtest portfolio (on ne "nettoie" pas le passé).
3. **Anti-Overfitting** : Pénalité `window_factor` pour rejeter les combos "chanceux" sur peu de fenêtres historiques.
4. **Cohérence Physique** : Rejet immédiat si les paramètres violent les limites de marge ou de corrélation.

---

## 3. Workflow complet — Validation de Production

### Étape 0 — Calcul leverage (AVANT tout WFO)
Calcul mathématique des limites basées sur le Stop Loss max et le Kill Switch.
**Règle critique** : Fixer le leverage dans `strategies.yaml` AVANT le WFO. Le changer après invalide tout le travail.

### Étape 1 — WFO mono-asset (21 assets)
Identifie les actifs ayant un "edge" statistique.
```bash
uv run python -m scripts.optimize --strategy grid_atr --all-symbols --subprocess -v
```

### Étape 2 — Sanity Check (Filtrage de Cohérence)
**Indispensable AVANT toute application de paramètres.**
```bash
uv run python -m scripts.analyze_wfo_deep --strategy grid_atr
```
**Critères de REJET AUTOMATIQUE (même si Grade A/B) :**
- **Risque de Ruine** : `SL % × Leverage > 100%`. L'actif est exclu s'il peut liquider sa propre marge.
- **Concentration de Régime** : Si > 80% du profit vient d'un seul régime (ex: uniquement les Crashs) avec < 10 trades ailleurs.
- **Significativité Bitget** : Si l'actif a moins de 5-10 trades réels sur l'historique Bitget récent.

### Étape 3 — Application des Paramètres (Le "Commit")
Seuls les actifs ayant passé l'Étape 1 (Grade A/B) **ET** l'Étape 2 (Sanity Check) sont injectés dans la configuration.
```bash
# Appliquer avec exclusion des rejetés de l'étape 2
uv run python -m scripts.optimize --strategy grid_atr --apply --exclude "SYM1,SYM2"
```

### Étape 4 — Portfolio backtest (Validation de Diversification)
Simule l'ensemble des actifs validés avec capital partagé.
**RÈGLE D'OR** : On ne modifie pas la liste des actifs après avoir vu le résultat. Si le résultat global est mauvais, c'est la **stratégie** ou le **levier** qui est à revoir à l'étape 0.
```bash
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days 365 --save --label "strat_V2_levX_DATE"
```

### Étape 5 — Robustesse Statistique (Le Juge Final)
Validation par Bootstrap et scénarios de stress.
```bash
uv run python -m scripts.portfolio_robustness --label "label_étape_4" --save
```
**Verdict Final :**
- **VIABLE** : Prêt pour déploiement.
- **FAIL** : Échec structurel. Retour à l'étape 0 (réduction levier ou refonte logique).

---

## 4. Maintenance et Dérive (Drift)
Les paramètres ont une date de péremption de **60 jours**. 
- Une alerte visuelle apparaît dans le frontend (`Périmé`) pour signaler les actifs nécessitant un re-run complet.
- Ne jamais "tuner" les paramètres à la main dans `strategies.yaml`. Tout changement doit repasser par le workflow.
