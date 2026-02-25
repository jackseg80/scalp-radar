# Scalp-Radar — Guide complet Backtest & WFO

## Architecture du système

Le projet est un bot de trading crypto automatisé tournant sur Bitget futures. Le pipeline complet va du backtest à la production :

```
Données historiques (Binance 3+ ans)
  → WFO (Walk-Forward Optimization) par asset
  → Grading (A/B/C/D/F)
  → Portfolio backtest (assets Grade A/B ensemble)
  → Stress test leverage
  → Paper trading → Live trading
```

---

## 1. Données

### Sources
- **Binance** : historique profond (3+ ans), utilisé pour le WFO et backtest
- **Bitget** : données récentes (~90j), utilisé pour la validation cross-exchange

### Fetch des données
```bash
# Tous les assets, Binance, 3 ans
uv run python -m scripts.fetch_history --exchange binance --days 1100

# Un asset spécifique
uv run python -m scripts.fetch_history --exchange binance --symbol BTC/USDT --days 1100

# Bitget pour validation
uv run python -m scripts.fetch_history --exchange bitget --days 90

# Vérifier les données disponibles
uv run python -m scripts.optimize --check-data
```

Les candles sont stockées en SQLite (DB locale) avec colonnes exchange, symbol, timeframe, timestamp, OHLCV.

Référence complète des commandes fetch : voir [COMMANDS.md § 3](../COMMANDS.md#3-données-historiques).

### Assets disponibles (21)
BTC, ETH, SOL, DOGE, LINK, ADA, AAVE, ARB, AVAX, BCH, BNB, CRV, DYDX, FET, GALA, ICP, NEAR, OP, SUI, UNI, XRP

---

## 2. Stratégies existantes

### Stratégies Grid (DCA) — les seules viables
Toutes sont des stratégies **mean-reversion** avec grille de niveaux DCA :

| Stratégie | Direction | Filtre | Statut |
|-----------|-----------|--------|--------|
| **grid_atr** | LONG only | Enveloppes ATR autour de SMA (V2 : min_grid_spacing_pct + min_profit_pct) | ✅ LIVE 7x, 14 assets |
| **grid_multi_tf** | LONG + SHORT | Supertrend 4h filtre + Grid ATR 1h | ✅ LIVE 3x (test, cible 6x), 14 assets |
| **grid_boltrend** | LONG + SHORT | Bollinger + Supertrend | PAPER, 2 assets en config, pas re-WFO depuis corrections moteur 40a |
| **grid_momentum** | LONG + SHORT | Donchian breakout + DCA pullback | Non validé (17e stratégie, WFO à lancer) |
| grid_range_atr | LONG + SHORT | Range-bound ATR | Non validé |
| grid_funding | LONG | Funding rate négatif | Non validé |
| grid_trend | LONG + SHORT | Trend following EMA + ADX | Échoue en forward test |

Pour le détail de chaque stratégie (logique, paramètres, indicateurs), voir [STRATEGIES.md](STRATEGIES.md).

### Principe de fonctionnement (toutes les grid)
1. **Niveaux d'entrée** : la stratégie calcule N niveaux de prix sous/au-dessus de la SMA
2. **DCA** : quand le prix touche un niveau → ouvrir une position (level 1, puis 2, puis 3...)
3. **TP** : quand le prix revient à la SMA → fermer TOUTE la grille (TP global)
4. **SL** : si le prix continue à s'éloigner au-delà du sl_percent → fermer en perte

### Paramètres typiques optimisés par le WFO
- `timeframe` : 1h (standard pour toutes les grid)
- `ma_period` : période de la SMA (ex: 50, 100, 200)
- `atr_period` : période de l'ATR (ex: 14, 21)
- `atr_mult` : multiplicateur ATR pour les niveaux (ex: 1.0, 1.5, 2.0)
- `sl_percent` : stop loss en % (ex: 15%, 20%, 25%)
- `num_levels` : nombre de niveaux DCA (ex: 2, 3, 4)
- Divers filtres selon la stratégie (trend_period, supertrend_period, etc.)

Les grilles de paramètres sont définies dans `config/param_grids.yaml`.

---

## 3. Workflow complet — Nouvelle stratégie ou re-validation

### Étape 0 — Calcul leverage (AVANT tout WFO)

Calcul purement mathématique, pas besoin de backtest :

```
Plancher = kill_switch_pct / (SL_max × margin_guard)
         = 45% / (25% × 70%) = 2.57 → arrondi 3x

Plafond  = 80% / (SL_max × avg_margin_usage)
         = 80% / (25% × 50%) = 6.4 → arrondi 6x

Fourchette typique : [3x, 6x]
```

Inputs depuis `param_grids.yaml` (SL_max, num_levels_max) et `risk.yaml` (kill_switch_pct, max_margin_ratio).

**Règle critique** : écrire le leverage dans `strategies.yaml` AVANT le WFO. Le WFO optimise en fonction du leverage — changer après invalide les résultats.

Si la stratégie fait du SHORT (grid_multi_tf, grid_range_atr) → rester dans le bas de la fourchette (risque short squeeze).

### Étape 1 — WFO mono-asset (21 assets)

```bash
# Tous les assets
uv run python -m scripts.optimize --strategy grid_atr --all-symbols --subprocess -v

# Un seul asset
uv run python -m scripts.optimize --strategy grid_atr --symbol BTC/USDT -v

# Reprendre après crash
uv run python -m scripts.optimize --strategy grid_atr --all-symbols --resume
```

Référence complète des flags : voir [COMMANDS.md § 2](../COMMANDS.md#2-lancer-des-optimisations-wfo).

Le WFO fait :
1. Découpe l'historique en fenêtres glissantes (config dans `param_grids.yaml`, section `wfo:` par stratégie) :
   - Stratégies 1h standard (grid_atr, grid_boltrend, grid_multi_tf, grid_momentum) : IS=180j, OOS=60j, step=60j → ~16 fenêtres sur 1130j
   - grid_funding, grid_trend : IS=360j, OOS=90j, step=90j → ~8 fenêtres
   - Stratégies 5m (legacy) : IS=120j, OOS=30j, step=30j → ~30 fenêtres
2. Grid search sur toutes les combinaisons de paramètres (IS = in-sample)
3. Teste le meilleur combo sur la fenêtre suivante (OOS = out-of-sample)
4. Répète sur toutes les fenêtres (~8 à ~16 selon la stratégie)
5. Calcule un score composite : ratio OOS/IS, Monte Carlo p-value, DSR, stabilité, consistency
6. Attribue un Grade (A/B/C/D/F) avec pénalité shallow si < 24 fenêtres

**Critère** : Grade A ou B sur ≥ 5 assets pour continuer.

**window_factor** (fix critique Sprint 38b) : pénalise les combos évalués sur peu de fenêtres. Sans ça, des combos "parfaits" sur 1-2 fenêtres polluent les résultats.

### Étape 2 — Appliquer les résultats (TOUS les Grade A/B)

```bash
# Appliquer les params Grade A/B dans strategies.yaml — TOUS les assets, sans filtre
uv run python -m scripts.optimize --strategy grid_atr --apply
```

Écrit les paramètres optimaux dans `config/strategies.yaml` sous `per_asset:` pour chaque asset Grade A/B. Les params convergents deviennent les défauts, les divergents vont dans per_asset.

**grid_atr V2 (Sprint 47)** : la grille WFO inclut `min_grid_spacing_pct` et `min_profit_pct`. Ces params sont propagés dans per_asset → strategies.yaml. Le WFO peut sélectionner 0.0 (classique) ou une valeur non-nulle pour activer les protections basse-volatilité. Après `--apply`, lancer `purge_wfo_duplicates` si la DB a des doublons `is_latest` (Sprint 47b fix).

```powershell
# Purger les doublons is_latest (après WFO multi-TF)
uv run python -m scripts.purge_wfo_duplicates --dry-run
uv run python -m scripts.purge_wfo_duplicates
```

> **Règle** : appliquer TOUS les Grade A/B sans filtrer. Les red flags individuels peuvent être
> compensés par la diversification (prouvé grid_boltrend : BCH SL×6=120% et DYDX DSR=0 → mais
> portfolio +552.2% avec les 6 assets. Le portfolio backtest décidera).

**Guard timeframe** : si un asset a un timeframe ≠ 1h, `--apply` bloque. Solutions :
```bash
# Re-tester en 1h
uv run python -m scripts.optimize --strategy grid_atr --symbols BCH/USDT --force-timeframe 1h

# Ou exclure (uniquement si timeframe incompatible, pas pour des red flags)
uv run python -m scripts.optimize --strategy grid_atr --apply --exclude BCH/USDT
```

### Étape 3 — Portfolio backtest (LE vrai filtre)

> **IMPORTANT — CHAQUE STRATÉGIE EST INDÉPENDANTE** : Chaque stratégie a son propre sous-compte
> Bitget avec son propre capital. On backteste chaque stratégie séparément.
> Ne PAS utiliser `--strategies "strat1:...+strat2:..."` — ce mode ne correspond pas à la production.

Simule TOUS les assets Grade A/B d'**une seule stratégie** avec capital partagé, comme en production :

```bash
# Auto-détection durée (max historique commun)
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days auto --capital 1000

# Durée spécifique
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days 365 --capital 1000

# Sauvegarder en DB (requis pour les étapes 3b, 5 et 6)
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days auto --save --label "grid_atr_7x"

# Comparer des leverages
uv run python -m scripts.portfolio_backtest --strategy grid_atr --leverage 5
uv run python -m scripts.portfolio_backtest --strategy grid_atr --leverage 7
```

Référence complète des flags : voir [COMMANDS.md § 12](../COMMANDS.md#12-portfolio-backtest-portfolio_backtest).

**Le portfolio backtest inclut** :
- Capital partagé entre N assets (sizing = capital / N / num_levels)
- Margin guard 70% (pas d'ouverture si marge > 70% du capital)
- Kill switch simulé (45% drawdown sur 24h glissantes)
- Taker fees sur toutes les exits (correction Sprint 40a)
- Embargo 7 jours IS→OOS (évite le leaking)
- Analyse par régime de marché (RANGE, BULL, BEAR, CRASH)
- Equity curve, drawdown curve, alpha vs BTC buy-and-hold

**Critères** : Return > 0, Max DD < -35%, 0 kill switch, W-SL < kill_switch - 5pts

> **C'est ici que la décision finale se prend.** Les red flags individuels (SL×leverage, DSR, régimes)
> vus en Deep Analysis peuvent être compensés par la diversification. Seul le résultat combiné compte.
> Exemple réel : grid_boltrend, 4/6 assets AT RISK individuellement → portfolio **+552.2%, DD -15.3%**.

### Étape 3b — Deep Analysis post-WFO (DIAGNOSTIC — pas un filtre)

```bash
uv run python -m scripts.analyze_wfo_deep --strategy grid_atr
```

**Ce n'est PAS un critère GO/NO-GO.** Utiliser après le portfolio backtest uniquement pour diagnostiquer.

| Utilisation | Description |
|-------------|-------------|
| Profil par asset | Sharpe par régime, SL×leverage, DSR, CI95 par asset |
| Portfolio échoue | Identifier quel(s) asset(s) retire, tester portfolio SANS eux |
| Monitoring | Documenter forces/faiblesses pour le suivi live |

**Si un asset semble problématique** : tester le portfolio SANS cet asset. Si le portfolio s'améliore → exclure. Sinon → garder (la diversification l'absorbe).

**Verdicts** :
- `[OK] VIABLE` : aucun red flag critique
- `[~~] BORDERLINE` : warnings seulement (DSR, trades faibles, OOS/IS élevé)
- `[!?] AT RISK` : flags critiques détectés — tester l'impact sur le portfolio avant de décider

### Étape 4 — Stress test leverage

Confirme que le leverage choisi tient sous pression :

```bash
uv run python -m scripts.stress_test_leverage --strategy grid_atr
```

Référence complète des flags : voir [COMMANDS.md § 16](../COMMANDS.md#16-stress-test-leverage).

Teste leverage ± 1x (ex: si 6x choisi → teste 5x, 6x, 7x) sur plusieurs fenêtres temporelles. Kill switch désactivé (99%) pour voir le vrai max drawdown.

**Critères** : Liq distance > 50%, KS@45 = 0, W-SL < kill_switch - 5pts

### Étape 5 — Portfolio Robustness

Valide la robustesse statistique du portfolio backtest sauvé à l'étape 3 :

```bash
uv run python -m scripts.portfolio_robustness --label "<label_étape_3>" --save
```

4 méthodes complémentaires :
- **Block Bootstrap** (5000 sims, blocs 7j) — CI95 sur return et max DD, probabilité de perte
- **Regime Stress** (5 scénarios × 1000 sims) — bear prolongé, double crash, range permanent, bull run, crypto winter
- **Historical Stress** — performance réelle pendant COVID, China ban, LUNA, FTX, Aug 2024 crash
- **CVaR** — tail risk journalier, compound 30j, par régime

**Critères GO/NO-GO** :
- CI95 return borne basse > 0%
- Probabilité de perte < 10%
- CVaR 5% 30j (compound) < kill_switch (45%)
- Survit à ≥ 3/5 crashes historiques avec DD < -40%

**Verdicts** :
- `VIABLE` (tout vert) → paper trading
- `CAUTION` (1 warning) → paper trading 1 mois avec surveillance renforcée
- `FAIL` (≥ 2 critères rouges) → investiguer avant de déployer

> Note : le `--label` doit correspondre exactement au label utilisé lors du `--save` à l'étape 3.

### Étape 6 — Corrélation inter-stratégies (si multi-stratégie)

> **Rappel** : chaque stratégie a son propre sous-compte Bitget avec son propre capital.
> L'objectif ici n'est pas de créer un portfolio unifié, mais de vérifier que les stratégies
> ne crashent pas au même moment. L'allocation = proportion de capital entre les sous-comptes.

```bash
# Comparer la corrélation DD entre deux labels sauvés
uv run python -m scripts.analyze_correlation --labels "grid_atr_7x,grid_multi_tf_6x"

# Lister tous les labels disponibles
uv run python -m scripts.analyze_correlation --list

# Avec noms personnalisés pour le rapport
uv run python -m scripts.analyze_correlation --labels "grid_atr_7x,grid_multi_tf_6x" --label-names "ATR,MTF"
```

Référence complète des flags : voir [COMMANDS.md § 20](../COMMANDS.md#20-analyze-correlation).

Mesure la corrélation des drawdowns entre stratégies. **Cible : r < 0.3**.

- Si r < 0.3 → bonne diversification → optimiser l'allocation (ex: 40/60 ATR/MTF)
- Si r ≥ 0.3 → les stratégies crashent ensemble → réévaluer l'allocation ou ne pas combiner

### Étape 7 — Paper trading (≥ 2 semaines)

Déployer avec `enabled: true, live_eligible: false` sur le serveur. Observer les signaux, les trades paper, vérifier la cohérence avec le backtest.

- Verdict **VIABLE** → paper minimum 2 semaines
- Verdict **CAUTION** → paper minimum 1 mois avec surveillance renforcée

### Étape 8 — Live trading (progressif)

1. Sous-compte Bitget dédié + API keys (voir [COMMANDS.md § 21](../COMMANDS.md#21-multi-executor-sprint-36b))
2. `live_eligible: true` + leverage réduit (3x)
3. Montée progressive : 3x → 5x → 6x (2 semaines par palier)

---

## 4. Commandes utiles

### Recalculer les grades sans re-WFO
```bash
uv run python -m scripts.optimize --regrade --strategy grid_atr
```

### Analyse de régression WFO
```bash
uv run python -m scripts.analyze_wfo_regression --strategy grid_atr --leverage 7
```

### ~~Multi-stratégie portfolio~~ (déprécié)

> **DÉPRÉCIÉ** : le flag `--strategies` ne correspond pas à la production (chaque stratégie a son
> propre sous-compte Bitget). Backtester chaque stratégie séparément (étape 3) puis comparer
> avec `analyze_correlation` (étape 6).

```bash
# Comparer la corrélation DD entre deux stratégies déjà backtestées
uv run python -m scripts.analyze_correlation --labels "label_strat1,label_strat2"
```

Pour toutes les commandes CLI : voir [COMMANDS.md](../COMMANDS.md).

---

## 5. Moteur de backtest — Corrections critiques (Sprint 40a)

Trois corrections majeures appliquées au moteur. Tout WFO/backtest fait AVANT ces corrections est invalide :

### 5a. Taker fees sur les exits
Les fast engines ne comptaient pas les taker fees sur les fermetures (TP, SL, force close). Impact : +2-5% de return fictif par asset.

### 5b. Margin guard 70%
Les fast engines ne bloquaient pas de marge à l'ouverture → capital jamais réduit → inflation illimitée. Fix : `capital -= notional/leverage` à l'ouverture.

### 5c. Embargo 7 jours IS→OOS
Les 7 premiers jours de chaque fenêtre OOS sont ignorés pour éviter le leaking de la dernière fenêtre IS.

### 5d. Window factor (Sprint 38b)
Les combos évaluées sur peu de fenêtres OOS recevaient des scores "parfaits". Fix : `window_factor = min(1.0, n_windows / total_windows)` pénalise les combos incomplètes.

---

## 6. Grading — Comment un asset obtient son Grade

Le score est calculé sur 100 points :

| Critère | Poids | Description |
|---------|-------|-------------|
| OOS/IS ratio | 20 | Performance out-of-sample vs in-sample |
| Monte Carlo p-value | 20 | Significativité statistique |
| DSR (Deflated Sharpe) | 15 | Sharpe corrigé du data snooping |
| Stabilité params | 10 | Les params sont-ils sur un plateau ? |
| Consistency | 20 | % de fenêtres OOS profitables |
| Bitget transfer | 15 | Validation cross-exchange |

**Pénalités** :
- Shallow validation : -10 (18-23 fenêtres), -20 (12-17), -25 (< 12)
- Cap trades : < 30 trades → cap Grade C, < 50 → cap Grade B

**Grades** :
- A : score ≥ 85
- B : score ≥ 70
- C : score ≥ 55
- D : score ≥ 40
- F : score < 40

Seuls les **Grade A et B** passent en portfolio/paper/live.

---

## 7. Régimes de marché

L'analyse par régime est intégrée au portfolio backtest :

| Régime | Définition | % temps typique |
|--------|-----------|-----------------|
| RANGE | BTC varie < ±20% sur 30j | ~83% |
| BULL | BTC > +20% sur 30j | ~13% |
| BEAR | BTC < -20% sur 30j | ~2% |
| CRASH | BTC DD > -30% en ≤ 14j | ~2% |

Les stratégies grid (mean-reversion) excellent en RANGE, sont rentables en BULL (mais sous-performent HODL), et sont quasi-neutres en BEAR/CRASH.

---

## 8. Résultats actuels de référence (post-corrections 40a)

*Ces résultats proviennent des portfolio backtests Sprint 40c (février 2026), post-corrections moteur 40a. Ils ne sont pas dans ROADMAP.md mais dans les transcripts de conversation.*

### grid_atr — 14 assets Grade A/B, 7x, 1130j

| Métrique | Valeur |
|----------|--------|
| Return | +275.7% |
| Max DD | -9.2% |
| Win rate | 77.7% |
| Alpha vs BTC | +55.4% |
| Kill switch | 0 |
| W-SL | 40.2% |

### grid_multi_tf — 14 assets Grade A/B, 6x (WFO validé), 1130j

*Note : WFO et backtests réalisés à 6x. Production actuelle à 3x (phase test, montée progressive vers 6x).*

| Métrique | Valeur |
|----------|--------|
| Return | +300.1% |
| Max DD | -9.9% |
| Win rate | 75.8% |
| Alpha vs BTC | +81.4% |
| Kill switch | 0 |
| W-SL | 31.1% |

### Combiné 40% ATR / 60% MTF

| Métrique | Valeur |
|----------|--------|
| Return | +291% |
| Max DD | **-5.52%** |
| Ratio return/DD | **52.6** |
| Corrélation DD | r=0.18 |

### Robustness — Sprint 44

| Stratégie | Verdict | CI95 borne basse | Prob. perte | CVaR 30j | Crashes |
|-----------|---------|-----------------|-------------|----------|---------|
| grid_atr 14 assets 7x | **VIABLE** | +157% | 0.0% | 26.9% < 45% | 1/1 OK |
| grid_multi_tf 14 assets 6x | **VIABLE** | +121% | 0.0% | 35.6% < 45% | 1/1 OK |
| grid_boltrend 6 assets 6x | **CAUTION** | +177% | 0.0% | 57.2% > 45% | 4/4 OK |

**Note grid_boltrend** : seul critère en échec = CVaR 30j (57.2% vs kill_switch 45%).
Tous les autres critères sont verts. Paper 1 mois avec surveillance renforcée.

### Corrélation inter-stratégies — Sprint 44b

| Paire | Corrélation DD (r) | Verdict | Allocation optimale |
|-------|-------------------|---------|---------------------|
| grid_atr / grid_multi_tf | **r = 0.18** | Bonne diversification (< 0.3) | 40% ATR / 60% MTF |

---

## 9. Fichiers clés

| Fichier | Rôle |
|---------|------|
| `config/strategies.yaml` | Config des stratégies (params, per_asset, leverage) |
| `config/param_grids.yaml` | Grilles de paramètres pour le WFO |
| `config/risk.yaml` | Kill switch, margin guard, risk params |
| `scripts/optimize.py` | CLI WFO (optimize, apply, regrade) |
| `scripts/portfolio_backtest.py` | CLI portfolio backtest |
| `scripts/stress_test_leverage.py` | CLI stress test leverage |
| `scripts/fetch_history.py` | Fetch candles Binance/Bitget |
| `backend/optimization/walk_forward.py` | Moteur WFO |
| `backend/optimization/fast_multi_backtest.py` | Fast engines (backtests rapides) |
| `backend/optimization/report.py` | Grading et rapports |
| `backend/backtesting/portfolio_engine.py` | Portfolio backtest engine |
| [STRATEGIES.md](STRATEGIES.md) | Détail de chaque stratégie (logique, paramètres) |
| [COMMANDS.md](../COMMANDS.md) | Toutes les commandes CLI |

---

## 10. Règles d'or

1. **Leverage AVANT le WFO** — toujours fixer le leverage avant d'optimiser
2. **Un changement à la fois** — ne pas tester nouveau param + nouveau leverage simultanément
3. **Données fraîches** — `fetch_history` avant un WFO important
4. **Grade minimum A/B** — seuls les A et B passent en portfolio/live
5. **Timeframe unifié 1h** — tous les Grade A/B doivent être en 1h
6. **Pas de raccourci** — chaque étape du workflow doit être validée
7. **Méfiance des scores parfaits** — un score 100% sur peu de fenêtres = suspect (window_factor)
8. **Mean-reversion ≠ bear market** — les grid strategies sous-performent en tendance directionnelle prolongée
