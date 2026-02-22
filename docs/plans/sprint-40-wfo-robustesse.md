# Sprint 40 WFO Robustesse — Plan d'implémentation

## Contexte

L'audit méthodologique du module WFO a identifié des biais qui gonflent artificiellement les Sharpe et grades. Ce sprint corrige les 3 biais les plus impactants (Sprint 40a) avant le déploiement paper de grid_multi_tf, puis complète avec 2 améliorations défensives (Sprint 40b).

**Impact estimé** : ~20-25 tests à ajuster sur 1716 (~1.5%). Les résultats WFO existants devront être relancés après les changements.

---

## Sprint 40a — 3 changements avant déploiement grid_multi_tf

### 1. Taker fee sur tous les exits (URGENT)

**Problème** : Les TP exits utilisent `maker_fee` (0.02%) + 0 slippage. Or en live, l'executor détecte la condition SMA sur la candle puis envoie un **market order** → c'est un taker. Sur 4292 trades, 0.04% de sous-estimation par trade = ~1.7% de return fantôme sur le portfolio.

**Fichier** : `backend/optimization/fast_multi_backtest.py`

**Changements** :

1. `_simulate_grid_common()` lignes 319-324 — supprimer la différenciation maker/taker sur `tp_global` :

```python
# Avant
if exit_reason == "tp_global":
    fee = maker_fee
    slip = 0.0
else:
    fee = taker_fee
    slip = slippage_pct

# Après
fee = taker_fee
slip = slippage_pct
```

2. `_simulate_grid_range()` lignes 564-572 — même changement sur `tp` individuel.

3. `_simulate_grid_trend()` — vérifier et aligner (trail_stop déjà taker, vérifier tp).

**Note** : grid_boltrend utilise déjà taker sur tous les exits (lignes 996-999) — ne pas toucher.

**Note** : Ne PAS toucher aux stratégies mono-position (swing) — elles ne sont pas déployées.

**Tests impactés** : ~10-15 tests qui vérifient des `net_pnl` exacts avec maker_fee sur TP. Ajustement des valeurs attendues uniquement (logique inchangée).

---

### 2. Margin guard 70% dans le fast engine

**Problème** : Le fast engine n'applique qu'un guard par niveau (`if capital < margin: continue`), pas le `max_margin_ratio=0.70` global de `risk.yaml`. En live, l'executor bloque au-delà de 70% — le WFO peut donc sélectionner des combos qui ouvrent plus de positions que le live ne le permettrait.

**Fichiers** :

- `backend/core/config.py` — ajouter `max_margin_ratio: float = 0.70` dans `BacktestConfig`
- `backend/optimization/fast_multi_backtest.py` — `_simulate_grid_common()` (lignes 375-389)

**Changements dans `_simulate_grid_common()`** :

```python
# Initialisation (début de la fonction)
total_margin_locked = 0.0

# Ouverture position (ligne ~382) — ajouter guard global
margin = notional / leverage
if capital < margin:
    continue
if capital > 0 and (total_margin_locked + margin) / capital > max_margin_ratio:
    continue  # Guard global 70% — dénominateur = capital COURANT (cf. simulator.py:1120)
capital -= margin
total_margin_locked += margin

# Fermeture position (ligne ~327) — libérer la marge
margin_to_return = sum(ep * qty / leverage for _l, ep, qty, _f in positions)
capital += margin_to_return
total_margin_locked -= margin_to_return
```

**Note** : le live executor (`risk_manager.py`) n'a PAS de guard `max_margin_ratio` — il utilise `min_free_margin_percent`. Le dénominateur `capital courant` s'aligne sur le simulateur paper ([simulator.py:1120](backend/backtesting/simulator.py#L1120)).

- Propager `max_margin_ratio` depuis `bt_config` dans les 3 fonctions : `_simulate_grid_common`, `_simulate_grid_range`, `_simulate_grid_boltrend`
- Lire `max_margin_ratio` depuis `risk.yaml` dans `BacktestConfig` (déjà présent dans `risk.yaml`, juste à propager)

**Tests impactés** : ~3-5 tests fast engine avec capital exact. Certains combos n'ouvriront plus tous leurs niveaux → PnL et trades counts légèrement différents.

---

### 3. Embargo IS→OOS 21 jours

**Problème** : `oos_start = is_end` sans tampon. Les positions grid ouvertes en IS (durée moyenne 3-7 jours, parfois 15-20j en drawdown) contaminent les premiers jours OOS.

**Fichiers** :

- `backend/optimization/walk_forward.py` — `_build_windows()` (lignes 1039-1063)
- `config/param_grids.yaml` — sections `wfo:` pour grid_atr, grid_boltrend, grid_multi_tf

**Changement dans `_build_windows()`** :

```python
def _build_windows(
    self,
    data_start: datetime,
    data_end: datetime,
    is_days: int,
    oos_days: int,
    step_days: int,
    embargo_days: int = 0,  # NOUVEAU paramètre, défaut 0 = rétrocompat
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    ...
    oos_start = is_end + timedelta(days=embargo_days)  # MODIFIÉ
    oos_end = oos_start + timedelta(days=oos_days)
```

**Changement dans `param_grids.yaml`** — pour chaque stratégie grid :

```yaml
grid_atr:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
    embargo_days: 7  # NOUVEAU

grid_boltrend:
  wfo:
    is_days: 180
    oos_days: 60
    step_days: 60
    embargo_days: 7  # NOUVEAU

grid_multi_tf:
  wfo:
    ...
    embargo_days: 7  # NOUVEAU
```

- Propager le paramètre depuis `param_grids.yaml` → `WalkForwardOptimizer.__init__` → `_build_windows()`
- Défaut `embargo_days=0` pour rétrocompatibilité (les stratégies qui ne spécifient pas = comportement identique à avant)

**Effet** : 7 jours couvre ~80-85% des positions (celles qui ferment dans la semaine). Perte négligeable en données vs 21j (~1 fenêtre sur 8-10). Résiduel : positions en drawdown > 7j tenues en fin d'IS contaminent légèrement — risque acceptable.

**Tests impactés** : ~2-3 tests qui vérifient les bornes et le nombre de fenêtres. Ajouter un test `embargo_days=0` qui vérifie le comportement identique à avant.

---

## Sprint 40b — Améliorations défensives (après 40a)

### 4. Stress-test fees dans le rapport

**Utilité** : Diagnostic uniquement — ne change pas les grades, n'influence pas la sélection. Ajoute de l'information au rapport pour détecter les stratégies fragiles aux frais.

**Fichier** : `backend/optimization/report.py`

**Changement** : Nouvelle fonction `fee_sensitivity_analysis(trades, best_params)` appelée dans `build_final_report()`.

```python
FEE_SCENARIOS = {
    "nominal":  {"taker": 0.0006, "maker": 0.0002, "slip": 0.0005},
    "degraded": {"taker": 0.0008, "maker": 0.0004, "slip": 0.0010},
    "stress":   {"taker": 0.0010, "maker": 0.0006, "slip": 0.0020},
}
```

- Recalcul du net_pnl de chaque trade OOS avec les fees alternatives (pas de relance backtest complet)
- Ajout au `FinalReport` : `fee_sensitivity: dict[str, float]` (scenario → Sharpe)
- Warning si Sharpe tombe sous 0.5 en scénario "degraded"

**Tests impactés** : 0 existants. ~5 nouveaux tests.

---

### 5. Guard max DD -80%

**Utilité** : Garde-fou défensif. Peu d'impact en pratique (portfolio 365j montre DD max -28.4%), mais élimine les combos pathologiques qui subsistent malgré un SL large.

**Fichier** : `backend/optimization/fast_multi_backtest.py` — `_simulate_grid_common()`

**Changement** :

```python
# Tracker le peak capital
peak_capital = capital
...
# Après chaque mise à jour de capital
peak_capital = max(peak_capital, capital)
current_dd = (capital - peak_capital) / peak_capital
if current_dd < -max_dd_threshold:  # -0.80 par défaut
    break  # Forcer arrêt, marquer combo comme "liquidated"
```

- Paramètre `max_wfo_drawdown_pct: 80` ajouté dans `risk.yaml` + `BacktestConfig`
- Les combos avec DD > 80% reçoivent Sharpe = -99.0 et sont exclus du scoring
- Si le best combo est marqué "liquidated" → Grade F automatique

**Tests impactés** : 0 existants. ~3-5 nouveaux tests.

---

## Décisions d'architecture

**#6 Capital fixe** : abandonné comme filtre de grading. Le compounding est la réalité live — le fixer donnerait un Sharpe moins prédictif du rendement réel. On ajoute `sharpe_fixed` en **champ informatif** dans `FinalReport` uniquement (sizing sur `initial_capital`, pas de changement au grading).

**#5 Correction BH** : abandonnée. Le DSR avec `n_trials=n_distinct_combos` (2000-3000) fait déjà le travail de corriger pour les tests multiples. BH serait redondant avec le DSR et potentiellement trop punitif.

---

## Fichiers modifiés

| Fichier | Sprint | Changements |
|---------|--------|-------------|
| `backend/optimization/fast_multi_backtest.py` | 40a | Taker fees (#1), margin guard (#2), DD guard (#5) |
| `backend/optimization/walk_forward.py` | 40a | Embargo (#3) |
| `backend/optimization/report.py` | 40b | Fee sensitivity (#4), sharpe_fixed info |
| `backend/core/config.py` | 40a | `BacktestConfig` : `max_margin_ratio`, `max_wfo_drawdown_pct` |
| `config/param_grids.yaml` | 40a | `embargo_days: 21` dans sections `wfo:` |
| `config/risk.yaml` | 40b | `max_wfo_drawdown_pct: 80` |

## Tests Sprint 40a

**Nouveaux** (~12) :

- `tests/test_wfo_embargo.py` : 3 tests (bornes fenêtres, nombre de fenêtres, rétrocompat embargo=0)
- `tests/test_margin_guard_wfo.py` : 4 tests (guard 70%, multi-niveaux, grid_range bidirectionnel, rétrocompat)
- `tests/test_taker_all_exits.py` : 3 tests (TP→taker, SL taker inchangé, grid_boltrend inchangé)

**Ajustements existants** (~15) :

- Tests fast engine PnL avec maker_fee → taker_fee sur TP (valeurs attendues uniquement)
- Tests fenêtres WFO avec nouvelles bornes
- Tests capital avec margin guard plus strict

## Tests Sprint 40b (~10 nouveaux)

- `tests/test_fee_sensitivity.py` : 5 tests (3 scénarios, warning seuil, format rapport)
- `tests/test_dd_guard.py` : 4 tests (seuil -80%, liquidation forcée, marquage combo, Grade F)

## Vérification

**Sprint 40a :**

1. `uv run pytest` — tous les tests passent (après ajustement ~15)
2. WFO grid_atr sur BTC (1 asset) : Sharpe attendu en baisse de 5-10%, grade stable ou -1 cran
3. WFO complet 21 assets grid_atr + 5 assets grid_boltrend + relance per_asset si grade change

**Sprint 40b :**

1. `uv run pytest` — tous les tests passent
2. Vérifier `fee_sensitivity` présent dans le rapport JSON/DB
3. Warning visible si Sharpe degraded < 0.5
