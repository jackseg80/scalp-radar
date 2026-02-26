# Sprint 50b — Impact Portfolio : Leverage Dynamique

## Contexte

Sprint 50a-bis a validé `ema_atr` comme détecteur binaire (F1 def=0.668, 1.6 trans/an).
Ce sprint intègre ce détecteur dans le portfolio backtest pour mesurer l'impact du leverage dynamique 7x/4x sur grid_atr (13 assets, 3 ans).

Modifications backward-compatible uniquement. Sans `--regime`, comportement identique.

## Fichiers

| Fichier | Action |
|---------|--------|
| `backend/regime/__init__.py` | Créer (vide) |
| `backend/regime/detectors.py` | Créer — copie sélective depuis `scripts/regime_detectors.py` |
| `backend/regime/btc_regime_signal.py` | Créer — RegimeSignal + compute_regime_signal() |
| `backend/backtesting/portfolio_engine.py` | Modifier — + regime_signal param + leverage dynamique |
| `scripts/portfolio_backtest.py` | Modifier — + --regime flags |
| `scripts/regime_backtest_compare.py` | Créer — orchestration A/B/C + rapport |
| `tests/test_regime_signal.py` | Créer — ~20 tests |
| `COMMANDS.md` | Modifier — + section régime |

---

## Étape 1 — `backend/regime/detectors.py` (copie sélective)

Copier depuis `scripts/regime_detectors.py` les éléments nécessaires pour éviter l'import fragile scripts→backend :

```python
# Constantes
SEVERITY, LABELS, BINARY_NORMAL, BINARY_DEFENSIVE, BINARY_LABELS
LABEL_CHAR_MAP, CHAR_LABEL_MAP

# Fonctions
to_binary_labels(labels)
apply_hysteresis(raw_labels, h_down, h_up)

# Helpers indicateurs
ema_series(series, period)
atr_series(high, low, close, period)
resample_4h_to_daily(df_4h)

# Classes
DetectorResult (dataclass)
BaseDetector (ABC) — avec run()
EMAATRDetector — detect_raw() + param_grid()
```

~250 lignes. Pas de lien symbolique, pas d'import circulaire.

## Étape 2 — `backend/regime/btc_regime_signal.py`

```python
@dataclass
class RegimeSignal:
    timestamps: list[datetime]      # timestamps 4h alignés
    regimes: list[str]              # "normal" ou "defensive"
    transitions: list[dict]         # {"timestamp", "from", "to"}
    params: dict

    def get_regime_at(self, dt: datetime) -> str:
        """Bisect sur timestamps. Retourne "normal" si dt < premier ts."""
        if not self.timestamps or dt < self.timestamps[0]:
            return "normal"
        idx = bisect.bisect_right(self.timestamps, dt) - 1
        return self.regimes[idx]

    def get_leverage(self, dt, lev_normal=7, lev_defensive=4) -> int:
        return lev_normal if self.get_regime_at(dt) == "normal" else lev_defensive
```

```python
async def compute_regime_signal(
    db_path: str = "data/scalp_radar.db",
    start: datetime | None = None,
    end: datetime | None = None,
    exchange: str = "binance",
    detector_params: dict | None = None,
) -> RegimeSignal:
```

**Pipeline :**
1. Charger BTC/USDT 4h via `Database.get_candles()` (pas aiosqlite direct — plus simple, déjà testé)
2. Convertir `list[Candle]` → DataFrame pandas (colonnes: timestamp_utc, open, high, low, close, volume)
3. `resample_4h_to_daily()` depuis `backend.regime.detectors`
4. `EMAATRDetector().run(df_4h, df_daily, **params)` → DetectorResult
5. `to_binary_labels(result.labels_4h)` → list[str] binaire
6. Construire RegimeSignal avec timestamps alignés sur les candles 4h

**Params par défaut** (Sprint 50a-bis best) :
```python
DEFAULT_PARAMS = {
    "h_down": 6, "h_up": 24,
    "ema_fast": 50, "ema_slow": 200,
    "atr_fast": 7, "atr_slow": 30,
    "atr_stress_ratio": 2.0,
}
```

**Erreur claire si BTC 4h manquant :**
```python
if not candles:
    raise ValueError(
        "BTC/USDT 4h not found in DB. "
        "Run: uv run python -m scripts.backfill_candles "
        "--symbol BTC/USDT --timeframe 4h --since 2017-01-01"
    )
```

## Étape 3 — Tests RegimeSignal + compute (~8 tests)

Fichier : `tests/test_regime_signal.py`

**TestRegimeSignal (5 tests) :**
1. `test_get_regime_before_first_ts` → "normal"
2. `test_get_regime_exact_ts` → régime correct
3. `test_get_regime_between_ts` → régime du précédent (bisect)
4. `test_get_leverage` → normal→7, defensive→4
5. `test_transitions_list` → nombre et format corrects

**TestComputeRegimeSignal (3 tests) :**

Helper `_create_test_db(tmp_path)` — crée une DB temp avec ~200 candles BTC 4h synthétiques via `Database.insert_candles_batch()`. Pattern existant : `tests/test_database.py` (in-memory DB + `@pytest_asyncio.fixture`).

1. `test_signal_computed` → résultat contient "normal" et "defensive"
2. `test_default_params` → params par défaut si None
3. `test_missing_btc_raises` → DB vide → ValueError avec message backfill

## Étape 4 — Intégration dans `portfolio_engine.py`

### 4a. Nouveau champ `PortfolioResult`

Après `regime_analysis` (ligne ~152) :
```python
# Leverage dynamique (Sprint 50b)
leverage_changes: list[dict] = field(default_factory=list)
# Format : {"timestamp": ..., "runner": ..., "old": 7, "new": 4, "regime": "defensive"}
```

### 4b. Nouveau paramètre `__init__`

```python
def __init__(
    self,
    ...existant...,
    regime_signal: "RegimeSignal | None" = None,  # NOUVEAU (Sprint 50b)
) -> None:
    ...existant...
    self._regime_signal = regime_signal
```

Import conditionnel (TYPE_CHECKING) pour éviter import lourd au boot :
```python
from __future__ import annotations  # déjà présent
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.regime.btc_regime_signal import RegimeSignal
```

### 4c. Helper `_update_runner_leverage`

Nouvelle méthode dans PortfolioBacktester :

```python
def _update_runner_leverage(self, runner_key: str, runner: GridStrategyRunner, new_leverage: int) -> None:
    """Met à jour le leverage d'un runner (3 emplacements)."""
    runner._leverage = new_leverage
    runner._gpm._config.leverage = new_leverage
    runner._strategy._config.leverage = new_leverage
```

**Correction critique vs le spec :** le spec mentionne `runner._position_manager._config` qui N'EXISTE PAS. C'est `runner._gpm._config`. Et il faut aussi `runner._leverage` (utilisé dans `_take_snapshot` ligne 892 et dans le calcul de marge lignes 1055, 1166, 1200).

### 4d. Modification de `_simulate()`

Ajouter un tracking de leverage et les changements dynamiques dans la boucle `for i, candle in enumerate(merged_candles):` (après ligne 797, avant snapshot) :

```python
# --- Leverage dynamique (Sprint 50b) ---
if self._regime_signal is not None:
    target_lev = self._regime_signal.get_leverage(candle.timestamp)
    for rk in runner_keys:
        r = runners[rk]
        if r._leverage != target_lev:
            # Ne PAS changer si positions ouvertes
            has_positions = any(
                positions for positions in r._positions.values()
            )
            if not has_positions:
                old_lev = r._leverage
                self._update_runner_leverage(rk, r, target_lev)
                leverage_changes.append({
                    "timestamp": candle.timestamp.isoformat(),
                    "runner": rk,
                    "old": old_lev,
                    "new": target_lev,
                    "regime": self._regime_signal.get_regime_at(candle.timestamp),
                })
```

**Note :** `leverage_changes` est initialisé en début de `_simulate()` comme `list[dict]` et retourné dans le tuple (ajouter un 4ème élément au return).

**Initialisation du leverage au démarrage :** avant la boucle principale, si `self._regime_signal` est défini, initialiser TOUS les runners au leverage correspondant au premier timestamp de simulation :

```python
# --- Init leverage au démarrage (Sprint 50b) ---
if self._regime_signal is not None:
    first_candle_ts = merged_candles[0].timestamp if merged_candles else None
    if first_candle_ts:
        init_lev = self._regime_signal.get_leverage(first_candle_ts)
        for rk, r in runners.items():
            if r._leverage != init_lev:
                self._update_runner_leverage(rk, r, init_lev)
```

Sans ça, le premier trade partirait avec le leverage du YAML au lieu de celui du régime.

### 4e. Passage de `leverage_changes` à `_build_result()`

Ajouter le paramètre `leverage_changes` à `_build_result()` et l'assigner à `PortfolioResult.leverage_changes`.

## Étape 5 — Tests intégration portfolio (~7 tests)

Dans `tests/test_regime_signal.py`, classe **TestPortfolioRegimeIntegration** :

1. `test_regime_none_backward_compatible` → regime_signal=None, pas de leverage_changes
2. `test_leverage_changes_no_position` → leverage change quand runner._positions est vide
3. `test_leverage_no_change_with_position` → leverage NE change PAS si positions ouvertes
4. `test_update_runner_leverage_all_three` → vérifie que _leverage, _gpm._config.leverage ET _strategy._config.leverage sont mis à jour
5. `test_leverage_changes_tracked` → leverage_changes dans PortfolioResult non vide
6. `test_leverage_changes_empty_without_regime` → regime_signal=None → leverage_changes=[]
7. `test_delayed_transition` → transition defensive→normal retardée si position ouverte, appliquée après fermeture

Tests avec mocks légers : créer un fake runner avec `_positions`, `_leverage`, `_gpm._config.leverage`, `_strategy._config.leverage`.

## Étape 6 — CLI `scripts/portfolio_backtest.py`

### 6a. Nouveaux arguments

Après le bloc `--params` (ligne ~458) :

```python
parser.add_argument("--regime", action="store_true",
    help="Leverage dynamique piloté par régime BTC (ema_atr)")
parser.add_argument("--regime-normal", type=int, default=7,
    help="Leverage en mode normal (défaut: 7)")
parser.add_argument("--regime-defensive", type=int, default=4,
    help="Leverage en mode defensive (défaut: 4)")
```

### 6b. Logique dans `main()`

**AVANT le bloc `--leverage`** (ligne 214), ajouter :

```python
# --regime override --leverage (leverage piloté par le signal)
if args.regime and args.leverage is not None:
    print("  ⚠ --leverage ignoré car --regime est actif")
    args.leverage = None
```

**APRÈS le bloc `--days`** (ligne ~281), ajouter :

```python
regime_signal = None
if args.regime:
    from backend.regime.btc_regime_signal import compute_regime_signal
    regime_signal = await compute_regime_signal(
        db_path=args.db, start=start, end=end, exchange=args.exchange
    )
    n_trans = len(regime_signal.transitions)
    print(f"  Regime signal       : {n_trans} transitions")
    print(f"  Normal leverage     : {args.regime_normal}x")
    print(f"  Defensive leverage  : {args.regime_defensive}x")
```

**Note :** il faut calculer `start` et `end` avant ce bloc (extraire de la logique existante).

**DANS la création du backtester** (ligne ~285) :

```python
backtester = PortfolioBacktester(
    ...existant...,
    regime_signal=regime_signal,
)
```

## Étape 7 — Script de comparaison `scripts/regime_backtest_compare.py`

CLI : `uv run python -m scripts.regime_backtest_compare --strategy grid_atr`

**Pipeline :**
1. `get_config()` pour charger la config
2. Calculer start/end (auto-détection identique à portfolio_backtest)
3. Calculer `regime_signal` via `compute_regime_signal()`
4. **Run A** : `PortfolioBacktester(leverage=7)` → result_a
5. **Run B** : `PortfolioBacktester(leverage=4)` → result_b
6. **Run C** : `PortfolioBacktester(regime_signal=regime_signal)` → result_c
7. Calculer métriques comparatives (Sharpe, Calmar depuis snapshots)
8. Générer rapport + plot

**Important :** ne PAS muter `config`. Utiliser le paramètre `leverage` du constructeur pour A/B (via `self._leverage_override`). Pour C, ne pas passer de leverage override, laisser le régime piloter.

### Calcul Sharpe/Calmar depuis snapshots

```python
def _calc_sharpe(snapshots: list[PortfolioSnapshot]) -> float:
    """Sharpe annualisé depuis equity snapshots 1h."""
    equities = [s.total_equity for s in snapshots]
    if len(equities) < 2:
        return 0.0
    returns = [(equities[i+1] - equities[i]) / equities[i]
               for i in range(len(equities) - 1) if equities[i] > 0]
    if not returns:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r)**2 for r in returns) / len(returns)
    std_r = var ** 0.5
    return (mean_r / std_r * (24 * 365)**0.5) if std_r > 0 else 0.0

def _calc_calmar(total_return_pct: float, max_dd_pct: float) -> float:
    return total_return_pct / abs(max_dd_pct) if max_dd_pct != 0 else 0.0
```

### Analyse des transitions (Test C)

Pour chaque transition dans `regime_signal.transitions` :
- Trouver les snapshots dans les 96h suivantes
- Calculer le return portfolio sur cette fenêtre
- Charger BTC candles pour le return BTC sur 96h
- Classer : transition "normal→defensive" (entrée en bear) ou "defensive→normal" (recovery)

### Breakdown par régime

Itérer les snapshots avec `regime_signal.get_regime_at(snap.timestamp)` :
- Séparer en blocs "normal" et "defensive"
- Calculer return et max DD par bloc

### Verdict Go/No-Go

```python
criteria = {
    "return_ok": result_c.total_return_pct > 0.8 * result_a.total_return_pct,
    "dd_ok": abs(result_c.max_drawdown_pct) < abs(result_a.max_drawdown_pct),
    "sharpe_ok": sharpe_c > sharpe_a,
}
score = sum(criteria.values())
verdict = "GO" if score >= 2 else ("NO-GO" if score == 0 else "BORDERLINE")
```

### Rapport : `docs/regime_impact_report.md`

Sections : résumé exécutif, tableau A/B/C, transitions, breakdown par régime, verdict.

### Plot : `docs/images/regime_equity_curves.png`

2 subplots :
1. Equity curves (A bleu, B gris, C vert) + bandes régime (alpha=0.15)
2. Drawdown curves (3 lignes)

## Étape 8 — Tests rapport/comparaison (~5 tests)

**TestReportMetrics :**
1. `test_sharpe_calculation` → formule correcte sur données synthétiques
2. `test_calmar_calculation` → return/DD
3. `test_verdict_go` → 2/3 critères → "GO"
4. `test_verdict_nogo` → 0/3 critères → "NO-GO"
5. `test_verdict_borderline` → 1/3 critères → "BORDERLINE"

## Étape 9 — COMMANDS.md

Ajouter dans la section 12 (Portfolio Backtest) :

```markdown
### Leverage dynamique (régime BTC)

# Backtest avec leverage piloté par régime BTC
uv run python -m scripts.portfolio_backtest --strategy grid_atr --regime --days auto

# Leverages custom
uv run python -m scripts.portfolio_backtest --strategy grid_atr --regime --regime-normal 7 --regime-defensive 4

# Comparaison complète A/B/C (baseline 7x, baseline 4x, dynamique 7x/4x)
uv run python -m scripts.regime_backtest_compare --strategy grid_atr
```

---

## Ordre d'implémentation

1. `backend/regime/__init__.py` + `detectors.py` (copie sélective)
2. `backend/regime/btc_regime_signal.py` (RegimeSignal + compute)
3. `tests/test_regime_signal.py` — TestRegimeSignal + TestComputeRegimeSignal (8 tests)
4. `backend/backtesting/portfolio_engine.py` (PortfolioResult + __init__ + _simulate + helper)
5. `tests/test_regime_signal.py` — TestPortfolioRegimeIntegration (7 tests)
6. `scripts/portfolio_backtest.py` (--regime flags)
7. `scripts/regime_backtest_compare.py` (orchestration + rapport + plot)
8. `tests/test_regime_signal.py` — TestReportMetrics (5 tests)
9. `COMMANDS.md`

## Vérification

```bash
# Tests unitaires (~20 nouveaux)
uv run pytest tests/test_regime_signal.py -x -q

# 0 régression
uv run pytest tests/ -x -q

# Run individuel avec --regime
uv run python -m scripts.portfolio_backtest --strategy grid_atr --regime --days auto

# Comparaison complète
uv run python -m scripts.regime_backtest_compare --strategy grid_atr

# Vérifier outputs
# - docs/regime_impact_report.md
# - docs/images/regime_equity_curves.png
```
