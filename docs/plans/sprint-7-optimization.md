# Sprint 7 — Parameter Optimization & Overfitting Detection

## Objectif

Module d'optimisation automatique des paramètres (par stratégie × par asset) avec walk-forward validation, détection d'overfitting (Monte Carlo, DSR, stabilité, convergence cross-asset), rapport de confiance gradé (A-F), et validation croisée Binance 2 ans → Bitget 90 jours.

---

## Phase 0 — Données, config per_asset, et assets supplémentaires

### 0.1 Ajout DOGE/USDT et LINK/USDT dans `config/assets.yaml`

```yaml
  - symbol: "DOGE/USDT"
    exchange: bitget
    type: futures
    timeframes: ["1m", "5m", "15m", "1h"]
    max_leverage: 20
    min_order_size: 10
    tick_size: 0.00001
    correlation_group: altcoins

  - symbol: "LINK/USDT"
    exchange: bitget
    type: futures
    timeframes: ["1m", "5m", "15m", "1h"]
    max_leverage: 20
    min_order_size: 0.1
    tick_size: 0.001
    correlation_group: altcoins
```

Nouveau groupe de corrélation :
```yaml
  altcoins:
    max_concurrent_same_direction: 2
    max_exposure_percent: 40
```

**Fichier modifié** : `config/assets.yaml`

### 0.2 Support `per_asset` overrides dans `strategies.yaml`

**Problème** : `strategies.yaml` est asset-agnostique (un seul set de paramètres par stratégie). Mais un SL de 0.3% est bon pour BTC et suicidaire pour SOL. Sans overrides par asset, tout le travail d'optimisation par asset est perdu au moment de l'application.

**Solution** : section `per_asset` optionnelle par stratégie dans `strategies.yaml`.

```yaml
vwap_rsi:
  enabled: true
  live_eligible: true
  timeframe: "5m"
  rsi_period: 14
  rsi_long_threshold: 30
  rsi_short_threshold: 70
  volume_spike_multiplier: 2.0
  vwap_deviation_entry: 0.3
  trend_adx_threshold: 25.0
  tp_percent: 0.8
  sl_percent: 0.3
  weight: 0.25
  per_asset:                      # NOUVEAU — overrides par asset
    SOL/USDT:
      sl_percent: 0.5
      tp_percent: 1.0
    DOGE/USDT:
      sl_percent: 0.6
      tp_percent: 1.2
```

**Fichiers modifiés** :

- **`config/strategies.yaml`** : ajouter section `per_asset: {}` (vide par défaut) sur vwap_rsi et momentum
- **`backend/core/config.py`** :
  - `VwapRsiConfig` et `MomentumConfig` : ajouter champ `per_asset: dict[str, dict[str, Any]] = {}`
  - Nouvelle méthode `get_params_for_symbol(symbol: str) → dict` sur chaque config :
    ```python
    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}
    ```
- **`backend/strategies/base.py`** : ajouter `symbol` au constructeur de base (optionnel, défaut `None`). Si fourni, la stratégie applique les overrides per_asset.
  - Nouvelle méthode concrète sur `BaseStrategy` :
    ```python
    def _resolve_param(self, param_name: str, symbol: str | None = None) -> Any:
        """Résout un paramètre avec override per_asset si applicable."""
        if symbol and hasattr(self._config, 'per_asset'):
            overrides = self._config.per_asset.get(symbol, {})
            if param_name in overrides:
                return overrides[param_name]
        return getattr(self._config, param_name)
    ```
- **`backend/strategies/vwap_rsi.py`** et **`backend/strategies/momentum.py`** : utiliser `_resolve_param()` pour `tp_percent`, `sl_percent`, et les paramètres qui peuvent varier par asset (volume, lookback...).

**Impact sur l'existant** : aucun — `per_asset` est un dict vide par défaut, donc tout le code existant fonctionne sans modification. Le Simulator et l'Executor passent déjà le symbole dans le contexte.

**IMPORTANT — Deux chemins distincts pour les paramètres** :

Il y a deux façons de résoudre les paramètres par asset, utilisées dans des contextes différents :

1. **Chemin optimisation** (Sprint 7 — `walk_forward.py`, `run_backtest_single`) :
   Le grid search fournit **directement** les paramètres spécifiques à l'asset. `create_strategy_with_params(strategy_name, params)` crée la stratégie avec les params explicites du grid — pas besoin de `_resolve_param()`. La stratégie ne connaît même pas le symbole ; elle reçoit des params déjà sélectionnés pour cet asset par l'optimizer.

2. **Chemin production** (Simulator, Executor — runtime live) :
   La stratégie est instanciée **une seule fois** depuis la config YAML et reçoit des signaux de **plusieurs assets** (BTC, ETH, SOL...). C'est ici que `_resolve_param(param_name, symbol)` intervient : quand `evaluate(ctx)` est appelé, le symbole vient de `ctx.symbol`, et la stratégie consulte `per_asset` pour résoudre le bon SL/TP.

En résumé :
- **Optimisation** : params explicites par asset → `per_asset` non utilisé, les params sont injectés dans le constructeur de la config
- **Production** : params résolus au runtime via `_resolve_param(name, ctx.symbol)` → `per_asset` est la source de vérité
- `--apply` écrit les résultats de l'optimisation dans `per_asset` du YAML → le chemin production les consomme automatiquement

### 0.3 Schéma DB : colonne `exchange` sur la table `candles`

**Fichier modifié** : `backend/core/database.py`

Changements :
- **Backup automatique** avant migration : `shutil.copy2(db_path, f"{db_path}.bak.{datetime.now():%Y%m%d_%H%M%S}")` si la colonne n'existe pas encore
- Ajouter colonne `exchange TEXT NOT NULL DEFAULT 'bitget'` à la table `candles`
- Modifier la clé primaire : `PRIMARY KEY (symbol, exchange, timeframe, timestamp)`
- Modifier l'index : `CREATE INDEX idx_candles_lookup ON candles (symbol, exchange, timeframe, timestamp)`
- Ajouter méthode `get_candles()` avec paramètre optionnel `exchange: str = "bitget"`
- `get_latest_candle_timestamp()` : ajouter paramètre `exchange`
- `insert_candles_batch()` : lire `exchange` depuis le champ `candle.exchange`
- `delete_candles()` : ajouter paramètre `exchange`
- Migration idempotente : `PRAGMA table_info(candles)` → vérifier si `exchange` existe avant `ALTER TABLE`

**Impact** : Aucun impact sur le DataEngine live (qui utilise toujours `exchange='bitget'`). Seuls `fetch_history.py` et le module d'optimisation utilisent `exchange='binance'`.

### 0.4 Modèle Candle : champ exchange

**Fichier modifié** : `backend/core/models.py`

Ajouter `exchange: str = "bitget"` au modèle `Candle`. Valeur par défaut = "bitget" pour backward compat.

### 0.5 Script `fetch_history.py` : support Binance

**Fichier modifié** : `scripts/fetch_history.py`

Changements :
- Nouveau paramètre CLI `--exchange binance|bitget` (défaut: `bitget`)
- `--days` : défaut 180 pour bitget, 720 pour binance (2 ans)
- Factory d'exchange :

```python
def _create_exchange(name: str) -> ccxt.Exchange:
    if name == "binance":
        return ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},  # Perpétuel USDT-M
        })
    return ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
```

- Passer `exchange_name` aux candles créées (`candle.exchange = exchange_name`)
- Pour Binance : symboles identiques (`BTC/USDT`, etc.) — ccxt normalise

**Lancement** :
```bash
# Binance 2 ans (5 assets × 4 TF)
uv run python -m scripts.fetch_history --exchange binance --days 720

# Bitget 90 jours (pour validation)
uv run python -m scripts.fetch_history --exchange bitget --days 90

# Un asset spécifique
uv run python -m scripts.fetch_history --exchange binance --symbol DOGE/USDT --days 720
```

**Estimation** : ~2h pour 5 assets × 4 TF × 720 jours sur Binance (rate limit 1200 req/min).

---

## Phase 1 — Walk-Forward Optimizer

### 1.1 Registre de stratégies optimisables

**Fichier créé** : `backend/optimization/__init__.py`

```python
"""Package d'optimisation des paramètres de stratégies."""

from backend.strategies.vwap_rsi import VwapRsiConfig, VwapRsiStrategy
from backend.strategies.momentum import MomentumConfig, MomentumStrategy

# Registre central — pas de switch/case, extensible par ajout de ligne
STRATEGY_REGISTRY: dict[str, tuple[type, type]] = {
    "vwap_rsi": (VwapRsiConfig, VwapRsiStrategy),
    "momentum": (MomentumConfig, MomentumStrategy),
    # Funding et Liquidation exclus : pas de données historiques OI/funding
}

def create_strategy_with_params(strategy_name: str, params: dict) -> "BaseStrategy":
    """Crée une stratégie avec paramètres custom depuis le registre."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Stratégie '{strategy_name}' non optimisable. "
            f"Disponibles : {list(STRATEGY_REGISTRY.keys())}"
        )
    config_cls, strategy_cls = STRATEGY_REGISTRY[strategy_name]
    cfg = config_cls(**{**config_cls().model_dump(), **params})
    return strategy_cls(cfg)
```

Ce registre est utilisé par `walk_forward.py`, `overfitting.py`, et `engine.py` (run_backtest_single). Plus de switch/case.

### 1.2 Fichier `config/param_grids.yaml` (nouveau)

Espaces de recherche par stratégie, avec sections `default` + overrides par asset.

```yaml
# Paramètres d'optimisation walk-forward
optimization:
  is_window_days: 120     # In-sample : 4 mois
  oos_window_days: 30     # Out-of-sample : 1 mois
  step_days: 30           # Avancement entre fenêtres
  metric: "sharpe_ratio"  # Métrique d'optimisation (sharpe_ratio | net_return_pct | profit_factor)
  max_workers: null        # null = os.cpu_count() (i9-14900HX = 24)
  main_exchange: "binance" # Exchange source pour le WFO
  validation_exchange: "bitget"  # Exchange pour la validation finale

vwap_rsi:
  default:
    rsi_period: [10, 14, 20]
    rsi_long_threshold: [25, 30, 35]
    rsi_short_threshold: [65, 70, 75]
    volume_spike_multiplier: [1.5, 2.0, 2.5]
    vwap_deviation_entry: [0.1, 0.2, 0.3, 0.5]
    trend_adx_threshold: [20, 25, 30]
    tp_percent: [0.4, 0.6, 0.8, 1.0]
    sl_percent: [0.2, 0.3, 0.4, 0.5]
  BTC/USDT:
    sl_percent: [0.2, 0.3, 0.4]
    tp_percent: [0.4, 0.6, 0.8]
  SOL/USDT:
    sl_percent: [0.4, 0.5, 0.7]
    tp_percent: [0.6, 0.8, 1.0, 1.2]

momentum:
  default:
    breakout_lookback: [15, 20, 30, 40]
    volume_confirmation_multiplier: [1.5, 2.0, 2.5]
    atr_multiplier_tp: [1.5, 2.0, 2.5]
    atr_multiplier_sl: [0.5, 1.0, 1.5]
    tp_percent: [0.4, 0.6, 0.8, 1.0]
    sl_percent: [0.2, 0.3, 0.5]

# Funding et liquidation : exclus du Sprint 7.
# Pas de données historiques OI/funding sur Binance.
# Paper trading avec paramètres manuels.
```

**Taille du grid search** :
- VWAP+RSI default : 3×3×3×3×4×3×4×4 = **15 552** combinaisons (trop)
- **Réduction** : grid search en 2 passes (coarse → fine) — voir section 1.3

### 1.3 Fichier `backend/optimization/walk_forward.py` (nouveau, ~350 lignes)

**Classes** :

```python
@dataclass
class WindowResult:
    """Résultat d'une fenêtre IS+OOS."""
    window_index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    best_params: dict[str, Any]
    is_sharpe: float
    is_net_return_pct: float
    is_profit_factor: float
    is_trades: int
    oos_sharpe: float
    oos_net_return_pct: float
    oos_profit_factor: float
    oos_trades: int
    top_n_params: list[dict]  # Top 5 pour analyse stabilité

@dataclass
class WFOResult:
    """Résultat complet du walk-forward."""
    strategy_name: str
    symbol: str
    windows: list[WindowResult]
    avg_is_sharpe: float
    avg_oos_sharpe: float
    oos_is_ratio: float          # avg_oos / avg_is
    consistency_rate: float      # % de fenêtres OOS positives
    recommended_params: dict[str, Any]  # Médiane des best_params sur toutes les fenêtres
    all_oos_trades: list[TradeResult]   # Pour Monte Carlo
    n_distinct_combos: int       # Nombre total de combinaisons distinctes testées (pour DSR)
```

**Classe `WalkForwardOptimizer`** :

```python
class WalkForwardOptimizer:
    def __init__(self, config_dir: str = "config"):
        self._config = get_config(config_dir)
        self._grids = _load_yaml("config/param_grids.yaml")

    async def optimize(
        self,
        strategy_name: str,
        symbol: str,
        exchange: str = "binance",
        is_window_days: int = 120,
        oos_window_days: int = 30,
        step_days: int = 30,
        max_workers: int | None = None,
    ) -> WFOResult:
        """Walk-forward optimization complète."""
        # 1. Charger les candles de la fenêtre depuis la DB
        # 2. Découper en fenêtres IS+OOS
        # 3. Pour chaque fenêtre :
        #    a. Découper les candles IS (seulement la fenêtre, pas tout l'historique)
        #    b. Grid search parallèle sur IS
        #    c. Découper les candles OOS
        #    d. Évaluer top N sur OOS
        # 4. Agréger les résultats
```

**Grid search en 2 passes** (pour limiter les combinaisons) :

1. **Coarse pass** : grid large, sous-ensemble de combinaisons (stratified sampling ~500 combos max)
   - Méthode : si grid > 1000 combos, prendre un échantillon Latin Hypercube de 500
   - Retourne top 20 combinaisons par IS Sharpe

2. **Fine pass** : autour des top 20, grid fin (±1 step sur chaque paramètre)
   - ~200 combinaisons supplémentaires
   - Retourne les 5 meilleurs

**Parallélisation — passage des candles par fenêtre uniquement** :

```python
def _run_single_backtest(args: tuple) -> tuple[dict, float, float, int]:
    """Fonction top-level pour ProcessPoolExecutor (picklable)."""
    params, window_candles_by_tf, strategy_name, symbol, bt_config = args
    # window_candles_by_tf contient SEULEMENT les candles de la fenêtre IS
    # PAS tout l'historique → réduit la mémoire par worker
    from backend.optimization import create_strategy_with_params
    strategy = create_strategy_with_params(strategy_name, params)
    engine = BacktestEngine(bt_config, strategy)
    result = engine.run(window_candles_by_tf, main_tf=strategy._config.timeframe)
    metrics = calculate_metrics(result)
    return (params, metrics.sharpe_ratio, metrics.net_return_pct, metrics.total_trades)
```

- `ProcessPoolExecutor(max_workers=max_workers or os.cpu_count())`
- **Seules les candles de la fenêtre IS** sont passées au worker, pas tout l'historique
- ~34k candles (120j en 5m) × ~100 bytes = ~3.4 MB par worker → 24 workers = ~80 MB peak (pas 2 GB)

**Sélection des paramètres recommandés** :
- Pour chaque paramètre, prendre la **médiane** des valeurs optimales sur toutes les fenêtres (pas la moyenne, pour résister aux outliers)
- Snapper la médiane à la valeur du grid la plus proche

### 1.4 Fonction `run_backtest_single` dans `engine.py`

**Fichier modifié** : `backend/backtesting/engine.py`

Ajouter une **fonction module-level** (pas une méthode) pour la sérialisation `ProcessPoolExecutor` :

```python
def run_backtest_single(
    strategy_name: str,
    params: dict[str, Any],
    candles_by_tf: dict[str, list[Candle]],
    bt_config: BacktestConfig,
    main_tf: str = "5m",
) -> BacktestResult:
    """Lance un backtest unique avec paramètres custom. Utilisé par l'optimizer."""
    from backend.optimization import create_strategy_with_params
    strategy = create_strategy_with_params(strategy_name, params)
    engine = BacktestEngine(bt_config, strategy)
    return engine.run(candles_by_tf, main_tf=main_tf)
```

Pas de modification de la classe `BacktestEngine` elle-même — on compose.

---

## Phase 2 — Détection d'overfitting

### 2.1 Fichier `backend/optimization/overfitting.py` (nouveau, ~280 lignes)

**Classe `OverfitDetector`** :

```python
class OverfitDetector:
    """Détection d'overfitting multi-méthodes."""

    def full_analysis(
        self,
        wfo_result: WFOResult,
        strategy_name: str,
        symbol: str,
        candles_by_tf: dict[str, list[Candle]],
        bt_config: BacktestConfig,
        all_symbols_results: dict[str, WFOResult] | None = None,
    ) -> OverfitReport:
        """Analyse complète : Monte Carlo + DSR + stabilité + convergence."""
```

**Méthode 1 : Monte Carlo block bootstrap**

```python
def monte_carlo_block_bootstrap(
    self,
    trades: list[TradeResult],
    n_sims: int = 1000,
    block_size: int = 7,
    seed: int | None = 42,
) -> MonteCarloResult:
    """
    Permute des blocs de trades consécutifs (pas individuels)
    pour respecter la corrélation temporelle.

    Args:
        seed: Graine pour reproductibilité. None = aléatoire
              (utile pour vérifier la stabilité avec différents seeds).

    Returns:
        p_value: float — probabilité que le Sharpe réel soit dû au hasard
        real_sharpe: float
        distribution: list[float] — 1000 Sharpe permutés
        significant: bool — p_value < 0.05
    """
```

Algorithme :
1. Découper les trades en blocs de `block_size` (dernier bloc potentiellement plus petit)
2. Pour chaque simulation : shuffler les blocs (pas les trades dans les blocs)
3. Recalculer le Sharpe sur la séquence permutée
4. p-value = % de simulations avec Sharpe ≥ Sharpe réel

**Méthode 2 : Deflated Sharpe Ratio (DSR)**

```python
def deflated_sharpe_ratio(
    self,
    observed_sharpe: float,
    n_trials: int,        # Nombre total de combinaisons DISTINCTES testées
    n_trades: int,        # Nombre de trades
    skewness: float,      # Asymétrie des rendements
    kurtosis: float,      # Kurtosis des rendements
) -> DSRResult:
    """
    Bailey & Lopez de Prado (2014).
    Corrige le Sharpe pour le data mining (multiple testing).

    n_trials : nombre de combinaisons DISTINCTES testées sur ce (strategy, symbol).
    C'est WFOResult.n_distinct_combos (~700 après coarse+fine).
    PAS multiplié par le nombre de fenêtres (les mêmes combos sont testées
    sur chaque fenêtre, ce n'est pas du cherry-picking entre fenêtres).
    PAS multiplié par le nombre d'assets (chaque asset est un test indépendant,
    le DSR est calculé par asset — la convergence cross-asset est un test séparé).

    Returns:
        dsr: float — 0 à 1, > 0.95 = confiance élevée
        max_expected_sharpe: float — E[max(SR)] sous H0
    """
```

Formule clé :
```python
# Expected max Sharpe under null (i.i.d. trials)
e_max_sr = expected_max_sharpe(n_trials)  # ≈ sqrt(2 * log(n_trials))

# PSR avec correction skew/kurtosis
psr = norm.cdf(
    (observed_sharpe - e_max_sr)
    * sqrt(n_trades - 1)
    / sqrt(1 - skewness * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2)
)
```

**Calcul de `n_trials`** :
- WFO avec coarse pass (500 combos) + fine pass (~200 combos) = **~700 combinaisons distinctes**
- Ce nombre est stocké dans `WFOResult.n_distinct_combos`
- C'est le nombre à passer au DSR — pas 700×20 fenêtres (même grid sur chaque fenêtre), pas 700×5 assets (DSR par asset)
- Si on sélectionne ensuite le "meilleur asset" parmi 5, c'est un second niveau de data mining → pour l'instant hors scope, capturé par le test de convergence cross-asset

**Méthode 3 : Parameter stability (perturbation analysis)**

```python
def parameter_stability(
    self,
    strategy_name: str,
    symbol: str,
    optimal_params: dict[str, Any],
    candles_by_tf: dict[str, list[Candle]],
    bt_config: BacktestConfig,
    perturbation_pcts: list[float] = [0.10, 0.20],
) -> StabilityResult:
    """
    Pour chaque paramètre optimal, perturbe de ±pct et mesure l'impact sur le Sharpe OOS.

    Returns:
        stability_map: dict[str, float] — score 0-1 par paramètre (1 = plateau, 0 = cliff)
        overall_stability: float — moyenne pondérée
        cliff_params: list[str] — paramètres avec score < 0.5
    """
```

Algorithme :
1. Pour chaque paramètre P avec valeur V :
   - Tester V×0.8, V×0.9, V×1.1, V×1.2
   - Calculer le Sharpe pour chaque perturbation
2. Score = 1 - max_drop / original_sharpe
   - Si le Sharpe chute de 30% pour ±10% → score = 0.7 (mauvais)
   - Si le Sharpe est stable ±5% → score ≈ 0.95 (bon)

**Méthode 4 : Cross-asset convergence**

```python
def cross_asset_convergence(
    self,
    optimal_params_by_symbol: dict[str, dict[str, Any]],
) -> ConvergenceResult:
    """
    Compare les paramètres optimaux entre assets pour une stratégie donnée.

    Returns:
        convergence_score: float — 0-1 (1 = paramètres identiques sur tous les assets)
        param_scores: dict[str, float] — convergence par paramètre
        divergent_params: list[str] — paramètres avec score < 0.5
    """
```

Algorithme :
- Pour chaque paramètre numérique, calculer le coefficient de variation (CV = std/mean)
- Score = 1 - min(CV, 1)  (borné entre 0 et 1)
- Un CV < 0.15 → score > 0.85 (bonne convergence)
- Un CV > 0.5 → score < 0.5 (pas de convergence)

### 2.2 Dataclasses résultats

```python
@dataclass
class MonteCarloResult:
    p_value: float
    real_sharpe: float
    distribution: list[float]  # Pour histogramme éventuel
    significant: bool          # p_value < 0.05

@dataclass
class DSRResult:
    dsr: float
    max_expected_sharpe: float
    observed_sharpe: float
    n_trials: int

@dataclass
class StabilityResult:
    stability_map: dict[str, float]  # param_name → score 0-1
    overall_stability: float
    cliff_params: list[str]

@dataclass
class ConvergenceResult:
    convergence_score: float
    param_scores: dict[str, float]
    divergent_params: list[str]

@dataclass
class OverfitReport:
    monte_carlo: MonteCarloResult
    dsr: DSRResult
    stability: StabilityResult
    convergence: ConvergenceResult | None  # None si un seul asset
```

---

## Phase 3 — Validation Bitget & rapport de confiance

### 3.1 Fichier `backend/optimization/report.py` (nouveau, ~250 lignes)

**Dataclasses** :

```python
@dataclass
class ValidationResult:
    """Résultat de la validation Bitget 90j."""
    bitget_sharpe: float
    bitget_net_return_pct: float
    bitget_trades: int
    bitget_sharpe_ci_low: float    # Bootstrap CI 95% borne basse
    bitget_sharpe_ci_high: float   # Bootstrap CI 95% borne haute
    binance_oos_avg_sharpe: float
    transfer_ratio: float          # bitget_sharpe / binance_oos_avg_sharpe
    transfer_significant: bool     # True si CI ne chevauche pas 0
    volume_warning: bool           # True si volume patterns divergent
    volume_warning_detail: str     # Description si warning

@dataclass
class FinalReport:
    """Rapport complet d'une optimisation stratégie × asset."""
    strategy_name: str
    symbol: str
    timestamp: datetime
    grade: str  # A, B, C, D, F

    # WFO
    wfo: WFOResult
    recommended_params: dict[str, Any]

    # Overfitting
    overfit: OverfitReport

    # Validation
    validation: ValidationResult

    # Métriques de décision
    oos_is_ratio: float
    mc_p_value: float
    dsr: float
    stability: float
    convergence: float | None
    bitget_transfer: float

    # Décision
    live_eligible: bool  # grade A ou B
    warnings: list[str]
```

**Grading** :

```python
def compute_grade(self) -> str:
    """Calcule le grade A-F selon les critères."""
    score = 0

    # OOS/IS ratio (max 25 points)
    if self.oos_is_ratio > 0.6: score += 25
    elif self.oos_is_ratio > 0.5: score += 20
    elif self.oos_is_ratio > 0.4: score += 15
    elif self.oos_is_ratio > 0.3: score += 10

    # Monte Carlo (max 25 points)
    if self.mc_p_value < 0.05: score += 25
    elif self.mc_p_value < 0.10: score += 15

    # DSR (max 20 points)
    if self.dsr > 0.95: score += 20
    elif self.dsr > 0.90: score += 15
    elif self.dsr > 0.80: score += 10

    # Stability (max 15 points)
    if self.stability > 0.80: score += 15
    elif self.stability > 0.60: score += 10
    elif self.stability > 0.40: score += 5

    # Bitget transfer (max 15 points)
    if self.bitget_transfer > 0.50: score += 15
    elif self.bitget_transfer > 0.30: score += 8

    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"
```

**Sauvegarde** :
- JSON dans `data/optimization/{strategy}_{symbol}_{YYYYMMDD_HHMMSS}.json`
- Contient toutes les métriques, le grade, les paramètres recommandés, les warnings
- Format lisible (indent=2, sort_keys)

**Validation Bitget 90 jours — avec bootstrap CI** :

```python
async def validate_on_bitget(
    self,
    strategy_name: str,
    symbol: str,
    recommended_params: dict[str, Any],
    binance_oos_avg_sharpe: float,
) -> ValidationResult:
    """
    Backtester les paramètres optimaux sur Bitget 90 jours.
    Calcule un intervalle de confiance bootstrap sur le Sharpe Bitget
    pour déterminer si la différence avec Binance est significative.
    """
    # 1. Charger candles Bitget
    # 2. Backtest avec recommended_params
    # 3. Bootstrap CI sur le Sharpe Bitget (1000 resamples des trades)
    # 4. Calculer transfer_ratio = bitget_sharpe / binance_oos_avg_sharpe
    # 5. transfer_significant = CI_low > 0 (le Sharpe Bitget est significativement positif)
    # 6. Volume warning si les indicateurs volume divergent significativement
    #    (comparer vol_sma Binance vs Bitget → si ratio > 2 ou < 0.5, warning)
    # 7. PAS de mini grid search automatique sur volume (risque overfitting sur 90j)
    #    → reporter le warning et laisser le trader décider
```

**Abandon du mini grid search volume sur Bitget 90j** :
90 jours c'est insuffisant pour optimiser un paramètre sans risque d'overfitting. Au lieu de recalibrer automatiquement, on :
- Reporte un **volume_warning** si les profils de volume divergent entre Binance et Bitget
- Log un conseil : "Le volume_multiplier pourrait nécessiter un ajustement manuel pour Bitget"
- Le transfer_ratio est accompagné d'un **intervalle de confiance bootstrap** pour distinguer le bruit statistique d'une vraie divergence

### 3.2 Application des paramètres (`--apply`)

Dans `report.py` :

```python
def apply_to_yaml(
    self,
    reports: list[FinalReport],  # Tous les rapports pour une stratégie
    strategies_yaml_path: str = "config/strategies.yaml",
) -> bool:
    """
    Écrit les paramètres grade A/B dans strategies.yaml avec per_asset.
    Crée un backup horodaté strategies.yaml.bak.YYYYMMDD_HHMMSS.
    Retourne True si au moins un asset appliqué.
    """
    eligible = [r for r in reports if r.grade in ("A", "B")]
    if not eligible:
        logger.warning("Aucun rapport grade A/B — paramètres NON appliqués")
        return False

    # 1. Backup horodaté (pas écrasé)
    backup_path = f"{strategies_yaml_path}.bak.{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(strategies_yaml_path, backup_path)

    # 2. Charger YAML
    # 3. Paramètres communs = médiane cross-asset des paramètres convergents (score > 0.7)
    # 4. Paramètres divergents → écrits dans per_asset par symbol
    # 5. Sauvegarder YAML
    # 6. Log des changements (avant → après pour chaque paramètre)
```

**Logique d'application per_asset** :
- Les paramètres avec convergence cross-asset score > 0.7 → écrits comme paramètre par défaut (médiane)
- Les paramètres divergents (score < 0.7) → écrits dans `per_asset[symbol]` pour chaque asset grade A/B
- Les assets grade C-F → pas de `per_asset` pour cet asset, paramètres par défaut conservés

---

## Phase 4 — CLI & orchestration

### 4.1 Fichier `scripts/optimize.py` (nouveau, ~180 lignes)

```bash
# Vérifier les données disponibles avant de lancer
uv run python -m scripts.optimize --check-data

# Optimiser une stratégie sur un asset
uv run python -m scripts.optimize --strategy vwap_rsi --symbol BTC/USDT

# Optimiser une stratégie sur tous les assets
uv run python -m scripts.optimize --strategy vwap_rsi --all-symbols

# Optimiser toutes les stratégies optimisables
uv run python -m scripts.optimize --all

# Dry run (affiche le grid + estimation temps, sans exécuter)
uv run python -m scripts.optimize --all --dry-run

# Appliquer les paramètres grade A/B dans strategies.yaml (avec per_asset)
uv run python -m scripts.optimize --all --apply

# Verbose (affiche les résultats par fenêtre)
uv run python -m scripts.optimize --strategy vwap_rsi --symbol BTC/USDT -v
```

**Commande `--check-data`** :

```
Vérification des données pour l'optimisation
─────────────────────────────────────────────
BTC/USDT  Binance : 718 jours (5m: 206k candles) ✓
ETH/USDT  Binance : 718 jours (5m: 206k candles) ✓
SOL/USDT  Binance : 715 jours (5m: 205k candles) ✓
DOGE/USDT Binance :   0 jours                    ✗  → uv run python -m scripts.fetch_history --exchange binance --symbol DOGE/USDT --days 720
LINK/USDT Binance :   0 jours                    ✗  → uv run python -m scripts.fetch_history --exchange binance --symbol LINK/USDT --days 720

BTC/USDT  Bitget  :  88 jours (5m: 25k candles)  ✓
ETH/USDT  Bitget  :  88 jours (5m: 25k candles)  ✓
SOL/USDT  Bitget  :  88 jours (5m: 25k candles)  ✓
DOGE/USDT Bitget  :   0 jours                    ✗  → uv run python -m scripts.fetch_history --exchange bitget --symbol DOGE/USDT --days 90
LINK/USDT Bitget  :   0 jours                    ✗  → uv run python -m scripts.fetch_history --exchange bitget --symbol LINK/USDT --days 90
```

**Flux du CLI** :

```
1. Charger param_grids.yaml
2. Vérifier données Binance disponibles en DB
   → Si absent : logger.error + commande fetch_history à copier-coller + exit
3. Afficher message explicite si --all : "Funding et Liquidation exclus (pas de données
   historiques OI/funding). Optimisation de vwap_rsi et momentum uniquement."
4. Pour chaque (stratégie, symbole) :
   a. WFO (Phase 1) → WFOResult
   b. Overfitting detection (Phase 2) → OverfitReport
   c. Validation Bitget 90j (Phase 3) → ValidationResult
   d. Grading → FinalReport (sauvé en JSON)
   e. Affichage console résumé
5. Si --apply : appliquer les paramètres grade A/B (avec per_asset)
6. Affichage récapitulatif
```

**Sortie console (exemple)** :

```
═══════════════════════════════════════════════════════════
  Optimisation VWAP+RSI × BTC/USDT
═══════════════════════════════════════════════════════════

  Walk-Forward (20 fenêtres, Binance 2 ans)
  ──────────────────────────────────────────
  IS Sharpe moyen     : 1.82
  OOS Sharpe moyen    : 0.94
  OOS/IS ratio        : 0.52
  Consistance OOS+    : 75% (15/20 fenêtres)
  Combinaisons testées: 712

  Paramètres recommandés
  ──────────────────────
  rsi_period           : 14
  rsi_long_threshold   : 30
  rsi_short_threshold  : 70
  volume_spike_mult    : 2.0
  vwap_deviation_entry : 0.2
  tp_percent           : 0.6
  sl_percent           : 0.3

  Détection d'overfitting
  ───────────────────────
  Monte Carlo p-value  : 0.012 ✓ (< 0.05)
  DSR (n=712)          : 0.96 ✓ (> 0.95)
  Stabilité paramètres : 0.84 ✓ (> 0.80)
  Convergence cross    : 0.78 ✓ (> 0.70)

  Validation Bitget 90j
  ─────────────────────
  Sharpe Bitget        : 0.71 [CI 95%: 0.32 — 1.14]
  Transfer ratio       : 0.76 ✓ (> 0.50)
  Volume divergence    : Non

  ════════════════════════
  GRADE : A
  LIVE ELIGIBLE : Oui
  ════════════════════════
```

### 4.2 Stratégies optimisables

| Stratégie | Optimisable Sprint 7 ? | Raison |
|-----------|----------------------|--------|
| **vwap_rsi** | Oui | Données price-based suffisantes |
| **momentum** | Oui | Données price-based suffisantes |
| **funding** | Non | Pas de données historiques funding rate sur Binance |
| **liquidation** | Non | Pas de données historiques OI sur Binance |

Les stratégies funding et liquidation restent en paper trading avec paramètres manuels. Le CLI affiche un message explicite quand `--all` est utilisé.

---

## Phase 5 — Tests

### 5.1 Fichier `tests/test_optimization.py` (nouveau, ~35 tests)

**Tests Walk-Forward** (~8 tests) :

1. `test_wfo_trending_data_finds_momentum_params` — Données synthétiques avec trend connu → l'optimiseur retrouve les paramètres de breakout adaptés
2. `test_wfo_random_data_no_edge` — Données random → Sharpe OOS ≈ 0, pas d'edge détecté
3. `test_wfo_window_count` — 720 jours avec IS=120, OOS=30, step=30 → ~20 fenêtres
4. `test_wfo_grid_merge_default_and_asset` — Grid merge default + override asset
5. `test_wfo_coarse_fine_reduces_combos` — Grid > 1000 → coarse pass réduit à ~500
6. `test_wfo_consistency_rate` — Calculée correctement (fenêtres OOS+ / total)
7. `test_wfo_recommended_params_median` — Paramètres recommandés = médiane snappée au grid
8. `test_wfo_n_distinct_combos_tracked` — `WFOResult.n_distinct_combos` = coarse + fine sans doublons

**Tests Overfitting** (~10 tests) :

9. `test_mc_profitable_strategy_significant` — Stratégie profitable → p-value < 0.05
10. `test_mc_random_trades_not_significant` — Trades random → p-value > 0.10
11. `test_mc_block_preserves_order_within_blocks` — Les blocs sont intacts après shuffle
12. `test_mc_seed_none_varies_results` — seed=None donne des résultats légèrement différents
13. `test_dsr_many_trials_deflates` — DSR < Sharpe brut avec n_trials élevé
14. `test_dsr_few_trials_close_to_sharpe` — DSR ≈ Sharpe si peu de trials
15. `test_stability_plateau_high_score` — Paramètre sur plateau → score > 0.8
16. `test_stability_cliff_low_score` — Paramètre sur cliff → score < 0.5
17. `test_convergence_same_params_high_score` — Mêmes paramètres 5 assets → score > 0.9
18. `test_convergence_divergent_params_low_score` — Params très différents → score < 0.5
19. `test_convergence_single_asset_returns_none` — Un seul asset → convergence = None

**Tests Report & Validation** (~9 tests) :

20. `test_grade_A_all_criteria_met` — Tous les critères remplis → grade A
21. `test_grade_F_low_oos_ratio` — OOS/IS < 0.3 → grade F
22. `test_grade_C_partial_criteria` — Critères partiels → grade C
23. `test_validation_bootstrap_ci` — Bootstrap CI calculé correctement
24. `test_validation_no_auto_recalibration` — Pas de mini grid search volume automatique
25. `test_apply_writes_yaml_grade_A_with_per_asset` — `--apply` écrit dans strategies.yaml avec per_asset
26. `test_apply_refuses_grade_D` — `--apply` refuse d'écrire pour grade D
27. `test_apply_backup_timestamped` — Backup horodaté créé (pas écrasé)
28. `test_report_json_saved` — Rapport JSON sauvé correctement dans data/optimization/

**Tests Config per_asset** (~3 tests) :

29. `test_per_asset_override_tp_sl` — per_asset override correctement appliqué
30. `test_per_asset_empty_uses_defaults` — per_asset vide → paramètres par défaut
31. `test_per_asset_unknown_symbol_uses_defaults` — Symbole absent de per_asset → défaut

**Tests Intégration** (~4 tests) :

32. `test_strategy_registry_known` — Registre contient vwap_rsi et momentum
33. `test_strategy_registry_unknown_raises` — Stratégie inconnue → ValueError explicite
34. `test_candles_separated_by_exchange` — Candles Binance et Bitget stockées séparément en DB
35. `test_check_data_reports_missing` — `--check-data` détecte les données manquantes

---

## Fichiers modifiés (récap)

| Fichier | Modification |
|---------|-------------|
| `config/assets.yaml` | +DOGE/USDT, +LINK/USDT, +groupe altcoins |
| `config/strategies.yaml` | +section `per_asset: {}` sur vwap_rsi et momentum |
| `backend/core/models.py` | +champ `exchange: str = "bitget"` sur Candle |
| `backend/core/config.py` | +`per_asset` sur VwapRsiConfig/MomentumConfig, +`get_params_for_symbol()` |
| `backend/core/database.py` | +colonne exchange, backup auto, migration idempotente, méthodes avec param exchange |
| `backend/strategies/base.py` | +`_resolve_param()` pour overrides per_asset |
| `backend/strategies/vwap_rsi.py` | Utiliser `_resolve_param()` pour tp/sl et paramètres per_asset |
| `backend/strategies/momentum.py` | Idem |
| `scripts/fetch_history.py` | +`--exchange binance\|bitget`, factory exchange |
| `backend/backtesting/engine.py` | +fonction `run_backtest_single()` (module-level) |

## Fichiers créés (récap)

| Fichier | Contenu | Lignes estimées |
|---------|---------|-----------------|
| `backend/optimization/__init__.py` | Registre STRATEGY_REGISTRY + `create_strategy_with_params()` | ~30 |
| `backend/optimization/walk_forward.py` | WalkForwardOptimizer, grid search 2 passes, parallélisation | ~350 |
| `backend/optimization/overfitting.py` | Monte Carlo, DSR, stabilité, convergence | ~280 |
| `backend/optimization/report.py` | FinalReport, grading, validation Bitget + bootstrap CI, apply YAML per_asset | ~250 |
| `config/param_grids.yaml` | Espaces de recherche par stratégie | ~60 |
| `scripts/optimize.py` | CLI orchestrateur + --check-data | ~180 |
| `tests/test_optimization.py` | 35 tests | ~500 |

---

## Contraintes & risques

1. **Performance** : Le grid search sur ~700 combos (après coarse pass) × 20 fenêtres × 5 assets = 70 000 backtests. Chaque backtest sur 120j de 5m (~34 000 bougies) prend ~50ms → ~58 min total. Avec 24 threads i9-14900HX → **~3 min**. Acceptable.

2. **Mémoire** : Seules les candles de la fenêtre IS (~34k candles, ~3.4 MB) sont passées à chaque worker. Avec 24 workers : ~80 MB peak côté workers. L'historique complet (~500 MB pour 5 assets × 4 TF × 720j) est en mémoire une seule fois dans le process principal.

3. **Funding/Liquidation** : Exclus du Sprint 7. Pas de données historiques OI/funding sur Binance facilement accessibles. Ces stratégies restent en paper trading. Message explicite dans le CLI.

4. **Sérialisation ProcessPoolExecutor** : Les Candle Pydantic v2 et les stratégies sont picklables (pas de state non-picklable). Vérifié.

5. **Backward compat DB** : Migration idempotente (`PRAGMA table_info` avant `ALTER TABLE`). Backup automatique horodaté avant migration. Les candles existantes auront `exchange='bitget'` par défaut. Pas de perte de données.

6. **Volume Binance ≠ Bitget** : Pas de recalibration automatique (risque d'overfitting sur 90j). Warning reporté avec bootstrap CI pour évaluer la significativité de la divergence. Le trader décide.

7. **per_asset overrides** : Changement de config non-breaking. `per_asset: {}` vide par défaut → tout le code existant fonctionne. Les stratégies utilisent `_resolve_param()` qui fallback sur le paramètre par défaut si pas d'override.

---

## Ordre d'implémentation

1. **Phase 0** : assets.yaml + per_asset config + DB migration/backup + Candle.exchange + fetch_history Binance
2. **Phase 1** : registre stratégies + param_grids.yaml + walk_forward.py + run_backtest_single
3. **Phase 2** : overfitting.py (Monte Carlo avec seed opt, DSR avec n_trials explicite, stabilité, convergence)
4. **Phase 3** : report.py (grading, validation Bitget + bootstrap CI, apply YAML per_asset, backup horodaté)
5. **Phase 4** : scripts/optimize.py (CLI + --check-data + message exclusions)
6. **Phase 5** : tests/test_optimization.py

Chaque phase dépend de la précédente. Tests écrits au fur et à mesure de chaque phase.

---

## Vérification finale

1. `uv run pytest` — tous les tests passent (284 existants + ~35 nouveaux = ~319)
2. `uv run python -m scripts.optimize --check-data` — vérifie les données disponibles
3. `uv run python -m scripts.optimize --strategy vwap_rsi --symbol BTC/USDT --dry-run` — affiche le grid sans exécuter
4. Optimisation complète VWAP+RSI sur BTC → rapport JSON avec grade, métriques WFO, Monte Carlo, DSR, stabilité, bootstrap CI
5. Paramètres VWAP+RSI convergent entre BTC/ETH/SOL → cross-asset score > 0.7
6. Paramètres divergents (tp/sl) écrits dans `per_asset` au lieu de médiane aplatie
7. Validation Bitget 90j avec bootstrap CI — transfer_ratio + significativité
8. `--apply` écrit les paramètres grade A/B dans strategies.yaml (avec per_asset), ignore grade C-F
9. Backup `strategies.yaml.bak.YYYYMMDD_HHMMSS` créé (non écrasé)
