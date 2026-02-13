# Sprint 10 — Moteur Multi-Position Modulaire

## Objectif

Ajouter un moteur de backtesting multi-position qui supporte les stratégies DCA/grid
tout en restant modulable pour de futures stratégies. Le moteur existant (mono-position)
reste inchangé.

---

## Architecture

### Principe de séparation

```
BacktestEngine          (existant, inchangé)
  └─ BaseStrategy       → 1 position à la fois, TP/SL fixes
  └─ PositionManager    → open/close/check_exit une position

MultiPositionEngine     (NOUVEAU)
  └─ BaseGridStrategy   → N positions simultanées, grille de niveaux
  └─ GridPositionManager → gère N positions, prix moyen, SL global
```

Deux moteurs indépendants, même interface de sortie (`BacktestResult`),
même intégration WFO. Le WFO choisit le bon moteur selon la stratégie.

### Pourquoi deux moteurs plutôt qu'un seul unifié

Le mono-position fait `if position: check_exit, else: evaluate`. Simple, rapide.
Le multi-position fait `for each level: check_entry, manage N positions, check global exit`.
Forcer les deux dans un seul moteur crée de la complexité inutile et ralentit le mono.

### Héritage — BaseGridStrategy hérite de BaseStrategy

`BaseGridStrategy` **hérite de `BaseStrategy`** pour rester compatible avec l'écosystème
existant (Arena, Simulator, Dashboard, factory). Les méthodes mono-position ont des
implémentations par défaut :

- `evaluate()` → retourne `None` (pas utilisé par le MultiPositionEngine)
- `check_exit()` → retourne `None`
- `get_current_conditions()` → retourne les niveaux de la grille et leur état

Le MultiPositionEngine ignore `evaluate()`/`check_exit()` et utilise à la place
`compute_grid()`, `should_close_all()`, `get_tp_price()`, `get_sl_price()`.

---

## Étape 1 — Abstractions (`backend/strategies/base_grid.py`, ~150 lignes)

### GridLevel — Un niveau de la grille

```python
@dataclass
class GridLevel:
    """Un niveau d'entrée dans la grille."""
    index: int              # 0, 1, 2, 3 (0 = plus proche de la MA)
    entry_price: float      # Prix d'entrée pour ce niveau
    direction: Direction    # LONG ou SHORT
    size_fraction: float    # Fraction du capital alloué (ex: 0.25 pour 4 niveaux)
```

### GridState — État courant de la grille

```python
@dataclass
class GridState:
    """État complet de toutes les positions de la grille."""
    positions: list[GridPosition]   # Positions ouvertes (0 à N)
    avg_entry_price: float          # Prix moyen pondéré
    total_quantity: float           # Quantité totale
    total_notional: float           # Valeur notionnelle totale
    unrealized_pnl: float           # P&L non réalisé global

@dataclass
class GridPosition:
    """Une position individuelle dans la grille."""
    level: int
    direction: Direction
    entry_price: float
    quantity: float
    entry_time: datetime
    entry_fee: float
```

### BaseGridStrategy — Hérite de BaseStrategy

```python
class BaseGridStrategy(BaseStrategy):
    """Stratégie multi-position avec grille de niveaux.

    Hérite de BaseStrategy pour la compatibilité Arena/Simulator/Dashboard.
    Les méthodes mono-position (evaluate, check_exit) ont des implémentations
    par défaut (return None). Le MultiPositionEngine utilise les méthodes
    grid-spécifiques à la place.

    Interface grid (nouvelles méthodes abstraites) :
    - compute_grid() : les niveaux d'entrée/sortie à chaque bougie
    - should_close_all() : condition de sortie globale (retour à la MA, etc.)
    - get_sl_price() : SL global basé sur le prix moyen
    - get_tp_price() : TP global dynamique

    Le moteur gère : ouverture/fermeture des positions, sizing, fees, equity.
    """

    name: str = "base_grid"

    # --- Implémentations par défaut BaseStrategy (mono-position) ---

    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Non utilisé par MultiPositionEngine. Retourne None."""
        return None

    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """Non utilisé par MultiPositionEngine. Retourne None."""
        return None

    def get_current_conditions(self, ctx: StrategyContext) -> list[dict]:
        """Retourne les niveaux de la grille pour le dashboard.

        Montre quels niveaux sont actifs et leur état (filled/pending).
        Compatible avec le format conditions Sprint 6.
        """
        grid_state = GridState(positions=[], avg_entry_price=0, total_quantity=0,
                               total_notional=0, unrealized_pnl=0)
        levels = self.compute_grid(ctx, grid_state)
        conditions = []
        for lvl in levels:
            conditions.append({
                "name": f"Level {lvl.index + 1} ({lvl.direction.value})",
                "met": False,
                "value": f"{lvl.entry_price:.2f}",
                "threshold": f"touch",
            })
        return conditions

    # --- Interface grid (abstraite) ---

    @abstractmethod
    def compute_indicators(
        self, candles_by_tf: dict[str, list[Candle]]
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Pré-calcule les indicateurs (identique à BaseStrategy)."""

    @abstractmethod
    def compute_grid(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> list[GridLevel]:
        """Calcule les niveaux de la grille pour la bougie courante.

        Appelé à chaque bougie. Retourne les niveaux auxquels on veut entrer.
        Le moteur vérifie si le prix a touché un niveau et ouvre la position.

        IMPORTANT : un seul côté actif à la fois. Si des positions LONG sont
        ouvertes dans grid_state, ne PAS retourner de niveaux SHORT (et inversement).
        Le premier niveau touché détermine la direction de toute la séquence.

        Args:
            ctx: Contexte (candles, indicateurs, capital)
            grid_state: Positions actuellement ouvertes

        Returns:
            Liste de GridLevel. Seuls les niveaux non encore remplis sont utilisés.
        """

    @abstractmethod
    def should_close_all(
        self, ctx: StrategyContext, grid_state: GridState
    ) -> str | None:
        """Vérifie si toutes les positions doivent être fermées.

        Returns:
            "tp_global" si retour à la MA (TP)
            "sl_global" si SL global touché
            None sinon
        """

    @abstractmethod
    def get_sl_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """Calcule le SL global basé sur le prix moyen d'entrée.

        Appelé par le moteur pour la vérification intra-candle OHLC.
        """

    @abstractmethod
    def get_tp_price(
        self, grid_state: GridState, current_indicators: dict
    ) -> float:
        """Calcule le TP global (ex: retour à la SMA).

        Dynamique : change à chaque bougie car la SMA bouge.
        """

    @property
    @abstractmethod
    def min_candles(self) -> dict[str, int]:
        """Nombre minimum de bougies par timeframe."""

    @property
    @abstractmethod
    def max_positions(self) -> int:
        """Nombre maximum de positions simultanées (= nombre de niveaux)."""

    def get_params(self) -> dict[str, Any]:
        """Retourne les paramètres pour le reporting."""
        return {}
```

---

## Étape 2 — GridPositionManager (`backend/core/grid_position_manager.py`, ~200 lignes)

Gère N positions ouvertes simultanément avec prix moyen pondéré.

### Sizing — allocation fixe par niveau (comme le live)

Pas de risk-based sizing pour le DCA. C'est une allocation fixe par niveau :
```
notional = capital × (1/num_levels) × leverage
quantity = notional / entry_price
```
Le SL global protège le capital total, pas le sizing par position.
C'est identique au live : `size = (params["size"] * balance) / len(envelopes) * leverage / price`.

### Sizing — crossed margin (capital total)

En crossed margin, le capital total reste disponible. Le sizing utilise
`capital × (1/num_levels) × leverage / entry_price`.
Pas de déduction progressive du capital entre les ouvertures — c'est cohérent avec
le fonctionnement réel de Bitget en mode cross.

```python
class GridPositionManager:
    """Gestion multi-position pour stratégies grid/DCA.

    Différences avec PositionManager :
    - Gère une LISTE de positions (pas une seule)
    - Calcule le prix moyen pondéré automatiquement
    - Ferme TOUTES les positions en un seul TradeResult agrégé
    - Support SL/TP global basé sur le prix moyen
    - Allocation fixe par niveau (pas risk-based) : notional = capital/levels × leverage
    """

    def __init__(self, config: PositionManagerConfig):
        self._config = config

    def open_grid_position(
        self,
        level: GridLevel,
        timestamp: datetime,
        capital: float,
        total_levels: int,
    ) -> GridPosition | None:
        """Ouvre une position pour un niveau de grille.

        Allocation fixe par niveau (comme le live, pas risk-based) :
        Notional = capital × (1/total_levels) × leverage
        Quantity = notional / entry_price
        Le SL global protège le capital total, pas le sizing par position.
        """

    def close_all_positions(
        self,
        positions: list[GridPosition],
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        regime: MarketRegime,
    ) -> TradeResult:
        """Ferme toutes les positions, retourne UN TradeResult agrégé.

        Le TradeResult utilise :
        - entry_price = prix moyen pondéré
        - quantity = somme des quantités
        - gross_pnl = somme des P&L individuels
        - fee_cost = somme des fees (entrée + sortie)
        """

    def check_global_tp_sl(
        self,
        positions: list[GridPosition],
        candle: Candle,
        tp_price: float,
        sl_price: float,
    ) -> tuple[str | None, float]:
        """Vérifie TP/SL global avec heuristique OHLC.

        Returns:
            (exit_reason, exit_price) ou (None, 0.0)
        """

    def compute_grid_state(
        self,
        positions: list[GridPosition],
        current_price: float,
    ) -> GridState:
        """Calcule l'état agrégé de la grille."""

    def unrealized_pnl(
        self,
        positions: list[GridPosition],
        current_price: float,
    ) -> float:
        """P&L non réalisé total."""
```

---

## Étape 3 — MultiPositionEngine (`backend/backtesting/multi_engine.py`, ~300 lignes)

```python
class MultiPositionEngine:
    """Moteur de backtesting multi-position.

    Boucle principale :
    1. Pré-calcul des indicateurs (une seule fois)
    2. Pour chaque bougie :
       a. Si positions ouvertes :
          - Calculer TP/SL global dynamique
          - Check TP/SL global (heuristique OHLC)
          - Check should_close_all() (sortie sur signal)
          - Si fermeture → enregistrer TradeResult agrégé
       b. Si grille pas pleine :
          - compute_grid() → niveaux d'entrée (un seul côté actif)
          - Pour chaque niveau non rempli :
            - Si low <= level.entry_price (LONG) → ouvrir position
            - Si high >= level.entry_price (SHORT) → ouvrir position
       c. Mise à jour equity curve
    3. Clôture forcée en fin de données

    Produit un BacktestResult (même format que BacktestEngine).
    """

    def __init__(self, config: BacktestConfig, strategy: BaseGridStrategy):
        self._config = config
        self._strategy = strategy
        self._gpm = GridPositionManager(PositionManagerConfig(
            leverage=config.leverage,
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
            slippage_pct=config.slippage_pct,
            high_vol_slippage_mult=config.high_vol_slippage_mult,
            max_risk_per_trade=config.max_risk_per_trade,
        ))

    def run(
        self,
        candles_by_tf: dict[str, list[Candle]],
        main_tf: str = "1h",
        precomputed_indicators=None,
    ) -> BacktestResult:
        """Lance le backtest multi-position."""

        # 1. Pré-calcul indicateurs
        # 2. Boucle principale
        #    Pour chaque bougie:
        #      a. Construire StrategyContext
        #      b. Construire GridState depuis positions ouvertes
        #      c. Si positions ouvertes:
        #         - tp_price = strategy.get_tp_price(grid_state, indicators)
        #         - sl_price = strategy.get_sl_price(grid_state, indicators)
        #         - Vérifier TP/SL global via OHLC heuristic
        #         - Vérifier should_close_all()
        #         - Si fermeture: close_all → trade_result, capital += pnl
        #      d. Si len(positions) < strategy.max_positions:
        #         - levels = strategy.compute_grid(ctx, grid_state)
        #         - CONTRAINTE : un seul côté actif. Si positions ouvertes en
        #           LONG, filtrer les niveaux SHORT (et inversement).
        #         - Pour chaque level non rempli:
        #           Si candle.low <= level.entry_price (LONG):
        #             position = gpm.open_grid_position(level, ...)
        #             positions.append(position)
        #      e. Equity curve: capital + unrealized_pnl
        # 3. Force close en fin de données
```

### Points critiques de la boucle

**Ordre d'évaluation intra-bougie (OHLC heuristic)** :
- Bougie verte (close > open) : on suppose low touché d'abord → check SL d'abord,
  puis high → check TP, puis levels d'entrée en dessous du close
- Bougie rouge : high d'abord → check TP, puis low → check SL
- Cela réutilise la même logique que BacktestEngine

**Entrées multiples sur la même bougie** :
- Si le prix traverse plusieurs niveaux en une bougie (ex: flash crash),
  on ouvre toutes les positions touchées dans l'ordre (du plus proche au plus éloigné)
- Sizing en crossed margin : capital total, pas de déduction progressive

**Un seul côté actif** :
- Si des positions LONG sont ouvertes, compute_grid() ne retourne que des niveaux LONG
- Le premier niveau touché dans une séquence vide détermine la direction
- Le moteur filtre en plus (double sécurité) les niveaux du mauvais côté

**TP/SL global vs par position** :
- Pas de TP/SL par position individuelle
- TP = retour à la SMA (dynamique, change à chaque bougie)
- SL = % fixe depuis le prix moyen pondéré de TOUTES les positions
- Quand TP ou SL touché → fermeture de TOUTES les positions en un bloc

---

## Étape 4 — EnvelopeDCAStrategy (`backend/strategies/envelope_dca.py`, ~200 lignes)

Première implémentation concrète de BaseGridStrategy :

### Formule des enveloppes asymétriques

Les bandes haute et basse ne sont PAS symétriques. La bande basse est un pourcentage
direct, mais la bande haute utilise la formule inverse pour que l'aller-retour soit
cohérent :

```python
# Bandes basses (LONG) : distance directe
lower_envelopes = [envelope_start + i * envelope_step for i in range(num_levels)]
# Ex: [0.07, 0.10, 0.13] pour start=0.07, step=0.03, levels=3

# Bandes hautes (SHORT) : formule inverse → round(1/(1-e) - 1, 3)
high_envelopes = [round(1 / (1 - e) - 1, 3) for e in lower_envelopes]
# Ex: [0.075, 0.111, 0.149] pour les mêmes valeurs

# Application
lower_band[i] = sma * (1 - lower_envelopes[i])
upper_band[i] = sma * (1 + high_envelopes[i])
```

Explication : si le prix descend de 7% (lower), il doit remonter de `1/(1-0.07)-1 = 7.5%`
pour revenir au même point. La bande haute compense cette asymétrie.

```python
class EnvelopeDCAStrategy(BaseGridStrategy):
    """Stratégie Envelope DCA (Mean Reversion Multi-Niveaux).

    Logique :
    - SMA sur le close
    - N enveloppes en dessous (LONG) et au-dessus (SHORT) — asymétriques
    - Entrée à chaque niveau touché (DCA)
    - TP = retour à la SMA
    - SL = % fixe depuis le prix moyen

    Paramètres optimisables :
    - ma_period: période SMA (5-30)
    - num_levels: nombre d'enveloppes (2-5)
    - envelope_start: écart du premier niveau (0.03-0.10)
    - envelope_step: écart entre niveaux (0.02-0.05)
    - sl_percent: SL global depuis prix moyen (10-30%)
    - sides: ["long"], ["short"], ["long", "short"]
    """

    name = "envelope_dca"

    def __init__(self, config: EnvelopeDCAConfig):
        self._config = config

    @property
    def max_positions(self) -> int:
        return self._config.num_levels

    @property
    def min_candles(self) -> dict[str, int]:
        return {
            self._config.timeframe: max(self._config.ma_period + 20, 50),
        }

    def compute_indicators(self, candles_by_tf):
        """Calcule SMA uniquement.

        Les bandes d'enveloppe sont calculées à la volée dans compute_grid()
        (c'est une simple multiplication, pas besoin de pré-calcul).
        """
        result: dict[str, dict[str, dict[str, float]]] = {}
        tf = self._config.timeframe
        if tf in candles_by_tf and candles_by_tf[tf]:
            candles = candles_by_tf[tf]
            closes = np.array([c.close for c in candles], dtype=float)
            sma_arr = sma(closes, self._config.ma_period)

            indicators: dict[str, dict[str, float]] = {}
            for i, c in enumerate(candles):
                ts = c.timestamp.isoformat()
                indicators[ts] = {
                    "sma": float(sma_arr[i]),
                    "close": c.close,
                }
            result[tf] = indicators
        return result

    def compute_grid(self, ctx, grid_state):
        """Retourne les niveaux non encore remplis.

        CONTRAINTE : un seul côté actif à la fois.
        Si des positions sont ouvertes, on ne retourne que les niveaux
        du même côté. Le premier niveau touché détermine la direction.
        """
        indicators = ctx.indicators.get(self._config.timeframe, {})
        sma_val = indicators.get("sma", float("nan"))

        if np.isnan(sma_val):
            return []

        levels = []
        filled_levels = {p.level for p in grid_state.positions}

        # Déterminer le côté actif
        if grid_state.positions:
            active_direction = grid_state.positions[0].direction
            active_sides = (
                ["long"] if active_direction == Direction.LONG else ["short"]
            )
        else:
            active_sides = self._config.sides

        # Calcul des enveloppes
        lower_envelopes = [
            self._config.envelope_start + i * self._config.envelope_step
            for i in range(self._config.num_levels)
        ]
        high_envelopes = [round(1 / (1 - e) - 1, 3) for e in lower_envelopes]

        for i in range(self._config.num_levels):
            if i in filled_levels:
                continue

            if "long" in active_sides:
                lower_price = sma_val * (1 - lower_envelopes[i])
                levels.append(GridLevel(
                    index=i,
                    entry_price=lower_price,
                    direction=Direction.LONG,
                    size_fraction=1.0 / self._config.num_levels,
                ))

            if "short" in active_sides:
                upper_price = sma_val * (1 + high_envelopes[i])
                levels.append(GridLevel(
                    index=i,
                    entry_price=upper_price,
                    direction=Direction.SHORT,
                    size_fraction=1.0 / self._config.num_levels,
                ))

        return levels

    def should_close_all(self, ctx, grid_state):
        """Fermer si le prix revient à la SMA."""
        if not grid_state.positions:
            return None

        indicators = ctx.indicators.get(self._config.timeframe, {})
        sma_val = indicators.get("sma", float("nan"))
        close = indicators.get("close", float("nan"))

        if np.isnan(sma_val) or np.isnan(close):
            return None

        direction = grid_state.positions[0].direction

        # TP: prix revient à la SMA
        if direction == Direction.LONG and close >= sma_val:
            return "tp_global"
        if direction == Direction.SHORT and close <= sma_val:
            return "tp_global"

        # SL: prix s'éloigne trop du prix moyen
        sl_pct = self._config.sl_percent / 100
        if direction == Direction.LONG:
            if close <= grid_state.avg_entry_price * (1 - sl_pct):
                return "sl_global"
        else:
            if close >= grid_state.avg_entry_price * (1 + sl_pct):
                return "sl_global"

        return None

    def get_tp_price(self, grid_state, indicators):
        """TP = SMA actuelle (dynamique)."""
        return indicators.get("sma", float("nan"))

    def get_sl_price(self, grid_state, indicators):
        """SL = % depuis prix moyen."""
        if not grid_state.positions:
            return float("nan")
        avg = grid_state.avg_entry_price
        sl_pct = self._config.sl_percent / 100
        direction = grid_state.positions[0].direction
        if direction == Direction.LONG:
            return avg * (1 - sl_pct)
        return avg * (1 + sl_pct)

    def get_params(self) -> dict[str, Any]:
        return self._config.model_dump(
            exclude={"enabled", "live_eligible", "weight", "per_asset"}
        )
```

### Config (`EnvelopeDCAConfig` dans `backend/core/config.py`)

```python
class EnvelopeDCAConfig(BaseModel):
    enabled: bool = True
    live_eligible: bool = False
    timeframe: str = "1h"
    ma_period: int = Field(default=7, ge=2, le=50)
    num_levels: int = Field(default=3, ge=1, le=6)
    envelope_start: float = Field(default=0.07, gt=0)  # Premier niveau à 7%
    envelope_step: float = Field(default=0.03, gt=0)    # +3% entre niveaux
    sl_percent: float = Field(default=25.0, gt=0)       # SL global 25%
    sides: list[str] = Field(default=["long"])           # ["long"], ["short"], ["long","short"]
    leverage: int = Field(default=6, ge=1, le=20)
    weight: float = Field(default=0.20, ge=0, le=1)
    per_asset: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:
        """Retourne les paramètres avec overrides per_asset appliqués."""
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}
```

**Modifications `StrategiesConfig`** :
```python
class StrategiesConfig(BaseModel):
    # ... existants ...
    envelope_dca: EnvelopeDCAConfig = Field(default_factory=EnvelopeDCAConfig)

    @model_validator(mode="after")
    def validate_weights(self) -> StrategiesConfig:
        enabled = [
            s for s in [
                self.vwap_rsi, self.liquidation, self.orderflow,
                self.momentum, self.funding,
                self.bollinger_mr, self.donchian_breakout, self.supertrend,
                self.envelope_dca,  # AJOUTÉ
            ]
            if s.enabled
        ]
        # ...
```

### Grid d'optimisation (`config/param_grids.yaml`)

```yaml
envelope_dca:
  wfo:
    is_days: 365      # 1 an IS (trades lents = besoin de plus de données)
    oos_days: 90       # 3 mois OOS
    step_days: 90      # Pas de 3 mois
  default:
    ma_period: [5, 7, 10, 15, 20]
    num_levels: [2, 3, 4]
    envelope_start: [0.05, 0.07, 0.10]
    envelope_step: [0.02, 0.03, 0.05]
    sl_percent: [15.0, 20.0, 25.0, 30.0]
  # sides fixé à ["long"] pour le premier test
  # 5 × 3 × 3 × 3 × 4 = 540 combinaisons
```

---

## Étape 5 — Fast Engine Multi-Position (`backend/optimization/fast_multi_backtest.py`, ~350 lignes)

Nouveau fichier parallèle à `fast_backtest.py`, optimisé pour le grid search multi-position.

### Cache — seulement SMA, enveloppes à la volée

Le cache ne stocke que `sma[period]` (un array par variante de `ma_period`).
Les enveloppes sont calculées à la volée dans la boucle de simulation :
`lower_band = sma[i] * (1 - offset)` — c'est une multiplication triviale,
pas besoin de pré-calcul.

**Extension IndicatorCache** :
```python
# Ajouter dans build_cache pour envelope_dca :
# Réutilise les champs existants :
#   cache.bb_sma: dict[int, np.ndarray]  (déjà utilisé par bollinger_mr)
# On stocke SMA[period] dans bb_sma — même type, même usage.
# OU un champ dédié si on veut la clarté :
#   cache.sma_by_period: dict[int, np.ndarray]  # {period: sma_array}
```

Dans `build_cache()`, section envelope_dca :
```python
if strategy_name == "envelope_dca":
    ma_periods: set[int] = set()
    if "ma_period" in param_grid_values:
        ma_periods.update(param_grid_values["ma_period"])
    if not ma_periods:
        ma_periods.add(7)
    for period in ma_periods:
        bb_sma_dict[period] = sma(closes, period)
```

### Boucle de simulation

```python
def run_multi_backtest_from_cache(
    strategy_name: str,
    params: dict,
    cache: IndicatorCache,
    bt_config: BacktestConfig,
) -> _ISResult:
    """Backtest rapide multi-position sur cache numpy.

    Différence avec run_backtest_from_cache :
    - Gère N positions ouvertes simultanément
    - TP = retour à SMA (dynamique, recalculé à chaque bougie)
    - SL = % depuis prix moyen
    - Un "trade" = ouverture/fermeture de TOUTES les positions
    """

def _simulate_grid_trades(
    cache: IndicatorCache,
    params: dict,
    initial_capital: float,
    leverage: int,
    maker_fee: float,
    taker_fee: float,
    slippage_pct: float,
    max_risk_per_trade: float,
) -> tuple[list[float], list[float], float]:
    """
    Simulation multi-position.

    La boucle est nécessaire (état = positions ouvertes),
    mais les indicateurs sont pré-calculés (SMA depuis le cache).
    Les enveloppes sont calculées à la volée (multiplication triviale).
    """
    trades_pnl = []
    trade_returns = []
    positions = []  # list of (level, entry_price, quantity, entry_fee, entry_idx)
    capital = initial_capital
    num_levels = params["num_levels"]

    sma_arr = cache.bb_sma[params["ma_period"]]

    # Pré-calculer les offsets d'enveloppe (constantes pour toute la simulation)
    lower_offsets = [
        params["envelope_start"] + lvl * params["envelope_step"]
        for lvl in range(num_levels)
    ]
    high_offsets = [round(1 / (1 - e) - 1, 3) for e in lower_offsets]

    for i in range(len(cache.closes)):
        if np.isnan(sma_arr[i]):
            continue

        # 1. Check TP/SL global si positions ouvertes
        if positions:
            avg_entry = sum(p[1]*p[2] for p in positions) / sum(p[2] for p in positions)
            total_qty = sum(p[2] for p in positions)
            direction = 1  # LONG (sides=["long"] par défaut)

            # SL check (prix moyen - sl%)
            sl_price = avg_entry * (1 - params["sl_percent"] / 100)
            if cache.lows[i] <= sl_price:
                pnl = _calc_grid_pnl(positions, sl_price, taker_fee, slippage_pct, direction)
                trades_pnl.append(pnl)
                if capital > 0:
                    trade_returns.append(pnl / capital)
                capital += pnl
                positions = []
                continue

            # TP check (retour à SMA — dynamique)
            tp_price = sma_arr[i]
            if cache.highs[i] >= tp_price:
                pnl = _calc_grid_pnl(positions, tp_price, maker_fee, 0.0, direction)
                trades_pnl.append(pnl)
                if capital > 0:
                    trade_returns.append(pnl / capital)
                capital += pnl
                positions = []
                continue

        # 2. Ouvrir de nouvelles positions si niveaux touchés
        filled_levels = {p[0] for p in positions}
        for lvl in range(num_levels):
            if lvl in filled_levels:
                continue
            # Enveloppe calculée à la volée
            entry_price = sma_arr[i] * (1 - lower_offsets[lvl])
            if np.isnan(entry_price):
                continue
            if cache.lows[i] <= entry_price:
                # Sizing : allocation fixe par niveau (comme le live)
                notional = capital * (1.0 / num_levels) * leverage
                qty = notional / entry_price
                if qty <= 0:
                    continue
                entry_fee = qty * entry_price * taker_fee
                positions.append((lvl, entry_price, qty, entry_fee, i))

    # Force close fin de données
    if positions:
        exit_price = cache.closes[-1]
        pnl = _calc_grid_pnl(positions, exit_price, taker_fee, slippage_pct, 1)
        trades_pnl.append(pnl)
        if capital > 0:
            trade_returns.append(pnl / capital)
        capital += pnl

    return trades_pnl, trade_returns, capital
```

---

## Étape 6 — WFO Integration

### `backend/optimization/__init__.py`

```python
from backend.strategies.envelope_dca import EnvelopeDCAStrategy
from backend.core.config import EnvelopeDCAConfig

# Registre existant — ajout envelope_dca
STRATEGY_REGISTRY: dict[str, tuple[type, type]] = {
    "vwap_rsi": (VwapRsiConfig, VwapRsiStrategy),
    "momentum": (MomentumConfig, MomentumStrategy),
    "funding": (FundingConfig, FundingStrategy),
    "liquidation": (LiquidationConfig, LiquidationStrategy),
    "bollinger_mr": (BollingerMRConfig, BollingerMRStrategy),
    "donchian_breakout": (DonchianBreakoutConfig, DonchianBreakoutStrategy),
    "supertrend": (SuperTrendConfig, SuperTrendStrategy),
    "envelope_dca": (EnvelopeDCAConfig, EnvelopeDCAStrategy),  # AJOUTÉ
}

# Set des stratégies grid (pour choisir le bon moteur)
GRID_STRATEGIES: set[str] = {"envelope_dca"}

def is_grid_strategy(name: str) -> bool:
    return name in GRID_STRATEGIES
```

Pas de registre séparé — `envelope_dca` est dans le `STRATEGY_REGISTRY` principal.
`GRID_STRATEGIES` est juste un set pour savoir quel moteur utiliser.
`--all` dans optimize.py itère `STRATEGY_REGISTRY.keys()` sans changement.

### `backend/optimization/walk_forward.py`

Modifications :

**1. `_INDICATOR_PARAMS`** — ajouter :
```python
_INDICATOR_PARAMS: dict[str, list[str]] = {
    # ... existants ...
    "envelope_dca": ["ma_period"],  # Seul ma_period affecte compute_indicators
}
```

**2. `_parallel_backtest()`** — ajouter envelope_dca dans la liste fast engine :
```python
if strategy_name in ("vwap_rsi", "momentum", "bollinger_mr",
                      "donchian_breakout", "supertrend", "envelope_dca"):
    try:
        results = self._run_fast(...)
```

**3. `_run_fast()`** — le fast engine appelle `run_multi_backtest_from_cache`
pour les grid strategies :
```python
from backend.optimization import is_grid_strategy
if is_grid_strategy(strategy_name):
    from backend.optimization.fast_multi_backtest import run_multi_backtest_from_cache
    # ... même pattern que run_backtest_from_cache
else:
    from backend.optimization.fast_backtest import run_backtest_from_cache
    # ... existant
```

**4. Leverage depuis la config stratégie** — Le WFO doit passer le leverage
de `EnvelopeDCAConfig` (6) au `BacktestConfig`, pas le défaut (15) :
```python
config_cls, _ = STRATEGY_REGISTRY[strategy_name]
default_cfg = config_cls()
# Si la config a un champ leverage, l'utiliser
strategy_leverage = getattr(default_cfg, "leverage", bt_config.leverage)
bt_config = BacktestConfig(
    ...,
    leverage=strategy_leverage,
)
```

**5. OOS evaluation** — pour les grid strategies, utiliser MultiPositionEngine :
```python
from backend.optimization import is_grid_strategy
if is_grid_strategy(strategy_name):
    from backend.backtesting.multi_engine import MultiPositionEngine
    strategy = create_strategy_with_params(strategy_name, best_params)
    engine = MultiPositionEngine(bt_config, strategy)
    oos_result = engine.run(oos_candles_by_tf, main_tf=main_tf)
else:
    oos_result = run_backtest_single(...)
```

---

## Étape 7 — Factory & run_backtest.py

### `backend/strategies/factory.py`

```python
from backend.strategies.envelope_dca import EnvelopeDCAStrategy

def create_strategy(name: str, config: AppConfig) -> BaseStrategy:
    mapping: dict[str, tuple] = {
        # ... existants ...
        "envelope_dca": (EnvelopeDCAStrategy, strategies_config.envelope_dca),
    }
    # ...

def get_enabled_strategies(config: AppConfig) -> list[BaseStrategy]:
    # ... existants ...
    if strats.envelope_dca.enabled:
        strategies.append(EnvelopeDCAStrategy(strats.envelope_dca))
    return strategies
```

### `scripts/run_backtest.py`

Étendre `STRATEGY_MAP` en le générant depuis les registres :

```python
from backend.optimization import STRATEGY_REGISTRY, is_grid_strategy

# Auto-generate STRATEGY_MAP depuis les registres
def _build_strategy_map():
    mapping = {}
    for name, (config_cls, strategy_cls) in STRATEGY_REGISTRY.items():
        # Closure pour capturer name/config_cls/strategy_cls
        def factory(config, _cls=strategy_cls, _ccls=config_cls, _name=name):
            strat_config = getattr(config.strategies, _name, _ccls())
            return _cls(strat_config)
        mapping[name] = factory
    return mapping

STRATEGY_MAP = _build_strategy_map()
```

Pour les grid strategies, `run_backtest.py` doit utiliser `MultiPositionEngine` :
```python
from backend.optimization import is_grid_strategy

if is_grid_strategy(strategy_name):
    from backend.backtesting.multi_engine import MultiPositionEngine
    engine = MultiPositionEngine(bt_config, strategy)
else:
    engine = BacktestEngine(bt_config, strategy)
result = engine.run(candles_by_tf, main_tf=main_tf)
```

---

## Étape 8 — Config YAML

### `config/strategies.yaml`

```yaml
envelope_dca:
  enabled: true
  live_eligible: false
  timeframe: "1h"
  ma_period: 7
  num_levels: 3
  envelope_start: 0.07
  envelope_step: 0.03
  sl_percent: 25.0
  sides: ["long"]
  leverage: 6
  weight: 0.20
  per_asset: {}
```

---

## Étape 9 — Tests (~40 tests)

### `tests/test_multi_engine.py`

**GridPositionManager (8 tests)** :
- open_grid_position sizing correct (allocation fixe par niveau)
- close_all_positions agrège P&L
- prix moyen pondéré correct (2 positions, 3 positions)
- unrealized_pnl avec N positions
- check_global_tp_sl OHLC heuristic
- position vide → state vide
- sizing allocation fixe crossed margin (capital total, pas de déduction)
- fees cumulées correctes

**MultiPositionEngine (10 tests)** :
- 1 niveau touché → 1 position ouverte
- 3 niveaux touchés progressivement → 3 positions
- TP global quand prix revient à SMA → fermeture toutes positions
- SL global quand prix dépasse seuil → fermeture toutes positions
- Flash crash traverse 3 niveaux en 1 bougie → 3 positions ouvertes
- Equity curve avec positions ouvertes = capital + unrealized
- Force close fin de données
- Aucun niveau touché → 0 trades
- OHLC heuristic multi-position (bougie verte/rouge)
- BacktestResult format identique au mono-position

**EnvelopeDCAStrategy (10 tests)** :
- compute_grid retourne bons niveaux
- niveaux déjà remplis exclus
- should_close_all TP quand close >= SMA
- should_close_all SL quand close < avg_entry - sl%
- compute_indicators SMA correcte
- sides=["long"] → pas de niveaux SHORT
- sides=["long","short"] → niveaux des deux côtés
- **un seul côté actif** : positions LONG ouvertes → pas de niveaux SHORT
- enveloppes asymétriques : upper != lower (formule 1/(1-e)-1)
- get_current_conditions retourne les niveaux pour le dashboard

**Fast Engine Multi (8 tests)** :
- Parité fast vs normal (même params, même données, même résultat ±1%)
- Performance : fast au moins 10x plus rapide
- Grid search 540 combos en < 2 min
- Trades identiques (nombre, PnL, timestamps)
- Enveloppes calculées à la volée (pas dans le cache)
- SMA cache hit (même period = même array)
- Leverage 6 (pas le défaut 15)
- Sizing : allocation fixe notional = capital/levels × leverage

**WFO Integration (4 tests)** :
- is_grid_strategy détecte correctement
- WFO utilise MultiPositionEngine pour envelope_dca
- Registry contient envelope_dca
- param_grids chargé correctement

---

## Résumé des fichiers

| Fichier | Type | Lignes | Notes |
|---------|------|--------|-------|
| `backend/strategies/base_grid.py` | NOUVEAU | ~150 | Hérite de BaseStrategy |
| `backend/core/grid_position_manager.py` | NOUVEAU | ~200 | Risk / num_levels |
| `backend/backtesting/multi_engine.py` | NOUVEAU | ~300 | Un seul côté actif |
| `backend/strategies/envelope_dca.py` | NOUVEAU | ~200 | Enveloppes asymétriques |
| `backend/optimization/fast_multi_backtest.py` | NOUVEAU | ~350 | Cache SMA only |
| `backend/optimization/__init__.py` | MODIFIÉ | +10 | GRID_STRATEGIES set |
| `backend/optimization/walk_forward.py` | MODIFIÉ | +40 | Leverage, fast, OOS |
| `backend/optimization/indicator_cache.py` | MODIFIÉ | +10 | SMA pour envelope_dca |
| `backend/core/config.py` | MODIFIÉ | +30 | EnvelopeDCAConfig + StrategiesConfig |
| `backend/strategies/factory.py` | MODIFIÉ | +5 | Ajout envelope_dca |
| `scripts/run_backtest.py` | MODIFIÉ | +20 | Auto STRATEGY_MAP + MultiEngine |
| `config/strategies.yaml` | MODIFIÉ | +12 | Section envelope_dca |
| `config/param_grids.yaml` | MODIFIÉ | +15 | Grid + WFO config |
| `tests/test_multi_engine.py` | NOUVEAU | ~500 | 40 tests |
| **Total** | | **~1840** | |

---

## Points d'attention pour Claude Code

1. **Ne PAS modifier BacktestEngine** — tout est dans de nouveaux fichiers
2. **Même BacktestResult** en sortie — le WFO et les métriques fonctionnent sans changement
3. **Heuristique OHLC** — réutiliser la même logique que PositionManager pour le TP/SL global
4. **Sizing : allocation fixe par niveau** — `notional = capital/levels × leverage`, pas risk-based. Capital total en crossed margin
5. **TP dynamique** — la SMA change à chaque bougie, le TP aussi. Le fast engine recalcule à chaque itération
6. **Enveloppes asymétriques** — `high_envelope = round(1/(1-e) - 1, 3)` (la bande haute n'est PAS le miroir de la bande basse)
7. **`iloc[-2]`** — le live utilise l'avant-dernière bougie (la dernière n'est pas clôturée). Le backtester utilise chaque bougie clôturée → pas de décalage nécessaire
8. **Pas de shorts initialement** — `sides: ["long"]` par défaut
9. **Un seul côté actif** — si positions LONG ouvertes, compute_grid ne retourne que des LONG. Le moteur filtre en double sécurité
10. **BaseGridStrategy hérite de BaseStrategy** — evaluate()/check_exit() retournent None, get_current_conditions() retourne les niveaux
11. **Cache minimal** — seulement sma[period], enveloppes à la volée (multiplication triviale)
12. **Leverage 6** — EnvelopeDCAConfig.leverage override le BacktestConfig.leverage (défaut 15)

---

## Après implémentation

```bash
# 1. Tests
uv run python -m pytest tests/test_multi_engine.py -x -q

# 2. Sanity check (pas WFO)
uv run python -m scripts.run_backtest --strategy envelope_dca --symbol BTC/USDT --days 1800

# 3. WFO
uv run python -m scripts.optimize --strategy envelope_dca --symbol BTC/USDT -v
uv run python -m scripts.optimize --strategy envelope_dca --symbol ETH/USDT -v
uv run python -m scripts.optimize --strategy envelope_dca --symbol SOL/USDT -v
uv run python -m scripts.optimize --strategy envelope_dca --symbol DOGE/USDT -v
uv run python -m scripts.optimize --strategy envelope_dca --symbol LINK/USDT -v
```
