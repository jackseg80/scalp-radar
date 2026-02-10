# Plan Sprint 2 — Backtesting & Stratégie VWAP+RSI

## Contexte

Sprint 1 terminé : infrastructure complète (models, database, config, data_engine, API, tests).
Sprint 2 = construire le moteur de backtesting event-driven + la première stratégie (VWAP+RSI mean reversion).

## Décisions architecturales clés

### 1. Backtester event-driven (pas vectorisé)

- Chaque bougie est traitée séquentiellement dans l'ordre chronologique
- La stratégie ne voit **que** les données disponibles au temps T → anti-look-ahead structurel
- Hybride : **numpy** pour les calculs d'indicateurs (RSI, VWAP, ADX, ATR), **boucle Python** pour l'évaluation stratégie

### 2. Synchronisation multi-timeframe

- Les bougies 15m ne sont disponibles qu'après leur clôture (la bougie 12:00 n'est accessible qu'à 12:15)
- Le backtester maintient des buffers par timeframe et aligne les timestamps
- La stratégie reçoit un `StrategyContext` avec les candles de chaque timeframe

### 3. 100% synchrone pour le backtesting

- Le backtesting est CPU-bound, pas I/O-bound
- `asyncio.run()` uniquement pour charger les données depuis la DB au démarrage
- Le moteur de backtest lui-même est une boucle `for` synchrone

### 4. Modèle de frais réaliste

- **Entry** : taker fee (0.06%) — on entre au marché
- **TP exit** : maker fee (0.02%) — le TP est un limit order
- **SL exit** : taker fee (0.06%) + slippage — le SL est un market order
- **Signal exit** : taker fee (0.06%) — market order déclenché par le code (check_exit)
- Fee cost = `quantity × price × fee_rate` (appliqué sur la taille notionnelle = quantity × price)
- Slippage : flat % (0.05%) avec multiplicateur ×2 en haute volatilité (ATR > 2× SMA(ATR))

### 5. Une position à la fois par symbole (Sprint 2)

- Simplifie le moteur et la validation
- Multi-positions = Sprint 3 (Arena)

### 6. VWAP rolling 24h pour crypto

- Crypto = marché 24/7, pas de "market open"
- VWAP calculé sur une fenêtre rolling de 288 bougies 5min (= 24h)
- `VWAP = Σ(typical_price × volume) / Σ(volume)` où `typical_price = (H+L+C)/3`

### 7. Pré-calcul des indicateurs (performance)

- Les indicateurs (RSI, VWAP, ADX, ATR, volume SMA) sont calculés **une seule fois** sur tout le dataset au début du backtest
- Stockés dans un dict indexé par timestamp : `indicators[timestamp] = {rsi: ..., vwap: ..., ...}`
- La stratégie accède aux indicateurs pré-calculés via le `StrategyContext`
- Pas de look-ahead car chaque indicateur au temps T n'utilise que des données ≤ T par construction (RSI, VWAP rolling, etc.)
- Évite de recalculer ~52k fois les indicateurs pour un backtest de 6 mois en 5min

### 8. Alignement indicateurs multi-TF géré par le moteur

- Le **moteur** (pas la stratégie) gère le mapping des timestamps entre timeframes
- Pour chaque bougie 5m, le moteur injecte dans `ctx.indicators["15m"]` les valeurs de la **dernière bougie 15m clôturée**
- Exemple : bougies 5m à 12:05 et 12:10 reçoivent les indicateurs 15m de 11:45 (la 15m de 12:00 n'est pas encore clôturée)
- La bougie 5m à 12:15 reçoit enfin les indicateurs 15m de 12:00
- `compute_indicators()` retourne un dict par TF, indexé par timestamp de chaque TF. Le moteur fait le `last_available_before(timestamp_5m)` pour le 15m
- Cela évite un lookup naïf par timestamp exact (qui retournerait None 2 fois sur 3 pour le 15m)

---

## Fichiers à créer (9 fichiers, ~1900 lignes estimées)

### Étape 1 — `backend/core/indicators.py` (~280 lignes)

Bibliothèque d'indicateurs techniques en **pur numpy**. Pas de dépendance externe (pas de ta-lib).

**Fonctions :**

```python
def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI avec lissage de Wilder (exponentiel). Retourne array de même taille, NaN au début."""

def vwap_rolling(highs, lows, closes, volumes, window: int = 288) -> np.ndarray:
    """VWAP rolling sur fenêtre glissante. window=288 pour 24h en 5min."""

def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""

def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""

def atr(highs, lows, closes, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder smoothing)."""

def adx(highs, lows, closes, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index.
    Retourne (adx, di_plus, di_minus) — les 3 sont nécessaires pour
    distinguer TRENDING_UP de TRENDING_DOWN dans detect_market_regime."""

def volume_sma(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """SMA du volume pour détecter les spikes."""

def detect_market_regime(
    adx_values: np.ndarray,
    di_plus: np.ndarray,
    di_minus: np.ndarray,
    atr_values: np.ndarray,
    atr_sma: np.ndarray,
) -> MarketRegime:
    """Détecte le régime de marché actuel :
    - ADX > 25 et DI+ > DI- → TRENDING_UP
    - ADX > 25 et DI- > DI+ → TRENDING_DOWN
    - ADX < 20 → RANGING
    - ATR > 2× SMA(ATR) → HIGH_VOLATILITY (prioritaire sur trending)
    - ATR < 0.5× SMA(ATR) → LOW_VOLATILITY
    """
```

**Choix de design :**

- Toutes les fonctions prennent et retournent des `np.ndarray`
- Les premières valeurs sont `NaN` (période d'échauffement)
- Pas d'état interne, pas de classes — fonctions pures
- RSI utilise le **lissage de Wilder** (pas SMA) : `avg_gain = prev_avg × (n-1)/n + current_gain/n`
- `adx()` retourne un **tuple (adx, di_plus, di_minus)** pour permettre la distinction trending up/down

---

### Étape 2 — `backend/strategies/base.py` (~130 lignes)

**`StrategyContext` dataclass :**

```python
@dataclass
class StrategyContext:
    symbol: str
    timestamp: datetime
    candles: dict[str, list[Candle]]  # {"5m": [...], "15m": [...], "1h": [...]}
    indicators: dict[str, Any]  # Indicateurs pré-calculés pour ce timestamp
    current_position: OpenPosition | None  # Position ouverte actuelle
    capital: float  # Capital disponible
    config: AppConfig
```

Le champ `indicators` contient les valeurs pré-calculées pour le timestamp courant :
```python
{
    "5m": {"rsi": 23.5, "vwap": 98500.0, "adx": 28.0, "di_plus": 30.0,
           "di_minus": 15.0, "atr": 120.0, "volume_sma": 500.0},
    "15m": {"rsi": 45.0, "adx": 22.0, "di_plus": 20.0, "di_minus": 18.0}
}
```

**`StrategySignal` dataclass :**

```python
@dataclass
class StrategySignal:
    direction: Direction
    entry_price: float
    tp_price: float
    sl_price: float
    score: float  # 0-1
    strength: SignalStrength
    market_regime: MarketRegime
    signals_detail: dict[str, float]  # Sous-scores détaillés
```

**`BaseStrategy` ABC :**

```python
class BaseStrategy(ABC):
    name: str  # Ex: "vwap_rsi"

    @abstractmethod
    def evaluate(self, ctx: StrategyContext) -> StrategySignal | None:
        """Évalue les conditions d'entrée. Retourne un signal ou None."""

    @abstractmethod
    def check_exit(self, ctx: StrategyContext, position: OpenPosition) -> str | None:
        """Vérifie les conditions de sortie anticipée.
        Retourne "signal_exit" ou None.
        Appelé UNIQUEMENT si ni TP ni SL n'ont été touchés sur cette bougie.
        Le TP/SL par prix est géré par le moteur (priorité supérieure)."""

    @abstractmethod
    def compute_indicators(self, candles_by_tf: dict[str, list[Candle]]) -> dict[str, dict[str, dict]]:
        """Pré-calcule tous les indicateurs sur le dataset complet.
        Appelé une fois au début du backtest.
        Retourne {tf: {timestamp_iso: {indicator: value}}}.
        Le moteur se charge de l'alignement multi-TF : pour chaque bougie 5m,
        il injecte les derniers indicateurs 15m clôturés dans ctx.indicators["15m"]."""

    @property
    @abstractmethod
    def min_candles(self) -> dict[str, int]:
        """Nombre minimum de bougies par timeframe. Ex: {"5m": 300, "15m": 50}"""
```

---

### Étape 3 — `backend/strategies/vwap_rsi.py` (~220 lignes)

Implémentation de la stratégie VWAP + RSI Mean Reversion.

**Logique d'entrée :**

1. Lire les indicateurs pré-calculés depuis `ctx.indicators` :
   - VWAP rolling 24h (288 bougies)
   - RSI(14) avec Wilder smoothing
   - Volume actuel vs volume_sma pour détecter les spikes
   - ADX + DI+/DI- + ATR pour le régime de marché
2. Filtrage multi-timeframe (15m) :
   - Lire RSI et DI+/DI- du 15m depuis `ctx.indicators["15m"]`
   - Si 15m trend baissier (DI- > DI+) → pas de LONG, si haussier → pas de SHORT
3. Conditions d'entrée LONG :
   - Prix < VWAP × (1 - deviation_entry/100) — prix sous le VWAP
   - RSI < rsi_long_threshold (25) — survente
   - Volume > volume_sma × volume_spike_multiplier (3.0) — spike de volume
   - 15m trend pas baissier
4. Conditions d'entrée SHORT : symétriques
5. Score composé (0-1) :
   - `rsi_score` : distance au seuil (RSI=10 score plus haut que RSI=24)
   - `vwap_score` : distance au VWAP
   - `volume_score` : ratio volume/sma
   - `trend_score` : alignement avec le 15m
   - Score final = moyenne pondérée

**`compute_indicators()` :**

- Extrait les arrays numpy depuis les listes de Candle (closes, highs, lows, volumes)
- Calcule RSI, VWAP, ADX+DI, ATR, volume SMA sur le TF principal (5m)
- Calcule RSI, ADX+DI sur le TF filtre (15m)
- Retourne `{"5m": {ts: {...}, ...}, "15m": {ts: {...}, ...}}`
- Chaque TF est indexé par ses propres timestamps (le 15m a 3× moins d'entrées)
- L'alignement cross-TF (bougie 5m → derniers indicateurs 15m clôturés) est géré par le **moteur**, pas par la stratégie

**Calcul TP/SL :**

- TP = entry × (1 + tp_percent/100) pour LONG
- SL = entry × (1 - sl_percent/100) pour LONG
- Inversé pour SHORT
- Le sl_percent est la distance pure. Le coût réel du SL (frais + slippage) est pris en compte dans le **position sizing** par le moteur.

**Sortie anticipée :**

- `check_exit()` : si RSI revient au-dessus de 50 (LONG) ou en-dessous de 50 (SHORT) et que le trade est en profit → retourne "signal_exit" (mean reversion accomplie)
- N'est appelée que si ni TP ni SL n'ont été touchés sur cette bougie

---

### Étape 4 — `backend/backtesting/engine.py` (~450 lignes)

Le moteur de backtesting event-driven. C'est le fichier le plus complexe du Sprint 2.

**`BacktestConfig` dataclass :**

```python
@dataclass
class BacktestConfig:
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10_000.0
    leverage: int = 15  # Depuis risk.yaml
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0006  # 0.06%
    slippage_pct: float = 0.0005  # 0.05%
    high_vol_slippage_mult: float = 2.0
    max_risk_per_trade: float = 0.02  # 2%
```

**`OpenPosition` dataclass :**

```python
@dataclass
class OpenPosition:
    direction: Direction
    entry_price: float
    quantity: float
    entry_time: datetime
    tp_price: float
    sl_price: float
    entry_fee: float  # Frais déjà payés à l'entrée
```

**`TradeResult` dataclass :**

```python
@dataclass
class TradeResult:
    direction: Direction
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    fee_cost: float  # entry_fee + exit_fee
    slippage_cost: float
    net_pnl: float
    exit_reason: str  # "tp", "sl", "signal_exit", "end_of_data"
    market_regime: MarketRegime
```

**`BacktestEngine` classe principale :**

```python
class BacktestEngine:
    def __init__(self, config: BacktestConfig, strategy: BaseStrategy):
        ...

    def run(self, candles_by_tf: dict[str, list[Candle]]) -> BacktestResult:
        """Boucle principale event-driven.

        0. Appeler strategy.compute_indicators(candles_by_tf) — pré-calcul une seule fois
        1. Trier les bougies du TF principal par timestamp
        2. Pour chaque bougie:
           a. Mettre à jour les buffers de chaque timeframe
           b. Vérifier l'alignement multi-TF (15m dispo seulement après clôture)
           c. Construire le StrategyContext (avec indicateurs pré-calculés)
           d. Si position ouverte:
              - Vérifier TP/SL hit (avec heuristique OHLC, voir ci-dessous)
              - Si ni TP ni SL touchés: appeler strategy.check_exit(ctx, position)
              - Si check_exit retourne "signal_exit": clôturer au close (taker fee)
           e. Si pas de position: évaluer strategy.evaluate(ctx)
              - Si signal: ouvrir une position (position sizing avec coût SL réel)
           f. Mettre à jour l'equity curve (point à chaque bougie)
        3. Forcer la clôture des positions ouvertes à la fin (exit_reason="end_of_data")
        4. Retourner BacktestResult
        """
```

**Détails critiques de la boucle :**

- **TP/SL check — heuristique OHLC intra-bougie :**
  Si TP et SL sont tous deux touchés dans la même bougie, on utilise l'heuristique OHLC pour déterminer lequel a été touché en premier :
  - **Bougie verte** (close > open) : le prix est probablement monté d'abord → mouvement inféré : open → high → low → close
  - **Bougie rouge** (close < open) : le prix est probablement descendu d'abord → mouvement inféré : open → low → high → close
  - Pour un LONG : si bougie verte → TP touché d'abord (high avant low) ; si bougie rouge → SL touché d'abord (low avant high)
  - Pour un SHORT : inversé
  - Si close == open : SL priorisé (hypothèse conservatrice)
  - Cette heuristique évite le biais pessimiste systématique (toujours SL) tout en restant réaliste

- **Position sizing — coût SL réel inclus :**
  ```
  sl_distance = |entry - sl_price| / entry  # distance en %
  sl_real_cost = sl_distance + taker_fee + slippage_pct  # coût réel en %
  risk_amount = capital × max_risk_per_trade
  notional = risk_amount / sl_real_cost
  quantity = notional / entry_price
  ```
  Le risk par trade est ainsi précisément respecté : en cas de SL, la perte réelle (distance + fee + slippage) = exactement `max_risk_per_trade` × capital.

- **Fee calculation :**
  - Entry fee = `quantity × entry_price × taker_fee` (market entry)
  - TP exit fee = `quantity × tp_price × maker_fee` (limit TP)
  - SL exit fee = `quantity × sl_price × taker_fee` (market SL)
  - Signal exit fee = `quantity × close_price × taker_fee` (market order, car c'est le code qui déclenche la sortie)

- **Slippage :** appliqué sur SL et signal_exit (market orders), pas sur TP (limit order).
  `actual_exit = sl_price × (1 - slippage)` pour LONG, `× (1 + slippage)` pour SHORT.
  Slippage doublé si haute volatilité : `atr > 2 × sma(atr)` au moment de la sortie.

- **Alignement multi-TF :** buffer séparé pour 5m et 15m. La bougie 15m à 12:00 n'est ajoutée au buffer 15m qu'au timestamp 12:15 de la bougie 5m.

- **Ordre de priorité dans la boucle :** TP/SL par prix → check_exit signal → evaluate entry. Les TP/SL sont des ordres sur l'exchange (priorité absolue), check_exit ne s'évalue que si aucun n'est touché.

**`BacktestResult` dataclass :**

```python
@dataclass
class BacktestResult:
    config: BacktestConfig
    strategy_name: str
    strategy_params: dict  # Paramètres de la stratégie pour l'optimisation future
    trades: list[TradeResult]
    equity_curve: list[float]  # Capital à chaque bougie (pas juste par trade)
    equity_timestamps: list[datetime]  # Un timestamp par bougie → drawdown duration calculable
    final_capital: float
```

L'equity curve a un point **par bougie** (pas par trade). Cela permet de calculer le max drawdown en durée de manière fiable — on sait exactement combien de temps s'écoule entre chaque point. Le capital évolue uniquement quand un trade se clôture, mais les timestamps intermédiaires sont nécessaires pour la durée du drawdown.

---

### Étape 5 — `backend/backtesting/metrics.py` (~220 lignes)

**`BacktestMetrics` dataclass :**

```python
@dataclass
class BacktestMetrics:
    # Performance
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # %

    # P&L
    gross_pnl: float
    total_fees: float
    total_slippage: float
    net_pnl: float
    net_return_pct: float  # % du capital initial

    # Ratios
    profit_factor: float  # net_wins / net_losses (choix scalping : fees incluses)
    gross_profit_factor: float  # gross_wins / gross_losses (pour comparaison benchmarks)
    avg_win: float
    avg_loss: float
    risk_reward_ratio: float  # avg_win / avg_loss
    expectancy: float  # (win_rate × avg_win) - (loss_rate × avg_loss)

    # Risk
    max_drawdown_pct: float
    max_drawdown_duration: timedelta  # Calculé via equity_timestamps (point par bougie)
    sharpe_ratio: float  # Annualisé 365j, rf=0
    sortino_ratio: float  # Annualisé 365j, downside deviation uniquement

    # Fee impact
    fee_drag_pct: float  # total_fees / gross_pnl — montre l'impact des fees

    # Breakdown par régime
    regime_stats: dict[str, dict]  # {regime: {trades, win_rate, net_pnl}}
```

**Fonctions :**

```python
def calculate_metrics(result: BacktestResult) -> BacktestMetrics:
    """Calcule toutes les métriques depuis un BacktestResult."""

def format_metrics_table(metrics: BacktestMetrics) -> str:
    """Formate les métriques en tableau lisible pour la console."""
```

**Points critiques :**

- `fee_drag_pct` : le ratio frais/gross_pnl est LE chiffre clé pour le scalping. À x20 sur des moves de 0.3%, il devrait être ~40%. Si > 60%, la stratégie n'est pas viable.
- `profit_factor` : calculé sur le **net** P&L (somme des wins / somme des losses en valeur absolue). < 1.0 = perte nette. C'est un choix conscient pour le scalping où les fees changent fondamentalement le résultat. `gross_profit_factor` est aussi calculé pour comparaison avec les benchmarks standards (1.5 = bon, 2.0 = excellent sont souvent des chiffres gross).
- `sharpe_ratio` : annualisé à 365 jours (crypto 24/7), risk-free rate = 0. `sharpe = mean(returns) / std(returns) × sqrt(periods_per_year)`. Les returns sont calculés par trade (net_pnl / capital_avant_trade).
- `sortino_ratio` : même formule que Sharpe mais le dénominateur est le **downside deviation** — l'écart-type des rendements négatifs uniquement. `sortino = mean(returns) / downside_std × sqrt(periods_per_year)`. downside_std = `sqrt(mean(min(returns, 0)²))`.
- `max_drawdown` : calculé sur l'equity curve (point par bougie), peak-to-trough. La durée est mesurée entre le peak et la recovery (ou la fin du backtest si pas de recovery).

**Note Sprint 3 :** Sauvegarder le BacktestResult complet en base (table `backtest_results`) pour affichage dans le dashboard. Pas bloquant pour Sprint 2.

---

### Étape 6 — `scripts/run_backtest.py` (~150 lignes)

**CLI :**

```
uv run python -m scripts.run_backtest --symbol BTC/USDT --days 30 --capital 10000 --json
```

**Arguments :**

- `--symbol` : symbole (défaut: BTC/USDT)
- `--strategy` : nom de la stratégie (défaut: vwap_rsi)
- `--days` : nombre de jours de données (défaut: 90)
- `--capital` : capital initial (défaut: 10000)
- `--leverage` : levier (défaut: depuis risk.yaml)
- `--json` : output JSON au lieu du tableau formaté
- `--output` : fichier de sortie (optionnel, sinon stdout)

**Workflow :**

1. Charger la config
2. Charger les candles depuis la DB (`asyncio.run()`)
3. Vérifier qu'il y a assez de données (warning si < 30 jours)
4. Instancier la stratégie + le moteur
5. Lancer `engine.run()` (synchrone)
6. Calculer les métriques
7. Afficher le résultat (table ou JSON)

**Output console exemple :**

```
═══════════════════════════════════════════════
  BACKTEST — VWAP+RSI · BTC/USDT · 90 jours
═══════════════════════════════════════════════

  Performance
  ───────────
  Trades         : 127
  Win rate       : 54.3%
  Net P&L        : +$823.45 (+8.23%)
  Profit factor  : 1.42

  Frais & Slippage
  ────────────────
  Gross P&L      : +$1,847.20
  Fees           : -$782.30 (42.3% du gross)
  Slippage       : -$241.45
  Net P&L        : +$823.45

  Risque
  ──────
  Max drawdown   : -3.2% (durée: 2j 4h)
  Sharpe ratio   : 1.85
  Sortino ratio  : 2.31

  Par régime de marché
  ────────────────────
  RANGING        : 45 trades, 62% win, +$520
  TRENDING_UP    : 38 trades, 47% win, +$180
  TRENDING_DOWN  : 30 trades, 50% win, +$123
  HIGH_VOLATILITY: 14 trades, 36% win, +$0.45
═══════════════════════════════════════════════
```

---

### Étape 7 — Tests (~450 lignes au total)

#### `tests/test_indicators.py` (~160 lignes, ~17 tests)

- RSI sur données connues (vérification manuelle)
- RSI période d'échauffement : les N premières valeurs sont NaN
- VWAP rolling : résultat connu sur 5 bougies
- ATR : valeur connue sur données simples
- ADX retourne tuple (adx, di_plus, di_minus)
- ADX : DI+ > DI- sur données en tendance haussière
- ADX : DI- > DI+ sur données en tendance baissière
- SMA/EMA : valeurs exactes
- `detect_market_regime` : ADX=30 + DI+ > DI- → TRENDING_UP
- `detect_market_regime` : ADX=30 + DI- > DI+ → TRENDING_DOWN
- `detect_market_regime` : ADX=15 → RANGING
- `detect_market_regime` : ATR spike → HIGH_VOLATILITY
- Edge cases : array vide, array trop court, volumes à 0

#### `tests/test_strategy_vwap_rsi.py` (~120 lignes, ~10 tests)

- Signal LONG : prix sous VWAP, RSI < 25, volume spike → Signal émis
- Signal SHORT : symétrique
- Pas de signal : RSI normal (40-60) → None
- Filtre multi-TF : 15m bearish (DI- > DI+) + 5m long conditions → None (filtré)
- Filtre multi-TF : 15m bullish (DI+ > DI-) + 5m long conditions → Signal émis
- Score composé : vérifier que les sous-scores sont calculés
- `check_exit` : RSI revenu > 50 en profit → "signal_exit"
- `check_exit` : RSI revenu > 50 en perte → None (pas de sortie)
- `compute_indicators` : retourne dict indexé par timestamp avec les bons champs
- `min_candles` : retourne les bonnes valeurs

#### `tests/test_backtesting.py` (~170 lignes, ~16 tests)

- Trade LONG complet : entry → TP hit → vérifie net_pnl (avec fees maker)
- Trade LONG SL : entry → SL hit → vérifie net_pnl négatif (avec fees taker + slippage)
- Trade signal_exit : vérifie taker fee + slippage appliqués (market order)
- Fee calculation : vérifier maker vs taker selon exit reason
- Slippage appliqué sur SL et signal_exit, PAS sur TP
- High volatility : slippage doublé
- **Heuristique OHLC** : bougie verte + LONG → TP/SL les deux touchés → TP d'abord (high avant low)
- **Heuristique OHLC** : bougie rouge + LONG → TP/SL les deux touchés → SL d'abord (low avant high)
- **Position sizing coût SL réel** : vérifie que quantity intègre taker_fee + slippage dans le calcul
- Ordre de priorité : TP/SL avant check_exit (si TP touché + RSI > 50, c'est le TP qui compte)
- Multi-TF alignment : bougie 15m pas disponible trop tôt
- Pas de trade si données insuffisantes
- `BacktestMetrics` : calculs de win_rate, profit_factor, drawdown
- `fee_drag_pct` : vérifie le ratio fees/gross
- Equity curve : un point par bougie, valeur finale = capital + sum(net_pnl)
- `strategy_params` dans BacktestResult : contient les paramètres de la stratégie
- Sortino ratio : vérifie que seuls les rendements négatifs comptent au dénominateur

---

## Dépendances d'implémentation

```
backend/core/indicators.py          (aucune dépendance interne, que numpy)
    ↓
backend/strategies/base.py          (dépend de models.py, config.py)
    ↓
backend/strategies/vwap_rsi.py      (dépend de base.py, indicators.py)
    ↓
backend/backtesting/engine.py       (dépend de base.py, models.py, config.py)
    ↓
backend/backtesting/metrics.py      (dépend de engine.py pour les types)
    ↓
scripts/run_backtest.py             (dépend de tout le stack)
    ↓
tests/                              (valide l'ensemble)
```

**Ordre d'implémentation optimal** : dans l'ordre ci-dessus, chaque fichier est testable indépendamment.

---

## Pré-requis données

Avant de lancer un backtest, il faut des données historiques en DB :

```bash
uv run python -m scripts.fetch_history --symbol BTC/USDT --days 90
uv run python -m scripts.fetch_history --symbol ETH/USDT --days 90
uv run python -m scripts.fetch_history --symbol SOL/USDT --days 90
```

Les tests utilisent des données synthétiques (pas besoin de fetch).

---

## Dépendances Python à ajouter

Aucune nouvelle dépendance nécessaire. Tout est déjà dans `pyproject.toml` :
- `numpy` — calcul indicateurs
- `pydantic` — modèles
- `aiosqlite` — chargement données
- `pyyaml` — config
- `loguru` — logging

---

## Récapitulatif des corrections intégrées (suite à la revue)

| # | Point | Correction |
|---|-------|-----------|
| 1 | TP/SL même bougie | Heuristique OHLC au lieu de toujours prioriser SL |
| 2 | Position sizing | Coût SL réel (distance + taker_fee + slippage) dans le calcul |
| 3 | Priorité TP/SL vs check_exit | TP/SL par prix d'abord, check_exit seulement si aucun touché |
| 4 | Fee type sur signal_exit | Taker fee (0.06%) — c'est un market order |
| 5 | Equity curve | Point par bougie (pas par trade) → drawdown duration fiable |
| 6 | strategy_params | Ajouté dans BacktestResult pour l'optimisation future |
| 7 | ADX sans DI+/DI- | adx() retourne tuple (adx, di_plus, di_minus) |
| 8 | Indicateurs recalculés | Pré-calcul une fois, indexé par timestamp via compute_indicators() |
| 9 | Mapping 15m → 5m | Le moteur injecte les derniers indicateurs 15m clôturés (pas lookup naïf) |
| 10 | Profit factor net vs gross | Les deux sont calculés : net (scalping) + gross (benchmarks) |

**Note Sprint 3 :** Sauvegarder BacktestResult en DB (table `backtest_results`) pour le dashboard.

---

## Résumé des fichiers

| # | Fichier | Lignes est. | Description |
|---|---------|-------------|-------------|
| 1 | `backend/core/indicators.py` | ~280 | RSI, VWAP, ADX+DI, ATR, SMA, EMA, régime marché |
| 2 | `backend/strategies/base.py` | ~130 | StrategyContext (avec indicators), StrategySignal, BaseStrategy ABC |
| 3 | `backend/strategies/vwap_rsi.py` | ~220 | Stratégie VWAP+RSI + compute_indicators |
| 4 | `backend/backtesting/engine.py` | ~450 | Moteur event-driven + OHLC heuristique + sizing SL réel |
| 5 | `backend/backtesting/metrics.py` | ~220 | Métriques + Sortino + format table console |
| 6 | `scripts/run_backtest.py` | ~150 | CLI runner |
| 7 | `tests/test_indicators.py` | ~160 | 17 tests indicateurs (dont ADX DI+/DI-) |
| 8 | `tests/test_strategy_vwap_rsi.py` | ~120 | 10 tests stratégie |
| 9 | `tests/test_backtesting.py` | ~170 | 16 tests moteur (OHLC, sizing, priorité, Sortino) |
| **Total** | | **~1900** | |
