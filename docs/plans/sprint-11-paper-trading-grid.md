# Sprint 11 — Paper Trading Envelope DCA

## Objectif

Faire tourner envelope_dca en paper trading live sur BTC/USDT avec les params
optimisés WFO. Pas d'ordres réels (Executor) pour l'instant — juste le
Simulator en capital virtuel.

---

## Architecture

```
Simulator
  ├─ LiveStrategyRunner     (existant, pour vwap_rsi, momentum, etc.)
  └─ GridStrategyRunner     (NOUVEAU, pour envelope_dca)
       └─ GridPositionManager  (existant, Sprint 10)
       └─ BaseGridStrategy     (existant, Sprint 10)
```

Le Simulator crée le bon type de runner selon la stratégie.
Les deux types de runners partagent la même interface duck-typed :
`on_candle()`, `get_status()`, `get_trades()`, `get_stats()`, `name`,
`is_kill_switch_triggered`, `build_context()`, `restore_state()`.

**Arena** : avec un seul runner actif (envelope_dca), l'Arena n'a pas de
classement utile. C'est voulu — on observe le DCA seul en paper trading.

---

## Étape 1 — GridStrategyRunner (~250 lignes, dans `backend/backtesting/simulator.py`)

Nouveau runner parallèle à `LiveStrategyRunner`, même interface pour le Simulator.

### Compatibilité duck-typing avec le Simulator

Le `Simulator` et le `StateManager` accèdent directement à des attributs internes
de `LiveStrategyRunner` :
- `runner._position` / `runner._position_symbol` (get_open_positions, get_conditions, save_runner_state)
- `runner._capital` / `runner._stats` / `runner._kill_switch_triggered` (save_runner_state)
- `runner._pending_events` (_dispatch_candle)

Pour éviter de refactorer ces accès dans ce sprint, `GridStrategyRunner` expose
les mêmes attributs :

```python
# COMPAT: duck typing pour Simulator/StateManager qui accèdent à ces attributs.
# Voir GridStrategyRunner._positions pour le vrai état grid.
self._position = None           # Toujours None — mono-position non utilisée
self._position_symbol = None    # Toujours None
```

`_capital`, `_stats`, `_kill_switch_triggered`, `_pending_events` existent
naturellement (mêmes noms, même sémantique).

### Buffer de closes interne (warm-up SMA)

L'`IncrementalIndicatorEngine` ne calcule pas la SMA. Plutôt que de modifier
l'engine (impact sur les autres stratégies), le runner maintient son propre
buffer de closes et calcule la SMA lui-même.

Les indicateurs SMA sont **mergés** dans le dict existant (pas écrasés) :
```python
indicators.setdefault(strategy_tf, {}).update({"sma": sma_val, "close": close})
```

### Warm-up depuis la DB

Au démarrage, le runner charge les N dernières bougies 1h depuis la DB locale
(données de `fetch_history`). Il les injecte aussi dans l'`IncrementalIndicatorEngine`
pour que `get_indicators()` et `build_context()` fonctionnent immédiatement :

```python
async def _warmup_from_db(self, db: Database, symbol: str) -> None:
    """Pré-charge les bougies historiques 1h depuis la DB."""
    tf = self._strategy_tf
    needed = max(self._ma_period + 20, 50)
    candles = await db.get_recent_candles(symbol, tf, needed)
    for candle in candles:
        self._close_buffer[symbol].append(candle.close)
        self._indicator_engine.update(symbol, tf, candle)
```

Le Simulator appelle `_warmup_from_db()` pour chaque symbole après la création
du runner, AVANT de câbler le callback `on_candle`.

**Note** : `db.get_recent_candles()` n'existe pas encore — à ajouter dans
`database.py` (simple `SELECT ... ORDER BY timestamp DESC LIMIT N`).

### Détection du régime marché

Le runner détecte le régime via les indicateurs ADX/DI/ATR de l'engine
(s'ils sont disponibles dans le dict 1h). Sinon fallback `RANGING`.
Le DCA ne filtre pas par régime, mais le `TradeResult` aura le bon
`market_regime` au lieu de toujours `RANGING`.

### Events format

Les events Executor utilisent `TradeEvent` / `TradeEventType` (pas des dicts),
cohérents avec le `LiveStrategyRunner`. Même si l'Executor n'écoute pas encore
les grid events, le format est prêt.

Pour un grid close, `TradeEvent` utilise :
- `entry_price` = prix moyen pondéré
- `quantity` = quantité totale
- `tp_price` / `sl_price` = valeurs globales
- `exit_reason` / `exit_price` = raison et prix de sortie

### Positions grid pour le dashboard

Méthode `get_grid_positions()` retourne la liste des positions grid ouvertes
dans le format attendu par `Simulator.get_open_positions()`.

### Code complet

```python
class GridStrategyRunner:
    """Exécute une stratégie grid/DCA sur données live (paper trading).

    Différences avec LiveStrategyRunner :
    - Gère N positions simultanées via GridPositionManager
    - Utilise compute_grid() au lieu de evaluate()
    - TP/SL global (pas par position)
    - Un "trade" = cycle complet (toutes positions ouvertes → toutes fermées)
    """

    def __init__(
        self,
        strategy: BaseGridStrategy,
        config: AppConfig,
        indicator_engine: IncrementalIndicatorEngine,
        grid_position_manager: GridPositionManager,
        data_engine: DataEngine,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._indicator_engine = indicator_engine
        self._gpm = grid_position_manager
        self._data_engine = data_engine

        self._initial_capital = 10_000.0
        self._capital = self._initial_capital
        self._positions: list[GridPosition] = []  # Positions grid ouvertes
        self._trades: list[tuple[str, TradeResult]] = []
        self._current_regime = MarketRegime.RANGING
        self._kill_switch_triggered = False
        self._stats = RunnerStats(
            capital=self._capital,
            initial_capital=self._initial_capital,
        )

        # Sprint 5a : queue d'événements pour l'Executor (drainée par le Simulator)
        self._pending_events: list[Any] = []

        # COMPAT: duck typing pour Simulator/StateManager qui accèdent à ces attributs.
        # Voir self._positions pour le vrai état grid.
        self._position = None
        self._position_symbol = None

        # Buffer de closes pour calcul SMA interne
        self._strategy_tf = getattr(strategy._config, "timeframe", "1h")
        self._ma_period = getattr(strategy._config, "ma_period", 7)
        self._close_buffer: dict[str, deque] = {}  # {symbol: deque of closes}

    @property
    def name(self) -> str:
        return self._strategy.name

    @property
    def is_kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered

    @property
    def strategy(self) -> BaseStrategy:
        return self._strategy

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    async def _warmup_from_db(
        self, db: Database, symbol: str
    ) -> None:
        """Pré-charge les bougies historiques depuis la DB pour le warm-up SMA."""
        needed = max(self._ma_period + 20, 50)
        candles = await db.get_recent_candles(
            symbol, self._strategy_tf, needed
        )
        if not candles:
            logger.info(
                "[{}] Warm-up: 0 bougies {} en DB pour {}",
                self.name, self._strategy_tf, symbol,
            )
            return

        if symbol not in self._close_buffer:
            self._close_buffer[symbol] = deque(
                maxlen=max(self._ma_period + 20, 50)
            )

        for candle in candles:
            self._close_buffer[symbol].append(candle.close)
            # Alimenter aussi l'IncrementalIndicatorEngine
            self._indicator_engine.update(symbol, self._strategy_tf, candle)

        logger.info(
            "[{}] Warm-up: {} bougies {} chargées pour {}",
            self.name, len(candles), self._strategy_tf, symbol,
        )

    async def on_candle(
        self, symbol: str, timeframe: str, candle: Candle
    ) -> None:
        """Traitement d'une nouvelle candle — logique grid."""
        if self._kill_switch_triggered:
            return

        # Filtre strict : seul le timeframe de la stratégie est traité
        if timeframe != self._strategy_tf:
            return

        # Maintenir le buffer de closes
        if symbol not in self._close_buffer:
            self._close_buffer[symbol] = deque(
                maxlen=max(self._ma_period + 20, 50)
            )
        self._close_buffer[symbol].append(candle.close)

        # Calculer SMA
        closes = list(self._close_buffer[symbol])
        if len(closes) < self._ma_period:
            return  # Pas assez de données

        sma_val = float(np.mean(closes[-self._ma_period:]))

        # Récupérer les indicateurs de l'engine et merger SMA
        indicators = self._indicator_engine.get_indicators(symbol)
        if not indicators:
            indicators = {}
        indicators.setdefault(self._strategy_tf, {}).update({
            "sma": sma_val,
            "close": candle.close,
        })

        # Détecter le régime (si ADX/ATR disponibles)
        main_ind = indicators.get(self._strategy_tf, {})
        self._current_regime = detect_market_regime(
            main_ind.get("adx", float("nan")),
            main_ind.get("di_plus", float("nan")),
            main_ind.get("di_minus", float("nan")),
            main_ind.get("atr", float("nan")),
            main_ind.get("atr_sma", float("nan")),
        )

        # Construire le contexte
        ctx = StrategyContext(
            symbol=symbol,
            timestamp=candle.timestamp,
            candles={},
            indicators=indicators,
            current_position=None,
            capital=self._capital,
            config=self._config,
        )

        # Construire le GridState
        grid_state = self._gpm.compute_grid_state(
            self._positions, candle.close
        )

        # 1. Si positions ouvertes → check TP/SL global
        if self._positions:
            tp_price = self._strategy.get_tp_price(grid_state, main_ind)
            sl_price = self._strategy.get_sl_price(grid_state, main_ind)

            # Check via OHLC heuristic
            exit_reason, exit_price = self._gpm.check_global_tp_sl(
                self._positions, candle, tp_price, sl_price
            )

            # Si pas de TP/SL OHLC, check should_close_all (signal)
            if exit_reason is None:
                close_reason = self._strategy.should_close_all(ctx, grid_state)
                if close_reason:
                    exit_reason = close_reason
                    exit_price = candle.close

            if exit_reason:
                trade = self._gpm.close_all_positions(
                    self._positions, exit_price,
                    candle.timestamp, exit_reason,
                    self._current_regime,
                )
                self._record_trade(trade, symbol)
                self._positions = []
                self._emit_close_event(symbol, trade)
                return

        # 2. Ouvrir de nouveaux niveaux si grille pas pleine
        if len(self._positions) < self._strategy.max_positions:
            levels = self._strategy.compute_grid(ctx, grid_state)

            for level in levels:
                if level.index in {p.level for p in self._positions}:
                    continue

                # Vérifier si le prix a touché ce niveau
                touched = False
                if level.direction == Direction.LONG:
                    touched = candle.low <= level.entry_price
                else:
                    touched = candle.high >= level.entry_price

                if touched:
                    position = self._gpm.open_grid_position(
                        level, candle.timestamp,
                        self._capital,
                        self._strategy.max_positions,
                    )
                    if position:
                        self._positions.append(position)
                        self._capital -= position.entry_fee
                        logger.info(
                            "[{}] GRID {} level {} @ {:.2f} ({}) — {}/{} positions",
                            self.name,
                            level.direction.value,
                            level.index,
                            level.entry_price,
                            symbol,
                            len(self._positions),
                            self._strategy.max_positions,
                        )
                        self._emit_open_event(symbol, level, position)

    def _record_trade(self, trade: TradeResult, symbol: str = "") -> None:
        """Enregistre un trade grid (fermeture de toutes les positions)."""
        self._capital += trade.net_pnl
        self._trades.append((symbol, trade))
        self._stats.total_trades += 1
        self._stats.net_pnl = self._capital - self._initial_capital
        self._stats.capital = self._capital

        if trade.net_pnl > 0:
            self._stats.wins += 1
        else:
            self._stats.losses += 1

        logger.info(
            "[{}] Grid trade clos : {} avg={:.2f} → {:.2f}, net={:+.2f} ({})",
            self.name,
            trade.direction.value,
            trade.entry_price,
            trade.exit_price,
            trade.net_pnl,
            trade.exit_reason,
        )

        # Kill switch
        session_loss_pct = (
            abs(min(0, self._stats.net_pnl)) / self._initial_capital * 100
        )
        max_session = self._config.risk.kill_switch.max_session_loss_percent
        if session_loss_pct >= max_session:
            self._kill_switch_triggered = True
            self._stats.is_active = False
            logger.warning(
                "[{}] KILL SWITCH : perte session {:.1f}% >= {:.1f}%",
                self.name, session_loss_pct, max_session,
            )

    def _emit_open_event(
        self, symbol: str, level: GridLevel, position: GridPosition
    ) -> None:
        """Crée un TradeEvent OPEN pour un niveau de grille."""
        from backend.execution.executor import TradeEvent, TradeEventType

        grid_state = self._gpm.compute_grid_state(
            self._positions, position.entry_price
        )
        self._pending_events.append(TradeEvent(
            event_type=TradeEventType.OPEN,
            strategy_name=self.name,
            symbol=symbol,
            direction=position.direction.value,
            entry_price=position.entry_price,
            quantity=position.quantity,
            tp_price=0.0,  # TP dynamique — pas fixe
            sl_price=0.0,  # SL global — pas par position
            score=0.0,
            timestamp=position.entry_time,
            market_regime=self._current_regime.value,
        ))

    def _emit_close_event(self, symbol: str, trade: TradeResult) -> None:
        """Crée un TradeEvent CLOSE pour la fermeture grid globale."""
        from backend.execution.executor import TradeEvent, TradeEventType

        self._pending_events.append(TradeEvent(
            event_type=TradeEventType.CLOSE,
            strategy_name=self.name,
            symbol=symbol,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            quantity=trade.quantity,
            tp_price=0.0,
            sl_price=0.0,
            score=0.0,
            timestamp=trade.exit_time,
            market_regime=self._current_regime.value,
            exit_reason=trade.exit_reason,
            exit_price=trade.exit_price,
        ))

    def restore_state(self, state: dict) -> None:
        """Restaure l'état du runner depuis un snapshot sauvegardé."""
        self._capital = state.get("capital", self._initial_capital)
        self._kill_switch_triggered = state.get("kill_switch", False)

        self._stats.capital = self._capital
        self._stats.net_pnl = state.get("net_pnl", 0.0)
        self._stats.total_trades = state.get("total_trades", 0)
        self._stats.wins = state.get("wins", 0)
        self._stats.losses = state.get("losses", 0)
        self._stats.is_active = state.get("is_active", True)

        # Restaurer les positions grid ouvertes
        grid_positions = state.get("grid_positions", [])
        self._positions = []
        for gp in grid_positions:
            self._positions.append(GridPosition(
                level=gp["level"],
                direction=Direction(gp["direction"]),
                entry_price=gp["entry_price"],
                quantity=gp["quantity"],
                entry_time=datetime.fromisoformat(gp["entry_time"]),
                entry_fee=gp["entry_fee"],
            ))

        logger.info(
            "[{}] État restauré : capital={:.2f}, trades={}, positions_grid={}, kill_switch={}",
            self.name,
            self._capital,
            self._stats.total_trades,
            len(self._positions),
            self._kill_switch_triggered,
        )

    def build_context(self, symbol: str) -> StrategyContext | None:
        """Construit un StrategyContext pour le dashboard (get_conditions)."""
        if not self._indicator_engine:
            return None

        indicators = self._indicator_engine.get_indicators(symbol)
        if not indicators:
            indicators = {}

        # Merger SMA depuis le buffer interne
        closes = list(self._close_buffer.get(symbol, []))
        if len(closes) >= self._ma_period:
            sma_val = float(np.mean(closes[-self._ma_period:]))
            indicators.setdefault(self._strategy_tf, {}).update({
                "sma": sma_val,
                "close": closes[-1] if closes else 0.0,
            })

        return StrategyContext(
            symbol=symbol,
            timestamp=datetime.now(tz=timezone.utc),
            candles={},
            indicators=indicators,
            current_position=None,
            capital=self._capital,
            config=self._config,
        )

    def get_grid_positions(self) -> list[dict]:
        """Retourne les positions grid ouvertes pour le dashboard."""
        result = []
        for p in self._positions:
            result.append({
                "symbol": "UNKNOWN",  # Sera set par le Simulator
                "strategy": self.name,
                "direction": p.direction.value,
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "entry_time": p.entry_time.isoformat(),
                "level": p.level,
                "type": "grid",
            })
        return result

    def get_status(self) -> dict:
        """Même interface que LiveStrategyRunner.get_status() + champs grid."""
        return {
            "name": self.name,
            "capital": self._capital,
            "net_pnl": self._stats.net_pnl,
            "total_trades": self._stats.total_trades,
            "wins": self._stats.wins,
            "losses": self._stats.losses,
            "win_rate": (
                self._stats.wins / self._stats.total_trades * 100
                if self._stats.total_trades > 0 else 0.0
            ),
            "is_active": self._stats.is_active,
            "kill_switch": self._kill_switch_triggered,
            "has_position": len(self._positions) > 0,
            # Champs grid spécifiques
            "open_positions": len(self._positions),
            "max_positions": self._strategy.max_positions,
            "avg_entry_price": (
                sum(p.entry_price * p.quantity for p in self._positions)
                / sum(p.quantity for p in self._positions)
                if self._positions else 0.0
            ),
        }

    def get_trades(self) -> list[tuple[str, TradeResult]]:
        return list(self._trades)

    def get_stats(self) -> RunnerStats:
        return self._stats
```

---

## Étape 2 — Simulator : créer le bon runner + adaptations

### 2a. `Simulator.start()` — détection grid strategy

```python
from backend.optimization import is_grid_strategy
from backend.core.grid_position_manager import GridPositionManager
from backend.strategies.base_grid import BaseGridStrategy

async def start(self, saved_state: dict | None = None) -> None:
    strategies = get_enabled_strategies(self._config)
    if not strategies:
        logger.warning("Simulator: aucune stratégie activée")
        return

    self._indicator_engine = IncrementalIndicatorEngine(strategies)

    # PositionManager pour les stratégies mono-position
    pm_config = PositionManagerConfig(
        leverage=self._config.risk.position.default_leverage,
        maker_fee=self._config.risk.fees.maker_percent / 100,
        taker_fee=self._config.risk.fees.taker_percent / 100,
        slippage_pct=self._config.risk.slippage.default_estimate_percent / 100,
        high_vol_slippage_mult=self._config.risk.slippage.high_volatility_multiplier,
        max_risk_per_trade=self._config.risk.position.max_risk_per_trade_percent / 100,
    )
    pm = PositionManager(pm_config)

    for strategy in strategies:
        if is_grid_strategy(strategy.name):
            # Grid strategy → GridStrategyRunner
            gpm_config = PositionManagerConfig(
                leverage=getattr(strategy._config, "leverage", 15),
                maker_fee=self._config.risk.fees.maker_percent / 100,
                taker_fee=self._config.risk.fees.taker_percent / 100,
                slippage_pct=self._config.risk.slippage.default_estimate_percent / 100,
                high_vol_slippage_mult=self._config.risk.slippage.high_volatility_multiplier,
                max_risk_per_trade=self._config.risk.position.max_risk_per_trade_percent / 100,
            )
            gpm = GridPositionManager(gpm_config)
            runner = GridStrategyRunner(
                strategy=strategy,
                config=self._config,
                indicator_engine=self._indicator_engine,
                grid_position_manager=gpm,
                data_engine=self._data_engine,
            )
        else:
            # Normal strategy → LiveStrategyRunner
            runner = LiveStrategyRunner(
                strategy=strategy,
                config=self._config,
                indicator_engine=self._indicator_engine,
                position_manager=pm,
                data_engine=self._data_engine,
            )
        self._runners.append(runner)
        logger.info("Simulator: stratégie '{}' ajoutée ({})",
                     strategy.name,
                     "grid" if is_grid_strategy(strategy.name) else "mono")

    # Restaurer l'état AVANT d'enregistrer le callback on_candle
    if saved_state is not None:
        runners_state = saved_state.get("runners", {})
        for runner in self._runners:
            if runner.name in runners_state:
                runner.restore_state(runners_state[runner.name])

    # Warm-up grid runners depuis la DB
    for runner in self._runners:
        if isinstance(runner, GridStrategyRunner):
            symbols = self._data_engine.get_all_symbols()
            for symbol in symbols:
                await runner._warmup_from_db(self._db, symbol)

    # Câblage DataEngine → Simulator (APRÈS restauration + warm-up)
    self._data_engine.on_candle(self._dispatch_candle)
    self._running = True
```

**Note** : le Simulator a besoin d'une référence `self._db` pour le warm-up.
Ajouter le paramètre `db: Database` dans `Simulator.__init__()` et le
passer depuis le lifespan de `server.py`.

### 2b. Type hint `_runners`

Changer le type hint :
```python
# Avant
self._runners: list[LiveStrategyRunner] = []

# Après
self._runners: list[LiveStrategyRunner | GridStrategyRunner] = []
```

Et la property :
```python
@property
def runners(self) -> list[LiveStrategyRunner | GridStrategyRunner]:
    return self._runners
```

### 2c. `get_open_positions()` — support grid

```python
def get_open_positions(self) -> list[dict]:
    """Retourne les positions ouvertes de tous les runners."""
    positions = []
    for runner in self._runners:
        # Positions grid
        if hasattr(runner, "get_grid_positions"):
            for gp in runner.get_grid_positions():
                # Injecter le symbole (le runner ne le connaît pas toujours)
                # Les grid positions ont déjà symbol="UNKNOWN" → on cherche
                # le symbole actif depuis le runner
                positions.append(gp)
        # Position mono-position classique
        elif runner._position is not None and runner._position_symbol:
            pos = runner._position
            positions.append({
                "symbol": runner._position_symbol,
                "strategy": runner.name,
                "direction": pos.direction.value,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "tp_price": pos.tp_price,
                "sl_price": pos.sl_price,
                "entry_time": pos.entry_time.isoformat(),
            })
    return positions
```

---

## Étape 3 — StateManager : sauvegarder les positions grid

Le `StateManager.save_runner_state()` accède à `runner._position` qui est
toujours `None` pour un grid runner (compat duck typing). Il sauvegarde
donc `"position": null`.

On ajoute la sérialisation des positions grid :

```python
async def save_runner_state(self, runners) -> None:
    state = {
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        "runners": {},
    }

    for runner in runners:
        # Position mono (LiveStrategyRunner)
        position_data = None
        if runner._position is not None:
            pos = runner._position
            position_data = {
                "direction": pos.direction.value,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "entry_time": pos.entry_time.isoformat(),
                "tp_price": pos.tp_price,
                "sl_price": pos.sl_price,
                "entry_fee": pos.entry_fee,
            }

        runner_state = {
            "capital": runner._capital,
            "net_pnl": runner._stats.net_pnl,
            "total_trades": runner._stats.total_trades,
            "wins": runner._stats.wins,
            "losses": runner._stats.losses,
            "kill_switch": runner._kill_switch_triggered,
            "is_active": runner._stats.is_active,
            "position": position_data,
            "position_symbol": runner._position_symbol,
        }

        # Positions grid (GridStrategyRunner)
        if hasattr(runner, "_positions") and runner._positions:
            runner_state["grid_positions"] = [
                {
                    "level": gp.level,
                    "direction": gp.direction.value,
                    "entry_price": gp.entry_price,
                    "quantity": gp.quantity,
                    "entry_time": gp.entry_time.isoformat(),
                    "entry_fee": gp.entry_fee,
                }
                for gp in runner._positions
            ]

        state["runners"][runner.name] = runner_state

    # ... écriture atomique (inchangée) ...
```

---

## Étape 4 — Database : `get_recent_candles()`

Ajouter une méthode simple dans `backend/core/database.py` :

```python
async def get_recent_candles(
    self,
    symbol: str,
    timeframe: str,
    limit: int = 50,
    exchange: str = "bitget",
) -> list[Candle]:
    """Retourne les N dernières bougies pour un (symbol, timeframe), triées par timestamp ASC."""
    async with aiosqlite.connect(self._db_path) as db:
        cursor = await db.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND exchange = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, timeframe, exchange, limit),
        )
        rows = await cursor.fetchall()

    # Inverser pour avoir ASC (les plus anciennes en premier)
    rows.reverse()

    return [
        Candle(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.fromisoformat(row[0]),
            open=row[1],
            high=row[2],
            low=row[3],
            close=row[4],
            volume=row[5],
            exchange=exchange,
        )
        for row in rows
    ]
```

---

## Étape 5 — Config YAML

### `config/strategies.yaml` — Activer uniquement envelope_dca

```yaml
vwap_rsi:
  enabled: false      # Grade F — désactivé
  # ... params inchangés ...

momentum:
  enabled: false      # Grade F — désactivé

funding:
  enabled: false      # Grade F — désactivé

liquidation:
  enabled: false      # Grade F — désactivé

bollinger_mr:
  enabled: false      # Grade F — désactivé

donchian_breakout:
  enabled: false      # Grade F — désactivé

supertrend:
  enabled: false      # Grade F — désactivé

envelope_dca:
  enabled: true
  live_eligible: false    # paper trading only
  timeframe: "1h"
  ma_period: 7            # Optimisé WFO
  num_levels: 2           # Optimisé WFO (à ajuster selon résultats)
  envelope_start: 0.05    # Optimisé WFO
  envelope_step: 0.02     # Optimisé WFO
  sl_percent: 20.0        # Optimisé WFO
  sides: ["long"]
  leverage: 6
  weight: 0.20
  per_asset: {}
```

---

## Étape 6 — Tests (~25 tests)

### `tests/test_grid_runner.py`

**GridStrategyRunner core (12 tests)** :
- `__init__` : capital=10k, positions=[], trades=[], compat attrs (position=None)
- `on_candle` wrong timeframe → ignored
- `on_candle` pas assez de données SMA → pas de signal
- 1 niveau touché → 1 position ouverte, capital diminué de entry_fee
- 2 niveaux touchés progressivement → 2 positions
- TP global (prix revient à SMA) → close all, trade enregistré, capital mis à jour
- SL global → close all, trade enregistré
- Kill switch déclenché après grosse perte
- `get_status()` format correct (open_positions, avg_entry_price, has_position)
- `get_trades()` retourne les trades clôturés
- Pending events émis (TradeEvent OPEN/CLOSE, pas des dicts)
- Pas de nouvelles positions si kill switch actif

**State persistence (4 tests)** :
- `restore_state()` : capital, stats, kill_switch restaurés
- `restore_state()` avec grid_positions : positions restaurées correctement
- Round-trip : save → load → état identique
- Pas de grid_positions dans state → positions vide (backward compat)

**Simulator integration (5 tests)** :
- Simulator crée `GridStrategyRunner` pour envelope_dca
- Simulator crée `LiveStrategyRunner` pour vwap_rsi
- `_dispatch_candle` atteint le GridStrategyRunner
- `get_all_status()` inclut le runner grid
- `get_open_positions()` retourne les positions grid via `get_grid_positions()`

**Warm-up (2 tests)** :
- `_warmup_from_db` charge les bougies et remplit le buffer
- `_warmup_from_db` alimente l'IncrementalIndicatorEngine

**Dashboard (2 tests)** :
- `build_context()` retourne un StrategyContext avec SMA
- `get_grid_positions()` retourne le bon format

---

## Résumé des fichiers

| Fichier | Type | Lignes | Notes |
|---------|------|--------|-------|
| `backend/backtesting/simulator.py` | MODIFIÉ | +250 | GridStrategyRunner + Simulator.start() + get_open_positions() |
| `backend/core/state_manager.py` | MODIFIÉ | +15 | Sérialisation grid_positions |
| `backend/core/database.py` | MODIFIÉ | +25 | get_recent_candles() |
| `backend/api/server.py` | MODIFIÉ | +2 | Passer db au Simulator |
| `config/strategies.yaml` | MODIFIÉ | ~20 | Enable envelope_dca, disable les F |
| `tests/test_grid_runner.py` | NOUVEAU | ~400 | 25 tests |
| **Total** | | **~710** | |

---

## Points d'attention pour Claude Code

1. **Ne PAS modifier LiveStrategyRunner** — le GridStrategyRunner est parallèle
2. **Duck typing** : `_position = None`, `_position_symbol = None` sur GridStrategyRunner
   avec commentaire `# COMPAT` pour la dette technique
3. **Merge indicators** : `indicators.setdefault(tf, {}).update(...)`, pas d'écrasement
4. **Warm-up DB** : charger les bougies 1h AVANT le câblage on_candle, injecter
   aussi dans l'IncrementalIndicatorEngine
5. **TradeEvent** format (pas dicts) pour les events grid → prêt pour l'Executor
6. **Leverage 6** : pris depuis `strategy._config.leverage`, pas le défaut risk.yaml (15)
7. **`Simulator.__init__`** a besoin de `db: Database` — impact sur `server.py`
8. **`get_recent_candles()`** : ORDER BY timestamp DESC LIMIT N puis reverse → ASC
9. **Régime marché** : calculé si ADX/ATR dispo, sinon RANGING (pas de filtre DCA)
10. **`StateManager`** type hint import : ajouter `GridStrategyRunner` dans `TYPE_CHECKING`

---

## Après implémentation

```bash
# 1. Tests
uv run python -m pytest tests/test_grid_runner.py -x -q

# 2. Régression complète
uv run python -m pytest -x -q

# 3. Vérifier que le DataEngine émet des candles 1h
uv run python -c "
from backend.core.config import get_config
config = get_config()
print('Timeframes:', config.assets.timeframes)
# Doit contenir '1h'
"

# 4. Démarrer le paper trading (dev local)
set ENABLE_WEBSOCKET=true
uv run python -m backend.api.server
# Logs attendus :
# "Simulator: stratégie 'envelope_dca' ajoutée (grid)"
# "[envelope_dca] Warm-up: N bougies 1h chargées pour BTC/USDT"
```
