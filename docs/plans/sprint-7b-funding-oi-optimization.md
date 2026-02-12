# Sprint 7b — Fetch Funding Rates + OI historiques & Optimisation Funding/Liquidation

## Objectif

Récupérer les données historiques de funding rates et d'open interest depuis Binance, les stocker en DB, adapter le moteur de backtest pour injecter ces données dans le `StrategyContext.extra_data`, puis rendre les stratégies funding et liquidation optimisables avec le WFO existant (Sprint 7).

## Contexte & analyse

Les 5 optimisations WFO du Sprint 7 (VWAP+RSI × 3 assets, Momentum × 2 assets) sont toutes Grade F — pas d'edge détectable sur les patterns techniques purs. Les deux seules stratégies avec un rationnel économique (funding arbitrage, liquidation hunting) sont exclues du WFO car il n'y a pas de données historiques OI/funding en DB.

La stratégie funding est à **+167$ en simulation** — c'est la seule stratégie en positif sur le bot live. Ce sprint débloque leur optimisation.

### Analyse du code existant

**Données manquantes** : Les stratégies funding et liquidation lisent leurs données via `ctx.extra_data` :
- `EXTRA_FUNDING_RATE` — float, stocké en % par le DataEngine (ex: `0.01` = 0.01%)
- `EXTRA_OI_CHANGE_PCT` — float, variation % OI vs snapshot précédent
- `EXTRA_OPEN_INTEREST` — liste de snapshots OI

Le `BacktestEngine` (`engine.py:157-165`) construit le `StrategyContext` **sans extra_data** (le champ a un `default_factory=dict`, donc c'est toujours `{}`). Les stratégies ne génèrent donc jamais de signal en backtest.

**Fast engine non nécessaire** : Le guard `if strategy_name in ("vwap_rsi", "momentum")` dans `_parallel_backtest()` (ligne 595) fait tomber directement les autres stratégies sur ProcessPool → séquentiel. Les grids funding/liquidation sont petits (~200-576 combos), le ProcessPool suffit (~8 min/stratégie/asset).

**Configs sans `per_asset`** : `FundingConfig` et `LiquidationConfig` n'ont pas de champ `per_asset` ni de `get_params_for_symbol()`. Il faut les ajouter pour que `apply_to_yaml()` puisse écrire les overrides par asset.

**Funding rate échelle** : Le DataEngine multiplie par 100 (`float(rate) * 100`) avant stockage. Les seuils dans `FundingConfig` sont en % : `extreme_positive_threshold: 0.03` = 0.03%. L'API Binance renvoie le taux brut (ex: `0.0001` = 0.01%), il faut multiplier par 100 pour cohérence.

---

## Phase 1 — Nouvelles tables DB + config

### 1.1 Tables DB

**Fichier modifié** : `backend/core/database.py`

Ajouter `_create_sprint7b_tables()` appelé dans `_create_tables()` :

```sql
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT 'binance',
    timestamp INTEGER NOT NULL,             -- epoch ms
    funding_rate REAL NOT NULL,             -- en % (ex: 0.01 = 0.01%), cohérent avec DataEngine
    PRIMARY KEY (symbol, exchange, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_funding_rates_lookup
    ON funding_rates (symbol, exchange, timestamp);

CREATE TABLE IF NOT EXISTS open_interest (
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT 'binance',
    timeframe TEXT NOT NULL,                -- "5m", "15m", "1h"
    timestamp INTEGER NOT NULL,             -- epoch ms
    oi REAL NOT NULL,                       -- OI en contrats/coins
    oi_value REAL NOT NULL,                 -- OI en USDT
    PRIMARY KEY (symbol, exchange, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_oi_lookup
    ON open_interest (symbol, exchange, timeframe, timestamp);
```

Tables créées avec `IF NOT EXISTS` — migration idempotente, pas de backup nécessaire (tables nouvelles, aucune altération).

### 1.2 Méthodes CRUD (~80 lignes)

Toutes `async` (aiosqlite) :

```python
# ─── FUNDING RATES ────────────────────────────────────────────

async def insert_funding_rates_batch(self, rates: list[dict]) -> int:
    """Insère un batch de funding rates (INSERT OR IGNORE)."""

async def get_funding_rates(
    self, symbol: str, exchange: str = "binance",
    start_ts: int | None = None, end_ts: int | None = None,
) -> list[dict]:
    """Retourne les funding rates triés par timestamp ASC."""

async def get_latest_funding_timestamp(
    self, symbol: str, exchange: str = "binance",
) -> int | None:
    """Retourne le timestamp du dernier funding rate (pour reprise incrémentale)."""

# ─── OPEN INTEREST ────────────────────────────────────────────

async def insert_oi_batch(self, records: list[dict]) -> int:
    """Insère un batch d'OI records (INSERT OR IGNORE)."""

async def get_open_interest(
    self, symbol: str, timeframe: str = "5m",
    exchange: str = "binance",
    start_ts: int | None = None, end_ts: int | None = None,
) -> list[dict]:
    """Retourne les records OI triés par timestamp ASC."""

async def get_latest_oi_timestamp(
    self, symbol: str, timeframe: str = "5m",
    exchange: str = "binance",
) -> int | None:
    """Retourne le timestamp du dernier record OI (pour reprise incrémentale)."""
```

### 1.3 Ajout `per_asset` à FundingConfig et LiquidationConfig

**Fichier modifié** : `backend/core/config.py`

```python
class FundingConfig(BaseModel):
    # ... champs existants ...
    per_asset: dict[str, dict[str, Any]] = {}   # NOUVEAU

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:  # NOUVEAU
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}

class LiquidationConfig(BaseModel):
    # ... champs existants ...
    per_asset: dict[str, dict[str, Any]] = {}   # NOUVEAU

    def get_params_for_symbol(self, symbol: str) -> dict[str, Any]:  # NOUVEAU
        base = self.model_dump(exclude={"per_asset", "enabled", "live_eligible", "weight"})
        overrides = self.per_asset.get(symbol, {})
        return {**base, **overrides}
```

**Fichier modifié** : `config/strategies.yaml`

Ajouter `per_asset: {}` aux sections `funding:` et `liquidation:`.

---

## Phase 2 — Scripts de fetch

### 2.1 `scripts/fetch_funding.py` (nouveau, ~120 lignes)

Fetch les funding rates historiques depuis Binance via **ccxt** (cohérent avec `fetch_history.py`).

**API sous-jacente** : `GET /fapi/v1/fundingRate` — max 1000 résultats.
Funding publié toutes les 8h (3/jour). Sur 720 jours = ~2160 points par asset. Très léger.

```bash
# Fetch 2 ans pour tous les assets
uv run python -m scripts.fetch_funding --days 720

# Un asset spécifique
uv run python -m scripts.fetch_funding --symbol BTC/USDT --days 720

# Force re-fetch (supprime puis refetch)
uv run python -m scripts.fetch_funding --force --days 720
```

**Flux** :

```python
async def fetch_funding_for_symbol(exchange, db, symbol, since_ms, end_ms):
    """Fetch incrémental des funding rates pour un symbol."""
    # Reprise incrémentale
    latest_ts = await db.get_latest_funding_timestamp(symbol, "binance")
    if latest_ts and latest_ts > since_ms:
        since_ms = latest_ts + 1

    batch = []
    current = since_ms
    while current < end_ms:
        # ccxt.fetchFundingRateHistory() retourne des dicts avec:
        #   {"timestamp": epoch_ms, "fundingRate": 0.0001, ...}
        rates = exchange.fetch_funding_rate_history(symbol, since=current, limit=1000)
        if not rates:
            break
        for r in rates:
            batch.append({
                "symbol": symbol,
                "exchange": "binance",
                "timestamp": r["timestamp"],
                "funding_rate": r["fundingRate"] * 100,  # → en %, cohérent DataEngine
            })
        current = rates[-1]["timestamp"] + 1
        time.sleep(0.1)  # rate limiting

    if batch:
        await db.insert_funding_rates_batch(batch)
    return len(batch)
```

**Conversion** : ccxt gère automatiquement `BTC/USDT` → le format Binance. Le taux brut est multiplié par 100 pour être en % (cohérent avec le DataEngine qui fait `float(rate) * 100`).

**Pagination** : 2160 rates / 1000 par requête = 3 requêtes par asset. Rapide.

**Fallback** : Si ccxt `fetchFundingRateHistory()` n'est pas supporté pour Binance, utiliser httpx avec l'API REST directe (`GET /fapi/v1/fundingRate`). Tester en Phase 2 avant d'implémenter.

### 2.2 `scripts/fetch_oi.py` (nouveau, ~140 lignes)

Fetch l'open interest historique depuis Binance via **ccxt**.

**API sous-jacente** : `GET /futures/data/openInterestHist` — max 500 résultats.
Periods disponibles : `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `1d`. On fetch en `5m` (résolution max).

Sur 720 jours en 5m = ~207k points par asset. Comparable aux candles.

```bash
# Fetch 2 ans pour tous les assets
uv run python -m scripts.fetch_oi --days 720

# Un asset spécifique avec timeframe spécifique
uv run python -m scripts.fetch_oi --symbol BTC/USDT --days 720 --timeframe 5m

# Force re-fetch
uv run python -m scripts.fetch_oi --force --days 720
```

**Flux** :

```python
async def fetch_oi_for_symbol(exchange, db, symbol, timeframe, since_ms, end_ms):
    """Fetch incrémental de l'OI pour un symbol."""
    latest_ts = await db.get_latest_oi_timestamp(symbol, timeframe, "binance")
    if latest_ts and latest_ts > since_ms:
        since_ms = latest_ts + 1

    batch = []
    current = since_ms
    while current < end_ms:
        records = exchange.fetch_open_interest_history(
            symbol, timeframe=timeframe, since=current, params={"limit": 500}
        )
        if not records:
            break
        for r in records:
            batch.append({
                "symbol": symbol,
                "exchange": "binance",
                "timeframe": timeframe,
                "timestamp": r["timestamp"],
                "oi": r["baseVolume"],         # OI en contrats/coins
                "oi_value": r["quoteVolume"],   # OI en USDT
            })
        current = records[-1]["timestamp"] + 1
        time.sleep(0.1)

        # Insert par batch de 5000 pour limiter la mémoire
        if len(batch) >= 5000:
            await db.insert_oi_batch(batch)
            batch = []

    if batch:
        await db.insert_oi_batch(batch)
```

**Volume** : 207k points / 500 par requête = ~414 requêtes par asset. Avec `sleep(0.1)` + rate limit ccxt : ~50s par asset, ~4 min pour 5 assets.

**Attention** : La profondeur historique OI peut varier par asset. Binance ne garantit pas 2 ans pour tous. Si un asset a moins de données, logguer un warning et continuer.

**Fallback** : Si ccxt `fetchOpenInterestHistory()` n'est pas supporté, utiliser httpx avec l'API REST directe. Tester en Phase 2.

### 2.3 `scripts/optimize.py --check-data` — Afficher les données funding/OI

**Modifications** (~15 lignes) :

Après la vérification des candles, ajouter :

```
Vérification des données pour l'optimisation
──────────────────────────────────────────────────
  BTC/USDT     binance candles  :  720 jours (5m: 207k) ✓
  BTC/USDT     binance funding  :  718 jours (2154 rates) ✓
  BTC/USDT     binance OI       :  720 jours (5m: 207k)  ✓
  ...
  DOGE/USDT    binance OI       :  180 jours (5m: 52k)   ⚠ (< 360j)
```

---

## Phase 3 — BacktestEngine : support `extra_data`

C'est le changement le plus critique. Les stratégies funding et liquidation lisent `ctx.extra_data` pour obtenir les taux de funding et l'OI. Le BacktestEngine doit injecter ces données dans le `StrategyContext` à chaque bougie.

### 3.1 Fonctions d'alignement extra_data

**Fichier créé** : `backend/backtesting/extra_data_builder.py` (~100 lignes)

```python
"""Construit les extra_data alignés par timestamp pour le backtest.

Les données funding/OI sont à des fréquences différentes des candles.
Ce module aligne et forward-fill pour que chaque bougie ait des extra_data.
"""

from backend.core.models import Candle
from backend.strategies.base import EXTRA_FUNDING_RATE, EXTRA_OI_CHANGE_PCT, EXTRA_OPEN_INTEREST

def build_extra_data_map(
    candles: list[Candle],
    funding_rates: list[dict] | None = None,
    oi_records: list[dict] | None = None,
) -> dict[str, dict[str, Any]]:
    """Construit un dict {timestamp_iso: {extra_data}} pour chaque bougie.

    Funding rates : forward-fill (le taux est valide jusqu'au prochain).
    OI : oi_change_pct calculé vs précédent (même logique que DataEngine).

    Retourne un dict clé = timestamp ISO de la bougie → valeurs extra_data.
    """
    result: dict[str, dict[str, Any]] = {}

    funding_sorted = sorted(funding_rates or [], key=lambda r: r["timestamp"])
    oi_sorted = sorted(oi_records or [], key=lambda r: r["timestamp"])

    f_idx = 0
    current_funding_rate = None
    o_idx = 0
    prev_oi_value = None

    for candle in candles:
        candle_ts_ms = int(candle.timestamp.timestamp() * 1000)
        ts_iso = candle.timestamp.isoformat()
        extra: dict[str, Any] = {}

        # --- Funding rate (forward-fill) ---
        while f_idx < len(funding_sorted) and funding_sorted[f_idx]["timestamp"] <= candle_ts_ms:
            current_funding_rate = funding_sorted[f_idx]["funding_rate"]
            f_idx += 1

        if current_funding_rate is not None:
            extra[EXTRA_FUNDING_RATE] = current_funding_rate

        # --- Open Interest ---
        current_oi_value = None
        while o_idx < len(oi_sorted) and oi_sorted[o_idx]["timestamp"] <= candle_ts_ms:
            current_oi_value = oi_sorted[o_idx]["oi_value"]
            o_idx += 1

        if current_oi_value is not None:
            oi_change = 0.0
            if prev_oi_value is not None and prev_oi_value > 0:
                oi_change = (current_oi_value - prev_oi_value) / prev_oi_value * 100
            extra[EXTRA_OI_CHANGE_PCT] = oi_change
            extra[EXTRA_OPEN_INTEREST] = [current_oi_value]
            prev_oi_value = current_oi_value

        result[ts_iso] = extra

    return result
```

### 3.2 BacktestEngine.run() — Paramètre extra_data

**Fichier modifié** : `backend/backtesting/engine.py`

Ajouter un paramètre `extra_data_by_timestamp` à `run()` :

```python
def run(
    self,
    candles_by_tf: dict[str, list[Candle]],
    main_tf: str = "5m",
    precomputed_indicators: dict | None = None,
    extra_data_by_timestamp: dict[str, dict[str, Any]] | None = None,  # NOUVEAU
) -> BacktestResult:
```

Et dans la boucle principale (~ligne 157), injecter les extra_data dans le context :

```python
# Récupérer extra_data pour cette bougie
extra = {}
if extra_data_by_timestamp:
    extra = extra_data_by_timestamp.get(ts_iso, {})

ctx = StrategyContext(
    symbol=self._config.symbol,
    timestamp=candle.timestamp,
    candles=candles_by_tf,
    indicators=ctx_indicators,
    current_position=position,
    capital=capital,
    config=None,
    extra_data=extra,  # NOUVEAU — était implicitement {}
)
```

**Impact nul sur les stratégies existantes** : Sans `extra_data_by_timestamp`, le paramètre vaut `None` et `extra` reste `{}` — comportement identique à avant.

### 3.3 `run_backtest_single()` — Propager extra_data

**Fichier modifié** : `backend/backtesting/engine.py`

```python
def run_backtest_single(
    strategy_name: str,
    params: dict[str, Any],
    candles_by_tf: dict[str, list[Candle]],
    bt_config: BacktestConfig,
    main_tf: str = "5m",
    precomputed_indicators: dict | None = None,
    extra_data_by_timestamp: dict[str, dict[str, Any]] | None = None,  # NOUVEAU
) -> BacktestResult:
    strategy = create_strategy_with_params(strategy_name, params)
    engine = BacktestEngine(bt_config, strategy)
    return engine.run(
        candles_by_tf, main_tf, precomputed_indicators,
        extra_data_by_timestamp=extra_data_by_timestamp,  # NOUVEAU
    )
```

---

## Phase 4 — Intégration WFO

### 4.1 Ajouter funding et liquidation au `STRATEGY_REGISTRY`

**Fichier modifié** : `backend/optimization/__init__.py`

```python
from backend.strategies.funding import FundingStrategy
from backend.strategies.liquidation import LiquidationStrategy

STRATEGY_REGISTRY = {
    "vwap_rsi": (VwapRsiConfig, VwapRsiStrategy),
    "momentum": (MomentumConfig, MomentumStrategy),
    "funding": (FundingConfig, FundingStrategy),           # NOUVEAU
    "liquidation": (LiquidationConfig, LiquidationStrategy), # NOUVEAU
}
```

### 4.2 Mirroring `extreme_negative_threshold`

**Fichier modifié** : `backend/optimization/__init__.py`

Modifier `create_strategy_with_params()` pour dériver automatiquement le seuil négatif :

```python
def create_strategy_with_params(strategy_name: str, params: dict[str, Any]) -> BaseStrategy:
    config_cls, strategy_cls = STRATEGY_REGISTRY[strategy_name]
    defaults = config_cls().model_dump()
    merged = {**defaults, **params}

    # Funding : mirror le seuil négatif si absent du grid
    if strategy_name == "funding" and "extreme_positive_threshold" in params:
        if "extreme_negative_threshold" not in params:
            merged["extreme_negative_threshold"] = -params["extreme_positive_threshold"]

    config = config_cls(**merged)
    return strategy_cls(config)
```

Cela permet de n'avoir qu'un seul paramètre `extreme_positive_threshold` dans le grid au lieu de créer un produit cartésien de 16 combos non-symétriques.

### 4.3 Grids dans `param_grids.yaml`

**Fichier modifié** : `config/param_grids.yaml`

Les noms de paramètres correspondent **exactement** aux champs des configs Pydantic (`FundingConfig`, `LiquidationConfig`).

```yaml
funding:
  default:
    extreme_positive_threshold: [0.01, 0.02, 0.03, 0.05]  # en %, mirror → extreme_negative auto
    entry_delay_minutes: [3, 5, 10, 15]
    tp_percent: [0.3, 0.4, 0.6, 0.8]
    sl_percent: [0.1, 0.2, 0.3]
    # extreme_negative_threshold : PAS dans le grid, dérivé automatiquement
    # 4 × 4 × 4 × 3 = 192 combinaisons (full grid, pas de LHS)

liquidation:
  default:
    oi_change_threshold: [3.0, 5.0, 7.0, 10.0]   # en % (default config = 5.0)
    zone_buffer_percent: [1.0, 1.5, 2.0, 3.0]     # en % (default config = 1.5)
    leverage_estimate: [10, 15, 20]
    tp_percent: [0.5, 0.8, 1.0, 1.5]
    sl_percent: [0.3, 0.4, 0.5]
    # 4 × 4 × 3 × 4 × 3 = 576 combinaisons → LHS à 200
```

### 4.4 `_INDICATOR_PARAMS` — Ajouter les stratégies

**Fichier modifié** : `backend/optimization/walk_forward.py`

```python
_INDICATOR_PARAMS: dict[str, list[str]] = {
    "vwap_rsi": ["rsi_period"],
    "momentum": ["breakout_lookback"],
    "funding": [],        # NOUVEAU — aucun param n'affecte compute_indicators()
    "liquidation": [],    # NOUVEAU — idem, toutes les données viennent d'extra_data
}
```

Avec une liste vide, tous les combos partagent les mêmes indicateurs → `compute_indicators()` appelé une seule fois → maximum de réutilisation dans le fallback séquentiel.

### 4.5 WFO — Charger les données funding/OI

**Fichier modifié** : `backend/optimization/walk_forward.py`

Dans `optimize()`, après le chargement des candles (~ligne 348), ajouter le chargement conditionnel :

```python
# Charger les données extra si nécessaire (funding/liquidation)
extra_data_full: dict[str, dict[str, Any]] | None = None
if strategy_name in ("funding", "liquidation"):
    db_extra = Database()
    await db_extra.init()
    try:
        funding_rates = await db_extra.get_funding_rates(symbol, exchange)
        oi_records = await db_extra.get_open_interest(symbol, "5m", exchange)
        if strategy_name == "funding" and not funding_rates:
            raise ValueError(
                f"Pas de funding rates pour {symbol} sur {exchange}. "
                f"Lancez: uv run python -m scripts.fetch_funding --symbol {symbol}"
            )
        if strategy_name == "liquidation" and not oi_records:
            raise ValueError(
                f"Pas de données OI pour {symbol} sur {exchange}. "
                f"Lancez: uv run python -m scripts.fetch_oi --symbol {symbol}"
            )
    finally:
        await db_extra.close()

    from backend.backtesting.extra_data_builder import build_extra_data_map
    extra_data_full = build_extra_data_map(
        all_candles_by_tf[main_tf], funding_rates, oi_records,
    )
```

### 4.6 WFO — Passer extra_data au worker et à l'OOS

L'`extra_data_full` doit être transmis à chaque backtest (IS via worker, OOS via `run_backtest_single()`).

**Worker globals** — ajouter `extra_data_map` :

```python
_worker_extra_data: dict[str, dict[str, Any]] | None = None

def _init_worker(
    candles_serialized, strategy_name, symbol, bt_config_dict, main_tf,
    extra_data_map=None,  # NOUVEAU
):
    global _worker_extra_data
    # ... code existant inchangé ...
    _worker_extra_data = extra_data_map  # NOUVEAU

def _run_single_backtest_worker(params):
    result = run_backtest_single(
        _worker_strategy, params, _worker_candles, _worker_bt_config, _worker_main_tf,
        extra_data_by_timestamp=_worker_extra_data,  # NOUVEAU
    )
    metrics = calculate_metrics(result)
    return (params, metrics.sharpe_ratio, metrics.net_return_pct,
            metrics.profit_factor, metrics.total_trades)
```

**Fallback séquentiel** :

```python
def _run_single_backtest_sequential(
    params, candles_by_tf, strategy_name, bt_config, main_tf,
    precomputed_indicators=None,
    extra_data_by_timestamp=None,  # NOUVEAU
):
    result = run_backtest_single(
        strategy_name, params, candles_by_tf, bt_config, main_tf,
        precomputed_indicators=precomputed_indicators,
        extra_data_by_timestamp=extra_data_by_timestamp,  # NOUVEAU
    )
    # ... suite identique ...
```

**`_parallel_backtest()`** — propager :

```python
def _parallel_backtest(
    self, grid, candles_by_tf, strategy_name, symbol,
    bt_config_dict, main_tf, n_workers, metric,
    extra_data_map=None,  # NOUVEAU
) -> list[_ISResult]:
```

Passer `extra_data_map` à `_run_pool()` (initargs worker) et `_run_sequential()`.

**OOS dans la boucle des fenêtres** (~ligne 476) :

```python
oos_result = run_backtest_single(
    strategy_name, best_params, oos_candles_by_tf, bt_config, main_tf,
    extra_data_by_timestamp=extra_data_full,  # NOUVEAU
)
```

**Note** : On passe le `extra_data_full` complet (toute la période). Le `BacktestEngine` ne lit que les timestamps correspondant aux bougies de la fenêtre (`.get(ts_iso, {})` retourne `{}` pour les timestamps hors fenêtre).

**Sérialisation workers** : Le dict pour funding (~100 Ko) et OI (~16 Mo) est léger par rapport aux candles (~30 Mo). Pas d'impact significatif.

### 4.7 `validate_on_bitget()` — Charger les données extra

**Fichier modifié** : `backend/optimization/report.py`

Dans `validate_on_bitget()`, charger les funding/OI si la stratégie en a besoin :

```python
async def validate_on_bitget(strategy_name, symbol, recommended_params, ...):
    # ... chargement candles Bitget existant ...

    # Charger extra_data si nécessaire
    extra_data_by_timestamp = None
    if strategy_name in ("funding", "liquidation"):
        # Données Binance (pas d'historique funding/OI sur Bitget)
        funding = await db.get_funding_rates(symbol, "binance")
        oi = await db.get_open_interest(symbol, "5m", "binance")
        from backend.backtesting.extra_data_builder import build_extra_data_map
        extra_data_by_timestamp = build_extra_data_map(candles_list, funding, oi)

    result = run_backtest_single(
        strategy_name, recommended_params, oos_candles_by_tf, bt_config, main_tf,
        extra_data_by_timestamp=extra_data_by_timestamp,  # NOUVEAU
    )
```

**Note** : La validation utilise les candles Bitget (prix réels) mais les funding/OI Binance (pas d'historique Bitget). Les funding rates sont corrélés entre exchanges (~95%), et l'OI Binance domine le marché. Un warning sera ajouté au rapport.

---

## Phase 5 — Tests

### Fichier `tests/test_funding_oi_data.py` (nouveau, ~22 tests, ~400 lignes)

**Tests DB funding** (~4 tests) :

1. `test_insert_and_get_funding_rates` — Round-trip insert + get, vérifier le tri ASC
2. `test_funding_incremental_fetch` — `get_latest_funding_timestamp` retourne le bon timestamp
3. `test_get_funding_rates_date_range` — Filtrage start_ts/end_ts correct
4. `test_insert_funding_rates_batch_duplicates` — INSERT OR IGNORE sur doublons

**Tests DB OI** (~4 tests) :

5. `test_insert_and_get_oi` — Round-trip insert + get
6. `test_oi_incremental_fetch` — `get_latest_oi_timestamp` retourne le bon timestamp
7. `test_get_oi_date_range` — Filtrage correct
8. `test_insert_oi_batch_large` — Batch de 5000+ records

**Tests extra_data_builder** (~5 tests) :

9. `test_build_extra_data_funding_forward_fill` — Funding rate forward-fillé entre mises à jour 8h
10. `test_build_extra_data_oi_change_pct` — Calcul oi_change_pct correct (même logique DataEngine)
11. `test_build_extra_data_no_funding` — Pas de funding → extra_data sans clé funding_rate
12. `test_build_extra_data_no_oi` — Pas d'OI → extra_data sans clé oi_change_pct
13. `test_build_extra_data_both` — Funding + OI combinés dans le même dict

**Tests BacktestEngine extra_data** (~3 tests) :

14. `test_engine_passes_extra_data_to_context` — Vérifier que evaluate() reçoit extra_data non vide
15. `test_funding_backtest_generates_trades` — Backtest funding avec données synthétiques → trades > 0
16. `test_liquidation_backtest_generates_trades` — Backtest liquidation avec données synthétiques → trades > 0

**Tests WFO intégration** (~3 tests) :

17. `test_funding_in_strategy_registry` — `STRATEGY_REGISTRY["funding"]` existe et crée une stratégie
18. `test_create_strategy_with_params_mirrors_negative` — `extreme_negative_threshold` = `-extreme_positive_threshold`
19. `test_indicator_params_funding_empty` — `_INDICATOR_PARAMS["funding"] == []`

**Tests config** (~3 tests) :

20. `test_funding_config_per_asset` — `FundingConfig.per_asset` existe et `get_params_for_symbol()` fonctionne
21. `test_liquidation_config_per_asset` — Idem pour LiquidationConfig
22. `test_param_grid_names_match_config` — Noms des params du grid ⊆ champs des configs Pydantic

---

## Phase 6 — Exécution des optimisations

Une fois les données fetchées et les tests passés :

```bash
# 1. Fetch funding rates 2 ans
uv run python -m scripts.fetch_funding --days 720

# 2. Fetch OI 2 ans
uv run python -m scripts.fetch_oi --days 720

# 3. Vérifier toutes les données
uv run python -m scripts.optimize --check-data

# 4. Optimiser funding sur tous les assets
uv run python -m scripts.optimize --strategy funding --all-symbols

# 5. Optimiser liquidation sur tous les assets
uv run python -m scripts.optimize --strategy liquidation --all-symbols

# 6. Si grade A/B, appliquer les params
uv run python -m scripts.optimize --strategy funding --all-symbols --apply
```

---

## Ordre d'implémentation détaillé

### Phase 1 — DB & config (fondations)

1. `backend/core/database.py` — 2 nouvelles tables + 6 méthodes CRUD async
2. `backend/core/config.py` — `per_asset` + `get_params_for_symbol()` sur FundingConfig et LiquidationConfig
3. `config/strategies.yaml` — `per_asset: {}` sur funding et liquidation

**Dépendances** : aucune — ne modifie pas le comportement existant.

### Phase 2 — Scripts de fetch

4. `scripts/fetch_funding.py` — Fetch via ccxt + reprise incrémentale
5. `scripts/fetch_oi.py` — Fetch via ccxt + reprise incrémentale + batch insert
6. `scripts/optimize.py` — `--check-data` affiche funding/OI

**Dépendances** : Phase 1 (tables + CRUD).

### Phase 3 — BacktestEngine extra_data

7. `backend/backtesting/extra_data_builder.py` — Alignement funding/OI sur timestamps bougies
8. `backend/backtesting/engine.py` — Paramètre `extra_data_by_timestamp` dans `run()` et `run_backtest_single()`

**Dépendances** : Phase 1 (DB pour charger les données).

### Phase 4 — Intégration WFO

9. `backend/optimization/__init__.py` — STRATEGY_REGISTRY + mirroring negative threshold
10. `config/param_grids.yaml` — Grids funding et liquidation
11. `backend/optimization/walk_forward.py` — Chargement funding/OI, propagation extra_data, `_INDICATOR_PARAMS`
12. `backend/optimization/report.py` — `validate_on_bitget()` charge extra_data

**Dépendances** : Phase 3 (extra_data dans BacktestEngine).

### Phase 5 — Tests

13. `tests/test_funding_oi_data.py` — 22 tests

**Dépendances** : Phases 1-4.

### Phase 6 — Exécution

14. Lancer les fetchs, vérifier, optimiser

**Dépendances** : Phase 5 (tests passants).

---

## Estimation en lignes par fichier

### Fichiers modifiés

| Fichier | Lignes ajoutées | Modification |
|---------|----------------:|--------------|
| `backend/core/database.py` | ~90 | 2 tables + 6 méthodes CRUD |
| `backend/core/config.py` | ~20 | `per_asset` + `get_params_for_symbol()` × 2 configs |
| `config/strategies.yaml` | ~2 | `per_asset: {}` × 2 |
| `config/param_grids.yaml` | ~20 | Grids funding + liquidation |
| `backend/optimization/__init__.py` | ~10 | Registry + mirroring |
| `backend/optimization/walk_forward.py` | ~60 | Chargement extra, propagation workers, `_INDICATOR_PARAMS` |
| `backend/optimization/report.py` | ~20 | `validate_on_bitget()` charge extra_data |
| `backend/backtesting/engine.py` | ~15 | Paramètre `extra_data_by_timestamp` + injection context |
| `scripts/optimize.py` | ~15 | `--check-data` funding/OI |

### Fichiers créés

| Fichier | Lignes | Contenu |
|---------|-------:|---------|
| `scripts/fetch_funding.py` | ~120 | Fetch funding rates via ccxt |
| `scripts/fetch_oi.py` | ~140 | Fetch OI via ccxt |
| `backend/backtesting/extra_data_builder.py` | ~100 | Alignement extra_data sur timestamps |
| `tests/test_funding_oi_data.py` | ~400 | 22 tests |

### Récapitulatif

| Catégorie | Lignes |
|-----------|-------:|
| Backend modifié | ~215 |
| Backend créé | ~100 |
| Scripts créé | ~260 |
| Scripts modifié | ~15 |
| Config modifié | ~22 |
| Tests | ~400 |
| **Total** | **~1 010** |

---

## Contraintes & risques

1. **Rate limit Binance OI** : ~414 requêtes par asset × 5 assets = 2070 requêtes. Avec rate limit ccxt + sleep 0.1s : ~4 min par asset = ~20 min total. Acceptable.

2. **Profondeur OI Binance** : Binance peut ne pas avoir 2 ans d'OI 5m pour les altcoins récents (DOGE, LINK). Logguer un warning et continuer avec ce qui est disponible. Le WFO s'adapte (moins de fenêtres si données plus courtes).

3. **Funding rate échelle** : Binance API retourne le taux brut (0.0001 = 0.01%). On multiplie par 100 pour stocker en % (cohérent avec DataEngine qui fait `float(rate) * 100`). Le grid et les seuils `FundingConfig` sont en %.

4. **Entry delay stateful** : La stratégie funding a un `entry_delay_minutes` qui est stateful (stocke `_signal_detected_at`). Le BacktestEngine appelle `evaluate()` séquentiellement → l'état interne du timer est maintenu correctement. Pas de problème avec le moteur normal (contrairement au fast engine qui serait vectorisé).

5. **Extra_data sérialisation workers** : Le dict `extra_data_full` est passé au worker initializer via pickle. Pour funding (~100 Ko) et OI (~16 Mo), c'est léger par rapport aux candles (~30 Mo). Pas d'impact mémoire significatif.

6. **Validation Bitget avec données Binance** : La validation utilise les candles Bitget mais les funding/OI Binance (pas d'API historique Bitget). Les funding rates sont corrélés entre exchanges (~95%), et l'OI Binance domine le marché. Un warning sera ajouté au rapport.

7. **Fast engine non implémenté** : Les grids sont petits (192-576 combos). Le ProcessPool + fallback séquentiel suffit (~8 min par stratégie/asset). Le guard `if strategy_name in ("vwap_rsi", "momentum")` (ligne 595 de walk_forward.py) fait tomber directement sur ProcessPool — aucun changement nécessaire dans le fast engine.

8. **Backward compat** : Le paramètre `extra_data_by_timestamp=None` a une valeur par défaut dans `run()` et `run_backtest_single()`. Les appels existants (vwap_rsi, momentum, tests) ne sont pas affectés — `extra` reste `{}`.

9. **ccxt `fetchFundingRateHistory`** : Vérifier que Binance supporte cette méthode dans ccxt. Si non, fallback sur httpx avec l'API REST directe. Le test en Phase 2 confirmera avant d'implémenter le script complet.

10. **Funding timeframe 15m** : La stratégie funding utilise `timeframe: "15m"`. Le WFO lit `default_cfg.timeframe` → `main_tf = "15m"`. Les candles chargées seront en 15m, et le `build_extra_data_map()` alignera les funding rates sur ces bougies 15m (forward-fill fonctionne identiquement).

---

## Hors scope

- **Fast engine funding/liquidation** : Grids petits, ProcessPool suffit. Sprint 7c si nécessaire.
- **Fetch OI/funding depuis Bitget** : Pas d'API historique Bitget. On utilise Binance.
- **Optimisation orderflow** : Stratégie de confirmation uniquement, pas optimisable seule.
- **Dashboard optimisation** : Sprint 8 (endpoints API + frontend).
- **Lancement optimisation depuis le dashboard** : Sprint 9.

---

## Vérification finale

1. `uv run pytest` — tous les tests passent (330 existants + ~22 nouveaux = ~352)
2. `uv run python -m scripts.fetch_funding --days 720` — funding rates stockés en DB
3. `uv run python -m scripts.fetch_oi --days 720` — OI stocké en DB
4. `uv run python -m scripts.optimize --check-data` — affiche candles + funding + OI par asset
5. `uv run python -m scripts.optimize --strategy funding --symbol BTC/USDT` — WFO produit un rapport avec grade
6. `uv run python -m scripts.optimize --strategy liquidation --symbol BTC/USDT` — idem
7. Les backtests vwap_rsi/momentum existants fonctionnent toujours identiquement (backward compat)
8. `uv run python -m scripts.run_backtest --strategy funding --symbol BTC/USDT --days 90` — génère des trades > 0
