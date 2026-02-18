# Sprint 34a — Lancement paper trading grid_boltrend

## Contexte

Grid_boltrend est validé en backtest (Grade B) et déjà `enabled: true` dans strategies.yaml avec 6 per_asset (BTC, ETH, DOGE, DYDX, LINK, SAND). Cependant, un **bug critique dans `_warmup_from_db()`** empêchait la stratégie de fonctionner : elle ne chargeait que 50 candles alors que grid_boltrend nécessite jusqu'à 420 candles (`long_ma_window=400`). Résultat : `compute_live_indicators()` retournait `{}` pendant ~15 jours après chaque restart.

Ce sprint corrige le warm-up, ajoute un filet de sécurité try/except avec alerte Telegram, améliore les alertes Telegram, et documente le rollback.

**Résultat : 6 tests ajoutés (1353 → 1359 tests)**

---

## Fichiers modifiés

### 1. `backend/backtesting/simulator.py` — Warm-up dynamique + try/except + TODO

**1a. `MAX_WARMUP_CANDLES` : 200 → 500**

Aligné avec le `max_buffer=500` de `IncrementalIndicatorEngine`. Couvre `long_ma_window=400 + 20 = 420`.

**1b. `_warmup_from_db()` : utiliser `strategy.min_candles`**

```python
# AVANT
needed = min(max(self._ma_period + 20, 50), self.MAX_WARMUP_CANDLES)

# APRÈS
strat_min = self._strategy.min_candles.get(self._strategy_tf, 50)
needed = min(max(strat_min, 50), self.MAX_WARMUP_CANDLES)
```

Impact rétrocompatible :
- grid_atr : `min_candles=50` → inchangé
- envelope_dca : `min_candles=50` → inchangé
- grid_boltrend : `min_candles=420` → **corrigé** (était 50)

**1c. try/except autour de `compute_live_indicators()`**

Log ERROR + signal `_last_indicator_error` pour que le Simulator envoie l'alerte Telegram :

```python
try:
    extra = self._strategy.compute_live_indicators(list(candle_buf))
    for tf_key, tf_data in extra.items():
        indicators.setdefault(tf_key, {}).update(tf_data)
except Exception as e:
    logger.error(
        "[{}] compute_live_indicators ERREUR pour {}: {}",
        self.name, symbol, e,
    )
    self._last_indicator_error = (symbol, str(e))
```

Attribut `self._last_indicator_error: tuple[str, str] | None = None` ajouté dans `__init__`.

**1d. `_dispatch_candle()` : forward des erreurs indicateurs vers Notifier**

```python
alert = getattr(runner, '_last_indicator_error', None)
if alert and self._notifier:
    err_symbol, err_msg = alert
    runner._last_indicator_error = None
    try:
        from backend.alerts.notifier import AnomalyType
        await self._notifier.notify_anomaly(
            AnomalyType.INDICATOR_ERROR,
            f"[{runner.name}] {err_symbol}: {err_msg}",
        )
    except Exception:
        pass
```

Cooldown 1h géré par `_ANOMALY_COOLDOWNS` du Notifier.

**1e. TODO sizing (commentaire)**

```python
# Equal allocation sizing (Sprint 20a)
# TODO Sprint Phase 2 : L'ajout de runners grid_boltrend dilue
# l'allocation grid_atr (10 assets → 16 runners = -37% par runner).
# Accepté pour le paper. Options pour le live multi-stratégie :
# 1. Capital séparé par stratégie (config.weight)
# 2. Compter les assets uniques au lieu des runners
# Marge fixe par niveau = capital / nb_assets / num_levels
```

### 2. `backend/alerts/notifier.py` — Nouveau type `INDICATOR_ERROR`

- `INDICATOR_ERROR = "indicator_error"` dans `AnomalyType`
- `AnomalyType.INDICATOR_ERROR: "Erreur compute_live_indicators"` dans `_ANOMALY_MESSAGES`
- `AnomalyType.INDICATOR_ERROR: 3600` dans `_ANOMALY_COOLDOWNS` (1h)

### 3. `backend/alerts/telegram.py` — Préfixe [TAG] stratégie

Mapping + helper ajoutés avant la classe :

```python
_STRATEGY_TAGS: dict[str, str] = {
    "grid_atr": "ATR",
    "grid_boltrend": "BOLT",
    "grid_range_atr": "RANGE",
    "grid_multi_tf": "MTF",
    "grid_funding": "FUND",
    "grid_trend": "TREND",
    "envelope_dca": "DCA",
    "envelope_dca_short": "DCA-S",
}

def _strategy_tag(strategy: str) -> str:
    return _STRATEGY_TAGS.get(strategy, strategy.upper())
```

4 méthodes préfixées : `send_live_order_opened`, `send_live_order_closed`, `send_grid_level_opened`, `send_grid_cycle_closed`.

Tests existants non cassés (cherchent des sous-chaînes `"GRID ENTRY"`, `"GRID CLOSE"`).

### 4. `COMMANDS.md` — Section 17 : Rollback d'urgence

Section ajoutée avec :
- Désactivation stratégie via `nano .env` (JAMAIS `echo >`)
- Rollback git complet
- Purge état simulator (`deploy.sh --clean`)
- Vérification post-rollback

### 5. Tests — 6 nouveaux/modifiés

**`tests/test_grid_runner.py`**
- `test_warmup_capped_at_max` : ajout `assert limit_arg == 50`
- `test_warmup_uses_strategy_min_candles` : mock `min_candles={"1h": 420}` → vérifie 420 candles demandées
- `test_compute_live_indicators_error_caught` : crash sur 1ère candle → runner survit, `_last_indicator_error` set ; 2ème candle sans erreur → runner récupère

**`tests/test_telegram.py`**
- `test_grid_level_opened_has_strategy_tag` : vérifie `[BOLT]` dans le message pour `strategy="grid_boltrend"`

---

## Fichiers NON modifiés

- `config/strategies.yaml` — déjà `enabled: true`, per_asset déjà configurés
- `config/assets.yaml` — 6 assets déjà présents avec 1h
- `backend/strategies/grid_boltrend.py` — pas de modification nécessaire
- `backend/core/incremental_indicators.py` — buffers déjà correctement dimensionnés via `strat.min_candles`

---

## Vérification avant déploiement

### Étape 0 — Vérifier les données en DB

```powershell
uv run python -c "import sqlite3; conn = sqlite3.connect('data/scalp_radar.db'); rows = conn.execute(\"SELECT symbol, COUNT(*) as cnt FROM candles WHERE timeframe='1h' GROUP BY symbol ORDER BY cnt ASC\").fetchall(); [print(f'{r[0]:14} {r[1]:6} candles 1h') for r in rows]; conn.close()"
```

Minimum requis : **420 candles 1h** par asset boltrend (BTC, ETH, DOGE, DYDX, LINK, SAND). Si insuffisant :

```powershell
uv run python -m scripts.fetch_history --exchange binance --days 60 --symbols LINK/USDT,SAND/USDT --timeframe 1h
```

### Étape 1 — Tests

```powershell
uv run python -m pytest --tb=short -q
```

→ 1359 tests passent

### Étape 2 — Post-deploy

```bash
docker compose logs backend --tail 50 | grep -i boltrend
```

→ 6 runners boltrend créés

### Étape 3 — Après 1-2h

```bash
docker compose logs backend | grep "boltrend.*indicators\|boltrend.*Warm-up"
```

→ Messages "Warm-up: 420 bougies" (pas 50)

### Étape 4 — Telegram

Vérifier que les préfixes `[ATR]` et `[BOLT]` apparaissent dans les alertes.

---

## Leçons apprises

- **Cause racine bug warm-up** : `GridStrategyRunner.__init__` utilisait `getattr(strategy._config, "ma_period", 7)` pour `_ma_period`. grid_boltrend n'a pas de `ma_period` → défaut 7 → 50 candles au lieu de 420
- **Fix élégant** : `strategy.min_candles.get(tf, 50)` — chaque stratégie connaît ses besoins, le runner se contente de les respecter
- **Architecture alerte Telegram** : signal `_last_indicator_error` sur le runner, lu par `Simulator._dispatch_candle()` — pas besoin de modifier le constructeur du runner
- **`echo > .env` DANGEREUX** : écrase tout le fichier (clés API, tokens Telegram). Toujours utiliser `nano`
