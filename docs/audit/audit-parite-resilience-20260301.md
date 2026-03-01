# Audit Parité Backtest↔Live & Résilience — 2026-03-01

## Périmètre

- **Audit 1** : Parité entre fast engine WFO (`fast_multi_backtest.py`) et simulator/live (`simulator.py`, `risk_manager.py`)
- **Audit 2** : Résilience au boot et recovery après crash (`sync.py`, `state_manager.py`, `data_engine.py`)

## Méthodologie

Lecture complète du code source des composants critiques, comparaison formule par formule
entre les 3 moteurs (fast engine WFO, simulator paper, executor live).

---

## Résultats — Audit 1 : Parité Backtest↔Live

### BUG CONFIRMÉ P1 : Kill switch — formule différente

| Moteur | Formule | Base |
|--------|---------|------|
| **Fast engine WFO** (avant fix) | `(peak_capital - capital) / peak_capital` | Drawdown depuis le **peak** |
| **Simulator paper** | `abs(min(0, realized_pnl)) / initial_capital` | Perte depuis le **capital initial** |
| **Live RiskManager** | `abs(min(0, session_pnl)) / initial_capital` | Perte depuis le **capital initial** |

**Impact** : Si une stratégie fait +50% puis recule à +20%, le fast engine voyait 20% de DD
(proche du seuil 25%) et stoppait le WFO. Le simulator et le live voyaient 0% de perte
(toujours profitable) et continuaient normalement. Résultat : le WFO rejetait des combos
paramètres qui performaient bien en paper/live.

**Exemple chiffré** (test `grid_atr`, seed=42, n=500) :
- Avant fix : 54 trades, Sharpe 33.3 (kill switch déclenché trop tôt)
- Après fix : 107 trades, Sharpe 54.6 (aligné avec le simulator)

**6 fonctions** corrigées dans `fast_multi_backtest.py` :
- `_simulate_grid_common` (grid_atr, grid_multi_tf, grid_trend, envelope_dca)
- `_simulate_grid_range_atr`
- `_simulate_grid_funding`
- `_simulate_grid_boltrend_v2`
- `_simulate_grid_momentum`

**Note** : Le hard break 80% (safety net) reste en drawdown depuis peak — c'est un filet
de sécurité, pas un kill switch opérationnel.

### FAUX POSITIF : Entry fees non déduites à l'entrée

**Verdict** : PAS un bug. Cohérent entre fast engine et simulator.
- Les deux déduisent la **marge** à l'entrée (`capital -= margin`)
- Les deux incluent l'**entry_fee** dans le `net_pnl` à la clôture
- Le capital final est correct ; seul le sizing intermédiaire est ~0.06% trop généreux (négligeable)

### FAUX POSITIF : Margin deduction inconsistante

**Verdict** : Cohérent. Fast engine et simulator déduisent tous deux `margin = notional / leverage` à l'entrée.

### FAUX POSITIF : Slippage haute volatilité absent des grids

**Verdict** : Intentionnel. `high_vol_slippage_mult` est appliqué aux stratégies mono (scalp/swing)
mais pas aux grids. Le live executor n'a pas non plus ce multiplicateur — il subit le slippage
réel du marché. L'absence dans le fast engine grid est donc **plus réaliste**.

---

## Résultats — Audit 2 : Résilience & Recovery

### BUG CONFIRMÉ P1 : Sync positions au boot — retour silencieux

**Fichier** : `backend/execution/sync.py:95-97`

**Avant** : Si `fetch_positions()` échoue (API Bitget indisponible), la fonction retourne
silencieusement avec `logger.error()`. L'executor démarre avec `_grid_states` vide.
Conséquence : positions ouvertes sur Bitget non monitorées, pas de SL/TP check, pas d'alerte.

**Fix** : Retry avec backoff exponentiel (3 tentatives, 2s/4s) + `logger.critical()` si
toutes les tentatives échouent. Le message indique explicitement le risque de positions orphelines.

### VÉRIFIÉ OK : State persistence — fsync

`state_manager.py:186` implémente déjà `os.fsync(f.fileno())` avant `os.replace()`.
L'écriture atomique est correcte et durable.

### VÉRIFIÉ OK : Race conditions async

Les modifications de `self._tasks` (DataEngine), `self._positions` (Executor), et
`self._session_pnl` (RiskManager) sont toutes dans le même event loop asyncio.
Aucun `await` entre lecture et écriture des structures partagées → pas de race condition.

---

## Corrections appliquées

| Fichier | Changement |
|---------|------------|
| `backend/optimization/fast_multi_backtest.py` | Kill switch : `capital < initial_capital × 0.75` au lieu de `(peak - capital) / peak > 0.25` (6 fonctions) |
| `backend/execution/sync.py` | Retry 3× avec backoff + `logger.critical` si échec total |
| `tests/test_fast_engine_refactor.py` | Valeurs attendues mises à jour (grid_atr) |

## Tests

- **2182 passants** (4 échecs pré-existants : config SUI, param_grids, regime_monitor)
- **0 régression** introduite par les corrections
- Test de parité `test_parity_grid_atr` : valeurs recalculées (Sharpe 33.3 → 54.6, 54 → 107 trades)

---

## Recommandations (non implémentées)

| Priorité | Recommandation |
|----------|----------------|
| P2 | Ajouter `high_vol_slippage_mult` au fast engine grid pour les sorties SL (cohérence avec mono) |
| P2 | Ajouter alerte Telegram dans sync.py si fetch_positions échoue après retries |
| P3 | Rate limiting sur endpoints CPU-intensifs (`/api/optimization/run`, `/api/portfolio/run`) |
| P3 | Zombie position detection dans watchdog (ouverte >24h sans mouvement) |
| P3 | Monitoring funding rate extrême (>0.1%) |
