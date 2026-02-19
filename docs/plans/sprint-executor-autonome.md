# Sprint — Executor Autonome : TP/SL indépendant + Réconciliation boot

## Contexte et problèmes résolus

### Problème 1 — Positions zombies (papier orphelin)
Quand le Simulator paper et l'Executor live divergent (ex : position live fermée par SL server-side Bitget, mais le Simulator ne reçoit jamais l'event CLOSE), la position paper devient zombie : elle reste ouverte indéfiniment dans le Simulator, affecte les métriques du journal, et la marge est bloquée virtuellement.

**Cause** : L'Executor dépend entièrement des events `TradeEvent` émis par le Simulator pour déclencher les ordres live. Si le circuit Simulator → Executor est rompu (restart, desync, SL hit directement sur Bitget), aucun mécanisme ne ferme la position paper.

### Problème 2 — Divergence paper ↔ live au restart
Après un restart, l'Executor restaure les positions live depuis `executor_state.json`, mais le Simulator repart à zéro (warm-up). Résultat : l'Executor a 3 grids actives, le Simulator n'en a aucune → pas de surveillance TP/SL paper, pas d'events live.

### Problème 3 — SL calibré pour 6x → dangereux en cross margin
Le SL était à 20% sur le prix d'entrée avec levier 6x = 120% de la marge. En cross margin, la position est liquidée bien avant d'atteindre le SL. Le levier réel optimal pour un SL 20% = 3x (20% × 3x = 60% marge, safe).

---

## Architecture des changements

### Bloc 0 — Fix leverage 6x → 3x

**Fichiers** :
- `config/risk.yaml` : `default_leverage: 15` → `default_leverage: 3`
- `config/strategies.yaml` : `leverage: 6` → `leverage: 3` pour `grid_atr` et `grid_boltrend`

**Raisonnement** : SL 20% × 3x = 60% marge (safe en cross margin, Bitget liquide à ~80-90%). Avec 6x, un mouvement adverse de 16% = liquidation avant le SL.

### Bloc 1 — Exit monitor autonome (Executor indépendant)

**Principe** : L'Executor tourne une boucle toutes les 60s qui appelle `strategy.should_close_all()` directement, sans dépendre des events du Simulator. Si should_close_all retourne "tp_global" ou "sl_global", l'Executor émet un `TradeEvent` synthétique et appelle `_close_grid_cycle()`.

**Méthodes ajoutées dans `executor.py`** :
- `set_data_engine(engine)` : enregistre le DataEngine pour accéder aux buffers de candles
- `set_strategies(strategies: dict[str, BaseGridStrategy])` : enregistre les instances stratégie par nom
- `start_exit_monitor()` : crée la tâche asyncio `_exit_monitor_loop()`
- `_exit_monitor_loop()` : boucle infinie avec `asyncio.sleep(60)` + error handling
- `_check_all_live_exits()` : itère `_grid_states` et appelle `_check_grid_exit()` pour chaque
- `_check_grid_exit(futures_sym)` :
  1. Récupère la stratégie depuis `_strategies[strategy_name]`
  2. Récupère les candles depuis `_data_engine._buffers[spot_sym][strategy_tf]`
  3. Appelle `strategy.compute_live_indicators(candles)` (avec fallback SMA simple)
  4. Construit un `GridState` depuis `_grid_states[futures_sym].positions`
  5. Construit un `StrategyContext` et appelle `strategy.should_close_all(ctx, grid_state)`
  6. Si retour non-None : crée `TradeEvent` synthétique et appelle `_close_grid_cycle()`

**Câblage dans `server.py`** :
```python
executor.set_data_engine(engine)
executor.set_strategies(simulator.get_strategy_instances())
await sync_live_to_paper(executor, simulator)
await executor.start_exit_monitor()
```

**Méthode ajoutée dans `simulator.py`** :
- `get_strategy_instances() -> dict[str, BaseGridStrategy]` : retourne `{runner.name: runner._strategy}` pour tous les `GridStrategyRunner`

### Bloc 2 — Réconciliation boot (sync live → paper)

**Fichier** : `backend/execution/sync.py` (NOUVEAU)

**Principe** : Au boot, après restore de l'Executor, on synchronise les positions paper (Simulator) avec les positions live (Executor). **Le LIVE fait autorité** (c'est l'argent réel).

**Règles** :
- Position live sans miroir paper → INJECTION : créer les positions paper correspondantes (depuis `GridLiveState.positions`)
- Position paper sans miroir live → SUPPRESSION : vider les positions, rendre la marge au capital paper
- Position live + paper → pas touchée (déjà en sync)

**Fonctions** :
- `sync_live_to_paper(executor, simulator)` : fonction principale async
- `_inject_live_to_paper(runner, spot_sym, live_state)` : crée des `GridPosition` depuis les `GridLivePosition`, déduit la marge du capital paper

**Ordre d'exécution dans server.py** :
1. `executor.restore_positions(executor_state)` — live positions restaurées
2. `simulator.start(saved_state)` — paper positions restaurées
3. `sync_live_to_paper()` — réconciliation (après les deux restores)
4. `executor.start_exit_monitor()` — monitoring démarre avec tout en place

### Bloc 3 — Hardening OPEN (déduplication exchange)

**But** : Empêcher l'Executor d'ouvrir un nouveau cycle de grid si Bitget a déjà une position sur ce symbol (ex : restart après ouverture, double event au même moment).

**Guard dans `_open_grid_position()`** :
```python
if is_first_level:
    positions = await self._fetch_positions_safe(futures_sym)
    has_exchange_position = any(float(p.get("contracts", 0)) > 0 for p in positions)
    if has_exchange_position:
        logger.warning("Executor: position déjà sur exchange, skip ouverture cycle")
        return
```

- Seulement au premier niveau (`is_first_level=True`) = nouveau cycle
- Les niveaux DCA supplémentaires (`is_first_level=False`) passent toujours

---

## Tests (20 nouveaux)

### `tests/test_executor_autonomous.py` (NOUVEAU)

**Classe `TestExitAutonomous` (10 tests)** :

| Test | Description |
|------|-------------|
| `test_exit_closes_grid_on_tp` | `should_close_all` retourne "tp_global" → `_close_grid_cycle()` appelé |
| `test_exit_no_close_when_none` | `should_close_all` retourne None → aucune action |
| `test_exit_sl_global` | `should_close_all` retourne "sl_global" → fermeture |
| `test_no_data_engine` | `_data_engine` absent → skip silencieux |
| `test_no_strategy` | Strategy inconnue dans `_strategies` → skip silencieux |
| `test_empty_buffer` | Buffer DataEngine vide pour ce symbol → skip |
| `test_correct_context_built` | StrategyContext passé à `should_close_all` a `symbol`, `candles`, `indicators` corrects |
| `test_correct_grid_state_built` | GridState construit depuis `_grid_states` avec les bonnes positions |
| `test_idempotent` | Deux appels consécutifs → une seule fermeture (state vidé après première) |
| `test_loop_catches_errors` | Exception dans `_check_grid_exit` → boucle continue (pas de crash) |

**Classe `TestSyncBoot` (7 tests)** :

| Test | Description |
|------|-------------|
| `test_inject_live_to_paper` | Position live sans miroir paper → positions créées dans `runner._positions` |
| `test_remove_paper_without_live` | Position paper sans miroir live → `runner._positions[sym] = []` |
| `test_keep_matching` | Position live + paper → pas touchée |
| `test_capital_adjusted_on_inject` | Injection déduit la marge du capital paper |
| `test_capital_returned_on_remove` | Suppression restitue la marge au capital paper |
| `test_multi_strategy` | 2 stratégies × 2 symbols → réconciliation correcte pour chacune |
| `test_no_live_positions` | Executor sans positions → paper nettoyé proprement |

**Classe `TestHardeningOpen` (3 tests)** :

| Test | Description |
|------|-------------|
| `test_rejected_if_exchange_has_position` | `_fetch_positions_safe` retourne position → `_open_grid_position` skip |
| `test_allowed_if_no_position` | `_fetch_positions_safe` retourne vide → ouverture normale |
| `test_dca_level_on_existing_grid_ok` | `is_first_level=False` → pas de check exchange, ouverture DCA autorisée |

---

## Fichiers modifiés

| Fichier | Type | Changement |
|---------|------|------------|
| `config/risk.yaml` | Config | `default_leverage: 15` → `3` |
| `config/strategies.yaml` | Config | `leverage: 6` → `3` pour grid_atr + grid_boltrend |
| `backend/execution/executor.py` | Backend | +5 méthodes exit monitor + guard OPEN + import Direction |
| `backend/backtesting/simulator.py` | Backend | +`get_strategy_instances()` |
| `backend/execution/sync.py` | Backend | NOUVEAU — `sync_live_to_paper()` + `_inject_live_to_paper()` |
| `backend/api/server.py` | Backend | Câblage exit monitor + sync dans lifespan |
| `tests/test_executor_autonomous.py` | Tests | NOUVEAU — 20 tests |

---

## Résultat

- **20/20 tests passants** (nouveau fichier)
- **1394 tests au total**, 0 régression
- Leverage 3x = SL sécurisé en cross margin Bitget (60% marge vs 120% avec 6x)
- Exit monitor : l'Executor peut fermer les positions live **même si le Simulator est désynchronisé**
- Sync boot : après restart, paper et live sont toujours alignés
- Guard OPEN : impossible d'ouvrir un doublon de cycle si Bitget a déjà une position

---

## Pièges à retenir

- **`datetime(2024, 1, 1, i)` heure invalide** : `i` de 0 à 49 → heure max 23. Fix : `base_ts + timedelta(hours=i)`
- **Early return dans sync bloque le cleanup** : si `executor._grid_states` vide, ne pas retourner tôt — le cleanup des positions paper doit quand même tourner
- **`getattr(runner._strategy, "strategy_tf", "1h")`** : utiliser getattr avec default pour compatibilité toutes stratégies
