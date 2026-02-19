# Sprint Backtest Réalisme — Liquidation, Funding, Leverage Validation

## Contexte

Les backtests actuels ne reflètent pas la réalité du trading cross margin sur Bitget :
- **Pas de simulation de liquidation** — un portfolio peut afficher -50% DD en backtest alors qu'il aurait été liquidé
- **Pas de funding costs** dans le GridStrategyRunner (paper/portfolio) — le fast engine WFO les gère (Sprint 26) mais pas le runner event-driven
- **Pas de validation leverage × SL** — `sl_percent=20 × leverage=6 = 120%` de la marge, chaque SL coûte plus que la marge allouée
- Le passage de 6x à 3x est justifié mais manque de données chiffrées

## Fix 1 — Simulation liquidation cross margin

### Fichier : `backend/backtesting/portfolio_engine.py`

**1a. Étendre `PortfolioSnapshot` (ligne 40)**

Ajouter 4 champs avec defaults (rétro-compatible) :

```python
total_notional: float = 0.0           # Somme entry_price × quantity (toutes positions)
maintenance_margin: float = 0.0       # 0.4% du notional (Bitget USDT-M tier 1)
liquidation_distance_pct: float = 0.0 # (equity - maintenance) / equity × 100
is_liquidated: bool = False
```

**1b. Modifier `_take_snapshot()` (ligne 557)**

Dans la boucle qui itère `runner._positions` (ligne 579-588), calculer aussi `total_notional` :

```python
total_notional = 0.0
# Dans la boucle existante (ligne 584-585) qui calcule margin :
for p in positions:
    notional_p = p.entry_price * p.quantity
    total_notional += notional_p
    # margin existant : += notional_p / leverage
```

Après le calcul de `total_equity` (ligne 591), ajouter :

```python
MAINTENANCE_MARGIN_RATE = 0.004  # 0.4% Bitget USDT-M
maintenance_margin = total_notional * MAINTENANCE_MARGIN_RATE
liquidation_distance_pct = (
    ((total_equity - maintenance_margin) / total_equity * 100)
    if total_equity > 0 else -100.0
)
is_liquidated = total_equity <= maintenance_margin and total_notional > 0
```

Passer ces valeurs au constructeur du `PortfolioSnapshot`.

**1c. Arrêter la simulation si liquidé — dans `_simulate()` (ligne 516)**

Après `snap = self._take_snapshot(...)` et `snapshots.append(snap)`, ajouter :

```python
if snap.is_liquidated:
    logger.critical(
        "LIQUIDATION à {} — equity={:.2f}, maintenance={:.2f}, notional={:.2f}",
        candle.timestamp, snap.total_equity, snap.maintenance_margin, snap.total_notional,
    )
    for runner in runners.values():
        runner._capital = 0.0
        runner._positions = {}
        runner._kill_switch_triggered = True
    liquidation_event = {
        "timestamp": candle.timestamp.isoformat(),
        "equity": snap.total_equity,
        "maintenance_margin": snap.maintenance_margin,
        "notional": snap.total_notional,
        "n_positions": snap.n_open_positions,
    }
    break
```

Déclarer `liquidation_event = None` avant la boucle, et le passer à `_build_result()`.

**1d. Étendre `PortfolioResult` (ligne 55)**

Ajouter après `funding_paid_total` (ligne 97) :

```python
was_liquidated: bool = False
liquidation_event: dict | None = None
min_liquidation_distance_pct: float = 100.0
```

**1e. Calculer `min_liquidation_distance_pct` dans `_build_result()` (ligne 740)**

```python
notional_snaps = [s for s in snapshots if s.total_notional > 0]
min_liq = min((s.liquidation_distance_pct for s in notional_snaps), default=100.0)
```

Passer `was_liquidated`, `liquidation_event`, `min_liquidation_distance_pct` au constructeur.

**1f. Worst-case SL — calculé à chaque snapshot, garder le max**

Le worst-case doit être calculé au moment du **peak positions**, pas à la fin de la simulation (où il peut n'y avoir aucune position). Approche : calculer à chaque snapshot et garder le max.

Dans `_take_snapshot()`, ajouter le calcul du worst-case SL instantané :

```python
# Worst-case SL : perte si TOUS les SL touchent en même temps
worst_case_sl_loss = 0.0
for runner_key, runner in runners.items():
    symbol = self._symbol_from_key(runner_key)
    sl_pct = runner._get_sl_percent(symbol) / 100  # ex: 20% → 0.20
    for pos_symbol, positions in runner._positions.items():
        for p in positions:
            notional = p.entry_price * p.quantity
            worst_case_sl_loss += notional * sl_pct
worst_case_sl_loss_pct = (
    (worst_case_sl_loss / self._initial_capital * 100)
    if self._initial_capital > 0 else 0.0
)
```

Note : `_take_snapshot` reçoit déjà `runners` en paramètre, pas besoin de modifier la signature.

Ajouter `worst_case_sl_loss_pct: float = 0.0` au `PortfolioSnapshot` (pour tracking par snapshot).

Dans `_build_result()`, prendre le max sur tous les snapshots :

```python
max_worst_case = max(
    (s.worst_case_sl_loss_pct for s in snapshots if s.n_open_positions > 0),
    default=0.0,
)
```

Ajouter `worst_case_sl_loss_pct: float = 0.0` à `PortfolioResult`.

## Fix 2 — Funding costs dans GridStrategyRunner

### Fichier : `backend/backtesting/simulator.py`

**2a. Ajouter dans `__init__` (après ligne 611)**

```python
self._total_funding_cost: float = 0.0
```

**2b. Ajouter le funding AVANT le check TP/SL dans `on_candle()` (avant ligne 901)**

Le funding settlement s'applique AVANT le TP/SL (si position ouverte à l'heure de settlement, on paie même si elle ferme ensuite).

Placer le bloc entre le calcul du `ctx` (ligne 884-892) et la section "Si positions ouvertes → check TP/SL" (ligne 901) :

```python
# Funding cost (settlement toutes les 8h : 00:00, 08:00, 16:00 UTC)
if not self._is_warming_up and positions and candle.timestamp.hour in (0, 8, 16):
    funding_rate = 0.0001  # 0.01% par session (approximation conservative)
    for pos in positions:
        notional = pos.entry_price * pos.quantity
        if pos.direction == Direction.LONG:
            cost = notional * funding_rate   # LONG paie
        else:
            cost = -notional * funding_rate  # SHORT reçoit
        self._capital -= cost
        self._total_funding_cost += cost
```

Note : `positions` est déjà défini juste avant (ligne 895 : `positions = self._positions.get(symbol, [])`).

**2c. Inclure le funding dans `get_status()` (ligne 1273)**

Ajouter un champ au dict retourné :

```python
"funding_cost": round(self._total_funding_cost, 2),
```

**2d. Agréger dans portfolio_engine `_build_result()`**

Le champ `funding_paid_total` existe déjà dans `PortfolioResult` (ligne 97). L'alimenter :

```python
total_funding = sum(
    getattr(r, '_total_funding_cost', 0.0) for r in runners.values()
)
```

Passer `funding_paid_total=round(total_funding, 2)` au constructeur.

### Config — PAS de modification de risk.yaml

Le taux 0.01% est hardcodé dans le runner (comme constante locale). Pas besoin de config YAML pour une approximation — les vrais funding rates sont dans la DB et utilisés par le fast engine WFO.

## Fix 3 — Validation leverage × SL dans le rapport WFO

### Fichier : `backend/optimization/report.py`

**3a. Ajouter une fonction helper (avant `build_final_report`, ~ligne 640)**

```python
def _validate_leverage_sl(strategy_name: str, params: dict) -> list[str]:
    """Valide que la combinaison SL × leverage est viable en cross margin."""
    from backend.optimization import STRATEGY_REGISTRY
    warnings = []

    sl_pct = params.get("sl_percent")
    if sl_pct is None:
        return warnings

    config_cls, _ = STRATEGY_REGISTRY.get(strategy_name, (None, None))
    if config_cls is None:
        return warnings

    default_cfg = config_cls()
    leverage = getattr(default_cfg, "leverage", 6)

    loss_per_margin = sl_pct * leverage / 100

    if loss_per_margin > 1.0:
        warnings.append(
            f"SL {sl_pct}% x leverage {leverage}x = {loss_per_margin:.0%} de la marge "
            f"(depasse 100% — chaque SL coute plus que sa marge en cross margin)"
        )
    elif loss_per_margin > 0.8:
        warnings.append(
            f"SL {sl_pct}% x leverage {leverage}x = {loss_per_margin:.0%} de la marge "
            f"(risque, peu de marge de securite)"
        )

    return warnings
```

**3b. Appeler dans `build_final_report()` (après les autres warnings, ~ligne 745)**

```python
leverage_warnings = _validate_leverage_sl(wfo.strategy_name, wfo.recommended_params)
warnings.extend(leverage_warnings)
```

Insérer juste avant le `return FinalReport(...)` (ligne 751).

## Fix 4 — Report enrichi portfolio backtest

### Fichier : `backend/backtesting/portfolio_engine.py`

**4a. Enrichir `format_portfolio_report()` (ligne 838)**

Ajouter une section "Cross-Margin Risk" après la section "Kill Switch" (après ligne 884) :

```python
# Cross-Margin Risk
lines.append("  --- Cross-Margin Risk ---")
lines.append(f"  Min liquidation distance  : {result.min_liquidation_distance_pct:.1f}%")
lines.append(f"  Worst-case SL loss        : {result.worst_case_sl_loss_pct:.1f}% du capital")
lines.append(f"  Funding costs total       : {result.funding_paid_total:+.2f}$")

if result.was_liquidated:
    lines.append(f"  LIQUIDE a {result.liquidation_event['timestamp']}")
    lines.append(f"     Equity={result.liquidation_event['equity']:.2f}, "
                 f"Maintenance={result.liquidation_event['maintenance_margin']:.2f}")
lines.append("")
```

### Fichier : `scripts/portfolio_backtest.py`

**4b. Enrichir `_result_to_dict()` (ligne 33)**

Ajouter après les champs existants :

```python
d["was_liquidated"] = result.was_liquidated
d["liquidation_event"] = result.liquidation_event
d["min_liquidation_distance_pct"] = result.min_liquidation_distance_pct
d["worst_case_sl_loss_pct"] = result.worst_case_sl_loss_pct
d["funding_paid_total"] = result.funding_paid_total
```

## Tests

### Nouveau fichier : `tests/test_backtest_realism.py`

| # | Test | Vérifie |
|---|------|---------|
| **Fix 1** | | |
| 1 | `test_snapshot_liquidation_distance` | Snapshot positions → `liquidation_distance_pct` calculé correctement |
| 2 | `test_snapshot_liquidated_when_equity_below_maintenance` | `equity < maintenance` → `is_liquidated=True` |
| 3 | `test_simulation_stops_on_liquidation` | Crash prix → equity < maintenance → simulation break, capital=0 |
| 4 | `test_no_liquidation_normal` | Conditions normales → `is_liquidated=False` partout |
| 5 | `test_maintenance_margin_calc` | notional 100k × 0.004 = 400$ |
| 6 | `test_worst_case_sl` | 3 positions × SL 20% × notional → perte correcte |
| **Fix 2** | | |
| 7 | `test_funding_at_8h_intervals` | Capital diminue à h=0,8,16 UTC |
| 8 | `test_funding_not_during_warmup` | Pas de funding pendant warm-up |
| 9 | `test_funding_long_pays_short_receives` | LONG → capital ↓, SHORT → capital ↑ |
| 10 | `test_funding_total_tracked` | `_total_funding_cost` correct après N heures |
| 11 | `test_no_funding_without_positions` | Pas de positions → pas de funding |
| **Fix 3** | | |
| 12 | `test_leverage_sl_warning_above_100pct` | SL 20% × 6x → warning "dépasse 100%" |
| 13 | `test_leverage_sl_warning_above_80pct` | SL 15% × 6x → warning "risqué" |
| 14 | `test_leverage_sl_ok` | SL 20% × 3x = 60% → pas de warning |
| 15 | `test_leverage_sl_in_final_report` | Warning apparaît dans FinalReport.warnings |

## Fichiers modifiés

| Fichier | Modification |
|---------|-------------|
| `backend/backtesting/portfolio_engine.py` | PortfolioSnapshot +4 champs, `_take_snapshot` (liquidation), `_simulate` (break on liquidation), PortfolioResult +3 champs, worst-case SL, `_build_result` (agréger funding + liquidation), `format_portfolio_report` (section Cross-Margin) |
| `backend/backtesting/simulator.py` | GridStrategyRunner: `_total_funding_cost` init, funding dans `on_candle()` AVANT TP/SL, `get_status()` funding_cost |
| `backend/optimization/report.py` | `_validate_leverage_sl()`, appel dans `build_final_report()` |
| `scripts/portfolio_backtest.py` | +5 champs dans `_result_to_dict()` |
| `tests/test_backtest_realism.py` | NOUVEAU — 15 tests |

## Ordre d'implémentation

1. **Fix 1** — Liquidation (le plus critique, révèle si les configs actuelles sont viables)
2. **Fix 2** — Funding (impact P&L significatif, change les résultats portfolio)
3. **Fix 3** — Leverage validation (warnings pour futurs WFO)
4. **Fix 4** — Report enrichi (affichage)
5. **Tests** — Fichier unique `test_backtest_realism.py`

## Vérification

```bash
# Tests du sprint
uv run python -m pytest tests/test_backtest_realism.py -x -v

# Suite complète (1394 tests existants + 15 nouveaux)
uv run python -m pytest -x -q

# Portfolio backtest avec nouvelles métriques
uv run python -m scripts.portfolio_backtest --days 90 --capital 1000
```

Résultats attendus avec leverage 6x + sl_percent=20% :
- `worst_case_sl_loss_pct` > 100% → warning
- Possible liquidation pendant crashes 2024
- Funding réduit le return de 5-15%

## Points d'attention

1. **MAINTENANCE_MARGIN_RATE = 0.004** — Bitget USDT-M tier 1, correct pour notre capital. Altcoins potentiellement 0.5-1% mais 0.4% = approximation basse acceptable.
2. **Funding rate fixe 0.01%** — approximation. Le fast engine WFO utilise les rates historiques DB (Sprint 26). Le runner utilise un taux fixe car le but est surtout de montrer l'impact global, pas la précision à 0.001%.
3. **Pas de persistence** de `_total_funding_cost` dans state_manager.py — en live, le P&L réel vient de Bitget (Hotfix 34). Le compteur est utile uniquement pour le backtest.
4. **Liquidation = ALL-OR-NOTHING** en cross margin — Bitget liquide TOUT, pas juste une position.
5. **Ne PAS modifier le fast engine WFO** — le funding y est déjà (Sprint 26), la liquidation n'est pas pertinente pour un seul asset.
