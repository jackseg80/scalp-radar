# Sprint 54 — Fix leverage WFO : source valeur YAML au lieu du default Pydantic

## Contexte

L'audit a confirmé que le WFO simulait grid_atr (et toutes les grids) avec **leverage=6** (default Pydantic hardcodé dans `GridATRConfig`) au lieu de **leverage=7** (valeur dans `strategies.yaml`).

**Bonne nouvelle** : `fast_multi_backtest.py` utilisait déjà `leverage` correctement dans ses calculs — `notional = capital / num_levels * leverage`, `margin = notional / leverage`, kill switch drawdown. **Aucune modification nécessaire sur ce fichier.**

**Le vrai bug** était dans la *source de la valeur* transmise à `bt_config.leverage` :

- `walk_forward.py:449` → `default_cfg = config_cls()` = instance Pydantic sans args = defaults hardcodés (leverage=6)
- `walk_forward.py:575` → `bt_config.leverage = default_cfg.leverage` = **6**, pas 7
- `report.py:703` → même pattern dans `_validate_leverage_sl()` → warning affichait "6x"

La fix correcte : lire `leverage` depuis `get_config().strategies.{strategy_name}` (qui charge `strategies.yaml`) au lieu de `config_cls()`.

## Fichiers modifiés

| Fichier | Lignes | Changement |
|---------|--------|------------|
| `backend/optimization/walk_forward.py` | 573-577 | Lire leverage depuis `get_config().strategies` |
| `backend/optimization/report.py` | 701-705 | Idem dans `_validate_leverage_sl()` |
| `tests/test_leverage_wfo.py` | nouveau | 6 tests ciblés |

## Fix 1 — walk_forward.py (lignes 573-577)

```python
# AVANT
if hasattr(default_cfg, 'leverage'):
    bt_config.leverage = default_cfg.leverage

# APRÈS
if hasattr(default_cfg, 'leverage'):
    from backend.core.config import get_config
    _yaml_strat = getattr(get_config().strategies, strategy_name, None)
    bt_config.leverage = getattr(_yaml_strat, 'leverage', default_cfg.leverage)
```

`get_config()` est un singleton qui charge `strategies.yaml` une seule fois — pas de rechargement à chaque fenêtre WFO. Le bloc est dans le processus principal, avant les workers ProcessPool, donc thread-safe.

## Fix 2 — report.py (lignes 701-705)

```python
# AVANT
config_cls, _ = entry
default_cfg = config_cls()
leverage = getattr(default_cfg, "leverage", 6)

# APRÈS
config_cls, _ = entry
default_cfg = config_cls()
from backend.core.config import get_config
_yaml_strat = getattr(get_config().strategies, strategy_name, None)
leverage = getattr(_yaml_strat, "leverage", None) or getattr(default_cfg, "leverage", 6)
```

## Tests — tests/test_leverage_wfo.py (6 tests, tous verts)

- `test_pydantic_default_is_6_not_7` — documente l'ancien bug (default Pydantic = 6)
- `test_yaml_value_overrides_pydantic_default` — la logique du fix retourne 7
- `test_first_trade_pnl_scales_linearly` — PnL 1er trade = 7× avec leverage=7 vs leverage=1
- `test_leverage_1x_runs_are_deterministic` — deux runs identiques avec leverage=1
- `test_warning_reads_yaml_7x` — warning affiche "7x" avec YAML leverage=7
- `test_pydantic_default_would_give_6x` — référence ancien comportement "6x"

## Ce qui N'est PAS dans ce sprint

- Pas de modification de `fast_multi_backtest.py` (déjà correct)
- Pas de margin deduction dans le capital (intentionnellement reporté)
- Pas d'ajout de `leverage` dans `param_grids.yaml` (levier reste global)
- Pas de modification de `fast_backtest.py` pour les scalp (hors scope)

## Résultats

- 6 nouveaux tests → **2034 tests, 2034 passants**, 0 régression
