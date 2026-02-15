# Sprint 20a : Sizing equal-allocation + margin guard

## Contexte

Le sizing actuel "equal risk" (`risk_budget / sl_pct`) fait que la taille de position dépend du SL — un SL serré (1%) donne une marge énorme, un SL large (30%) une marge minuscule. Résultat : certains assets mobilisent une marge disproportionnée (jusqu'à 4.9× le capital). On remplace par "equal allocation" : la marge est fixe (`capital / nb_assets / levels`), et le SL contrôle uniquement la perte en $, pas la taille.

## Fichiers modifiés

| Fichier | Changement |
|---------|-----------|
| [config.py](backend/core/config.py) | +1 champ `max_margin_ratio` dans `RiskConfig` |
| [risk.yaml](config/risk.yaml) | +1 ligne `max_margin_ratio: 0.70` |
| [simulator.py](backend/backtesting/simulator.py) | Remplacement bloc sizing (L804-814) + ajout margin guard |
| [test_grid_runner.py](tests/test_grid_runner.py) | 4 tests mis à jour + 2 nouveaux tests |

## Changements détaillés

### 1. `backend/core/config.py` — RiskConfig (L336)

Ajouter après `initial_capital` :
```python
max_margin_ratio: float = Field(default=0.70, ge=0.1, le=1.0)
```

### 2. `config/risk.yaml` — Après `initial_capital`

```yaml
max_margin_ratio: 0.70  # Max 70% du capital en marge simultanée
```

### 3. `backend/backtesting/simulator.py` — Bloc sizing (L804-814)

**Supprimer** (6 lignes) :
```python
# Equal risk sizing : ajuster la marge au SL de l'asset
# risk_budget = perte max par position en $
num_levels = self._strategy.max_positions
sl_pct = self._get_sl_percent(symbol) / 100
risk_budget = pos_per_asset / num_levels
margin = risk_budget / sl_pct
# Garde-fou : un seul asset ne prend jamais > 25% du capital total
margin = min(margin, self._capital * 0.25)
# Reconvertir en pos_capital pour open_grid_position
# (qui fait notional = capital / levels * leverage)
pos_capital = margin * num_levels
```

**Remplacer par** :
```python
# Equal allocation sizing (Sprint 20a)
# Marge fixe par niveau = capital / nb_assets / num_levels
# Le SL contrôle le risque en $, PAS la taille de position
num_levels = self._strategy.max_positions
margin_per_level = pos_per_asset / num_levels

# Cap de sécurité : jamais plus de 25% du capital sur un seul asset
max_margin_per_asset = pos_raw * 0.25
margin_per_level = min(margin_per_level, max_margin_per_asset / num_levels)

# Margin guard (Sprint 20a) — skip si marge totale dépasse le seuil
max_margin_ratio = getattr(self._config.risk, "max_margin_ratio", 0.70)
if not isinstance(max_margin_ratio, (int, float)):
    max_margin_ratio = 0.70
total_margin_used = sum(
    p.entry_price * p.quantity / self._leverage
    for positions_list in self._positions.values()
    for p in positions_list
)
if total_margin_used + margin_per_level > pos_raw * max_margin_ratio:
    continue  # Skip ce niveau, pas assez de marge

pos_capital = margin_per_level * num_levels
```

**`_get_sl_percent()`** reste en place (pas supprimée), juste plus appelée ici.

Le `continue` est dans la boucle `for level in levels:` (L788) — il saute seulement ce niveau, pas le symbol.

### 4. `tests/test_grid_runner.py`

**`_make_mock_config()`** : ajouter `config.risk.max_margin_ratio = 0.70`

**4 tests mis à jour :**

| Test | Avant | Après |
|------|-------|-------|
| `test_margin_scales_with_nb_assets` | `margin ≈ risk_budget / sl_pct` (476.19$) | `margin ≈ capital/21/4` (119.05$) |
| `test_margin_with_small_capital` | `margin ≈ 1.19/0.25` (4.76$) | `margin ≈ 100/21/4` (1.19$) |
| `test_equal_risk_sizing` → **renommé** `test_equal_allocation_sizing` | ETH margin > SOL margin | ETH margin = SOL margin = capital/21/4 |
| `test_margin_cap_25pct` | Mêmes assertions, commentaires mis à jour | Idem (cap 25% fonctionne pareil) |

**2 nouveaux tests :**
- `test_margin_guard_blocks_when_full` : 58 positions pré-remplies (~69% marge), la 59ème est bloquée (>70%)
- `test_total_margin_not_exceed_capital` : 21 assets × 4 niveaux → marge totale ≤ 70% du capital

## Ce qu'on NE touche PAS

- **fast_multi_backtest.py** — déjà en equal allocation (`notional = capital/levels * leverage`), pas de nb_assets/cap/guard (single-asset WFO, correct)
- **executor.py** — reçoit `event.quantity` du Simulator, pas de sizing propre
- **GridPositionManager** — `open_grid_position()` reçoit `pos_capital` et fait `notional = capital/levels * leverage`, agnostique au sizing
- **Grades/résultats WFO** — inchangés
- **Frontend** — inchangé

## Formule vérifiée

`pos_capital = margin_per_level * num_levels` → `open_grid_position` fait `notional = pos_capital / num_levels * leverage = margin_per_level * leverage` → `margin_used = notional / leverage = margin_per_level` ✓

## Vérification

```powershell
uv run python -m pytest tests/test_grid_runner.py -v
uv run python -m pytest --tb=short -q
```
