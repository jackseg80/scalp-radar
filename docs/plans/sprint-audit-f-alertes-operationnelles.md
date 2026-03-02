# Sprint Audit F — Alertes opérationnelles

**Date** : 2 mars 2026
**Commit** : HEAD (suite Sprint D, commit 8ded147)

---

## Contexte

Sprint F du plan d'audit global (A-F). Les sprints A-D sont terminés.
Ce sprint adresse deux situations silencieuses en production :
- **F1** : Marge saturée — l'executor refuse des positions sans alerte
- **F2** : Funding rate extrême — coût carry invisible sur les shorts (>0.1%)

---

## F1 — Margin Proximity Alert

**Problème** : Quand la marge utilisée dépasse 90% du solde exchange, le margin guard dans
`_open_grid_level()` refuse silencieusement les nouvelles positions (log WARNING local, pas de Telegram).
L'opérateur ne sait pas que le bot est en mode "survie marge".

**Solution** :
- `Executor._check_margin_proximity(balance)` — appelée dans `_balance_refresh_loop()` (toutes les 5 min)
- Calcule `total_margin = grid_margin + mono_margin + _pending_notional`
  - Grid : `gs.avg_entry_price * gs.total_quantity / max(gs.leverage, 1)`
  - Mono : `pos.entry_price * pos.quantity / max(default_leverage, 1)`
- Alerte `AnomalyType.MARGIN_PROXIMITY` si `ratio > _MARGIN_ALERT_THRESHOLD (0.90)`
- Cooldown 30 min (état persistant, pas besoin de spam)

**Fichiers** : `backend/execution/executor.py`, `backend/alerts/notifier.py`

---

## F2 — Funding Rate Extremes Alert

**Problème** : Le DataEngine poll les funding rates toutes les 5 min, mais aucune alerte si le rate
dépasse 0.1% (coût carry annualisé ~52.5% sur un SHORT). Les grids short peuvent perdre
silencieusement sur le carry.

**Solution** :
- `Executor._check_funding_rates()` — appelée dans `_balance_refresh_loop()` (toutes les 5 min)
- Itère sur `config.assets`, lit `DataEngine.get_funding_rate(symbol)` (retourne déjà en %)
- Alerte `AnomalyType.FUNDING_ALERT` si `|rate| > _FUNDING_ALERT_THRESHOLD (0.1%)`
- Cooldown 15 min (transitoire, peut se normaliser rapidement)
- Skip silencieux si `_data_engine is None` (executor non connecté au DataEngine)

**Fichiers** : `backend/execution/executor.py`, `backend/alerts/notifier.py`

---

## Implémentation

### Constantes module-level (executor.py)

```python
_MARGIN_ALERT_THRESHOLD = 0.90  # 90% du solde utilisé → alerte
_FUNDING_ALERT_THRESHOLD = 0.1  # |funding| > 0.1% → alerte
```

Niveau module (pas classe) pour éviter l'accès via `self.` qui casse les mocks dans les tests.

### AnomalyType ajoutés (notifier.py)

```python
MARGIN_PROXIMITY = "margin_proximity"   # cooldown 30 min
FUNDING_ALERT = "funding_alert"         # cooldown 15 min
```

### Intégration dans _balance_refresh_loop

```python
if new_balance is not None:
    self._risk_manager.record_balance_snapshot(new_balance)
    ...
    # Sprint F : alertes margin proximity + funding extremes
    await self._check_margin_proximity(new_balance)
await self._check_funding_rates()
```

`_check_funding_rates` est appelé en dehors du `if new_balance is not None` car le funding
ne dépend pas du solde — les deux checks sont indépendants.

---

## Tests

**Fichier** : `tests/test_sprint_f_alerts.py` (13 tests)

| Test | Vérifie |
|------|---------|
| `test_anomaly_type_margin_proximity_exists` | Enum existe avec valeur correcte |
| `test_anomaly_type_funding_alert_exists` | Enum existe avec valeur correcte |
| `test_anomaly_messages_margin_proximity` | Message contient "90" |
| `test_anomaly_cooldown_funding_alert` | Cooldown < 1800s |
| `test_margin_proximity_alert_triggered` | Alerte si ratio > 90% |
| `test_margin_proximity_no_alert_below_threshold` | Pas d'alerte si ratio < 90% |
| `test_margin_proximity_zero_balance_skipped` | balance <= 0 → skip (no div/0) |
| `test_margin_proximity_includes_pending_notional` | `_pending_notional` inclus |
| `test_funding_alert_triggered_positive` | Alerte si funding > 0.1% |
| `test_funding_alert_triggered_negative` | Alerte si funding < -0.1% |
| `test_funding_alert_not_triggered_normal` | Pas d'alerte si |funding| <= 0.1% |
| `test_funding_alert_skips_none_rate` | None → skip |
| `test_funding_check_skipped_when_no_data_engine` | `_data_engine is None` → no crash |

---

## Résultats

| Sprint | Tests ajoutés | Total | Régressions |
|--------|--------------|-------|-------------|
| F | 13 | 2226 | 0 |

5 pré-existants non liés (SUI/XTZ/JUP/param_grids/resample_gaps).

---

## Ce qui reste

| Sprint | Contenu |
|--------|---------|
| E | Frontend tests (Vitest + React Testing Library, 9 composants critiques) |
| C suite | `entry_handler.py` — fusion `_open_position` + `_open_grid_position` |
| S1 | Persist regime snapshots (table `regime_snapshots` en DB) |
