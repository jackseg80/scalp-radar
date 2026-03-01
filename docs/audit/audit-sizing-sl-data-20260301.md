# Audit Sizing, SL Coverage & Data Quality — 2026-03-01

## Périmètre

- **Audit 7** : Parité de sizing (WFO / Simulator / Executor)
- **Audit 8** : Couverture SL — règle "jamais de position sans SL"
- **Audit 9** : Qualité des données — gaps candles → signaux corrompus

---

## Résultats — Audit 7 : Parité de Sizing

### FAUX POSITIF : Margin guard executor utilise notional

**Verdict** : PAS un bug. `level.size_fraction × allocated_balance` EST la marge réelle
(pas la notional). Puisque `notional = size_fraction × balance × leverage`, alors
`margin = notional / leverage = size_fraction × balance`. Le calcul est correct.

### VÉRIFIÉ OK : Margin guard fast engine

Avec `num_levels=4` et `max_margin_ratio=0.70`, le fast engine ouvre les 4 niveaux.
Les tailles décroissent géométriquement (proportionnelles au capital résiduel), donc
`used_margin / total_equity` après 4 niveaux = 68.4% < 70% → guard jamais déclenché
en configuration standard.

### DIVERGENCE DOCUMENTÉE P2 : Formules de sizing différentes

| Moteur | Formule | Taille des niveaux |
|--------|---------|-------------------|
| **Fast engine WFO** | `margin = capital_courant × (1/num_levels)` | **Décroissante** (proportionnelle au capital résiduel) |
| **Simulator paper** | `margin = per_asset_capital / num_levels` | **Constante** (capital initial par asset) |
| **Executor live** | `margin = size_fraction × allocated_balance` | **Constante** (fractions fixes via `compute_grid()`) |

**Impact** : Le fast engine simule des niveaux 2-3-4 avec une taille légèrement inférieure
au premier. La différence est faible (~6% pour 4 niveaux) mais crée une légère sous-estimation
du PnL sur les niveaux tardifs en WFO. Pas critique — les signaux et les SL/TP restent alignés.

**Recommandation** : Aligner le fast engine sur le simulator (`initial_capital / num_levels`
constant). Faible priorité.

### VÉRIFIÉ OK : Double guard simulator en mode portfolio

Le simulator applique deux guards margin en portfolio mode (local + global). Comportement
conservateur mais cohérent — pas un bug.

---

## Résultats — Audit 8 : Couverture SL

### COUVERTURE FORTE : Ouvertures normales

- Retry 3× avec backoff sur `place_sl_order()`
- Si retry échoue → **emergency close immédiat** (`logger.critical` + `create_order("market")`)
- Notification Telegram si SL impossible
- Appliqué aux deux paths : mono-position et grid multi-niveaux

### COUVERTURE FORTE : Exit monitor autonome

L'exit monitor dans `_check_grid_exit()` calcule le SL indépendamment via
`strategy.get_sl_price(grid_state, tf_indicators)` — **aucune dépendance à `sl_order_id`**.
Si le prix atteint le SL calculé intra-candle, la position est fermée même sans ordre SL
exchange. Constitue un filet de sécurité robuste.

### BUG CONFIRMÉ P1 : SL manquant après boot via sync.py

**Fichier** : `backend/execution/executor.py:_reconcile_grid_symbol()`

**Avant** : Si `state.sl_order_id` est None (positions restaurées depuis Bitget via
`_populate_grid_states_from_exchange()` au boot), `_reconcile_grid_symbol()` restaure
le leverage et log "cycle restauré (SL=None)" — **sans placer de SL exchange**.

**Conséquence** : Position live avec `sl_price=0.0`, sans ordre SL sur Bitget.
Fenêtre de vulnérabilité jusqu'à la prochaine candle (max 1h pour grid 1h) pendant
laquelle seul l'exit monitor papier offre une protection (pas d'ordre exchange).

**Fix** : Après la restauration du leverage, si `not state.sl_order_id and state.total_quantity > 0`,
appeler `_update_grid_sl()` qui :
- Calcule le nouveau SL depuis `avg_entry_price` (propriété calculée depuis les positions)
- Place un ordre SL market sur Bitget (retry 3×)
- Si échec : emergency close

```python
# Ajouté dans _reconcile_grid_symbol() :
if not state.sl_order_id and state.total_quantity > 0:
    logger.info("Executor: SL manquant pour {} — replacement en cours", futures_sym)
    await self._update_grid_sl(futures_sym, state)
```

**Fréquence du bug** : Seulement lors de redémarrages avec perte du state file
(boot sans `executor_{strategy}_state.json`, ou fichier corrompu). Le cas normal
(redémarrage propre) conserve `sl_order_id` dans le state file.

---

## Résultats — Audit 9 : Qualité des Données

### VÉRIFIÉ OK : Validation OHLC

`DataEngine.validate_candle()` vérifie low ≤ high, volume ≥ 0, prix > 0. Candles
invalides rejetées avant insertion en buffer.

### VÉRIFIÉ OK : Indicateurs NaN prédictibles

Tous les indicateurs (`compute_rsi`, `compute_atr`, `compute_adx`, etc.) retournent
`float("nan")` explicitement si données insuffisantes. Pur Python, zéro segfault.

### VÉRIFIÉ OK : Régime market fallback

`_detect_regime()` retourne `MarketRegime.RANGING` si n'importe quel indicateur est NaN.
Comportement sûr — pas d'entrée aléatoire.

### LACUNE DOCUMENTÉE P2 : Gaps détectés mais non bloqués

**Fichier** : `backend/core/data_engine.py:806-813`

`check_gap()` détecte les gaps (candle manquante) et log un WARNING, mais **ajoute
quand même la candle** au buffer. Les indicateurs calculés après un gap sont corrompus
(ATR/Wilder-smoothing continu sur une discontinuité temporelle).

**Scénario** :
1. WebSocket Bitget timeout → reconnexion → 3 candles 1h manquées
2. DataEngine log "gap détecté" et insère les nouvelles candles
3. `compute_atr()` continue le Wilder-smoothing sur le gap → ATR invalide
4. `grid_atr.compute_grid()` utilise cet ATR → niveaux de grille faux
5. Signal émis avec `entry_price` et `sl_price` incohérents

**Impact** : Faible en pratique — les gaps >1.5× timeframe sont rares sur Bitget
(infrastructure solide). Et les indicateurs se "recalibrent" après ~N nouvelles candles.

**Recommandation** : Marquer le symbol comme stale pendant `atr_period × 2` candles
après un gap détecté, bloquer les signaux pendant cette période.

### LACUNE DOCUMENTÉE P2 : Pas de validation NaN avant evaluate()

**Fichier** : `backend/backtesting/simulator.py:_on_candle_inner()`

Le guard vérifie que `main_tf` est présent dans les indicateurs, mais pas que les
valeurs individuelles ne sont pas NaN. Un dict `{"adx": nan, "rsi": nan}` passe
le guard et atteint `strategy.evaluate()`.

**Impact** : Faible — les stratégies gèrent les NaN via `math.isnan()` dans leur logique
(ex: `if math.isnan(atr_value): return`). Mais certains chemins pourraient ne pas vérifier.

**Recommandation** : Ajouter un guard `math.isnan()` sur les indicateurs critiques
(ATR, BB, SMA) avant `strategy.evaluate()`.

---

## Corrections appliquées

| Fichier | Changement |
|---------|------------|
| `backend/execution/executor.py` | `_reconcile_grid_symbol()` : placement SL si `sl_order_id is None` au boot |

## Tests

- **2182 passants**, 0 régression, 5 pré-existants inchangés

---

## Recommandations (non implémentées)

| Priorité | Recommandation |
|----------|----------------|
| P2 | Aligner sizing fast engine : `initial_capital / num_levels` constant (comme simulator) |
| P2 | DataEngine : marquer symbol stale après gap, bloquer signaux pendant N candles |
| P2 | Simulator : guard NaN sur indicateurs critiques avant `strategy.evaluate()` |
| P3 | Vérifier NaN sur TP/SL dans le TradeEvent avant envoi à `_open_grid_position()` |
