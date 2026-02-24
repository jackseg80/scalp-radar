# Trading Strategies — Scalp Radar

Guide complet des 17 stratégies de trading implémentées dans Scalp Radar.
Tout ce qui est documenté ici est extrait du code source réel (`backend/strategies/`).

---

## Vue d'ensemble

| # | Stratégie | Timeframe | Type | Direction | Enabled | Statut |
|---|-----------|-----------|------|-----------|---------|--------|
| 1 | `grid_atr` | 1h | Grid/DCA | Long + Short | **true** | LIVE 7x, 14 assets |
| 2 | `grid_trend` | 1h | Grid/DCA | Long + Short | false | Echoue en forward test |
| 3 | `grid_multi_tf` | 1h + 4h | Grid/DCA | Long + Short | **true** | LIVE 3x (test), 14 assets |
| 4 | `grid_funding` | 1h | Grid/DCA | Long only | false | **ABANDONNÉ** (0/17 Grade B) |
| 5 | `grid_range_atr` | 1h | Grid/DCA | Long + Short | false | WFO à lancer |
| 6 | `grid_boltrend` | 1h | Grid/DCA | Long + Short | **true** | Paper, 2 assets (mise en pause Sprint 38b) |
| 7 | `envelope_dca` | 1h | Grid/DCA | Long only | false | Remplacé par grid_atr |
| 8 | `envelope_dca_short` | 1h | Grid/DCA | Short only | false | Validation en attente |
| 9 | `boltrend` | 1h | Swing | Long + Short | false | Grade C (trop peu de trades) |
| 10 | `bollinger_mr` | 1h | Swing | Long + Short | false | Désactivé |
| 11 | `donchian_breakout` | 1h | Swing | Long + Short | false | Désactivé |
| 12 | `supertrend` | 1h | Swing | Long + Short | false | Désactivé |
| 13 | `vwap_rsi` | 5m + 15m | Scalp | Long + Short | false | Désactivé |
| 14 | `momentum` | 5m + 15m | Scalp | Long + Short | false | Désactivé |
| 15 | `funding` | 15m | Scalp | Long + Short | false | Paper only (pas de backtest) |
| 16 | `liquidation` | 5m | Scalp | Long + Short | false | Paper only (pas de backtest) |
| 17 | `grid_momentum` | 1h | Grid/DCA | Long + Short | false | **ABANDONNÉ** (1/21 Grade B) |

**Live trading actif** :
- `grid_atr` sur 14 assets — leverage 7x
- `grid_multi_tf` sur 14 assets — leverage 3x (phase test, cible 6x)

**Paper trading** :
- `grid_boltrend` sur 2 assets (BTC, DYDX) — leverage 8x (mise en pause Sprint 38b)

---

## Architecture commune

Toutes les stratégies héritent de `BaseStrategy` (mono-position) ou `BaseGridStrategy` (multi-position DCA).

### Stratégies mono-position (BaseStrategy)

Méthodes clés :
- `evaluate(ctx)` → `StrategySignal | None` : conditions d'entrée, retourne un signal avec direction, TP, SL, score
- `check_exit(ctx, position)` → `"signal_exit" | None` : sortie anticipée (appelé seulement si ni TP ni SL touchés)
- `compute_indicators(candles_by_tf)` : pré-calcul des indicateurs sur tout le dataset

### Stratégies grid/DCA (BaseGridStrategy)

`evaluate()` et `check_exit()` retournent toujours `None` (non utilisés). Le `MultiPositionEngine` utilise :
- `compute_grid(ctx, grid_state)` → `list[GridLevel]` : niveaux d'entrée à chaque bougie
- `should_close_all(ctx, grid_state)` → `"tp_global" | "sl_global" | "direction_flip" | None`
- `get_tp_price()` / `get_sl_price()` : TP/SL global dynamique

Règle fondamentale : **un seul côté actif à la fois**. Si des positions LONG sont ouvertes, pas de niveaux SHORT (et inversement).

### Sizing (equal allocation)

```
margin_par_level = capital / nb_assets / num_levels
```
Cap 25% du capital par asset. Margin guard global 70% (`max_margin_ratio` dans `risk.yaml`).

**Politique leverage** :
- Stratégies grid : leverage défini par stratégie dans `strategies.yaml` (grid_atr = 7x, grid_multi_tf = 3x, grid_boltrend = 8x)
- Stratégies non-grid : `risk.yaml` `default_leverage: 3` (fallback)
- Résultats stress test (Sprint 35) : grid_atr optimal à 4-6x (Calmar 11.2), grid_boltrend optimal à 8x (Calmar 20.4)

---

## Stratégies Grid/DCA (actives)

### 1. grid_atr — Stratégie principale (paper trading actif)

**Concept** : Mean-reversion DCA avec enveloppes adaptatives basées sur l'ATR (volatilité). Quand le prix s'éloigne de la SMA, on accumule des positions (DCA). Quand il revient à la SMA, on prend les profits.

**Fichier** : `backend/strategies/grid_atr.py`

#### Indicateurs

- **SMA** (Simple Moving Average) sur les closes, période `ma_period`
- **ATR** (Average True Range) sur H/L/C, période `atr_period`

#### Logique d'entrée (compute_grid)

Les niveaux de la grille sont calculés comme des enveloppes autour de la SMA :

```
multiplier[i] = atr_multiplier_start + i × atr_multiplier_step

LONG :  entry_price[i] = SMA - ATR × multiplier[i]
SHORT : entry_price[i] = SMA + ATR × multiplier[i]
```

**Exemple numérique** : Si `SMA = 100`, `ATR = 5`, `atr_multiplier_start = 1.5`, `atr_multiplier_step = 1.0`, `num_levels = 3` :
- Level 0 LONG : `100 - 5 × 1.5 = 92.50` (multiplier = 1.5)
- Level 1 LONG : `100 - 5 × 2.5 = 87.50` (multiplier = 2.5)
- Level 2 LONG : `100 - 5 × 3.5 = 82.50` (multiplier = 3.5)

Quand le prix touche un niveau, une position est ouverte à ce prix. Le DCA accumule : level 0 → 1 → 2 au fur et à mesure que le prix descend.

L'ATR rend les enveloppes **adaptatives** : en haute volatilité, les niveaux s'écartent ; en basse volatilité, ils se resserrent.

#### Logique de sortie (should_close_all)

- **TP** : `close >= SMA` (LONG) ou `close <= SMA` (SHORT) → `"tp_global"`. Le TP est dynamique car la SMA bouge.
- **SL** : `close <= avg_entry_price × (1 - sl_percent/100)` → `"sl_global"`. Le SL est fixe en % du prix moyen d'entrée.

**Exemple** : 3 positions LONG ouvertes à 92.50, 87.50 et 82.50. Prix moyen = 87.50. Avec `sl_percent = 20`, SL = `87.50 × 0.80 = 70.00`.

#### Paramètres clés

| Paramètre | Rôle | Valeurs WFO |
|-----------|------|-------------|
| `ma_period` | Période SMA (centre des enveloppes) | 7, 10, 14, 20 |
| `atr_period` | Période ATR (largeur des enveloppes) | 10, 14, 20 |
| `atr_multiplier_start` | Multiplicateur ATR du 1er niveau | 1.0 — 3.0 |
| `atr_multiplier_step` | Ecart entre les niveaux | 0.5, 1.0, 1.5 |
| `num_levels` | Nombre de niveaux DCA | 2, 3, 4 |
| `sl_percent` | SL en % du prix moyen | 15, 20, 25, 30 |
| `sides` | Directions autorisées | ["long"] par défaut |
| `leverage` | Levier | 7 |

**WFO** : 3240 combos, fenêtres IS=180j / OOS=60j / step=60j. Grade A/B sur 14 assets.

#### Limites

- Piégé en bear market soutenu sans recovery (le prix ne revient pas à la SMA)
- 7/21 assets avec Sharpe négatif sur les 90 derniers jours en conditions bear

#### Note sur le SL max (Sprint 38)

La grille de recherche WFO pour grid_atr est contrainte à `sl_percent ≤ 25%` (retiré 30% de la grille).

Avec les régimes de marché correctement analysés (Hotfix 37c), le WFO tend à sélectionner des SL très larges (30%) pour survivre aux crashes inclus dans les fenêtres OOS. 10/21 assets plafonnaient à SL=30%, ce qui :

- dégrade le ratio rendement/drawdown (worst-case SL loss 38.9% du capital par asset)
- rapproche le worst SL du kill switch global (45%)
- sélectionne des params "survivants" plutôt que "performants"

La contrainte à 25% est un compromis validé en portfolio backtest 365j : DD -26.4% vs -33.3% avec SL=30%, pour un rendement quasi-identique (+43.6%). Worst SL réduit de 38.9% → 31.8%.

#### Note sur le window_factor (Sprint 38b)

Le `combo_score` inclut un `window_factor = min(1.0, n_windows / max_windows)` qui pénalise les combos évaluées sur peu de fenêtres OOS. Ce facteur est critique pour le 2-pass WFO : les combos fine (générées autour du top 20 de chaque fenêtre) sont spécifiques à chaque fenêtre et n'apparaissent que dans 1-5 fenêtres sur 30. Sans window_factor, ces combos gagnaient le scoring avec une consistency triviale (1/1 = 100%). Le fix a amélioré le rendement portfolio de +43.6% → +57.7% tout en réduisant le drawdown de -26.4% → -24.1%.

---

### 2. grid_trend — Trend Following DCA

**Concept** : Inverse de grid_atr. Au lieu de trader contre la tendance (mean-reversion), grid_trend trade AVEC la tendance. Filtre directionnel EMA cross + ADX, entrée en pullback, sortie par trailing stop ATR.

**Fichier** : `backend/strategies/grid_trend.py`

#### Indicateurs

- **EMA fast** et **EMA slow** (Exponential Moving Average) — croisement = signal de direction
- **ADX** (Average Directional Index) — force du trend. ADX < seuil = zone neutre
- **ATR** — largeur des pullbacks et trailing stop

#### Logique d'entrée (compute_grid)

1. **Filtre directionnel** :
   - `EMA_fast > EMA_slow` ET `ADX > adx_threshold` → trend UP → DCA LONG
   - `EMA_fast < EMA_slow` ET `ADX > adx_threshold` → trend DOWN → DCA SHORT
   - `ADX < adx_threshold` → **zone neutre**, pas de trades

2. **Niveaux pullback** (entrées dans le sens du trend) :
```
offset[i] = ATR × (pull_start + i × pull_step)

LONG (trend UP) :  entry_price[i] = EMA_fast - offset[i]
SHORT (trend DOWN) : entry_price[i] = EMA_fast + offset[i]
```

**Exemple** : Trend UP, `EMA_fast = 100`, `ATR = 3`, `pull_start = 1.0`, `pull_step = 0.5`, `num_levels = 3` :
- Level 0 LONG : `100 - 3 × 1.0 = 97.00` (pullback léger)
- Level 1 LONG : `100 - 3 × 1.5 = 95.50`
- Level 2 LONG : `100 - 3 × 2.0 = 94.00` (pullback profond)

On achète les pullbacks dans un trend haussier, en espérant que le trend continue.

#### Logique de sortie (should_close_all)

- **Direction flip** : `EMA_fast` croise `EMA_slow` dans l'autre sens → `"direction_flip"`. Force close immédiat.
- **SL** : `close <= avg_entry_price × (1 - sl_percent/100)` → `"sl_global"`
- **Trailing stop** : géré par le fast engine (pas dans `should_close_all()`). Le trailing stop suit le prix avec un offset de `ATR × trail_mult`. HWM (High Water Mark) ne fait que monter (LONG) ou descendre (SHORT).
- **Pas de TP SMA** : contrairement à grid_atr, il n'y a pas de TP fixe. Le trailing stop laisse courir les profits.

#### Différences clés avec grid_atr

| Aspect | grid_atr | grid_trend |
|--------|----------|------------|
| Philosophie | Mean-reversion (contre le trend) | Trend following (avec le trend) |
| Centre des enveloppes | SMA | EMA fast |
| Filtre d'entrée | Aucun (toujours actif) | EMA cross + ADX |
| TP | Retour à la SMA (dynamique) | Trailing stop ATR |
| Force close | Non | Oui (EMA cross flip) |
| Zone neutre | Non | Oui (ADX < seuil) |

#### Paramètres clés

| Paramètre | Rôle | Valeurs WFO |
|-----------|------|-------------|
| `ema_fast` | Période EMA rapide | 20, 30, 40 |
| `ema_slow` | Période EMA lente | 50, 100 |
| `adx_period` | Période ADX | 14 |
| `adx_threshold` | Seuil minimum ADX pour trader | 15, 20, 25 |
| `atr_period` | Période ATR | 14 |
| `pull_start` | Multiplicateur ATR du 1er pullback | 0.5, 1.0, 1.5 |
| `pull_step` | Ecart entre les pullbacks | 0.5, 1.0 |
| `num_levels` | Nombre de niveaux | 2, 3 |
| `trail_mult` | Multiplicateur ATR du trailing stop | 1.5, 2.0, 2.5, 3.0 |
| `sl_percent` | SL en % du prix moyen | 10, 15, 20 |

**WFO** : 2592 combos, fenêtres IS=360j / OOS=90j / step=90j.

#### Statut

`enabled: false`. Le forward test 365j montre -28% (seulement 1/5 assets profitables). Le trend following DCA souffre en bear market sans trends soutenus.

---

### 3. grid_multi_tf — Supertrend 4h + Grid ATR 1h

**Concept** : Combine un filtre directionnel Supertrend sur le 4h avec l'exécution grid ATR sur le 1h. Le Supertrend décide si on trade LONG ou SHORT, le grid ATR gère les niveaux.

**Fichier** : `backend/strategies/grid_multi_tf.py`

#### Indicateurs

- **SMA + ATR** sur le 1h (identique à grid_atr)
- **Supertrend 4h** : resampleé depuis les candles 1h → OHLC 4h, puis Supertrend calculé dessus

Le resampling est **anti-lookahead** : chaque candle 1h utilise la direction du bucket 4h **précédent** (pas le courant, qui n'est pas encore clôturé). Buckets 4h = frontières UTC 00h/04h/08h/12h/16h/20h.

#### Logique d'entrée

1. Lire `st_direction` du Supertrend 4h :
   - `+1` (UP) → DCA LONG sous la SMA
   - `-1` (DOWN) → DCA SHORT au-dessus de la SMA
   - `NaN` → pas de trading

2. Calcul des niveaux : identique à grid_atr (`SMA ± ATR × multiplier`)

#### Logique de sortie

- **TP** : retour à la SMA (identique à grid_atr)
- **SL** : % fixe depuis prix moyen
- **Direction flip** : Supertrend 4h passe de UP à DOWN (ou inversement) → `"direction_flip"`, force close

#### Paramètres spécifiques

| Paramètre | Rôle |
|-----------|------|
| `st_atr_period` | Période ATR pour le Supertrend 4h |
| `st_atr_multiplier` | Multiplicateur du Supertrend 4h |

Les autres paramètres sont les mêmes que grid_atr. **WFO** : 384 combos.

**Statut** : `enabled: true`, `live_eligible: true`. LIVE 3x sur 14 assets (phase test, cible 6x).

---

### 4. grid_funding — DCA sur Funding Rate Négatif

**Concept** : Exploite les funding rates négatifs (les shorts paient les longs). Quand le funding est très négatif, c'est un signal structurel indépendant du prix : pression short excessive → opportunité LONG. Multi-niveaux : plus le funding est négatif, plus on accumule.

**Fichier** : `backend/strategies/grid_funding.py`

#### Indicateurs

- **SMA** sur les closes (pour le TP mode `sma_cross`)
- **Funding rate** : vient de `ctx.extra_data["funding_rate"]` (pas un indicateur technique classique)

#### Logique d'entrée

Contrairement aux autres grids, l'entrée n'est **pas basée sur le prix** mais sur le funding rate :

```
threshold[i] = -(funding_threshold_start + i × funding_threshold_step)

Si funding_rate <= threshold[i] → ouvrir LONG au prix courant
```

**Exemple** : `funding_threshold_start = 0.0005`, `funding_threshold_step = 0.0005`, `num_levels = 3` :
- Level 0 : funding ≤ -0.0005 (= -0.05%)
- Level 1 : funding ≤ -0.0010 (= -0.10%)
- Level 2 : funding ≤ -0.0015 (= -0.15%)

Le funding rate est en DB en pourcentage, converti en décimal (`/100`) pour la comparaison.

#### Logique de sortie

- **SL** : `close <= avg_entry × (1 - sl_percent/100)` → `"sl_global"` (toujours actif, même pendant min_hold)
- **Min hold** : les `min_hold_candles` premières bougies après la première entrée bloquent le TP (mais pas le SL)
- **TP** selon `tp_mode` :
  - `"funding_positive"` : funding rate redevient positif → `"tp_funding"`
  - `"sma_cross"` : prix >= SMA → `"tp_sma"`
  - `"funding_or_sma"` : l'un ou l'autre

#### Paramètres clés

| Paramètre | Rôle | Valeurs WFO |
|-----------|------|-------------|
| `funding_threshold_start` | Seuil funding du 1er niveau | 0.0003 — 0.001 |
| `funding_threshold_step` | Ecart entre les seuils | 0.0003 — 0.001 |
| `num_levels` | Nombre de niveaux | 2, 3 |
| `ma_period` | Période SMA pour TP | 7, 14, 21 |
| `tp_mode` | Mode de TP | funding_positive, sma_cross, funding_or_sma |
| `min_hold_candles` | Bougies minimum avant TP | 4, 8, 16 |
| `sl_percent` | SL en % | 10, 15, 20, 25 |

**WFO** : 2592 combos, fenêtres IS=360j / OOS=90j.

**Statut** : `enabled: false`, LONG-only. **ABANDONNÉ** — WFO terminé Sprint 42 (0/17 Grade B, tous Grade F). Funding extrême corrélé avec stress marché : LONG entre dans un effondrement, funding collecté (~0.03%/8h) dérisoire face au mouvement de prix (-5% à -20%).

---

### 5. envelope_dca — Mean Reversion Multi-Niveaux (LONG)

**Concept** : Ancêtre de grid_atr. Enveloppes à pourcentages fixes autour de la SMA (pas adaptatives à la volatilité). Remplacé par grid_atr qui est plus performant.

**Fichier** : `backend/strategies/envelope_dca.py`

#### Logique d'entrée

Enveloppes calculées en pourcentage de la SMA avec **asymétrie log-return** :

```
offset_lower[i] = envelope_start + i × envelope_step
offset_upper[i] = round(1 / (1 - offset_lower[i]) - 1, 3)

LONG :  entry_price[i] = SMA × (1 - offset_lower[i])
SHORT : entry_price[i] = SMA × (1 + offset_upper[i])
```

L'asymétrie garantit un aller-retour cohérent : si le prix baisse de 5%, il faut +5.26% pour revenir au même niveau (pas 5%).

**Exemple** : `SMA = 100`, `envelope_start = 0.05`, `envelope_step = 0.05`, `num_levels = 3` :
- Level 0 LONG : `100 × (1 - 0.05) = 95.00` (offset = 5%)
- Level 1 LONG : `100 × (1 - 0.10) = 90.00` (offset = 10%)
- Level 2 LONG : `100 × (1 - 0.15) = 85.00` (offset = 15%)

#### Sortie

Identique à grid_atr : TP = retour à la SMA, SL = % du prix moyen.

**Statut** : `enabled: false`, remplacé par grid_atr.

---

### 6. envelope_dca_short — Miroir SHORT

**Fichier** : `backend/strategies/envelope_dca_short.py`

Réutilise 100% du code de `envelope_dca`. Seuls le nom (`"envelope_dca_short"`) et `sides = ["short"]` changent.

**Statut** : `enabled: false`, validation WFO en attente.

---

### 7. grid_boltrend — DCA Event-Driven sur Breakout Bollinger (paper trading actif)

**Fichier** : `backend/strategies/grid_boltrend.py`

**Concept** : Hybride boltrend (signal de breakout) + grid_atr (exécution DCA multi-niveaux). La grille est OFF par défaut ; elle s'active uniquement sur un breakout Bollinger filtré par tendance long terme, puis ouvre des niveaux DCA dans la direction du breakout. TP inversé : le signal de sortie est le retour du prix sous/sur la SMA Bollinger (prix revient à la valeur fondamentale).

**Logique d'entrée** :

1. Breakout LONG : `prev_close < bb_upper` ET `close > bb_upper` ET spread > `min_bol_spread` ET `close > sma_long`
2. Breakout SHORT : `prev_close > bb_lower` ET `close < bb_lower` ET spread > `min_bol_spread` ET `close < sma_long`
3. Level 0 = prix au breakout (entre immédiatement), Level k = `close ∓ k × ATR × atr_spacing_mult`

**Logique de sortie** :

- **TP inverse** : `close < bb_sma` (LONG) ou `close > bb_sma` (SHORT) → `"signal_exit"`
- **SL global** : prix moyen ± `sl_percent` → `"sl_global"`
- Si SL et TP simultanés sur même candle : bougie verte = TP, bougie rouge = SL

**Paramètres** :

| Paramètre | Défaut | Rôle |
|-----------|--------|------|
| `bol_window` | 100 | Période Bollinger |
| `bol_std` | 2.0 | Écart-type Bollinger |
| `long_ma_window` | 200 | SMA filtre tendance |
| `min_bol_spread` | 0.0 | Spread min bandes (% price) |
| `atr_period` | 14 | Période ATR (espacement niveaux) |
| `atr_spacing_mult` | 1.0 | Multiplicateur espacement |
| `num_levels` | 3 | Nombre de niveaux DCA |
| `sl_percent` | 15.0 | SL global (%) |
| `sides` | [long, short] | Côtés actifs |

**WFO** : 1296 combos (2×3×2×2×2×3×3×3), fenêtres 180j IS / 60j OOS.

**Différence vs boltrend** : boltrend = mono-position (4-6 trades/OOS → DSR 0.00 → Grade C max). grid_boltrend = 3-4× plus de trades par grille DCA → DSR > 0 → Grade B possible.

**Différence vs grid_atr** : grid_atr = grille toujours active (mean reversion). grid_boltrend = grille activée uniquement sur signal directionnel (trend following DCA).

**Statut** : `enabled: true`, `live_eligible: false`. Paper, 2 assets (BTC, DYDX) avec leverage 8x.

**Résultats WFO** : Grade B (83/100), Sharpe +1.58 (post-Hotfix 33a). Bugs corrigés : TP inverse via `get_tp_price()` retournant NaN (Hotfix 33a), exit_price/fees fast engine (Hotfix 33b), divergence fast/event réduite à 2.62%.

**Statut Sprint 38b — Mise en pause** : après correction du biais de sélection (`window_factor`), seuls 2 assets sur 21 atteignent Grade B (BTC, DYDX). Les 7 Grade B antérieurs avaient une consistency apparente de 100% sur 3-5 fenêtres (biais trivial), tombée à 62-77% réels sur 30 fenêtres. DSR = 0/15 sur tous les assets (trop peu de trades par OOS window). Portfolio 2 assets non viable (+7.4%, DD -38.6%). Stratégie mise en pause — à revisiter quand plus de données disponibles.

---

### 8. grid_momentum — Breakout Donchian + DCA Pullback + Trailing Stop ATR

**Fichier** : `backend/strategies/grid_momentum.py`

**Concept** : Stratégie breakout/trend-following à profil de payoff **convexe** (petites pertes sur faux breakouts, gros gains sur vrais trends). Intentionnellement décorrélée de grid_atr (profil concave, mean-reversion).

**Signal d'activation** :
- Breakout Donchian : `close > donchian_high` (LONG) ou `close < donchian_low` (SHORT)
- Filtre volume : `volume > volume_sma × vol_multiplier`
- Filtre ADX optionnel : `adx > adx_threshold` (si threshold > 0, désactivé par défaut)

**Exécution** : Grid DCA pullback depuis le prix du breakout.
- Level 0 = prix du breakout (trigger immédiat)
- Level k (k≥1) = prix ∓ ATR × (pullback_start + (k-1) × pullback_step)

**Sorties** :
- **Trailing stop ATR** : `close < HWM - ATR × trailing_atr_mult` (LONG) → `"trail_stop"`
- **Direction flip** : breakout inverse détecté SANS filtre volume/ADX (sortie de protection) → `"direction_flip"`
- **SL global** : `avg_entry ± sl_percent%` (géré par `get_sl_price()` + OHLC heuristic du runner)
- **Pas de TP fixe** : `get_tp_price()` retourne `float("nan")`

**HWM tracking** :
- Fast engine : variable locale dans `_simulate_grid_momentum()`
- Live : tracké dans `GridStrategyRunner._hwm` (dict par symbol), injecté via `indicators["hwm"]`

**Paramètres** :

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `donchian_period` | 30 | Lookback Donchian (rolling max/min) |
| `vol_sma_period` | 20 | Période SMA volume |
| `vol_multiplier` | 1.5 | Multiplicateur volume pour breakout |
| `adx_period` | 14 | Période ADX |
| `adx_threshold` | 0.0 | Seuil ADX (0 = désactivé) |
| `atr_period` | 14 | Période ATR |
| `pullback_start` | 1.0 | Offset ATR du 1er niveau DCA |
| `pullback_step` | 0.5 | Incrément ATR entre niveaux DCA |
| `num_levels` | 3 | Nombre de niveaux de la grille |
| `trailing_atr_mult` | 3.0 | Multiplicateur ATR pour trailing stop |
| `sl_percent` | 15.0 | Stop-loss global (%) |
| `cooldown_candles` | 3 | Bougies d'attente entre trades |
| `leverage` | 6 | Levier |

**Décorrélation** :
- grid_atr / grid_multi_tf = profil concave (mean-reversion, 84% du temps en range)
- grid_momentum = profil convexe (breakout, capture les 16% de trends directionnels)
- grid_trend (échoué) : EMA trop lent, pas de volume filter. grid_momentum utilise Donchian + volume spike = signal plus réactif

**Statut** : `enabled: false`, `live_eligible: false`. **ABANDONNÉ** — WFO terminé Sprint 41 (1 Grade B / 21 assets, NO-GO). Faux breakouts trop fréquents en crypto (83% range), profil convexe insuffisant pour compenser.

---

### 9. grid_range_atr — Range Trading Bidirectionnel

**Fichier** : `backend/strategies/grid_range_atr.py`

Range trading LONG + SHORT simultanés, TP/SL individuels par position, spacing ATR.

**Statut** : `enabled: false`, WFO à lancer.

---

## Stratégies Scalp 5 minutes

### 7. vwap_rsi — Mean Reversion VWAP + RSI

**Concept** : Trade contre le mouvement quand le prix s'éloigne du VWAP avec un RSI extrême et un spike de volume, en range (pas en tendance).

**Fichier** : `backend/strategies/vwap_rsi.py`

#### Indicateurs (5m)

RSI, VWAP rolling 24h, ADX + DI+/DI-, ATR, Volume SMA

#### Filtre 15m (anti-trend)

- ADX 15m > `trend_adx_threshold` → marché en tendance → **pas de trade** (mean-reversion = range only)
- Régime 5m doit être `RANGING` ou `LOW_VOLATILITY`

#### Conditions d'entrée

**LONG** (toutes requises) :
1. `RSI < rsi_long_threshold` (défaut 30) — survendu
2. `vwap_deviation < -vwap_deviation_entry` — prix sous le VWAP
3. Volume > `volume_sma × volume_spike_multiplier` — spike de volume
4. 15m pas bearish (DI- < DI+)

**SHORT** : conditions symétriques (RSI > 70, prix au-dessus du VWAP, 15m pas bullish)

#### Sortie anticipée

RSI revient > 50 (LONG) ou < 50 (SHORT) ET trade en profit → `"signal_exit"`

#### TP / SL

Pourcentage fixe : `entry × (1 ± tp_percent/100)`, `entry × (1 ∓ sl_percent/100)`

**Statut** : `enabled: false`.

---

### 8. momentum — Breakout avec Volume

**Concept** : Inverse de vwap_rsi. Trade AVEC la tendance quand le prix casse le range des N dernières bougies avec confirmation de volume et tendance 15m.

**Fichier** : `backend/strategies/momentum.py`

#### Conditions d'entrée

**LONG** (toutes requises) :
1. `close > rolling_high` (plus haut des `breakout_lookback` dernières bougies)
2. ADX 15m ≥ 25 (tendance confirmée) + DI+ > DI- (bullish)
3. Volume > `volume_sma × volume_confirmation_multiplier`

**SHORT** : `close < rolling_low`, DI- > DI+ (bearish)

#### TP / SL

Basé sur l'ATR : `TP = ATR × atr_multiplier_tp`, `SL = ATR × atr_multiplier_sl`, cappé par les % config.

#### Sortie anticipée

ADX 5m chute sous 20 → momentum essoufflé → `"signal_exit"`

**Statut** : `enabled: false`.

---

### 9. funding — Arbitrage Funding Rate (Scalp)

**Concept** : Trade les extrêmes de funding rate. Funding très négatif → LONG (shorts paient). Funding très positif → SHORT (longs paient).

**Fichier** : `backend/strategies/funding.py`

#### Logique

1. `funding_rate < extreme_negative_threshold` (-0.03%) → signal LONG
2. `funding_rate > extreme_positive_threshold` (+0.03%) → signal SHORT
3. **Entry delay** : attendre `entry_delay_minutes` après la première détection (éviter les faux signaux)

#### Sortie anticipée

Funding revient à neutre (`|funding_rate| < 0.01%`) → `"signal_exit"`

**Note** : Pas de backtest possible (pas d'historique funding en DB). Paper trading uniquement.

**Statut** : `enabled: false`.

---

### 10. liquidation — Zones de Liquidation

**Concept** : Estime les zones de liquidation via l'open interest et le levier moyen estimé. Trade quand le prix approche une zone de cascade.

**Fichier** : `backend/strategies/liquidation.py`

#### Logique

1. Estimation des zones :
   - `liq_long_zone = prix × (1 - 1/leverage_estimate)` — où les longs se font liquider
   - `liq_short_zone = prix × (1 + 1/leverage_estimate)` — où les shorts se font liquider
2. `oi_change > oi_change_threshold` — les leviers sont chargés
3. Prix approche une zone (distance < `zone_buffer_percent`) :
   - Proche zone shorts → anticipation short squeeze → **LONG**
   - Proche zone longs → anticipation cascade → **SHORT**

#### Sortie anticipée

OI chute de plus de 3% → cascade terminée → `"signal_exit"`

**Note** : Paper trading uniquement (pas d'historique OI pour backtest).

**Statut** : `enabled: false`.

---

## Stratégies Swing 1 heure

### 11. bollinger_mr — Bollinger Band Mean Reversion

**Concept** : Trade les extrêmes des bandes de Bollinger avec TP dynamique au retour à la SMA.

**Fichier** : `backend/strategies/bollinger_mr.py`

#### Indicateurs

Bandes de Bollinger : `SMA(bb_period) ± bb_std × écart-type`

#### Conditions d'entrée

- **LONG** : `close < bande basse`
- **SHORT** : `close > bande haute`

#### Sortie

- **TP dynamique** (dans `check_exit()`) : close croise la SMA → `"signal_exit"`. Le TP "fixe" est mis très loin (`entry × 2.0` ou `entry × 0.5`) pour ne jamais être touché — c'est `check_exit()` qui gère le vrai TP.
- **SL** : % fixe depuis l'entrée

**Statut** : `enabled: false`. WFO config IS=180j / OOS=60j.

---

### 12. donchian_breakout — Canal Donchian

**Concept** : Trade les cassures du canal Donchian (plus haut/bas des N dernières bougies). TP et SL basés sur des multiples d'ATR.

**Fichier** : `backend/strategies/donchian_breakout.py`

#### Conditions d'entrée

- **LONG** : `close > rolling_high` (plus haut des `entry_lookback` dernières bougies, candle courante exclue)
- **SHORT** : `close < rolling_low`

#### TP / SL

```
TP = entry ± ATR × atr_tp_multiple
SL = entry ∓ ATR × atr_sl_multiple
```

Pas de sortie anticipée (`check_exit()` retourne toujours `None`).

**Statut** : `enabled: false`.

---

### 13. supertrend — SuperTrend Flip

**Concept** : Trade les retournements de direction du SuperTrend. Entrée uniquement sur les flips (changement de direction).

**Fichier** : `backend/strategies/supertrend.py`

#### Indicateurs

SuperTrend calculé à partir de l'ATR : bande supérieure/inférieure adaptative. Direction = +1 (UP) ou -1 (DOWN).

#### Conditions d'entrée

- **LONG** : direction précédente = -1 (DOWN), direction courante = +1 (UP) → flip haussier
- **SHORT** : direction précédente = +1 (UP), direction courante = -1 (DOWN) → flip baissier

Pas de trade si la direction n'a pas changé.

#### TP / SL

Pourcentages fixes (`tp_percent`, `sl_percent`). Pas de sortie anticipée.

**Statut** : `enabled: false`.

---

### 14. boltrend — Bollinger Trend Following

**Fichier** : `backend/strategies/boltrend.py`

**Concept** : Version mono-position de la logique grid_boltrend. Trade les breakouts des bandes de Bollinger filtrés par tendance long terme (SMA). Sortie dynamique au retour à la SMA Bollinger (TP inverse : close < SMA = exit LONG).

**Logique d'entrée** :

- LONG : `prev_close < bb_upper` ET `close > bb_upper` ET spread > `min_bol_spread` ET `close > sma_long`
- SHORT : `prev_close > bb_lower` ET `close < bb_lower` ET spread > `min_bol_spread` ET `close < sma_long`

**Logique de sortie** :

- **TP inverse** : close croise la SMA Bollinger (retour au centre) → `"signal_exit"`
- **SL** : % fixe depuis l'entrée (filet de sécurité)

**Paramètres** :

| Paramètre | Défaut | Rôle |
|-----------|--------|------|
| `bol_window` | 100 | Période Bollinger |
| `bol_std` | 2.0 | Écart-type Bollinger |
| `long_ma_window` | 200 | SMA filtre tendance |
| `min_bol_spread` | 0.0 | Spread min bandes (% price) |
| `sl_percent` | 5.0 | SL fixe (%) |

**Statut** : `enabled: false`. Grade C (trop peu de trades par fenêtre OOS → DSR 0.00). La version DCA `grid_boltrend` résout ce problème.

---

## Comment ajouter une nouvelle stratégie

### Checklist (11 étapes)

1. Créer `backend/strategies/my_strategy.py` (hérite `BaseGridStrategy` ou `BaseStrategy`)
2. Ajouter la config Pydantic dans `backend/core/config.py`
3. Enregistrer dans `backend/strategies/factory.py`
4. Ajouter dans `backend/optimization/__init__.py` (`STRATEGY_REGISTRY` + `GRID_STRATEGIES`)
5. Ajouter le fast engine dans `backend/optimization/fast_multi_backtest.py`
6. Ajouter les indicateurs dans `backend/optimization/indicator_cache.py` si nécessaire
7. Ajouter `_INDICATOR_PARAMS` dans `backend/optimization/walk_forward.py`
8. Config YAML : `config/strategies.yaml` + `config/param_grids.yaml`
9. Tests : signaux, fast engine parité, WFO integration
10. WFO : `uv run python -m scripts.optimize --strategy my_strategy --all-symbols`
11. Deep Analysis : `uv run python -m scripts.analyze_wfo_deep --strategy my_strategy` — critère ≥ 5 VIABLE/BORDERLINE
12. Apply (VIABLE + BORDERLINE seulement) : voir [WORKFLOW_WFO.md § Étape 1c](WORKFLOW_WFO.md)

### Pattern de code — Stratégie Grid

```python
from backend.strategies.base_grid import BaseGridStrategy, GridLevel, GridState

class MyGridStrategy(BaseGridStrategy):
    name = "my_grid"

    def compute_grid(self, ctx, grid_state) -> list[GridLevel]:
        # Calculer les niveaux d'entrée
        # Un seul côté actif à la fois
        ...

    def should_close_all(self, ctx, grid_state) -> str | None:
        # "tp_global", "sl_global", "direction_flip", ou None
        ...

    def get_tp_price(self, grid_state, indicators) -> float:
        ...

    def get_sl_price(self, grid_state, indicators) -> float:
        ...

    @property
    def max_positions(self) -> int:
        return self._config.num_levels
```

### Pattern de code — Stratégie Mono-position

```python
from backend.strategies.base import BaseStrategy, StrategySignal

class MyStrategy(BaseStrategy):
    name = "my_strat"

    def evaluate(self, ctx) -> StrategySignal | None:
        # Conditions d'entrée → StrategySignal avec direction, TP, SL, score
        ...

    def check_exit(self, ctx, position) -> str | None:
        # Sortie anticipée → "signal_exit" ou None
        ...
```

---

## Validation Workflow

> **Workflow complet (pipeline nouvelle stratégie / re-validation)** : voir [WORKFLOW_WFO.md](WORKFLOW_WFO.md).
>
> Les sous-workflows B et C ci-dessous sont des variantes allégées pour des changements ponctuels.

### B. Nouveau paramètre / feature (ex: max_hold_candles)

Workflow A/B test — on isole l'impact d'une seule variable :

```text
1. Implémentation
   └─ Code + tests (backward compat avec défaut = désactivé)

2. WFO A/B mono-coin
   └─ Run A : WFO existant (baseline, paramètre désactivé)
   └─ Run B : WFO avec params_override={paramètre: [valeur_test]}
   └─ Comparer par asset : Sharpe OOS, Max DD, Win Rate, Consistance
   └─ Critère : Amélioration Sharpe sur ≥ 60% des assets

3. Si concluant → WFO avec plusieurs valeurs
   └─ params_override={paramètre: [val1, val2, val3]}
   └─ Trouver la valeur optimale par asset
   └─ Appliquer via per_asset dans strategies.yaml

4. Portfolio backtest comparatif
   └─ Run baseline (params actuels) vs Run avec nouveau paramètre
   └─ Critère : Sharpe portfolio amélioré, Max DD pas dégradé

5. Paper trading validation (même workflow que A.5)

6. Live deployment (même workflow que A.6)
```

### C. Changement de leverage ou risk params

```text
1. Recalculer la fourchette théorique (Étape 0 de Workflow A)
   └─ Vérifier que le nouveau leverage est dans la fourchette

2. Stress test leverage multi-fenêtre
   └─ uv run python -m scripts.stress_test_leverage --leverages <ancien>,<nouveau>
   └─ Critères : Liq > 50%, DD < -40%, KS@45 = 0, W-SL < kill_switch - 5%

3. Si le leverage change de ±2x par rapport à l'actuel :
   └─ Re-WFO obligatoire (le best combo peut changer)
   └─ Sinon : portfolio backtest seul suffit

4. Portfolio backtest au leverage choisi
   └─ uv run python -m scripts.portfolio_backtest --leverage <N>

5. Paper trading validation
```

### Règles générales

> Voir [WORKFLOW_WFO.md § 10 — Règles d'or](WORKFLOW_WFO.md#10-règles-dor) pour les règles complètes.

- **Backward compat** : tout nouveau paramètre a un défaut qui préserve le comportement existant

---

## Glossaire

| Terme | Définition |
|-------|-----------|
| **DCA** | Dollar-Cost Averaging — accumuler des positions à différents prix pour moyenner le prix d'entrée |
| **Grid** | Grille de niveaux de prix prédéfinis. Quand le prix touche un niveau, on entre en position |
| **Mean-reversion** | Pari que le prix reviendra vers sa moyenne (SMA). On achète quand il s'éloigne, on vend quand il revient |
| **Trend following** | Pari que la tendance va continuer. On entre dans le sens du trend sur les pullbacks |
| **Envelope** | Bande de prix autour d'une moyenne mobile. Définit les niveaux d'entrée |
| **Pull / Pullback** | Recul temporaire du prix dans le sens opposé au trend. Opportunité d'entrée en trend following |
| **Level** | Un niveau individuel dans la grille. Level 0 = le plus proche de la moyenne, Level N = le plus éloigné |
| **Force-close** | Fermeture immédiate de toutes les positions (ex: flip de direction EMA ou Supertrend) |
| **Kill switch** | Mécanisme de protection : arrêt automatique si perte > X% du capital |
| **WFO** | Walk-Forward Optimization — optimisation des paramètres sur fenêtres glissantes IS/OOS |
| **IS / OOS** | In-Sample (entraînement) / Out-of-Sample (validation). L'OOS simule des conditions futures |
| **ATR** | Average True Range — mesure de volatilité basée sur les highs/lows/closes |
| **SMA** | Simple Moving Average — moyenne arithmétique des N derniers closes |
| **EMA** | Exponential Moving Average — moyenne pondérée donnant plus de poids aux données récentes |
| **ADX** | Average Directional Index — mesure la force du trend (pas sa direction). > 25 = trend fort |
| **RSI** | Relative Strength Index — oscillateur 0-100. < 30 = survendu, > 70 = suracheté |
| **VWAP** | Volume Weighted Average Price — prix moyen pondéré par le volume |
| **SuperTrend** | Indicateur de tendance basé sur ATR. Direction +1 (UP) ou -1 (DOWN) |
| **Funding rate** | Taux de financement périodique (8h) sur les futures perpétuels. Négatif = shorts paient les longs |
| **Trailing stop** | SL dynamique qui suit le prix. Monte avec le prix (LONG), descend (SHORT). Ne recule jamais |
| **HWM** | High Water Mark — plus haut atteint par le prix depuis l'entrée (pour le trailing stop) |
| **Margin guard** | Limite de marge totale utilisée (70% par défaut) pour éviter la liquidation |

---

## Stratégies abandonnées

| Stratégie | Sprint | Résultat WFO | Raison échec |
|-----------|--------|--------------|--------------|
| `grid_momentum` | 41 | 1/21 Grade B (CRV) | Faux breakouts, crypto = 83% range |
| `grid_funding` | 42 | 0/17 Grade B (tous F) | Funding extrême = stress marché = prix contre nous |
| `grid_boltrend` | 38b | DSR 0/15 | Non viable (pré-corrections 40a, jamais re-testé) |
| `grid_trend` | 20 | Échoue forward test | Trends crypto trop courts, bear market 2025 |
| `vwap_rsi` | 9 | Grade F | Pas d'edge sans DCA |
| `momentum` | 9 | Grade F | Pas d'edge sans DCA |
| `bollinger_mr` | 9 | Grade F | Pas d'edge sans DCA |
| `donchian_breakout` | 9 | Grade F | Pas d'edge sans DCA |
| `supertrend` | 9 | Grade F | Pas d'edge sans DCA |
| `funding` (mono) | 9 | 0 trades | Données live requises |
| `liquidation` (mono) | 9 | 0 trades | Données live requises |

---

## Prochaines stratégies — Ordre de priorité

1. **Grid adaptatif** — ATR multiplier dynamique sur `grid_atr` (Workflow B, A/B test)
   - Moduler la largeur de grille selon le régime de volatilité
   - Objectif : améliorer résilience crash, pas décorrélation

2. **Pairs trading** — Spread mean-reversion ETH/BTC (Phase 3, refonte archi)
   - Market-neutral par construction (fonctionne tous régimes)
   - Refonte architecturale nécessaire (2 positions synchronisées)

---

## Insight clé

L'edge en crypto vient du **mécanisme DCA mean-reversion** en régime RANGE (83% du temps).
Toutes les tentatives de capter d'autres régimes (breakout, trend, funding) ont échoué.
Les deux seules stratégies viables (`grid_atr`, `grid_multi_tf`) partagent ce mécanisme.
