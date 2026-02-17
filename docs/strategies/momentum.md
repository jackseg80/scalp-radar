# momentum — Momentum Breakout

[Retour à l'index](INDEX.md)

## Identité

| Champ | Valeur |
|-------|--------|
| Nom technique | `momentum` |
| Catégorie | Scalp mono-position |
| Timeframe | 5m (principal) + 15m (filtre) |
| Sprint d'origine | Sprint 3 |
| Grade actuel | F |
| Statut | **Désactivé** |
| Fichier source | `backend/strategies/momentum.py` |
| Config class | `MomentumConfig` (`backend/core/config.py:92`) |

## Description

Trade AVEC la tendance (complémentaire à vwap_rsi). Le prix casse le max/min des N dernières bougies avec un volume élevé et un ADX 15m confirmant la tendance. Stratégie de breakout/continuation.

**Régime ciblé** : Marchés en tendance (ADX 15m >= 25). Inverse de vwap_rsi.

## Logique d'entrée

Méthode : `evaluate(ctx) -> StrategySignal | None`

### Filtre préalable

- ADX 15m >= 25 obligatoire (on veut de la tendance)

### Conditions LONG

- `close > rolling_high(breakout_lookback)` — cassure du plus haut des N bougies
- 15m bullish : DI+ > DI-
- Volume spike : `volume > volume_sma × volume_confirmation_multiplier` (défaut 2.0)

### Conditions SHORT

- `close < rolling_low(breakout_lookback)` — cassure du plus bas
- 15m bearish : DI- > DI+
- Volume spike identique

### Score composé

- Breakout score : 40% (base 0.6)
- Volume score : 30%
- Trend score : 30% (ADX 15m / 40, plafonné à 1.0)

## Logique de sortie

**TP** : `min(ATR × atr_multiplier_tp, close × tp_percent / 100)` — défaut min(ATR×2.0, 0.6%)

**SL** : `min(ATR × atr_multiplier_sl, close × sl_percent / 100)` — défaut min(ATR×1.0, 0.3%)

Le TP/SL est hybride ATR + cap par %, prenant le minimum des deux.

**Sortie anticipée** (`check_exit()`) :
- `ADX 5m < 20` → momentum essoufflé → `"signal_exit"`

## Paramètres

| Paramètre | Type | Défaut | Contraintes | WFO Range | Description |
|-----------|------|--------|-------------|-----------|-------------|
| `breakout_lookback` | int | 20 | >= 2 | [15, 20, 30, 40] | Bougies pour rolling high/low |
| `volume_confirmation_multiplier` | float | 2.0 | > 0 | [1.5, 2.0, 2.5] | Multiplicateur volume spike |
| `atr_multiplier_tp` | float | 2.0 | > 0 | [1.5, 2.0, 2.5] | Multiplicateur ATR pour TP |
| `atr_multiplier_sl` | float | 1.0 | > 0 | [0.5, 1.0, 1.5] | Multiplicateur ATR pour SL |
| `tp_percent` | float | 0.6 | > 0 | [0.4, 0.6, 0.8, 1.0] | Cap TP (%) |
| `sl_percent` | float | 0.3 | > 0 | [0.2, 0.3, 0.5] | Cap SL (%) |
| `timeframe` | str | "5m" | — | — | Timeframe principal |
| `trend_filter_timeframe` | str | "15m" | — | — | Timeframe filtre |

**Config WFO** : IS = 120j, OOS = 30j, step = 30j, **1 296 combinaisons**.

## Indicateurs utilisés

| Indicateur | Timeframe | Rôle |
|------------|-----------|------|
| Rolling High/Low | 5m | Signal breakout |
| ATR + ATR SMA(20) | 5m | TP/SL adaptatif |
| ADX + DI+/DI- | 5m, 15m | Filtre tendance + sortie anticipée |
| Volume SMA(20) | 5m | Confirmation volume |
| Régime de marché | 5m | Information (pas de filtre) |

## Résultats

**Grade F** sur tous les assets testés. Aucun edge démontré en WFO.

### Points forts

- TP/SL hybride ATR + cap % — bon concept
- Complémentaire avec vwap_rsi (tendance vs range)

### Points faibles

- Les breakouts en crypto sont souvent des faux breakouts suivis de mèches
- TP/SL trop serrés (0.6% / 0.3%) pour survivre au bruit
- Stratégie mono-position sans avantage structurel

## Remarques

- Inverse exact de vwap_rsi : momentum requiert ADX >= 25, vwap_rsi rejette ADX > 25
- La combinaison vwap_rsi + momentum devait couvrir range + tendance, mais aucune des deux n'a d'edge
- Le `breakout_lookback` affecte `compute_indicators()` (pas seulement `evaluate()`) — paramètre qui influe sur le pré-calcul
