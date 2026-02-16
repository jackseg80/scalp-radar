# Hotfix 20e — Kill switch grid + Warm-up fixes

## Contexte

Après un restart Docker en production, 5 bugs critiques découverts :
- Kill switch se redéclenche immédiatement au restart (seuils inadaptés + warm-up qui trade)
- Le state sauvegardé est écrasé par le warm-up bloqué
- Le warm-up génère des trades phantom sur des bougies historiques (-1 409$ fictifs, 183 trades)
- Config prod patchée temporairement à 99% pour contourner

## Analyse pré-implémentation

| # | Bug | Statut |
|---|-----|--------|
| 1 | Positions grid pas sérialisées dans le state | **Déjà fait** (state_manager.py:92-107 + simulator.py:644-669) |
| 2 | StateManager écrase le bon état quand warm-up bloqué par kill switch | **Fixé** |
| 3 | Kill switch se redéclenche immédiatement après restart | **Fixé** (grace period) |
| 4 | Seuil kill switch session 5% incompatible avec grid | **Fixé** (seuils grid 25%) |
| 5 | Warm-up génère des trades phantom | **Fixé** (guard temporel) |
| 6 | Spam Telegram (102 messages identiques) | **Déjà fait** (Hotfix 20d) |

## Changements

### 1. config.py — KillSwitchConfig
- Ajout `grid_max_session_loss_percent: Optional[float] = None`
- Ajout `grid_max_daily_loss_percent: Optional[float] = None`
- Backward compatible (None = fallback sur seuils standard)

### 2. risk.yaml
- `grid_max_session_loss_percent: 25.0`
- `grid_max_daily_loss_percent: 25.0`
- `max_session_loss_percent: 5.0` (restauré, pour mono)
- `max_daily_loss_percent: 10.0` (restauré, pour mono)

### 3. simulator.py — GridStrategyRunner

**Bug 2 — _end_warmup() forcé sous kill switch global :**
Dans `Simulator.start()`, après restauration du kill switch global, forcer `_end_warmup()` sur tous les grid runners en warm-up. Garantit que le pending_restore (capital, trades, positions) est appliqué avant que le StateManager ne sauvegarde.

**Bug 3 — Grace period post-warmup :**
- Compteur `_candles_since_warmup` incrémenté à chaque bougie live
- `_grace_period_candles = 10` (10 bougies 1h = ~10h)
- Kill switch runner skippé pendant la grace period
- Kill switch global (30%) reste toujours actif (dernière ligne de défense)

**Bug 4 — Kill switch grid re-enabled avec seuils adaptés :**
- `_record_trade()` : kill switch réactivé (était commenté "disabled for grid")
- Utilise `grid_max_session_loss_percent` (25%) au lieu de `max_session_loss_percent` (5%)
- Fallback MagicMock-safe avec `isinstance(value, (int, float))`
- Basé sur `_realized_pnl` (pas `capital - initial_capital` qui inclut la marge)

**Bug 5 — Guard anti-phantom trades :**
- Après `_end_warmup()`, pendant 5 minutes (`real_elapsed < 300`), les bougies > 2h sont skippées
- Protège contre le batch initial de `watch_ohlcv()` qui envoie des candles historiques
- Ne s'applique que si `_warmup_ended_at` existe (pas d'impact sur tests existants)
- Le compteur grace period ne s'incrémente que pour les bougies live

## Tests (22 nouveaux)

- **5 tests** sérialisation grid positions (roundtrip, backward compat, skip empty)
- **2 tests** warm-up forcé sous kill switch global
- **3 tests** grace period (bloque/autorise kill switch runner, global non affecté)
- **4 tests** seuils grid vs mono (grid 25%, mono 5%, fallback)
- **3 tests** guard phantom trades (old skipped, live OK, counter)
- **3 tests** anti-spam Telegram (cooldown, types différents, expiration)
- **2 tests** KillSwitchConfig Pydantic

## Résultat

847 tests passants (825 + 22 nouveaux), 0 régression.
