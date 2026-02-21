# Sprint 36a — ACTIVE_STRATEGIES + Circuit Breaker Runner Isolation

## Contexte

Grid_atr (10 assets, LIVE) et grid_boltrend (6 assets, PAPER) tournent en parallèle dans le même container Docker. Avant le multi-executor (Sprint 36b), on durcit l'isolation des runners.

**Résultat** : 17 tests nouveaux (1624 total), 0 régression.

## Étape 1 — ACTIVE_STRATEGIES env var

- `SecretsConfig.active_strategies: str | None` — env var comma-separated
- `AppConfig.active_strategies` property — parse, retourne list[str] (vide = toutes)
- `get_enabled_strategies()` filtre APRÈS construction de la liste
- `.env.example` documenté

## Étape 2 — Circuit Breaker Runner

- Attributs `_crash_times`, `_circuit_breaker_open`, MAX=3, WINDOW=600s
- `_record_crash()` : enregistre, purge fenêtre, déclenche si seuil
- `on_candle()` refactoré : guard + try/except → `_on_candle_inner()`
- `get_status()` enrichi : `circuit_breaker`, `crash_count`
- `_dispatch_candle()` : alerte Telegram CIRCUIT_BREAKER (cooldown 1h)
- Frontend : badge DISABLED clignotant dans Scanner.jsx

## Fichiers modifiés/créés

| Fichier | Action |
|---------|--------|
| backend/core/config.py | +env var, +property |
| backend/strategies/factory.py | +filtre ACTIVE_STRATEGIES |
| backend/backtesting/simulator.py | +circuit breaker Grid+Live runners |
| backend/alerts/notifier.py | +AnomalyType.CIRCUIT_BREAKER |
| frontend/src/components/Scanner.jsx | +badge DISABLED |
| frontend/src/styles.css | +styles circuit-breaker |
| .env.example | +doc ACTIVE_STRATEGIES |
| tests/test_config.py | +6 tests |
| tests/test_circuit_breaker.py | +11 tests |
| docs/ROADMAP.md | +sprint entry, état actuel |
