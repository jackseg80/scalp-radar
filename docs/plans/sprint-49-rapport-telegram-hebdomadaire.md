# Sprint 49 — Rapport Telegram Hebdomadaire

## Contexte
Le système envoie des alertes Telegram par trade (ouverture/fermeture) et un heartbeat horaire.
On ajoute un rapport hebdomadaire automatique résumant la performance de toutes les stratégies actives,
envoyé chaque lundi à 08:00 UTC et disponible en CLI `--dry-run`.

## Fichiers à créer/modifier

### 1. Créer `backend/alerts/weekly_reporter.py` — Logique principale

Classe `WeeklyReporter` suivant le pattern `Heartbeat` :
- `__init__(telegram, db, config)` — stocke les dépendances
- `async start()` → `asyncio.create_task(_loop())`
- `async _loop()` — calcule next Monday 08:00 UTC (pattern CandleUpdater), sleep, génère, envoie
- `async stop()` — cancel proprement
- `async generate_report(db, config) -> str` — **fonction statique** réutilisable par le script CLI

**Collecte de données (fonctions DB existantes réutilisées) :**

| Métrique | Source LIVE | Source PAPER |
|----------|-------------|--------------|
| P&L semaine + trades + WR | `db.get_live_stats(period="7d", strategy=name)` | SQL direct sur `simulation_trades` WHERE `strategy_name=? AND exit_time >= ?` |
| P&L total | `db.get_daily_pnl_summary(strategy=name).total_pnl` | SQL SUM(net_pnl) sur `simulation_trades` |
| Top/Worst assets | `db.get_live_per_asset_stats(period="7d", strategy=name)` | SQL GROUP BY symbol |
| Balance | Dernier `balance_snapshots` par stratégie | N/A (paper) |
| Max DD | `db.get_max_drawdown_from_snapshots(strategy=name, period="7d")` | N/A |
| Uptime | Comptage `balance_snapshots` des 7 derniers jours vs 168 attendus | Omis si pas de snapshots |

**Classification stratégies** — helper `_classify_strategies(config)` :
- Itère `config.strategies.model_fields` (même logique que `_get_live_eligible_strategies` dans server.py)
- Retourne `(live_list, paper_list)` selon `enabled` + `live_eligible`

**Format du message** — texte brut avec emoji Unicode, envoyé en HTML parse_mode (pas de balises HTML) :
```
📊 SCALP-RADAR — Rapport Hebdo ({date_debut} - {date_fin})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 GLOBAL
Solde total     : {balance} USDT
P&L Semaine     : {pnl_week}$ ({pnl_week_pct}%)
P&L Total       : {pnl_total}$
Trades          : {trades} (WR {wr}%)
⚡ {STRATEGY} ({balance}$, x{leverage})
...
👁️ {PAPER_STRATEGY} (paper)
...
⚙️ Uptime : {uptime}%
```

### 2. Créer `scripts/weekly_report.py` — CLI entry point

Pattern standard scripts/ : `argparse` + `asyncio.run()` + `get_config()` + `Database()`
- `--dry-run` : affiche le rapport dans le terminal, ne l'envoie pas
- Sans flag : crée `TelegramClient` depuis config.secrets et envoie
- Import : `from backend.alerts.weekly_reporter import generate_report`

```bash
uv run python -m scripts.weekly_report --dry-run   # aperçu terminal
uv run python -m scripts.weekly_report              # envoi Telegram
```

### 3. Modifier `backend/api/server.py` — Scheduling automatique

Dans le lifespan, après le démarrage du heartbeat (ligne ~249) :
```python
weekly_reporter = None
if telegram:
    from backend.alerts.weekly_reporter import WeeklyReporter
    weekly_reporter = WeeklyReporter(telegram, db, config)
    await weekly_reporter.start()
```

Dans le shutdown (avant heartbeat.stop) :
```python
if weekly_reporter:
    await weekly_reporter.stop()
```

### 4. Créer `tests/test_weekly_report.py` — 5 tests

Tous les tests mockent la DB pour éviter toute dépendance SQLite :

1. **test_weekly_report_format** — mock DB retourne des stats normales → vérifie que le message contient les sections attendues (GLOBAL, nom stratégie, P&L)
2. **test_weekly_report_no_trades** — mock DB retourne 0 trades → vérifie `P&L Semaine : +0.00$`, `Trades : 0`
3. **test_weekly_report_multiple_strategies** — 2 stratégies (1 live, 1 paper) → sections séparées avec icônes ⚡ et 👁️
4. **test_weekly_report_dry_run** — test que `generate_report()` retourne un str sans appeler Telegram
5. **test_weekly_report_top_worst_assets** — mock per-asset stats avec 3 assets → vérifie Top/Worst correctement extraits

### 5. Modifier `COMMANDS.md` — Ajouter section

Après la dernière section, ajouter :
```markdown
### Rapport Telegram hebdomadaire (Sprint 49)
uv run python -m scripts.weekly_report --dry-run   # aperçu terminal
uv run python -m scripts.weekly_report              # envoi Telegram
```

## Vérification

1. `uv run pytest tests/test_weekly_report.py -x -q` — 5 tests passent
2. `uv run pytest tests/ -x -q` — zéro régression sur les ~1933 tests existants
3. `uv run python -m scripts.weekly_report --dry-run` — affiche le rapport formaté dans le terminal (fonctionne sans DB live, graceful sur tables vides)
