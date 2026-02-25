# COMMANDS.md — Scalp Radar Commandes & Méthodologie

## Règles pour l'IA

**Ce fichier est la référence pour toutes les commandes CLI du projet.**
Quand l'utilisateur demande "montre-moi les résultats", "lance un backtest", "vérifie l'état du serveur", etc., utiliser les commandes ci-dessous. Ne pas improviser — ces commandes sont testées et fonctionnent.

**PowerShell (Windows local) :** Toutes les commandes locales utilisent PowerShell.
Attention aux quotes : utiliser `"` pour les strings Python, créer un fichier .py temporaire si les quotes imbriquées posent problème.

**Bash (serveur Linux prod) :** Préfixer avec `docker exec scalp-radar-backend-1` pour exécuter dans le container.

---

## 🚨 COMMANDES D'URGENCE LIVE

> Toutes les commandes suivantes s'exécutent **sur le serveur** (`ssh jack@192.168.1.200`, dans `~/scalp-radar`).
> Pour le rollback git, la purge état et la désactivation stratégie → voir **§ 18. Rollback d'urgence**.
> Pour un guide de dépannage complet (symptômes → diagnostic → fix) → voir **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**.

### Reset Kill Switch Live

```bash
curl -X POST http://localhost:8000/api/executor/kill-switch/reset \
  -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)"
```

### Status Executor (JSON formaté)

```bash
curl -s http://localhost:8000/api/executor/status \
  -H "X-API-Key: $(grep SYNC_API_KEY .env | cut -d= -f2)" | python3 -m json.tool
```

### Vérifier Kill Switch dans les logs

```bash
docker compose logs backend --since 5m | grep -i "kill_switch"
```

### Vérifier Exit Monitor

```bash
docker compose logs backend --since 10m | grep -E "(Exit monitor|EXIT AUTONOME|no exit)"
```

### Logs positions ouvertes/fermées + erreurs

```bash
# Activité grids récente
docker compose logs backend --since 1h | grep -E "(GRID CLOSE|GRID ENTRY|EXIT AUTONOME)" | tail -20
# Erreurs critiques
docker compose logs backend --since 1h | grep -E "(ERROR|CRITICAL)" | tail -20
# Boot sync
docker compose logs backend --since 5m | grep -i "sync\|restore\|warm-up"
```

### Reset d'urgence de l'état executor (DERNIER RECOURS)

> ⚠️ Utiliser uniquement si le bot est arrêté et que l'état JSON est corrompu.
> Fermer toutes les positions sur Bitget manuellement AVANT.

```bash
docker compose stop backend
python3 -c "
import json
with open('data/executor_state.json') as f:
    state = json.load(f)
state.setdefault('executor', {}).update({
    'risk_manager': {'kill_switch': False, 'session_pnl': 0.0},
    'grid_states': {},
    'positions': {},
})
with open('data/executor_state.json', 'w') as f:
    json.dump(state, f, indent=2)
print('State reset OK')
"
docker compose start backend
```

---

## 1. Résultats WFO / Grades

### Voir tous les grades (toutes stratégies, derniers résultats)
```powershell
uv run python -c "import sqlite3; conn = sqlite3.connect('data/scalp_radar.db'); rows = conn.execute('SELECT strategy_name, asset, grade, total_score, oos_sharpe, consistency FROM optimization_results WHERE is_latest=1 ORDER BY strategy_name, total_score DESC').fetchall(); [print(f'{r[0]:20} {r[1]:14} Grade {r[2]}  ({r[3]:3.0f})  Sharpe {r[4]:5.2f}  Consist {r[5]:.0%}') for r in rows]; conn.close()"
```

### Voir les grades d'une stratégie spécifique
```powershell
uv run python -c "import sqlite3; conn = sqlite3.connect('data/scalp_radar.db'); rows = conn.execute(""SELECT asset, grade, total_score, oos_sharpe, consistency FROM optimization_results WHERE strategy_name='envelope_dca' AND is_latest=1 ORDER BY total_score DESC"").fetchall(); [print(f'{r[0]:14} Grade {r[1]}  ({r[2]:3.0f})  Sharpe {r[3]:5.2f}  Consist {r[4]:.0%}') for r in rows]; conn.close()"
```

### Voir l'analyse par régime (créer un fichier temporaire si quotes complexes)
```powershell
@"
import sqlite3, json
conn = sqlite3.connect('data/scalp_radar.db')
rows = conn.execute(
    "SELECT asset, regime_analysis FROM optimization_results "
    "WHERE strategy_name='envelope_dca' AND is_latest=1 AND regime_analysis IS NOT NULL "
    "ORDER BY total_score DESC LIMIT 10"
).fetchall()
for asset, ra_json in rows:
    if ra_json:
        ra = json.loads(ra_json)
        print(f"\n=== {asset} ===")
        for regime in ["bull","bear","range","crash"]:
            d = ra.get(regime, {})
            if d:
                sharpe = d.get("avg_oos_sharpe", 0)
                consist = d.get("consistency", 0) * 100
                wins = d.get("n_windows", 0)
                print(f"  {regime:6}: Sharpe {sharpe:6.2f}  Consist {consist:4.0f}%  Windows {wins}")
conn.close()
"@ | Out-File -Encoding utf8 _tmp_query.py
uv run python _tmp_query.py
del _tmp_query.py
```

### Voir les résultats sur le serveur prod
```bash
ssh jack@192.168.1.200
docker exec scalp-radar-backend-1 uv run python -c "import sqlite3; conn = sqlite3.connect('data/scalp_radar.db'); rows = conn.execute('SELECT strategy_name, asset, grade, total_score FROM optimization_results WHERE is_latest=1 ORDER BY strategy_name, total_score DESC').fetchall(); [print(f'{r[0]:20} {r[1]:14} Grade {r[2]} ({r[3]:3.0f})') for r in rows]; conn.close()"
```

---

## 2. Lancer des optimisations WFO

### Optimiser une stratégie sur un asset
```powershell
uv run python -m scripts.optimize --strategy envelope_dca --symbol BTC/USDT -v
```

### Optimiser une stratégie sur tous les assets
```powershell
uv run python -m scripts.optimize --strategy envelope_dca --all-symbols -v
```

### Optimiser toutes les stratégies, tous les assets
```powershell
uv run python -m scripts.optimize --all
```

### Boucle sur plusieurs assets (PowerShell)
```powershell
foreach ($s in @("BTC/USDT","ETH/USDT","SOL/USDT","DOGE/USDT","LINK/USDT")) {
    Write-Host "=== envelope_dca $s ==="
    uv run python -m scripts.optimize --strategy envelope_dca --symbol $s
}
```

### Boucle sur les 15 altcoins (hors Top 6)
```powershell
foreach ($s in @("ADA/USDT","AAVE/USDT","ARB/USDT","AVAX/USDT","BCH/USDT","BNB/USDT","CRV/USDT","DYDX/USDT","FET/USDT","GALA/USDT","ICP/USDT","NEAR/USDT","OP/USDT","SUI/USDT","UNI/USDT")) {
    Write-Host "=== envelope_dca $s ==="
    uv run python -m scripts.optimize --strategy envelope_dca --symbol $s
}
```

### Appliquer les résultats A/B dans strategies.yaml
```powershell
uv run python -m scripts.optimize --all --apply
```

### Flags Sprint 37 — Timeframe Coherence Guard

| Flag | Description |
| ---- | ----------- |
| `--force-timeframe 1h` | Restreint la grid WFO à un seul TF (override `param_grids.yaml`) |
| `--symbols A,B,C` | Optimise plusieurs assets séparés par virgule (mutex avec `--symbol` / `--all-symbols`) |
| `--exclude A,B` | Exclut des assets de `--apply` (les retire du YAML s'ils y sont) |
| `--ignore-tf-conflicts` | Force `--apply` en excluant silencieusement les outliers TF |

**Workflow résolution conflit timeframe** (affiché automatiquement si `--apply` détecte un outlier) :

```powershell
# 1. Voir les outliers → --apply affiche le message bloquant avec les 3 actions
uv run python -m scripts.optimize --strategy grid_atr --apply

# 2a. Re-tester les outliers en 1h (recommandé)
uv run python -m scripts.optimize --strategy grid_atr --symbols BCH/USDT,BNB/USDT --force-timeframe 1h

# 2b. Exclure les outliers et appliquer
uv run python -m scripts.optimize --strategy grid_atr --apply --exclude BCH/USDT,BNB/USDT

# 2c. Forcer (exclut les outliers silencieusement, sans re-tester)
uv run python -m scripts.optimize --strategy grid_atr --apply --ignore-tf-conflicts

# Portfolio backtest — bloque si un runner a un timeframe ≠ 1h (TimeframeConflictError)
# Affiche les assets conflictuels + suggestions --force-timeframe
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days 365
```

---

### Reprendre un WFO interrompu (--resume)

Skippe les assets déjà en DB (`is_latest=1`) — utile après un crash OOM/segfault.

```powershell
# Reprendre grid_range_atr après crash (ex: planté à UNI/USDT)
uv run python -m scripts.optimize --strategy grid_range_atr --all-symbols --resume

# Voir quels assets sont déjà faits avant de lancer
uv run python -m scripts.optimize --strategy grid_range_atr --all-symbols --resume --dry-run
```

| Argument   | Défaut | Description                                          |
|------------|--------|------------------------------------------------------|
| `--resume` | false  | Skipper les assets déjà en DB (reprend après crash)  |

---

### Recalculer les grades sans re-WFO (--regrade)

Recalcule le score et le grade à partir des résultats OOS déjà en DB. Utile après un changement de formule combo_score ou de seuil de grading.

```powershell
uv run python -m scripts.optimize --regrade --strategy grid_atr
uv run python -m scripts.optimize --regrade --strategy grid_multi_tf
```

> Incompatible avec `--apply`, `--symbol`, `--symbols`, `--all-symbols`, `--all`, `--resume`.

---

### Optimiser grid_boltrend (source binance)

```powershell
# Asset unique
uv run python -m scripts.optimize --strategy grid_boltrend --symbol ETH/USDT --exchange binance -v

# Boucle Top assets
foreach ($s in @("BTC/USDT","ETH/USDT","DOGE/USDT","DYDX/USDT","LINK/USDT")) {
    Write-Host "=== grid_boltrend $s ==="
    uv run python -m scripts.optimize --strategy grid_boltrend --symbol $s --exchange binance
}

# Tous les assets
uv run python -m scripts.optimize --strategy grid_boltrend --all-symbols --exchange binance
```

---

## 3. Données historiques

### Vérifier les données disponibles
```powershell
uv run python -m scripts.optimize --check-data
```

### Fetch données 1h (backtest principal)

```powershell
# Tous les 21 assets depuis assets.yaml (sans --symbol = tous par défaut)
uv run python -m scripts.fetch_history --exchange binance --days 1100 --timeframe 1h

# Assets spécifiques
uv run python -m scripts.fetch_history --exchange binance --days 1100 --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1h

# Bitget pour validation cross-exchange (~90j)
uv run python -m scripts.fetch_history --exchange bitget --days 90 --timeframe 1h
```

### Fetch données courtes (indicateurs Scanner — 7 jours)
```powershell
uv run python -m scripts.fetch_history --exchange binance --days 7 --symbols ADA/USDT,XRP/USDT --timeframe 5m
uv run python -m scripts.fetch_history --exchange binance --days 7 --symbols ADA/USDT,XRP/USDT --timeframe 15m
```

### Backfill candles Binance (API publique, sans clé)
```powershell
uv run python -m scripts.backfill_candles --symbol BTC/USDT --timeframe 1h --days 1800
# Depuis une date précise
uv run python -m scripts.backfill_candles --symbol ETH/USDT --timeframe 1h --since 2022-01-01
```

### Fetch funding rates historiques (requis pour grid_funding)

```powershell
# Tous les assets depuis assets.yaml (720 jours)
uv run python -m scripts.fetch_funding

# Asset spécifique
uv run python -m scripts.fetch_funding --symbol BTC/USDT --days 720

# Force re-fetch (supprime existant)
uv run python -m scripts.fetch_funding --symbol ETH/USDT --force
```

### Fetch open interest historique (requis pour momentum/liquidation)

```powershell
# Tous les assets (5m, 720 jours)
uv run python -m scripts.fetch_oi

# Asset + timeframe spécifique
uv run python -m scripts.fetch_oi --symbol BTC/USDT --timeframe 1h --days 365

# Timeframes disponibles : 5m, 15m, 30m, 1h, 4h, 1d
uv run python -m scripts.fetch_oi --symbol SOL/USDT --timeframe 4h --force
```

---

## 4. Simulator / Paper Trading

### Reset simulator (efface state, repart à initial_capital)
```powershell
uv run python scripts/reset_simulator.py
```

### Lancer le serveur local (dev)
```powershell
.\dev.bat
# ou manuellement :
uv run uvicorn backend.api.server:app --reload --host 127.0.0.1 --port 8000
```

### Vérifier l'état du grid (positions ouvertes)
```
GET http://127.0.0.1:8000/api/simulator/grid-state
```

### Vérifier la santé
```
GET http://127.0.0.1:8000/health
```

### Voir les trades récents
```
GET http://127.0.0.1:8000/api/simulator/trades?limit=20
```

### Voir les conditions de marché
```
GET http://127.0.0.1:8000/api/simulator/conditions
```

---

## 5. Tests

### Lancer tous les tests
```powershell
uv run python -m pytest --tb=short -q
```

### Lancer un fichier de test spécifique
```powershell
uv run python -m pytest tests/test_multi_engine.py --tb=short -q
```

### Lancer un test spécifique
```powershell
uv run python -m pytest tests/test_multi_engine.py::TestEnvelopeDCA::test_asymmetric_envelopes -v
```

### Lancer avec couverture
```powershell
uv run python -m pytest --cov=backend --cov-report=term-missing --tb=short -q
```

---

## 6. Déploiement Production

### Déployer (préserve le state)
```bash
ssh jack@192.168.1.200
cd ~/scalp-radar
git pull
bash deploy.sh
```

### Déployer (fresh start — efface le state JSON, pas la DB)
```bash
bash deploy.sh --clean
```

### Voir les logs prod
```bash
docker compose logs -f backend
docker compose logs --tail 100 backend
```

### Vérifier le health
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

### Vérifier le sync
```bash
docker exec scalp-radar-backend-1 env | grep SYNC
```

---

## 7. Requêtes DB courantes

### Compter les tests dans la DB
```powershell
uv run python -c "import sqlite3; conn = sqlite3.connect('data/scalp_radar.db'); print('Trades:', conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]); print('WFO results:', conn.execute('SELECT COUNT(*) FROM optimization_results').fetchone()[0]); conn.close()"
```

### Voir les trades paper récents
```powershell
uv run python -c "import sqlite3; conn = sqlite3.connect('data/scalp_radar.db'); rows = conn.execute('SELECT symbol, direction, entry_price, exit_price, net_pnl, exit_reason, created_at FROM trades ORDER BY created_at DESC LIMIT 10').fetchall(); [print(f'{r[0]:14} {r[1]:5} entry={r[2]:10.4f} exit={r[3]:10.4f} pnl={r[4]:+8.2f} {r[5]:10} {r[6]}') for r in rows]; conn.close()"
```

### Voir le P&L par asset
```powershell
uv run python -c "import sqlite3; conn = sqlite3.connect('data/scalp_radar.db'); rows = conn.execute('SELECT symbol, COUNT(*) as n, SUM(net_pnl) as total_pnl, AVG(net_pnl) as avg_pnl FROM trades GROUP BY symbol ORDER BY total_pnl DESC').fetchall(); [print(f'{r[0]:14} {r[1]:4} trades  P&L={r[2]:+10.2f}  avg={r[3]:+8.2f}') for r in rows]; conn.close()"
```

### Voir le nombre de combos WFO par run
```powershell
uv run python -c "import sqlite3; conn = sqlite3.connect('data/scalp_radar.db'); rows = conn.execute('SELECT r.strategy_name, r.asset, r.grade, COUNT(c.id) as combos FROM optimization_results r LEFT JOIN wfo_combo_results c ON r.id = c.result_id WHERE r.is_latest=1 GROUP BY r.id ORDER BY r.strategy_name, r.total_score DESC').fetchall(); [print(f'{r[0]:20} {r[1]:14} Grade {r[2]}  {r[3]:4} combos') for r in rows]; conn.close()"
```

---

## 8. Config vérifications

### Voir les assets actifs
```powershell
uv run python -c "from backend.core.config import get_config; c = get_config(); print(f'{len(c.assets)} assets:'); [print(f'  {a.symbol}') for a in c.assets]"
```

### Voir les stratégies activées
```powershell
uv run python -c "from backend.strategies.factory import get_enabled_strategies; from backend.core.config import get_config; strats = get_enabled_strategies(get_config()); [print(f'  {s.name}') for s in strats]"
```

### Voir le capital et risk config
```powershell
uv run python -c "from backend.core.config import get_config; c = get_config(); print(f'Initial capital: {c.risk.initial_capital}'); print(f'Max concurrent: {c.risk.max_concurrent_positions}'); print(f'Kill switch: {c.risk.global_kill_switch_pct}/{c.risk.global_kill_switch_window_hours}h')"
```

### Voir les per_asset d'une stratégie
```powershell
uv run python -c "from backend.core.config import get_config; c = get_config(); pa = c.strategies.envelope_dca.per_asset; print(f'{len(pa)} per_asset overrides:'); [print(f'  {k}: sl={v.get(\"sl_percent\",\"?\")}% levels={v.get(\"num_levels\",\"?\")}') for k,v in sorted(pa.items())]"
```

---

## 9. Patterns PowerShell pour queries complexes

### Pattern "fichier temporaire" (évite les problèmes de quotes)
```powershell
@"
# Code Python ici, avec des quotes normales
import sqlite3
conn = sqlite3.connect("data/scalp_radar.db")
# ... requête ...
conn.close()
"@ | Out-File -Encoding utf8 _tmp_query.py
uv run python _tmp_query.py
del _tmp_query.py
```

### Pattern "boucle parallèle" (3 terminaux)
```powershell
# Terminal 1
foreach ($s in @("BTC/USDT","ETH/USDT","SOL/USDT")) { uv run python -m scripts.optimize --strategy grid_atr --symbol $s }

# Terminal 2
foreach ($s in @("DOGE/USDT","LINK/USDT","ADA/USDT")) { uv run python -m scripts.optimize --strategy grid_atr --symbol $s }

# Terminal 3
foreach ($s in @("AAVE/USDT","ARB/USDT","AVAX/USDT")) { uv run python -m scripts.optimize --strategy grid_atr --symbol $s }
```

---

## 10. Méthodologie de travail

### Workflow Claude (discussion) + Claude Code (implémentation)
1. **Claude (ce chat)** : discute, analyse, rédige les briefs/plans pour Claude Code
2. **Claude Code** : implémente le code, lance les tests, fait les commits
3. **Jack** : valide visuellement, teste en local, déploie en prod

### Avant chaque sprint
1. Vérifier les tests : `uv run python -m pytest --tb=short -q`
2. Vérifier le nombre de tests actuel (dans le output pytest)
3. Lire CLAUDE.md et ROADMAP.md pour le contexte

### Après chaque sprint
1. Mettre à jour CLAUDE.md (test count, nouvelles features)
2. Mettre à jour ROADMAP.md (sprint complété, résultats, leçons)
3. Déployer en prod : `git pull && bash deploy.sh`
4. Vérifier visuellement le dashboard

### Pour ajouter une nouvelle stratégie grid

Voir [STRATEGIES.md § Comment ajouter une nouvelle stratégie](docs/STRATEGIES.md#comment-ajouter-une-nouvelle-stratégie) (checklist 11 étapes).

Pour le workflow complet validation (WFO → Grade → Portfolio → Paper → Live), voir [WORKFLOW_WFO.md](docs/WORKFLOW_WFO.md).

---

## 11. Backtest individuel (run_backtest)

Lance un backtest simple sur une stratégie + asset avec le BacktestEngine event-driven.

```powershell
# Backtest vwap_rsi sur BTC/USDT, 90 jours
uv run python -m scripts.run_backtest --strategy vwap_rsi --symbol BTC/USDT --days 90

# Avec capital et levier personnalisés
uv run python -m scripts.run_backtest --strategy bollinger_mr --symbol ETH/USDT --days 180 --capital 5000 --leverage 5

# Sortie JSON (pour inspection programmatique)
uv run python -m scripts.run_backtest --strategy grid_atr --symbol BTC/USDT --json

# Écrire dans un fichier
uv run python -m scripts.run_backtest --strategy grid_atr --symbol BTC/USDT --output results/grid_atr_btc.json --json
```

**Arguments disponibles :**

| Argument | Défaut | Description |
| --- | --- | --- |
| `--strategy` | `vwap_rsi` | Stratégie à tester |
| `--symbol` | `BTC/USDT` | Paire à tester |
| `--days` | `90` | Nombre de jours de données |
| `--capital` | `10000` | Capital initial ($) |
| `--leverage` | depuis risk.yaml | Levier |
| `--json` | false | Sortie JSON au lieu du tableau |
| `--output` | — | Fichier de sortie |

---

## 12. Portfolio Backtest (portfolio_backtest)

Simule N assets avec capital partagé (même code que la prod). Voir section 2 pour les exemples WFO.

```powershell
# Auto-détection historique (défaut) — affiche le goulot par asset
uv run python -m scripts.portfolio_backtest --strategy grid_atr --capital 10000

# grid_atr sur les Top 10 assets paper, 365 jours
uv run python -m scripts.portfolio_backtest --strategy grid_atr --assets BTC/USDT,AVAX/USDT,CRV/USDT,DOGE/USDT,DYDX/USDT,FET/USDT,GALA/USDT,ICP/USDT,NEAR/USDT --days 365 --capital 10000

# Forward test grid_atr (365 derniers jours)
uv run python -m scripts.portfolio_backtest --strategy grid_atr --assets BTC/USDT,ETH/USDT,DOGE/USDT --days 365 --capital 10000 --save --label "forward_test_2025"

# grid_boltrend sur 6 assets, 730 jours
uv run python -m scripts.portfolio_backtest --strategy grid_boltrend --assets BTC/USDT,ETH/USDT,DOGE/USDT,DYDX/USDT,LINK/USDT --capital 1000 --days 730

# Multi-stratégie (grid_atr + grid_boltrend)
uv run python -m scripts.portfolio_backtest --strategies "grid_atr:BTC/USDT,ETH/USDT+grid_boltrend:DOGE/USDT,LINK/USDT" --capital 10000 --days 365

# Comparaison de leverages sans toucher au YAML
uv run python -m scripts.portfolio_backtest --capital 1000 --leverage 3
uv run python -m scripts.portfolio_backtest --capital 1000 --leverage 5
uv run python -m scripts.portfolio_backtest --capital 1000 --leverage 6

# Sortie JSON avec sauvegarde en DB
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days 90 --json --save --label "q1_2025"

# A/B test d'un paramètre (params WFO existants, un seul changement)
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days auto --save --label "grid_atr_baseline_hold0"
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days auto --save --label "grid_atr_test_hold48" --params "max_hold_candles=48"
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days auto --save --label "grid_atr_test_hold96" --params "max_hold_candles=96"
```

**Arguments disponibles :**

| Argument | Défaut | Description |
| --- | --- | --- |
| `--strategy` | `grid_atr` | Stratégie (mono-stratégie) |
| `--strategies` | — | Multi-stratégie : `strat1:sym1,sym2+strat2:sym3` |
| `--preset` | — | Preset prédéfini (ex: `combined`) |
| `--assets` | tous per_asset | Assets séparés par virgule |
| `--days` | `auto` | Période (jours ou `auto` = max historique commun) |
| `--capital` | `10000` | Capital initial ($) |
| `--exchange` | `binance` | Source des candles |
| `--leverage` | depuis strategies.yaml | Override leverage de tous les runners |
| `--params` | — | Override params stratégie : `key=val,key2=val2` |
| `--kill-switch-pct` | `45.0` | Seuil kill switch (%) |
| `--kill-switch-window` | `24` | Fenêtre kill switch (heures) |
| `--json` | false | Sortie JSON |
| `--output` | — | Fichier de sortie |
| `--save` | false | Sauvegarder en DB |
| `--label` | — | Label du run |

---

## 13. Sync vers le serveur de production (sync_to_server)

Pousse les résultats WFO et/ou portfolio backtests du local vers le serveur prod. Idempotent.

```powershell
# Sync tout (WFO + portfolio)
uv run python -m scripts.sync_to_server

# Sync uniquement les résultats WFO
uv run python -m scripts.sync_to_server --only wfo

# Sync uniquement les portfolio backtests
uv run python -m scripts.sync_to_server --only portfolio

# Dry-run (affiche ce qui serait envoyé sans envoyer)
uv run python -m scripts.sync_to_server --dry-run
```

**Prérequis :** Variables `.env` configurées : `SYNC_ENABLED=true`, `SYNC_SERVER_URL`, `SYNC_API_KEY`.

---

## 14. Scripts Diagnostic

### Sweep timeframes — comparer 1h / 4h / 1d

```powershell
# grid_atr sur ETH/USDT, 730 jours
uv run python -m scripts.test_timeframe_sweep --symbol ETH/USDT --days 730 --strategy grid_atr

# grid_boltrend sur BTC/USDT
uv run python -m scripts.test_timeframe_sweep --symbol BTC/USDT --days 730 --strategy grid_boltrend

# Avec capital personnalisé
uv run python -m scripts.test_timeframe_sweep --symbol SOL/USDT --days 365 --strategy grid_atr --capital 5000
```

**Stratégies supportées :** `boltrend`, `envelope_dca`, `envelope_dca_short`, `grid_atr`, `grid_range_atr`, `grid_trend`

### Diagnostic Grid Range ATR (fast engine)

```powershell
# Run basique sur BTC/USDT
uv run python -m scripts.test_grid_range_fast --symbol BTC/USDT --days 365

# Avec paramètres custom
uv run python -m scripts.test_grid_range_fast --symbol ETH/USDT --spacing 0.5 --num-levels 3 --tp-mode fixed_center

# Sweep de spacings automatique
uv run python -m scripts.test_grid_range_fast --symbol BTC/USDT --sweep

# Long-only ou Short-only
uv run python -m scripts.test_grid_range_fast --symbol BTC/USDT --sides long
```

### Diagnostic Grid BolTrend (trade log détaillé)

```powershell
# Log des 10 premiers trades (défaut)
uv run python -m scripts.grid_boltrend_diagnostic

# Plus de candles et de trades
uv run python -m scripts.grid_boltrend_diagnostic --n-candles 1000 --max-trades 20

# Sauvegarder le log
uv run python -m scripts.grid_boltrend_diagnostic --output results/boltrend_log.txt
```

### Parité Fast Engine vs BacktestEngine (parity_check)

Vérifie la divergence entre le fast engine (Numba/numpy) et le BacktestEngine classique.
Utile pour valider les résultats WFO après une modification du fast engine.
Compare aussi les résultats Bitget vs Binance.

```powershell
# Vérification de parité sur vwap_rsi (défaut)
uv run python scripts/parity_check.py

# Stratégie spécifique via DEFAULT_PARAMS dans le script
# (modifier les params en tête de fichier)
```

---

## 15. Benchmark Fast Engine

Mesure les performances du fast engine (compilation Numba + simulation). Exclut le 1er run (compilation).

```powershell
# Benchmark toutes les stratégies grid (200 combos, 5000 candles)
uv run python -m scripts.benchmark_fast_engine

# Benchmark avec plus de combos
uv run python -m scripts.benchmark_fast_engine --combos 500 --candles 10000

# Benchmark une stratégie spécifique
uv run python -m scripts.benchmark_fast_engine --strategies grid_atr grid_boltrend

# Plus de runs pour la moyenne
uv run python -m scripts.benchmark_fast_engine --runs 6
```

---

## 16. Stress Test Leverage

Compare les performances d'une stratégie à différents leverages sur plusieurs fenêtres temporelles. Kill switch désactivé pour voir le vrai max drawdown.

```powershell
# Lancer tous les tests (20 runs par défaut)
uv run python -m scripts.stress_test_leverage

# Une seule stratégie
uv run python -m scripts.stress_test_leverage --strategy grid_boltrend

# Leverages custom (pour affiner entre deux valeurs)
uv run python -m scripts.stress_test_leverage --leverages 5,7

# Une seule fenêtre
uv run python -m scripts.stress_test_leverage --days 180

# Combiné
uv run python -m scripts.stress_test_leverage --strategy grid_boltrend --leverages 3,5 --days 90

# Ajouter des résultats à un CSV existant
uv run python -m scripts.stress_test_leverage --strategy grid_atr --leverages 6,8 --append
```

**IMPORTANT :** Sur Windows avec Python 3.13, Numba segfault sur les fenêtres longues. Désactiver le JIT :

```powershell
$env:NUMBA_DISABLE_JIT=1
uv run python -m scripts.stress_test_leverage
```

Résultats CSV sauvegardés dans `data/stress_test_results.csv`

---

## 17. Maintenance / Migration

### Migrer les JSON WFO vers la DB (Sprint 13 — one-shot)

```powershell
# Dry-run : liste les fichiers sans importer
uv run python -m scripts.migrate_optimization --dry-run

# Import réel
uv run python -m scripts.migrate_optimization
```

---

## 18. Rollback d'urgence (production)

**ATTENTION : ne JAMAIS utiliser `echo "..." > .env`** — ça écrase tout le fichier (clés Bitget, tokens, etc.). Toujours éditer avec `nano`.

### Désactiver une stratégie sans redéployer

```bash
# Sur le serveur (SSH) — éditer .env manuellement
ssh jack@192.168.1.200
cd ~/scalp-radar
nano .env
# Modifier la ligne FORCE_STRATEGIES pour retirer grid_boltrend :
#   FORCE_STRATEGIES=grid_atr
# Sauvegarder et quitter (Ctrl+O, Ctrl+X)
docker compose restart backend
# Vérifier que seul grid_atr tourne :
docker compose logs backend --tail 20 | grep "runner"
```

### Rollback git complet

```bash
cd ~/scalp-radar && docker compose down
git log --oneline -5       # identifier le commit stable
git checkout <commit-hash>
bash deploy.sh
```

### Purger l'état simulator (fresh start)

```bash
cd ~/scalp-radar && bash deploy.sh --clean
```

### Vérifier après rollback

```bash
docker compose logs backend --tail 20 | grep "runner"
curl -s http://localhost:8000/health | python3 -m json.tool
```

---

## 19. Maintenance WFO

### Purger les doublons WFO (Sprint 47b)

```powershell
uv run python -m scripts.purge_wfo_duplicates --dry-run   # aperçu
uv run python -m scripts.purge_wfo_duplicates             # correction
```

Détecte et corrige les doublons `is_latest=1` par (strategy, asset) en base. Conserve le meilleur score (`MAX(total_score)`). Supporte `--db-path` pour cibler une base spécifique.

---

## 20. Scripts d'Audit & Analyse

Scripts d'analyse ponctuelle. Ne font aucune modification — lecture seule.

### Deep Analysis post-WFO (outil DIAGNOSTIQUE — pas un filtre)

Analyse le profil de risque par régime des assets Grade A/B. Produit un verdict : VIABLE / BORDERLINE / AT RISK.
Détecte les red flags (SL×leverage, régimes négatifs, DSR, CI95) — à des fins de diagnostic uniquement.

```powershell
# Analyser une stratégie
uv run python -m scripts.analyze_wfo_deep --strategy grid_boltrend

# Analyser toutes les stratégies avec résultats Grade A/B en DB
uv run python -m scripts.analyze_wfo_deep --all
```

Sortie : tableau récapitulatif (VIABLE/BORDERLINE/AT RISK) + détail par asset + note prochaine étape.

**Workflow post-WFO** :
```
1. WFO      → uv run python -m scripts.optimize --strategy <n> --all-symbols
2. Apply    → uv run python -m scripts.optimize --strategy <n> --apply   (TOUS les Grade A/B)
3. Portfolio → uv run python -m scripts.portfolio_backtest --strategy <n> --days auto   ← LE vrai filtre
4. Deep (si portfolio échoue) → uv run python -m scripts.analyze_wfo_deep --strategy <n>
```

> Note : les verdicts AT RISK individuels peuvent être compensés par la diversification.
> Exemple : grid_boltrend — 4/6 assets AT RISK, mais portfolio +552%, DD -15.3%.

---

### Analyser une régression WFO (compare les 2 derniers runs)

```powershell
# grid_atr avec levier 7 et kill switch 45%
uv run python -m scripts.analyze_wfo_regression --strategy grid_atr --leverage 7 --kill-switch 45

# Spécifier les assets à approfondir (deep dive)
uv run python -m scripts.analyze_wfo_regression --strategy grid_atr --losers NEAR/USDT DOGE/USDT
```

Sortie : diff paramétrique, analyse par régime, risque SL, recommandations.

### Auditer le scoring WFO (compare 4 formules combo_score)

```powershell
uv run python -m scripts.audit_combo_score --strategy grid_atr > data/analysis/combo_score_audit.txt
uv run python -m scripts.audit_combo_score --strategy grid_boltrend
```

Sortie : changements best combo, déltas Grade/Score pour chaque variante.

### Vérifier l'alignement des fees Bitget vs modèle backtest

```powershell
# Derniers 7 jours avec détail par symbol
uv run python -m scripts.audit_fees --days 7 -v

# Dump JSON des 3 premiers trades bruts (debug)
uv run python -m scripts.audit_fees --days 30 --debug
```

### Auditer la cohérence grid_states vs Bitget (fantômes, orphelins, SL manquants)

```powershell
# Via API (serveur en cours — recommandé)
uv run python -m scripts.audit_grid_states --mode api -v

# Via fichier JSON local (serveur arrêté)
uv run python -m scripts.audit_grid_states --mode file --state-file data/executor_state.json
```

### Vérifier le sizing avant déploiement live (validation minimums Bitget)

```powershell
# Lit config/strategies.yaml et .env automatiquement
uv run python scripts/check_live_sizing.py
```

Affiche capital/levels, quantité, notional, warning margin worst-case par asset.

### Comparer les performances WFO avant/après un re-run

```powershell
uv run python scripts/compare_wfo.py
```

Sortie : tableau Grade/Score/Sharpe/Consistency delta (vert=amélioration, rouge=régression).

### Diagnostiquer la marge portfolio (skips par margin guard)

```powershell
# Analyse grid_atr 365 jours, levier 7 vs 6 (référence)
uv run python -m scripts.diagnose_margin --strategy grid_atr --leverage 7 --days 365

# grid_boltrend 90 jours
uv run python -m scripts.diagnose_margin --strategy grid_boltrend --leverage 8 --days 90
```

Sortie : skip counts par asset/guard, timeline margin utilization, comparaison leverages.

### Analyser la robustesse d'un portfolio backtest (bootstrap, stress, CVaR)

```powershell
# Analyser un backtest sauvé par label
uv run python -m scripts.portfolio_robustness --label "grid_boltrend_6x_all_6B"

# Comparer plusieurs backtests
uv run python -m scripts.portfolio_robustness --labels "grid_atr_14assets_7x_post40a,grid_boltrend_6x_all_6B"

# Paramètres optionnels
uv run python -m scripts.portfolio_robustness --label "grid_boltrend_6x_all_6B" \
  --n-simulations 10000 --block-size 7 --confidence 95 --seed 42 --save
```

4 méthodes : Block Bootstrap (CI return/DD), Regime Stress (scénarios marché), Historical Stress (crashes réels), CVaR.
Verdict automatique GO/NO-GO (VIABLE / CAUTION / FAIL). `--save` sauvegarde en DB table `portfolio_robustness`.

### Analyser la corrélation entre stratégies

```powershell
# Lister les labels disponibles en DB
uv run python -m scripts.analyze_correlation --list

# Comparer la corrélation DD entre deux labels sauvés (étape 3)
uv run python -m scripts.analyze_correlation --labels "grid_atr_7x,grid_multi_tf_6x"

# Trois stratégies
uv run python -m scripts.analyze_correlation --labels "label1,label2,label3"
```

Mesure la corrélation des drawdowns entre stratégies (2 ou 3 labels max).
Cible : r < 0.3. Calcule aussi l'allocation optimale minimisant le DD combiné.

**Workflow post-WFO complet (étapes 0-8)** :
```
0. Leverage → calcul mathématique (AVANT le WFO)
1. WFO      → uv run python -m scripts.optimize --strategy <n> --all-symbols --subprocess -v
2. Apply    → uv run python -m scripts.optimize --strategy <n> --apply
3. Portfolio → uv run python -m scripts.portfolio_backtest --strategy <n> --days auto --save --label "<label>"
3b. Deep (DIAGNOSTIC) → uv run python -m scripts.analyze_wfo_deep --strategy <n>
4. Stress   → uv run python -m scripts.stress_test_leverage --strategy <n>
5. Robust.  → uv run python -m scripts.portfolio_robustness --label "<label>" --save
6. Corrél.  → uv run python -m scripts.analyze_correlation --labels "<label1>,<label2>"
7. Paper    → enabled: true, live_eligible: false (2 semaines min, 1 mois si CAUTION)
8. Live     → live_eligible: true (leverage progressif 3x → 5x → 6x)
```

---

## 20. Maintenance production

### Vérifier la taille de la DB et du WAL

```bash
# Sur le serveur (dans le container)
docker exec scalp-radar-backend-1 ls -lh data/scalp_radar.db data/scalp_radar.db-wal data/scalp_radar.db-shm 2>/dev/null

# En local (PowerShell)
Get-Item data/scalp_radar.db, data/scalp_radar.db-wal -ErrorAction SilentlyContinue | Select-Object Name, Length
```

### WAL checkpoint manuel (réduire le fichier .db-wal)

```bash
# Sur le serveur (dans le container)
docker exec scalp-radar-backend-1 python3 -c "
import sqlite3
conn = sqlite3.connect('data/scalp_radar.db')
result = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
print(f'WAL checkpoint TRUNCATE: busy={result[0]} log={result[1]} checkpointed={result[2]}')
conn.close()
"

# TRUNCATE = vide complètement le WAL (nécessite 0 reader actif → faire avec backend arrêté)
docker compose stop backend
sqlite3 data/scalp_radar.db "PRAGMA wal_checkpoint(TRUNCATE);"
docker compose start backend
```

> **Note :** Le WAL checkpoint PASSIVE est exécuté automatiquement toutes les heures par le backend.
> TRUNCATE est plus agressif (vide le WAL en entier) mais nécessite l'arrêt du backend.

### Backup manuel de la DB

```bash
# Sur le serveur
cd ~/scalp-radar
bash scripts/backup_db.sh

# Vérifier les backups existants
ls -lh data/backups/
```

### Installer le cron backup (Linux prod — 1x/jour à 3h00)

```bash
# Ouvrir le crontab
crontab -e

# Ajouter cette ligne (adapter le chemin)
0 3 * * * cd ~/scalp-radar && bash scripts/backup_db.sh >> logs/backup.log 2>&1

# Vérifier
crontab -l | grep backup
```

### Vérifier l'espace disque

```bash
# Via l'endpoint /health (inclut le champ "disk")
curl -s http://localhost:8000/health | python3 -m json.tool | grep -A5 '"disk"'

# Directement
df -h ~/scalp-radar/data/
du -sh ~/scalp-radar/data/

# Taille DB + WAL + backups
du -sh ~/scalp-radar/data/scalp_radar.db* ~/scalp-radar/data/backups/ 2>/dev/null
```

## 21. Multi-Executor (Sprint 36b)

### Architecture

Un Executor par stratégie live (enabled + live_eligible). Chaque executor peut utiliser :
- **Clés globales** (fallback) : `BITGET_API_KEY`, `BITGET_SECRET`, `BITGET_PASSPHRASE`
- **Sous-compte dédié** : `BITGET_API_KEY_GRID_ATR`, `BITGET_SECRET_GRID_ATR`, `BITGET_PASSPHRASE_GRID_ATR`

### Configurer un sous-compte Bitget

1. Bitget → API Management → Créer un sous-compte
2. Générer les clés API pour le sous-compte (futures read + trade, no withdrawal)
3. Ajouter dans `.env` (serveur) :
```bash
BITGET_API_KEY_GRID_ATR=votre_api_key
BITGET_SECRET_GRID_ATR=votre_secret
BITGET_PASSPHRASE_GRID_ATR=votre_passphrase
```

### Ajouter une nouvelle stratégie live

1. Dans `config/strategies.yaml` : mettre `enabled: true` + `live_eligible: true`
2. (Optionnel) Créer un sous-compte Bitget et ajouter les clés dans `.env`
3. Redéployer : `ssh prod "cd ~/scalp-radar && bash deploy.sh"`

### API endpoints

```bash
# Statut agrégé (tous les executors)
curl -s -H "X-API-Key: $KEY" http://localhost:8000/api/executor/status | python3 -m json.tool

# Statut d'un executor spécifique
curl -s -H "X-API-Key: $KEY" "http://localhost:8000/api/executor/status?strategy=grid_atr" | python3 -m json.tool

# Reset kill switch d'un executor spécifique
curl -X POST -H "X-API-Key: $KEY" "http://localhost:8000/api/executor/kill-switch/reset?strategy=grid_atr"

# Reset kill switch de tous les executors
curl -X POST -H "X-API-Key: $KEY" http://localhost:8000/api/executor/kill-switch/reset
```

### Fichiers state per-executor

```
data/executor_grid_atr_state.json       # État grid_atr
data/executor_grid_multi_tf_state.json  # État grid_multi_tf
data/executor_state.json                # Legacy (migration auto au 1er boot)
```

### Diagnostic

```bash
# Vérifier quels executors tournent (dans les logs)
docker compose logs backend 2>&1 | grep "Executor\[" | tail -20

# Vérifier si clés dédiées vs partagées
docker compose logs backend 2>&1 | grep "sous-compte" | tail -5
```
