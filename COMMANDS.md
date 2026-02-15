# COMMANDS.md — Scalp Radar Commandes & Méthodologie

## Règles pour l'IA

**Ce fichier est la référence pour toutes les commandes CLI du projet.**
Quand l'utilisateur demande "montre-moi les résultats", "lance un backtest", "vérifie l'état du serveur", etc., utiliser les commandes ci-dessous. Ne pas improviser — ces commandes sont testées et fonctionnent.

**PowerShell (Windows local) :** Toutes les commandes locales utilisent PowerShell.
Attention aux quotes : utiliser `"` pour les strings Python, créer un fichier .py temporaire si les quotes imbriquées posent problème.

**Bash (serveur Linux prod) :** Préfixer avec `docker exec scalp-radar-backend-1` pour exécuter dans le container.

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
uv run python -m scripts.optimize --strategy envelope_dca --all-assets -v
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

### Boucle sur les 16 nouveaux altcoins
```powershell
foreach ($s in @("ADA/USDT","APE/USDT","AR/USDT","AVAX/USDT","CRV/USDT","DYDX/USDT","ENJ/USDT","FET/USDT","GALA/USDT","ICP/USDT","IMX/USDT","NEAR/USDT","SAND/USDT","SUSHI/USDT","UNI/USDT","XTZ/USDT")) {
    Write-Host "=== envelope_dca $s ==="
    uv run python -m scripts.optimize --strategy envelope_dca --symbol $s
}
```

### Appliquer les résultats A/B dans strategies.yaml
```powershell
uv run python -m scripts.optimize --all --apply
```

---

## 3. Données historiques

### Vérifier les données disponibles
```powershell
uv run python -m scripts.optimize --check-data
```

### Fetch données 1h (backtest principal — 1800 jours)
```powershell
uv run python -m scripts.fetch_history --exchange binance --days 1800 --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1h
```

### Fetch données courtes (indicateurs Scanner — 7 jours)
```powershell
uv run python -m scripts.fetch_history --exchange binance --days 7 --symbols ADA/USDT,APE/USDT --timeframe 5m
uv run python -m scripts.fetch_history --exchange binance --days 7 --symbols ADA/USDT,APE/USDT --timeframe 15m
uv run python -m scripts.fetch_history --exchange binance --days 2 --symbols ADA/USDT,APE/USDT --timeframe 1m
```

### Backfill candles Binance (API publique, sans clé)
```powershell
uv run python -m scripts.backfill_candles --symbol BTC/USDT --timeframe 1h --days 1800
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
foreach ($s in @("APE/USDT","AR/USDT","AVAX/USDT")) { uv run python -m scripts.optimize --strategy grid_atr --symbol $s }
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
1. Créer `backend/strategies/my_strategy.py` (hérite BaseGridStrategy)
2. Ajouter config dans `backend/core/config.py`
3. Ajouter dans `backend/strategies/factory.py`
4. Ajouter dans `backend/optimization/__init__.py` (STRATEGY_REGISTRY + GRID_STRATEGIES)
5. Ajouter fast engine dans `backend/optimization/fast_multi_backtest.py`
6. Ajouter indicateurs dans `backend/optimization/indicator_cache.py` si nécessaire
7. Ajouter `_INDICATOR_PARAMS` dans `backend/optimization/walk_forward.py`
8. Config YAML : `config/strategies.yaml` + `config/param_grids.yaml`
9. Tests : signaux, fast engine parité, WFO integration
10. WFO : `uv run python -m scripts.optimize --strategy my_strategy --all-assets`
11. Résultats : commande section 1 ci-dessus