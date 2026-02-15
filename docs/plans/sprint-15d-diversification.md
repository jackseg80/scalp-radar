# Sprint 15d — Consistance + Diversification 21 Assets

**Date** : 15 février 2026
**Tests** : 698 → 707 passants
**Objectif** : Intégrer la consistance WFO dans le grade, diversifier sur 18 nouvelles paires, automatiser le déploiement.

---

## 1. Consistance dans le grade (20 pts/100)

### Problème

Le grade ne prenait pas en compte la consistance (% fenêtres OOS positives). Une stratégie pouvait avoir un Sharpe OOS élevé sur quelques fenêtres et un Grade A, tout en étant inconsistante.

### Nouveau système — 6 critères

| Critère | Points | Ancien |
| ------- | ------ | ------ |
| OOS/IS Ratio | 20 | 20 |
| Monte Carlo | 20 | 20 |
| **Consistance** | **20** | **(nouveau)** |
| DSR | 15 | 15 |
| Stabilité | 10 | 15 |
| Bitget Transfer | 15 | 15 |
| **Total** | **100** | **85 (+15 OOS Sharpe supprimé)** |

**Impact** : ETH passe de 100/100 à 88/100 (68% consistance pénalisée).

### Top 5 trié par combo_score

Le Top 5 affiché dans le CLI et le frontend est maintenant trié par `combo_score` décroissant. Le #1 = le best combo sélectionné pour le grading.

---

## 2. Diversification — 18 nouvelles paires

### fetch_history --symbols

Nouveau flag `--symbols ADA/USDT,AVAX/USDT,...` pour télécharger des paires sans les ajouter à assets.yaml. Permet de backfill pour WFO avant de décider si l'asset est viable.

### Paires ajoutées

ADA, APE, AR, AVAX, BNB, CRV, DYDX, ENJ, FET, GALA, ICP, IMX, NEAR, SAND, SUSHI, THETA, UNI, XTZ

**717k candles** téléchargées depuis l'API publique Binance.

### Résultats WFO — 23 assets

- **3 Grade A** : ETH (88), DOGE (85), SOL (85)
- **18 Grade B** : LINK, UNI, APE, SAND, AR, NEAR, DYDX, CRV, IMX, FET (81), AVAX, SUSHI (78), GALA (77), ENJ (74), ADA, THETA (73), ICP, XTZ (71)
- **2 Grade D** (exclus) : BNB (50), BTC (47)

### THETA — WebSocket Bitget

THETA obtient Grade B (73) mais le WebSocket Bitget refuse l'abonnement au symbol `THETAUSDT`. L'asset est commenté dans assets.yaml en attendant investigation.

---

## 3. Automatisation Apply

### optimize.py --apply

Écrit les `per_asset` dans `strategies.yaml` depuis la DB pour tous les assets Grade A ou B avec `is_latest=1`.

### apply_from_db() — auto-add assets.yaml

Quand un asset a des params optimisés mais n'est pas dans `assets.yaml`, la fonction :
1. Se connecte à Bitget via ccxt
2. Récupère les specs du marché (min_order_size, tick_size, max_leverage)
3. Ajoute l'asset dans `assets.yaml` avec timeframes `["1h"]` et `correlation_group: altcoins`

### Bouton "Appliquer A/B" frontend

Nouveau bouton dans la page Recherche : `POST /api/optimization/apply` → un clic = per_asset strategies.yaml + assets.yaml mis à jour. Retour JSON avec le nombre d'assets ajoutés/mis à jour.

---

## 4. Déploiement prod

- 21 assets en paper trading live (BTC inclus avec params par défaut, THETA exclu)
- deploy.sh sans --clean (conservation du state)
- Pas de kill switch déclenché

---

## Fichiers modifiés

### Backend

- `backend/optimization/report.py` — `compute_grade()` : consistance 20 pts, stabilité 10 pts
- `backend/optimization/walk_forward.py` — Top 5 trié par combo_score
- `backend/api/optimization_routes.py` — `POST /api/optimization/apply`
- `backend/optimization/optimization_db.py` — `apply_from_db()`, auto-add assets

### Scripts

- `scripts/fetch_history.py` — flag `--symbols`
- `scripts/optimize.py` — flag `--apply`

### Frontend

- `frontend/src/pages/ResearchPage.jsx` — bouton "Appliquer A/B"

### Config

- `config/assets.yaml` — 21 assets (16 ajoutés automatiquement)
- `config/strategies.yaml` — 21 per_asset overrides pour envelope_dca

### Tests

- `tests/test_combo_score.py` — 7 tests (combo_score, tri, seuils)
- `tests/test_optimization_routes.py` — test endpoint apply
