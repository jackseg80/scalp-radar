# Sprint 50a — Calibration Détecteur de Régime BTC (Phase 1 — Analyse pure)

## Contexte

On veut automatiser l'ajustement du leverage en fonction du régime de marché BTC.
Ce sprint est un sprint d'ANALYSE : valider par les données qu'un détecteur de régime apporte un edge mesurable.
Aucune modification du code de trading/backtest/executor. Tous les livrables vont dans `scripts/`, `data/`, `docs/`, `tests/`.

Le grid_atr performe bien en crash (DCA sur dips, TP sur rebond). On ne cherche PAS à couper mais à RÉDUIRE le leverage pour limiter le risque de liquidation.

---

## Étape 0 — Dépendances

**Modifier `pyproject.toml`** — Ajouter un groupe optionnel `analysis` :

```toml
analysis = [
    "matplotlib>=3.8",
    "pandas>=2.2",
]
```

- `pandas` 3.0.0 est déjà disponible en transitive (via ccxt) mais non déclaré
- `matplotlib` n'est PAS installé — nécessite `uv sync --group analysis`
- Groupe optionnel pour ne pas alourdir l'image Docker prod

Chaque script d'analyse inclura un guard :
```python
try:
    import pandas as pd
    import matplotlib
except ImportError:
    sys.exit("pandas et matplotlib requis. Installer: uv sync --group analysis")
```

---

## Étape 1 — Export des données (Phase 0)

### 1a. Backfill BTC/USDT 4h depuis 2017

Commande manuelle (script existant) :
```powershell
uv run python -m scripts.backfill_candles --symbol BTC/USDT --timeframe 4h --since 2017-01-01
```
Attend ~17 500+ candles. L'API Binance spot `/api/v3/klines` a des données depuis août 2017.

### 1b. Créer `scripts/export_btc_4h.py`

Lit les candles BTC/USDT 4h depuis la DB (SQL brut, pas `get_candles()` qui limite à 500) et exporte vers `data/btc_4h_2017_2025.csv`.

**Format CSV** : `timestamp_utc,open,high,low,close,volume`

Pattern standard : `argparse` + `asyncio.run(main())` + `Database()`.
Args : `--output` (défaut `data/btc_4h_2017_2025.csv`), `--exchange` (défaut `binance`).

Utilise `aiosqlite` directement (pas `db._conn` qui est un attribut privé) :
```python
import aiosqlite
async with aiosqlite.connect("data/scalp_radar.db") as conn:
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM candles WHERE exchange=? AND symbol='BTC/USDT' AND timeframe='4h' "
        "ORDER BY timestamp ASC",
        [args.exchange],
    )
    rows = await cursor.fetchall()
```
Script standalone — pas besoin de modifier `database.py` pour un script d'analyse one-shot.

---

## Étape 2 — Ground Truth (Phase 1)

### 2a. Créer `data/btc_regime_events.yaml`

Fichier d'annotations manuellesavec ~20 événements (fourni dans le brief).
Structure : `events[]` avec `name, start, end, type, severity, notes`.
`type` : bull, bear, range, crash.

Gestion des overlaps : crash > bear > range > bull (priorité).
Les crashs LUNA/FTX chevauchent le bear 2022 — voulu.
Inclut la "Correction avril 2021" (2021-04-14 → 2021-05-10, type bear, severity minor).

**YAML complet** : reprendre exactement les 20 événements du brief utilisateur (Bull 2017 → Correction Q1 2025) + la "Correction avril 2021" ajoutée entre "Bull Market 2020-2021" et "China Ban Crash".

### 2b. Créer `scripts/regime_labeler.py`

**Fonctions principales :**

- `load_events(yaml_path) -> (list[dict], str)` — charge le YAML, retourne événements + default_regime
- `label_candles(df, events, default_regime) -> pd.DataFrame` — assigne `regime_label` par candle 4h
  - Pour chaque candle, trouve tous les événements dont [start, end] contient la date
  - Si multiple : priorité crash > bear > range > bull
  - Si aucun : `default_regime` ("range")
- `print_summary(df)` — affiche % de temps par régime, nombre de transitions
- `plot_ground_truth(df, output_path)` — BTC prix + bandes colorées → `docs/images/btc_ground_truth_regimes.png`

**CLI** : `uv run python -m scripts.regime_labeler`
Args : `--csv` (défaut `data/btc_4h_2017_2025.csv`), `--yaml` (défaut `data/btc_regime_events.yaml`), `--output` (défaut `data/btc_4h_labeled.csv`)

Produit :
- `data/btc_4h_labeled.csv` (CSV enrichi avec colonne `regime_label`)
- `docs/images/btc_ground_truth_regimes.png`

---

## Étape 3 — Détecteurs (Phase 2)

### Créer `scripts/regime_detectors.py`

Module contenant :
1. Fonction utilitaire `resample_4h_to_daily(df_4h) -> df_daily`
2. Helpers indicateurs pandas-natifs (`sma_series`, `ema_series`, `atr_series`, `rolling_max_drawdown`, `realized_volatility`)
3. `BaseDetector` ABC + 3 implémentations
4. Fonctions de métriques (accuracy, F1, confusion matrix, crash delay, etc.)

### Resampling 4h → Daily

```python
def resample_4h_to_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
```
- open = open de la première candle 4h du jour (00:00 UTC)
- high = max(high) des 6 candles
- low = min(low) des 6 candles
- close = close de la dernière candle 4h (20:00 UTC)
- volume = sum(volume)
- Filtre les jours incomplets (< 6 candles)
- Sanity check : vérifier qu'on ne perd pas plus de 5 jours sur ~3000 (log warning sinon)

### Architecture des détecteurs

```python
@dataclass
class DetectorResult:
    labels_4h: list[str]        # Un label par candle 4h (après hysteresis)
    labels_daily: list[str]     # Un label par candle daily (après hysteresis, pour debug)
    raw_labels_daily: list[str] # Labels bruts daily (avant hysteresis, pour debug)
    params: dict[str, Any]
    warmup_end_idx: int         # Index 4h de la première candle avec signal valide

class BaseDetector(ABC):
    name: str
    def detect_raw(self, df_daily: pd.DataFrame, **params) -> list[str]:
        """Labels bruts sur daily (sans hysteresis)."""
    def param_grid(cls) -> list[dict]:
        """Combinaisons de paramètres pour grid search."""
    def run(self, df_4h, df_daily, **params) -> DetectorResult:
        """Pipeline complet : detect_raw → remap 4h → hysteresis 4h."""
```

**IMPORTANT — Hysteresis sur 4h (pas daily) :**
1. `detect_raw()` produit les labels bruts sur le daily
2. `run()` remap les labels daily → 4h (chaque candle 4h hérite du jour)
3. `_apply_hysteresis()` s'applique sur les labels 4h remappés (les params H_down/H_up sont en candles 4h)

Hysteresis asymétrique :
- `h_down` : candles 4h pour transition vers état plus sévère (bull→bear, *→crash)
- `h_up` : candles 4h pour transition vers état moins sévère (bear→bull, crash→*)
- Sévérité : bull(0) < range(1) < bear(2) < crash(3)

### Détecteur 1 : SMA200 + Stress rapide (`SMAStressDetector`)

**detect_raw (daily)** :
- Calcule SMA(sma_period, daily) sur close
- Calcule drawdown glissant sur stress_window jours
- Matrice : stress ON → crash, sinon close > SMA → bull, close < SMA → bear

**Grid** : sma_period [150,200,250] × stress_window [5,7,10,14] × stress_threshold [-15,-20,-25] × h_down [6,12,18] × h_up [24,36,48] = **324 combos**

### Détecteur 2 : EMA 50/200 + ATR Ratio (`EMAATRDetector`)

**detect_raw (daily)** :
- EMA(fast) vs EMA(slow) pour tendance
- ATR(fast) / ATR(slow) > seuil pour stress

**Grid** : ema_fast [30,50] × ema_slow [150,200] × atr_fast [5,7] × atr_slow [20,30] × atr_stress_ratio [1.5,2.0,2.5] × h_down [6,12,18] × h_up [24,36,48] = **432 combos**

### Détecteur 3 : Multi-MA + Vol Percentile (`MultiMAVolDetector`)

**detect_raw (daily)** :
- 3 états tendance : close > SMA50 ET SMA50 > SMA200 → bull, opposé → bear, sinon → range
- Stress : vol réalisée > percentile historique rolling

**Grid** : vol_window [7,14,21] × vol_percentile [90,95,97] × vol_lookback [180,365] × h_down [6,12] × h_up [24,36] = **72 combos**

**Limitation connue** : sma_fast=50, sma_slow=200 sont fixés (pas explorés). Si le Détecteur 3 gagne, explorer d'autres MAs en Sprint 50b. Le noter dans le rapport.

**Total : 828 combinaisons** sur ~17k candles 4h / ~2900 daily. Simple for loop, ~2-5 min.

### Gestion du warmup

SMA(200) daily nécessite 200 jours de données (~1200 candles 4h). Les candles avant ce warmup n'ont pas de signal fiable.

**Solution** : chaque détecteur retourne aussi `warmup_end_idx` (index de la première candle 4h avec signal valide). `regime_analysis.py` exclut les candles avant `max(warmup_end_idx)` de tous les détecteurs pour comparer sur la même période. En pratique, ça exclut ~200 premiers jours (août 2017 → mars 2018).

### Métriques (dans `regime_detectors.py`, pas de scikit-learn)

- `accuracy(y_true, y_pred)` — % correct
- `f1_per_class(y_true, y_pred, labels)` — precision/recall/F1 par régime
- `confusion_matrix_manual(y_true, y_pred, labels)` — matrice 4×4
- `crash_detection_delay(y_true, y_pred)` — délai moyen/max par crash, faux positifs
- `n_transitions(labels)` — nombre de changements
- `avg_regime_duration(labels, candle_hours)` — durée moyenne par régime
- `stability_score(labels, min_duration_candles)` — fraction de segments stables

---

## Étape 4 — Analyse et Rapport (Phase 3)

### Créer `scripts/regime_analysis.py`

Script principal d'orchestration.

**CLI** : `uv run python -m scripts.regime_analysis`
Args : `--detector` (optionnel, filtre un seul), `--skip-plots`

**Pipeline :**
1. Charge `data/btc_4h_labeled.csv` + resample daily
2. Pour chaque détecteur × chaque combinaison :
   - `detector.run(df_4h, df_daily, **params)`
   - Calcul de toutes les métriques vs ground truth
   - Calcul du macro F1 (moyenne des F1 par classe)
3. Tri par macro F1 décroissant → top 3 par détecteur
4. Analyse de robustesse : pour la meilleure config, vérifier que les configs voisines (±1 step par dimension) donnent F1 dans ±5%. WARNING si pic isolé.
5. Génération des plots (timeline prix + régime détecté vs ground truth) → `docs/images/`
6. Génération du rapport → `docs/regime_detector_report.md`

**Contenu du rapport :**
1. Résumé exécutif : quel détecteur recommandé et pourquoi
2. Tableau comparatif top 3 par détecteur (trié par F1 macro)
3. Par détecteur : meilleure config, métriques, confusion matrix, timeline plot
4. Analyse des crashs : tableau par crash (délai, faux positifs, détection avant/après point bas)
5. Analyse de robustesse : robuste ou WARNING pic isolé
6. Recommandation finale avec seuils optimaux

**Plots** (dans `docs/images/`, créer le répertoire) :
- `btc_ground_truth_regimes.png` (généré par regime_labeler)
- `regime_sma_stress_best.png`
- `regime_ema_atr_best.png`
- `regime_multi_ma_vol_best.png`

Tous utilisent `matplotlib.use("Agg")` (backend non-interactif).

---

## Étape 5 — Tests

### Créer `tests/test_regime_detectors.py`

**~25-30 tests** organisés en classes :

**TestResample4hToDaily** (~3 tests) :
- Agrégation basique 6 candles → 1 daily correct
- Multiple jours
- Jour incomplet filtré

**TestLabelCandles** (~4 tests) :
- Crash override bear quand overlap
- Candles hors événements → default_regime "range"
- Gap entre événements → default_regime
- Priorité crash > bear > range > bull

**TestSMAStressDetector** (~3 tests) :
- Prix en hausse → bull après warmup
- Crash sharp → détecté
- Bear sous SMA

**TestEMAATRDetector** (~2 tests) :
- Tendance haussière → bull
- ATR spike → crash

**TestMultiMAVolDetector** (~2 tests) :
- 3 états détectés (bull/range/bear)
- Vol spike → crash

**TestHysteresis** (~3 tests) :
- Prévient l'oscillation rapide (alternance raw → peu de transitions smoothed)
- Asymétrique : entre en bear vite (h_down), sort lentement (h_up)
- crash → range : nécessite h_up candles

**TestMetrics** (~8 tests) :
- accuracy parfaite / zéro
- F1 parfaite
- Confusion matrix diagonale
- Crash delay immédiat
- n_transitions correct
- stability_score 1.0 vs 0.0
- avg_regime_duration
- Crash false positives comptés correctement

**Total : ~25 tests minimum**, aucune dépendance à la DB (données synthétiques).

---

## Étape 6 — Documentation

### Modifier `COMMANDS.md`

Ajouter section **22. Analyse de Régime BTC (Sprint 50a)** :

```markdown
## 22. Analyse de Régime BTC (Sprint 50a)

### Prérequis : dépendances d'analyse
uv sync --group analysis

### Phase 0 — Données
uv run python -m scripts.backfill_candles --symbol BTC/USDT --timeframe 4h --since 2017-01-01
uv run python -m scripts.export_btc_4h

### Phase 1 — Ground Truth
uv run python -m scripts.regime_labeler

### Phase 2-3 — Détection et Analyse
uv run python -m scripts.regime_analysis
uv run python -m scripts.regime_analysis --detector sma_stress
uv run python -m scripts.regime_analysis --skip-plots
```

---

## Fichiers créés / modifiés

| Fichier | Action | Lignes estimées |
|---------|--------|-----------------|
| `pyproject.toml` | Modifier (ajouter groupe analysis) | +4 |
| `data/btc_regime_events.yaml` | Créer | ~120 |
| `scripts/export_btc_4h.py` | Créer | ~60 |
| `scripts/regime_labeler.py` | Créer | ~150 |
| `scripts/regime_detectors.py` | Créer (module principal) | ~500 |
| `scripts/regime_analysis.py` | Créer (orchestration) | ~350 |
| `tests/test_regime_detectors.py` | Créer | ~400 |
| `COMMANDS.md` | Modifier (ajouter section 22) | +25 |
| `docs/regime_detector_report.md` | Généré par script | auto |
| `docs/images/*.png` | Générés par scripts | auto |

**Fichiers NON modifiés** : tout `backend/`, `frontend/`, `config/`

---

## Ordre d'implémentation

1. `pyproject.toml` + `uv sync --group analysis`
2. `data/btc_regime_events.yaml`
3. `scripts/export_btc_4h.py`
4. `scripts/regime_labeler.py`
5. `scripts/regime_detectors.py` (module : détecteurs + métriques + resampling)
6. `scripts/regime_analysis.py` (orchestration + rapport + plots)
7. `tests/test_regime_detectors.py`
8. `COMMANDS.md`

---

## Vérification

1. `uv sync --group analysis` — installe matplotlib
2. Backfill : `uv run python -m scripts.backfill_candles --symbol BTC/USDT --timeframe 4h --since 2017-01-01`
3. Export : `uv run python -m scripts.export_btc_4h` — vérifie ~17500+ lignes dans le CSV
4. Labeler : `uv run python -m scripts.regime_labeler` — vérifie le résumé et le plot
5. Tests : `uv run pytest tests/test_regime_detectors.py -x -q` — ~25 tests passent
6. Régression : `uv run pytest tests/ -x -q` — 0 régression (les nouveaux scripts ne touchent pas au code existant)
7. Analyse complète : `uv run python -m scripts.regime_analysis` — génère rapport + plots
8. Vérifier `docs/regime_detector_report.md` et `docs/images/*.png`

---

## Fonctions existantes réutilisées

| Fonction | Fichier | Usage |
|----------|---------|-------|
| `backfill_candles.py` | `scripts/backfill_candles.py` | Fetch BTC 4h depuis Binance (Phase 0) |
| `Database()` + `db.init()` | `backend/core/database.py` | Lecture des candles pour export |
| `get_config()` | `backend/core/config.py` | Chargement config dans export |

**PAS de réutilisation** de `backend/core/indicators.py` — les scripts utilisent leurs propres helpers pandas-natifs pour éviter la conversion numpy↔pandas.
