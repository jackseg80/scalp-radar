# Plan — Sprint 37 : Timeframe Coherence Guard

## Contexte

Le WFO peut sélectionner 4h ou 1d comme meilleur timeframe pour certains assets, mais
le portfolio backtest et le paper/live tournent exclusivement en 1h. BCH/BNB (grid_atr, 4h)
produisaient 0 trades en portfolio malgré Grade A. Ce sprint ajoute des gardes-fous
bloquants pour détecter ces conflits tôt et guider l'utilisateur vers la résolution.

## Fichiers à modifier

| Fichier | Modifications |
|---------|--------------|
| `scripts/optimize.py` | Nouveaux flags argparse, `apply_from_db()` étendu |
| `backend/backtesting/portfolio_engine.py` | `TimeframeConflictError` + guard dans `run()` |
| `scripts/portfolio_backtest.py` | Catch + affichage `TimeframeConflictError` |
| `backend/api/optimization_routes.py` | Retour HTTP 409 si bloqué |
| `tests/test_timeframe_coherence.py` | Nouveau fichier, 11 tests |
| `docs/STRATEGIES.md` | Workflow mis à jour avec étape 2b |

## Partie A — scripts/optimize.py

### A1. Nouveaux arguments argparse (après la ligne 752)

```python
parser.add_argument(
    "--force-timeframe", type=str, default=None,
    help="Forcer le timeframe WFO (ex: 1h). Override la grid."
)
parser.add_argument(
    "--symbols", type=str, default=None,
    help="Liste de symbols séparés par virgule (ex: BCH/USDT,BNB/USDT)"
)
parser.add_argument(
    "--exclude", type=str, default=None,
    help="Assets à exclure de --apply (CSV). Ex: BCH/USDT,BNB/USDT"
)
parser.add_argument(
    "--ignore-tf-conflicts", action="store_true", default=False,
    help="Forcer --apply en ignorant les outliers timeframe (les exclut silencieusement)"
)
```

### A2. --force-timeframe dans run_optimization() / section param_grid

Chercher où `param_grid` est chargé depuis `param_grids.yaml`, APRÈS ce chargement :

```python
if args.force_timeframe:
    ft = args.force_timeframe
    if "timeframe" in param_grid:
        original_tfs = param_grid["timeframe"]
        if ft in original_tfs:
            param_grid["timeframe"] = [ft]
            logger.info("Timeframe forcé à [{}] (original: {})", ft, original_tfs)
        else:
            logger.error("Timeframe '{}' absent de la grid (disponibles: {})", ft, original_tfs)
            sys.exit(1)
    else:
        param_grid["timeframe"] = [ft]
        logger.info("Timeframe [{}] injecté dans la grid", ft)
```

### A3. --symbols dans main() — résolution des assets à optimiser

**Mutex obligatoire** — avant la résolution des assets, vérifier l'exclusion mutuelle :

```python
if sum(bool(x) for x in [args.symbol, args.symbols, args.all_symbols]) > 1:
    parser.error("Utilisez --symbol, --symbols OU --all-symbols (pas plusieurs)")
```

Dans la boucle qui détermine quels symbols optimiser (où `--symbol` et `--all-symbols` sont
traités), ajouter :

```python
elif args.symbols:
    symbols_to_run = [s.strip() for s in args.symbols.split(",") if s.strip()]
```

### A4. Signature étendue de apply_from_db()

```python
def apply_from_db(
    strategy_names: list[str],
    config_dir: str = "config",
    db_path: str | None = None,
    exclude_symbols: list[str] | None = None,
    ignore_tf_conflicts: bool = False,
) -> dict:
```

### A5. Requête SQL — ajouter `timeframe`

Ligne ~527, modifier :
```python
# AVANT
f"""SELECT strategy_name, asset, grade, total_score, best_params
    FROM optimization_results
    WHERE is_latest = 1 AND strategy_name IN ({placeholders})
    ORDER BY strategy_name, asset"""

# APRÈS
f"""SELECT strategy_name, asset, timeframe, grade, total_score, best_params
    FROM optimization_results
    WHERE is_latest = 1 AND strategy_name IN ({placeholders})
    ORDER BY strategy_name, asset"""
```

### A6. Entry dict — ajouter timeframe

```python
entry = {
    "asset": row["asset"],
    "grade": row["grade"],
    "total_score": row["total_score"],
    "best_params": json.loads(row["best_params"]) if row["best_params"] else {},
    "timeframe": row["timeframe"] or "1h",   # ← AJOUT
}
```

### A7. Détection et blocage dans la boucle par stratégie (après construction de eligible)

Juste AVANT l'écriture des paramètres dans `new_per_asset`, après que `eligible_assets`
est constitué de tous les Grade A/B :

```python
from collections import Counter

# 1. Appliquer exclusions manuelles
if exclude_symbols:
    eligible = [r for r in results if r["asset"] not in exclude_symbols and r["grade"] in ("A", "B")]
else:
    eligible = [r for r in results if r["grade"] in ("A", "B")]

# 2. Timeframe majoritaire
if eligible:
    tf_counts = Counter(r["timeframe"] for r in eligible)
    majority_tf, majority_count = tf_counts.most_common(1)[0]

    # Tiebreak : prendre le plus petit TF (1h < 4h < 1d)
    TF_ORDER = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4, "1d": 5}
    if len(tf_counts) > 1 and tf_counts.most_common(2)[0][1] == tf_counts.most_common(2)[1][1]:
        majority_tf = min(tf_counts.keys(), key=lambda tf: TF_ORDER.get(tf, 99))

    outliers = [r for r in eligible if r["timeframe"] != majority_tf]

    if outliers and not ignore_tf_conflicts:
        # BLOQUER
        print(f"\n  ❌  TIMEFRAME CONFLICT — --apply BLOQUÉ\n")
        print(f"  Timeframe majoritaire : {majority_tf} "
              f"({tf_counts[majority_tf]}/{len(eligible)} assets A/B)\n")
        print(f"  Outliers :")
        outlier_symbols = []
        for r in outliers:
            print(f"    {r['asset']:15s} Grade {r['grade']} ({r['total_score']})  "
                  f"timeframe={r['timeframe']}")
            outlier_symbols.append(r["asset"])
        symbols_csv = ",".join(outlier_symbols)
        print(f"\n  Actions requises :")
        print(f"    1. Re-tester en {majority_tf} :")
        print(f"       uv run python -m scripts.optimize --strategy {strat_name} "
              f"--symbols {symbols_csv} --force-timeframe {majority_tf}")
        print(f"    2. Exclure :")
        print(f"       uv run python -m scripts.optimize --strategy {strat_name} "
              f"--apply --exclude {symbols_csv}")
        print(f"    3. Forcer (exclut les outliers silencieusement) :")
        print(f"       uv run python -m scripts.optimize --strategy {strat_name} "
              f"--apply --ignore-tf-conflicts")
        print(f"\n  Aucune modification effectuée.\n")
        return {
            "changed": False,
            "blocked": True,
            "reason": "tf_conflict",
            "majority_tf": majority_tf,
            "tf_outliers": outlier_symbols,
            "applied": [],
            "removed": [],
            "excluded": [],
            "grades": {},
            "backup": None,
            "assets_added": [],
        }

    if outliers and ignore_tf_conflicts:
        outlier_assets = {r["asset"] for r in outliers}
        eligible = [r for r in eligible if r["asset"] not in outlier_assets]
        print(f"  ℹ️  {len(outliers)} outliers timeframe exclus (--ignore-tf-conflicts)")
```

**Positionnement :** Le check timeframe est PAR STRATÉGIE, dans la boucle
`for strat_name in strategy_names:`. Le `return` sort directement de `apply_from_db()`
au premier conflit détecté. C'est correct car `--apply` traite une stratégie à la fois
en pratique (le CLI passe `[args.strategy]`). Si plusieurs stratégies sont passées,
on bloque à la première conflictuelle — l'utilisateur doit résoudre avant de continuer.

### A8. Passage des flags dans main()

Dans le bloc `if args.apply:` standalone (ligne ~768) :
```python
exclude_list = [s.strip() for s in args.exclude.split(",")] if args.exclude else None
result = apply_from_db(
    [args.strategy] if args.strategy else available_strategies,
    args.config_dir,
    exclude_symbols=exclude_list,
    ignore_tf_conflicts=args.ignore_tf_conflicts,
)
if result.get("blocked"):
    sys.exit(1)
```

Idem dans le bloc après optimisation (ligne ~895).

---

## Partie B — backend/backtesting/portfolio_engine.py

### B1. Nouvelle exception (après les imports, avant la classe)

```python
class TimeframeConflictError(Exception):
    """Levée si un runner a un timeframe incompatible avec le portfolio (1h)."""
    def __init__(
        self,
        mismatched: list[tuple[str, str]],
        expected_tf: str,
        all_runner_keys: list[str],
    ):
        self.mismatched = mismatched
        self.expected_tf = expected_tf
        # Clés des runners valides — permet au script d'afficher --assets suggestion
        bad_keys = {k for k, _ in mismatched}
        self.valid_keys = [k for k in all_runner_keys if k not in bad_keys]
        super().__init__(
            f"{len(mismatched)} runners avec timeframe incompatible "
            f"(attendu {expected_tf})"
        )
```

### B2. Guard dans run() après _create_runners() (ligne ~253)

```python
# Check cohérence timeframe (portfolio = 1h seulement)
# TODO: si le portfolio supporte multi-TF à l'avenir,
# déduire expected_tf depuis la majorité des runners
expected_tf = "1h"
mismatched = []
for runner_key, runner in runners.items():
    runner_tf = getattr(runner._strategy._config, "timeframe", expected_tf)
    if runner_tf != expected_tf:
        mismatched.append((runner_key, runner_tf))

if mismatched:
    raise TimeframeConflictError(mismatched, expected_tf, list(runners.keys()))
```

---

## Partie C — scripts/portfolio_backtest.py

### C1. Import

```python
from backend.backtesting.portfolio_engine import (
    PortfolioBacktester,
    PortfolioResult,
    TimeframeConflictError,   # ← AJOUT
    format_portfolio_report,
)
```

### C2. Catch dans la boucle principale (autour du `backtester.run()`)

```python
try:
    result = await backtester.run(start, end, db_path=args.db)
except TimeframeConflictError as e:
    print(f"\n  ❌  TIMEFRAME CONFLICT — portfolio backtest ANNULÉ\n")
    print(f"  {len(e.mismatched)} runners incompatibles "
          f"(portfolio = {e.expected_tf}) :\n")
    for key, tf in e.mismatched:
        print(f"    {key} (WFO timeframe = {tf})")
    bad_strats = sorted({key.split(":", 1)[0] for key, _ in e.mismatched})
    print(f"\n  💡 Corrigez avec --force-timeframe :")
    for strat in bad_strats:
        strat_bads = sorted({key.split(":", 1)[1] for key, _ in e.mismatched
                             if key.startswith(strat + ":")})
        print(f"     uv run python -m scripts.optimize --strategy {strat} "
              f"--symbols {','.join(strat_bads)} "
              f"--force-timeframe {e.expected_tf}")
    # Suggestion --assets via e.valid_keys (peuplé par TimeframeConflictError)
    if e.valid_keys:
        valid_assets = sorted({
            k.split(":", 1)[1] if ":" in k else k
            for k in e.valid_keys
        })
        print(f"\n  Ou relancez sans les assets conflictuels :")
        print(f"     --assets {','.join(valid_assets)}")
    print()
    sys.exit(1)
```

---

## Partie D — backend/api/optimization_routes.py

### D1. Retour HTTP 409 si bloqué

Dans `POST /apply`, après `result = apply_from_db(...)` :

```python
if result.get("blocked"):
    raise HTTPException(
        status_code=409,
        detail={
            "error": "tf_conflict",
            "message": "Conflit de timeframe détecté — apply bloqué",
            "majority_tf": result.get("majority_tf"),
            "tf_outliers": result.get("tf_outliers", []),
        },
    )
return result
```

### D2. Nouveaux paramètres query

```python
@router.post("/apply")
async def apply_optimization_params(
    strategy_name: str | None = Query(default=None, ...),
    ignore_tf_conflicts: bool = Query(default=False),
    exclude: str | None = Query(default=None),
) -> dict:
    ...
    exclude_list = [s.strip() for s in exclude.split(",")] if exclude else None
    result = apply_from_db(
        strategy_names,
        exclude_symbols=exclude_list,
        ignore_tf_conflicts=ignore_tf_conflicts,
    )
```

---

## Partie E — tests/test_timeframe_coherence.py (nouveau fichier)

11 tests unitaires :

1. `test_majority_tf_simple` — Counter 19×1h + 2×4h → majority_tf="1h"
2. `test_majority_tf_all_same` — 20×1h → 0 outliers
3. `test_majority_tf_tie` — 10×1h + 10×4h → prend "1h" (plus petit TF)
4. `test_apply_blocked_on_conflict` — mock DB retourne 1h+4h → `{"blocked": True, "reason": "tf_conflict"}`
5. `test_apply_blocked_exit_code` — `sys.exit(1)` via pytest.raises(SystemExit) quand bloqué via main()
6. `test_apply_with_ignore_flag` — `ignore_tf_conflicts=True` → outliers exclus, apply réussit
7. `test_apply_with_exclude_flag` — `exclude_symbols=["BCH/USDT"]` → BCH absent, apply réussit
8. `test_force_timeframe_filters_grid` — param_grid["timeframe"] = ["1h", "4h"] → filtré à ["1h"]
9. `test_force_timeframe_invalid_value` — "2h" pas dans la grid → sys.exit(1)
10. `test_portfolio_raises_on_tf_conflict` — runner mock avec `_config.timeframe="4h"` → `TimeframeConflictError` raised, `valid_keys` peuplé
11. `test_apply_succeeds_after_conflict_resolved` — DB avec tous les TF alignés (1h) → `blocked` absent, `changed=True`

**Pattern des tests DB** : utiliser `sqlite3.connect(":memory:")` avec le schéma exact
de `optimization_results` (copié de `database.py` lignes ~240-250). Insérer des fixtures
avec timeframes mixtes. Voir `tests/test_optimization_db.py` pour le pattern existant
de fixture `temp_db`.

---

## Partie F — Frontend ResearchPage.jsx + ResearchPage.css

### F1. Badge timeframe dans le tableau

**Vérifié** : `GET /api/optimization/results` retourne déjà `r.timeframe` dans chaque
résultat (`optimization_db.py` ligne 494 : `SELECT r.id, r.strategy_name, r.asset, r.timeframe, ...`).
Aucun changement backend nécessaire pour cette partie.

**Tableau header** — ajouter une colonne "TF" (5%) après la colonne "Asset" :
```jsx
<th style={{ width: '5%' }}>TF</th>
```

**Cellule dans chaque ligne** — badge orange si `timeframe !== "1h"`, gris sinon :
```jsx
<td>
  <span
    className={`timeframe-badge ${r.timeframe && r.timeframe !== '1h' ? 'timeframe-badge--warn' : ''}`}
    title={r.timeframe !== '1h' ? `Optimisé en ${r.timeframe}. Incompatible avec paper/live (1h). Re-testez avec --force-timeframe 1h.` : `Timeframe : ${r.timeframe}`}
  >
    {r.timeframe || '1h'}
  </span>
</td>
```

**CSS dans ResearchPage.css** :
```css
.timeframe-badge {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 600;
  background: #1e293b;
  color: #94a3b8;
}
.timeframe-badge--warn {
  background: #431407;
  color: #fb923c;
  border: 1px solid #9a3412;
}
```

### F2. Modale conflit 409 dans handleApply

Ajouter un état pour le conflit :
```jsx
const [tfConflict, setTfConflict] = useState(null)
// tfConflict = { majority_tf, tf_outliers, strategy_name }
```

Modifier le handler pour catcher le 409 :
```jsx
const handleApply = async () => {
  // ...
  try {
    const qs = new URLSearchParams()
    if (filters.strategy) qs.set('strategy_name', filters.strategy)
    const resp = await fetch(`/api/optimization/apply?${qs}`, { method: 'POST' })
    if (resp.status === 409) {
      const err = await resp.json().catch(() => ({}))
      const detail = err.detail || {}
      setTfConflict({
        majority_tf: detail.majority_tf || '?',
        tf_outliers: detail.tf_outliers || [],
        strategy_name: filters.strategy || 'toutes',
      })
      return
    }
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }
    const json = await resp.json()
    setApplyResult(json)
    setTimeout(() => setApplyResult(null), 15000)
  } catch (err) {
    alert(`Erreur apply: ${err.message}`)
  } finally {
    setApplying(false)
  }
}
```

Modale affichée quand `tfConflict !== null` (juste après le banner applyResult) :
```jsx
{tfConflict && (
  <div className="tf-conflict-modal">
    <div className="tf-conflict-content">
      <h3>❌ Conflit de timeframe — apply bloqué</h3>
      <p>
        Timeframe majoritaire : <strong>{tfConflict.majority_tf}</strong><br/>
        Outliers ({tfConflict.tf_outliers.length} asset{tfConflict.tf_outliers.length > 1 ? 's' : ''}) :
      </p>
      <ul>
        {tfConflict.tf_outliers.map(s => <li key={s}><code>{s}</code></li>)}
      </ul>
      <p>Actions :</p>
      <ol>
        <li>Re-tester en {tfConflict.majority_tf} :
          <code className="cmd">
            uv run python -m scripts.optimize --strategy {tfConflict.strategy_name}{' '}
            --symbols {tfConflict.tf_outliers.join(',')} --force-timeframe {tfConflict.majority_tf}
          </code>
        </li>
        <li>
          <button
            className="btn-secondary"
            onClick={() => {
              const excludeParam = tfConflict.tf_outliers.join(',')
              const qs = new URLSearchParams()
              if (filters.strategy) qs.set('strategy_name', filters.strategy)
              qs.set('exclude', excludeParam)
              // Relancer apply avec exclude
              fetch(`/api/optimization/apply?${qs}`, { method: 'POST' })
                .then(r => r.json()).then(j => { setApplyResult(j); setTfConflict(null) })
            }}
          >
            Exclure les outliers et appliquer
          </button>
        </li>
        <li>
          <button
            className="btn-secondary"
            onClick={() => {
              const qs = new URLSearchParams()
              if (filters.strategy) qs.set('strategy_name', filters.strategy)
              qs.set('ignore_tf_conflicts', 'true')
              fetch(`/api/optimization/apply?${qs}`, { method: 'POST' })
                .then(r => r.json()).then(j => { setApplyResult(j); setTfConflict(null) })
            }}
          >
            Forcer (exclure silencieusement)
          </button>
        </li>
      </ol>
      <button className="btn-close" onClick={() => setTfConflict(null)}>Fermer</button>
    </div>
  </div>
)}
```

**CSS** pour la modale dans ResearchPage.css :
```css
.tf-conflict-modal {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.7); z-index: 1000;
  display: flex; align-items: center; justify-content: center;
}
.tf-conflict-content {
  background: #1e293b; border: 1px solid #9a3412;
  border-radius: 8px; padding: 24px; max-width: 600px;
  color: #f1f5f9;
}
.tf-conflict-content h3 { color: #fb923c; margin-bottom: 12px; }
.tf-conflict-content code.cmd {
  display: block; background: #0f172a; padding: 8px;
  border-radius: 4px; font-size: 11px; margin: 4px 0 8px;
  color: #7dd3fc; word-break: break-all;
}
.btn-secondary { background: #334155; color: #f1f5f9; border: none;
  padding: 6px 12px; border-radius: 4px; cursor: pointer; }
.btn-close { margin-top: 16px; background: #475569; color: #f1f5f9;
  border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
```

---

## Partie G — docs/STRATEGIES.md

Ajouter l'étape **2b** dans le workflow A, et la règle "Timeframe unifié" dans
"Règles générales", conformément au spec du sprint.

---

## Ordre d'implémentation

1. `scripts/optimize.py` — A1 (argparse) + A2 (--force-timeframe) + A3 (--symbols + mutex)
2. `scripts/optimize.py` — A4-A8 (apply_from_db étendu + main)
3. `backend/backtesting/portfolio_engine.py` — B1-B2 (TimeframeConflictError + guard)
4. `scripts/portfolio_backtest.py` — C1-C2 (catch + affichage via e.valid_keys)
5. `backend/api/optimization_routes.py` — D1-D2 (HTTP 409 + params)
6. `tests/test_timeframe_coherence.py` — E (11 tests)
7. `frontend/src/ResearchPage.jsx` + `.css` — F (badge TF + modale 409)
8. `docs/STRATEGIES.md` — G (documentation)

---

## Vérification

```bash
# Tests
uv run pytest tests/test_timeframe_coherence.py -v
uv run pytest tests/ -x -q

# --force-timeframe (dry-run pour ne pas lancer un vrai WFO)
uv run python -m scripts.optimize --strategy grid_atr \
    --symbol BCH/USDT --force-timeframe 1h --dry-run

# --apply BLOQUÉ si conflit (exit code 1)
uv run python -m scripts.optimize --strategy grid_atr --apply
# Si pas de conflit actuel : tester avec mock DB ou insérer manuellement une ligne 4h

# --apply avec --exclude
uv run python -m scripts.optimize --strategy grid_atr \
    --apply --exclude BCH/USDT,BNB/USDT

# --apply avec --ignore-tf-conflicts
uv run python -m scripts.optimize --strategy grid_atr --apply --ignore-tf-conflicts

# Portfolio guard
uv run python -m scripts.portfolio_backtest --strategy grid_atr --days 365
```
