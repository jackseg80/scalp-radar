# Audit Complet — Scalp Radar
**Date :** 23 février 2026
**Scope :** Tests, Configuration, Code mort, Sécurité, Documentation

---

## Résumé Exécutif

| Domaine | État | Score |
|---------|------|-------|
| Tests | 1747/1749 passed (2 flaky) | ✅ 99.9% |
| Configuration YAML | 2 anomalies | ⚠️ |
| Code mort | ~15 scripts orphelins + 4 fichiers non trackés | ⚠️ |
| Sécurité | 2 endpoints critiques sans auth | 🔴 |
| Documentation | +10 tests non documentés | ✅ 99% |
| Architecture | 100% cohérent | ✅ |

**Niveau de risque global : MOYEN** — Le projet est solide mais 2 failles de sécurité et quelques incohérences config méritent attention.

---

## 1. Tests — 1749 collectés, 1747 passed, 2 flaky

**Résultat :** `2 failed, 1747 passed, 5 warnings` (159s)

### Tests en échec (flaky — passent en isolation)
| Test | Fichier | Cause |
|------|---------|-------|
| `test_boot_applies_strategy_leverage` | test_executor_boot_leverage.py | Interaction inter-tests (état partagé config) |
| `test_boot_populates_leverage_applied` | test_executor_boot_leverage.py | Idem |

**Diagnostic :** Les 2 tests passent à 100% quand exécutés seuls (`pytest tests/test_executor_boot_leverage.py` → 8/8 passed). Le problème vient d'un test précédent dans la suite qui pollue l'état global (probablement la config YAML chargée en mémoire).

**Action recommandée :** Ajouter une fixture `autouse` pour isoler la config dans ce module de tests.

### Warnings (5)
- 4× `ResourceWarning: Event loop is closed` (aiosqlite) — cosmétique, pas d'impact
- 1× `PytestUnhandledThreadExceptionWarning` — thread DB fermé après le test

---

## 2. Configuration YAML — 2 anomalies

### 🔴 CRITIQUE : grid_boltrend enabled=true MAIS live_eligible=false
- **Fichier :** config/strategies.yaml
- **Impact :** La stratégie tourne en paper trading (5 assets) mais ne pourra JAMAIS être déployée en live tant que `live_eligible` reste false
- **Action :** Décider si grid_boltrend doit être live-eligible ou non

### 🟠 ÉLEVÉE : boltrend leverage=2 (vs 3-8 pour les autres)
- **Fichier :** config/strategies.yaml
- **Impact :** Sizing ~2x moins agressif que les autres stratégies swing
- **Action :** Aligner à 3 (conservateur) ou 6 (standard grid)

### Observations mineures
- **grid_boltrend per_asset** : seulement 2/5 assets ont des overrides (BTC, DYDX) — les 3 autres (ETH, DOGE, LINK) utilisent les défauts. Probablement intentionnel
- **Poids (weight)** non normalisés (somme=3.45) — l'Arena normalise automatiquement
- **vwap_rsi, momentum, envelope_dca** : live_eligible=true mais enabled=false — pas d'erreur fonctionnelle

---

## 3. Code Mort — ~15 scripts orphelins

### Fichiers non trackés par git (4)
| Fichier | Lignes | Verdict |
|---------|--------|---------|
| `check_deploy.py` | 11 | Script debug one-shot → supprimer |
| `fix_assets.py` | 11 | Script one-shot obsolète → supprimer |
| `scripts/analyze_dd_correlation.py` | 313 | Recherche ad-hoc → archiver ou supprimer |
| `scripts/analyze_multi_tf_hypothesis.py` | 332 | Recherche ad-hoc → archiver ou supprimer |

### Scripts non documentés dans COMMANDS.md (~12)
`analyze_wfo_regression.py`, `audit_combo_score.py`, `audit_fees.py`, `audit_grid_states.py`, `check_history.py`, `check_live_sizing.py`, `compare_wfo.py`, `diagnose_margin.py`, `parity_check.py`, `wfo_worker.py`, etc.

**Action :** Supprimer les obsolètes, documenter les utiles dans COMMANDS.md section "Scripts Diagnostic"

### Points positifs
- ✅ **Zéro** module backend orphelin (tous importés)
- ✅ **Zéro** composant frontend orphelin (tous importés)
- ✅ Imports bien nettoyés dans les fichiers principaux
- ✅ Un seul TODO dans le code production (`simulator_routes.py:115`)

---

## 4. Sécurité — 2 endpoints critiques sans auth

### 🔴 P0 — Endpoints SANS authentification (modifications sensibles)

| Endpoint | Risque | Impact |
|----------|--------|--------|
| `POST /api/simulator/kill-switch/reset` | Désactive la protection pertes | **CRITIQUE** — ordres live sans filet |
| `POST /api/optimization/apply` | Modifie strategies.yaml | **CRITIQUE** — change les paramètres de trading |

### 🟠 P1 — Endpoints sans auth (consommation ressources)

| Endpoint | Risque |
|----------|--------|
| `POST /api/optimization/run` | Lance jobs CPU-intensifs |
| `POST /api/portfolio/run` | Lance backtests |
| `DELETE /api/optimization/jobs/{id}` | Annule des jobs |
| `DELETE /api/portfolio/backtests/{id}` | Supprime des résultats |
| `POST /api/data/backfill` | Déclenche téléchargements |

### ✅ Bien sécurisé
- Secrets dans `.env` (gitignored) — aucun secret hardcodé
- `.gitignore` complet (data/, .env, *.db, logs/)
- Pas d'injection SQL (paramètres via `?` partout)
- Pas d'eval/exec dangereux
- Subprocess sans `shell=True`
- CORS restreint à localhost:5173 (dev)

### Actions recommandées
1. **P0 :** Ajouter `dependencies=[Depends(verify_executor_key)]` sur kill-switch/reset et /apply (~5 lignes)
2. **P1 :** Ajouter rate-limiting ou auth sur les POST/DELETE non-auth
3. **P2 :** Configurer CORS dynamique via `.env` pour prod
4. **P2 :** Implémenter whitelist IP (mentionnée dans CLAUDE.md mais absente du code)

---

## 5. Documentation — 99% cohérente

| Document | État | Écart |
|----------|------|-------|
| ROADMAP.md | ⚠️ Mineur | Mentionne 1739 tests, réalité = 1749 (+10) |
| ARCHITECTURE.md | ✅ 100% | Tous composants existent |
| STRATEGIES.md | ✅ 100% | 16/16 stratégies documentées |
| COMMANDS.md | ✅ 100% | Tous scripts documentés existent |
| pyproject.toml | ✅ v1.0.0 | Cohérent avec docs |

**Action :** Mettre à jour le compteur de tests dans ROADMAP.md (1739 → 1749)

---

## 6. Métriques Projet

| Métrique | Valeur |
|----------|--------|
| Tests | 1749 collectés |
| Stratégies | 16 implémentées |
| Assets | 21 configurés |
| Fichiers Python backend | ~60 |
| Composants frontend | 47 JSX |
| Sprints complétés | 40+ (incluant hotfixes) |
| Version | v1.0.0 |
| Durée tests | 2min 40s |

---

## Actions Prioritaires

### P0 — Immédiat
1. Sécuriser `POST /api/simulator/kill-switch/reset` (ajouter auth)
2. Sécuriser `POST /api/optimization/apply` (ajouter auth)

### P1 — Court terme
3. Décider si grid_boltrend → live_eligible=true
4. Aligner leverage boltrend (2 → 3 ou 6)
5. Fixer les 2 tests flaky (isolation config)

### P2 — Maintenance
6. Nettoyer scripts orphelins (~12 fichiers)
7. Supprimer fichiers non trackés (check_deploy.py, fix_assets.py)
8. Mettre à jour compteur tests ROADMAP.md
9. CORS dynamique via .env pour prod
10. Rate-limiting sur endpoints non-auth

---

*Audit effectué par Claude Code — aucune modification de fichier effectuée.*
