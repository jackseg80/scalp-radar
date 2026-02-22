# Audit #4 — Fees réelles Bitget vs modèle backtest — 2026-02-22

## Résumé exécutif

- **Écart fees : 0%** — le modèle backtest est parfaitement aligné avec la réalité Bitget
- **544 fills analysés** sur 30 jours (11 → 22 février 2026), ~51k$ notional, 19 assets
- **100% taker** — l'Executor n'utilise que des market orders, aucun maker fill détecté
- **Aucune correction de `risk.yaml` nécessaire**

---

## Contexte

Le backtest utilise les hypothèses suivantes (`risk.yaml`) :

| Paramètre | Valeur | Usage |
|---|---|---|
| `maker_percent` | 0.02% | TP limit orders |
| `taker_percent` | 0.06% | Entry market + SL market + signal_exit market |
| `slippage.default_estimate_percent` | 0.05% | SL + signal_exit (market orders) |

L'objectif était de vérifier que ces hypothèses ne rendent pas les backtests structurellement optimistes.

---

## Méthodologie

Script `scripts/audit_fees.py` (read-only, standalone) :

1. Connexion ccxt Bitget swap avec les credentials `.env`
2. Fetch `fetch_my_trades()` pour les 19 assets actifs `grid_atr` (avec pagination)
3. Détection taker/maker via `takerOrMaker` (champ ccxt fiable — `type` est toujours `null` sur Bitget swap futures)
4. Comparaison fee réelle (`fee.cost / notional`) vs taux modèle

**Finding technique** : Bitget swap retourne `type: null` dans l'objet unifié ccxt. Le champ fiable est `takerOrMaker` (`"taker"` ou `"maker"`) ou `info.tradeScope` en fallback.

---

## Résultats fees

### Par type d'ordre

| Type | Modèle | Réel moyen | Réel médian | Écart |
|---|---|---|---|---|
| **Taker** | 0.060% | 0.0600% | 0.0600% | **0%** |
| Maker | 0.020% | N/A (0 fills) | N/A | — |

### Global

| Métrique | Valeur |
|---|---|
| Fills analysés | 544 |
| Notional total | 50 925 $ |
| Fee totale réelle | 30.56 USDT |
| Fee modèle estimée | 30.56 USDT |
| **Écart global** | **+0.0%** |

### Par asset (extrait)

Tous les 19 assets affichent exactement 0.0600% de fee réelle. Aucun asset ne dévie du modèle.

---

## Résultats slippage

**Non mesurable** avec les données actuelles. Le paper trading et le live ne tradent pas aux mêmes instants exacts — un matching temporel produirait des faux positifs.

**Méthode recommandée** : analyser les logs Executor (lignes `slippage detected`) sur 2-4 semaines quand le volume de trades live sera suffisant.

Modèle backtest conservé à **0.05%**.

---

## Conclusions

### Ce que cet audit confirme

1. **Les grades WFO ne sont pas surestimés côté fees** — la taker fee réelle = 0.06% = modèle exact
2. **L'Executor utilise exclusivement des market orders** — cohérent avec l'architecture (grid entries, SL serveur, signal_exit)
3. **Pas de discount BGB actif** sur le compte — fees standard Bitget VIP0
4. **Pas de fees en devise non-USDT** détectées

### Ce que cet audit ne couvre pas

- **Slippage réel** — non mesurable (voir méthode ci-dessus)
- **Maker fees** — aucun limit order exécuté sur la période (TP grid déclenché par SMA côté client, pas par limit order Bitget)
- **Funding costs** — couvert séparément dans le backtest (Sprint 26), pas dans cet audit

### Recommandation

Aucune modification de `risk.yaml`. Relancer cet audit dans 2-3 mois avec un historique plus long pour confirmer la stabilité.

---

## Script

```bash
uv run python -m scripts.audit_fees           # 30 derniers jours
uv run python -m scripts.audit_fees --days 7  # 7 jours
uv run python -m scripts.audit_fees --debug   # dump 3 trades JSON bruts
```

Commits : `7da069e` (création) · `f38b87d` (fix taker/maker + slippage)
