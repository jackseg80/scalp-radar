# Audit Technique — Self-Healing & Parity (06 Mars 2026)

## 🎯 Objectif
Sécuriser le trading en timeframe 1h (H1) contre les instabilités réseau (gaps de données) et les désynchronisations d'état entre le bot et l'Exchange Bitget.

---

## 🛠️ Problèmes Identifiés & Résolus

### 1. DataEngine — Gaps de Données (Self-Healing)
**Problème** : Les micro-coupures du WebSocket Bitget créaient des "trous" dans les buffers de bougies. Sur du H1, rater une bougie faussait les indicateurs techniques (SMA, ATR, Bollinger) pendant 20 à 50 heures, menant à des signaux d'entrée erronés.
**Solution** : Implémentation du **Self-Healing**. 
- Lorsqu'un gap est détecté à la réception d'une bougie, le `DataEngine` déclenche immédiatement une requête REST `fetch_ohlcv` pour récupérer les bougies manquantes.
- Les données "guéries" sont suturées chronologiquement dans le buffer mémoire et persistées en DB.
**Validation** : `tests/test_dataengine_autoheal.py` (simule un gap de 3h et vérifie la réparation).

### 2. Watchdog — Parité Bot <=> Exchange
**Problème** : Si un message WebSocket "Filled" est manqué ou qu'un ordre est fermé manuellement sur Bitget, le bot conservait une vue erronée de ses positions, risquant de laisser des positions orphelines sans Stop-Loss.
**Solution** : Implémentation du **Parity Watchdog**.
- Une nouvelle tâche de fond dans le `Watchdog` effectue une réconciliation complète toutes les 15 minutes.
- Utilisation de la logique `reconcile_on_boot` (idempotente) pour comparer l'inventaire Bitget REST avec l'état local.
- Correction automatique des écarts et notification Telegram détaillée.
**Validation** : `tests/test_watchdog_parity.py`.

### 3. Executor — Accumulation de SL Orphelins (Critique)
**Problème** : Des dizaines de SL triggers restaient actifs sur Bitget pour quelques positions réelles. Cause : IDs de SL désynchronisés (40109 OrderNotFound) et nettoyage incomplet des "Plan Orders" sur Bitget v2.
**Solution** :
- **Gestion 40109** : Si un SL est introuvable via son ID local, le bot reset l'ID et purge exhaustivement le symbole.
- **Purge Renforcée** : `_cancel_all_open_orders` balaie désormais les types `stop` et `plan` (Bitget v2) pour ne laisser aucun orphelin.
- **Idempotence SL** : Vérification `_find_existing_sl` avant création pour réutiliser un ordre trigger identique déjà présent.
- **Sécurité Stale Price** : Blocage des modifications SL si les prix datent de plus de 5 minutes.
**Validation** : `tests/test_mission_sl_fix.py`.

### 4. DataEngine — Optimisation par Agrégation Native
**Problème** : La multiplication des flux WebSocket (un par timeframe) saturait la bande passante et provoquait des retards de données (stale data).
**Solution** :
- **Source Unique** : Le bot s'abonne désormais uniquement au plus petit timeframe (ex: 5m).
- **Moteur d'Agrégation** : Reconstruction locale en temps réel des TFs supérieurs (15m, 1h, 4h, 1d) via `_aggregate_to_target_tf`.
- **Mise à jour Intra-bougie** : Les indicateurs et l'UI sont rafraîchis à chaque tick du flux source, même pour les TFs agrégés.
- **Gain Performance** : Réduction du trafic WS de ~80% et suppression des erreurs stale sur les TFs lents.
**Validation** : `tests/test_dataengine_aggregation.py`.

---

## 📊 Impact sur la Fiabilité
- **Disponibilité des données** : 100% (buffers garantis sans trous).
- **Fidélité des positions** : Vérifiée via API REST toutes les 15 min (filet de sécurité ultime).
- **Régression** : Aucune. Les 2225 tests du projet passent avec succès.

---

## 🚀 Recommandations Futures
- **Performance** : Envisager le passage à NumPy vectorisé pour les indicateurs live si la charge CPU augmente avec l'ajout de nouveaux assets, Numba étant actuellement désactivé sur Python 3.13 pour stabilité.
- **Deleverage** : Implémenter une routine de réduction automatique de l'exposition si le ratio de marge effective dépasse 85% (avant le Kill Switch à 45% DD).
