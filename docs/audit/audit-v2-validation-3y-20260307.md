# Audit de Validation Stratégie Grid ATR v2 — Mars 2026

## 1. Objectif de l'étude
Comparer la performance long-terme (3 ans) entre la version standard (**v1**) et la version avec protection range/volatilité (**v2**).

**Données :** 1095 jours (Mars 2023 - Mars 2026)
**Portfolio :** 13 assets majeurs, capital 10 000 $, levier 6x.

## 2. Résultats Comparatifs

| Métrique | v1 (Standard) | v2 (Range Protected) | Écart / Impact |
| :--- | :--- | :--- | :--- |
| **Rendement Total** | **+111.6 %** | +107.4 % | -4.2 % (absolu) |
| **Nombre de trades** | 2731 | **1788** | **-943 trades (-35%)** |
| **Win Rate** | 78.1 % | **80.0 %** | +1.9 % |
| **Max Drawdown (3y)** | -7.3 % | **-6.3 %** | +1.0 % |
| **Efficience (Profit/Trade)** | 4.12 $ | **6.04 $** | **+46.6 %** |
| **Worst-case SL loss** | 25.8 % | 26.0 % | Identique |

## 3. Analyse Technique
La v2 démontre une supériorité opérationnelle majeure :
1. **Élimination du bruit :** Près de 1000 trades "inutiles" ont été supprimés sans dégrader significativement le profit final.
2. **Récupération d'actifs :** L'ETH/USDT, perdant en v1 (-195$), devient rentable en v2 (+144$) grâce au filtre ATR.
3. **Réduction de la fatigue :** Moins de trades signifie moins de frais de transaction cumulés (66$ vs 96$), moins de slippage et moins de risque d'exécution sur Bitget.

## 4. Note sur le Levier et le Risque
* **Le mirage du Drawdown :** Bien que le DD affiche -6.3% sur 3 ans, le risque réel sur capital frais est estimé à **~12%** (basé sur le test 1 an). 
* **Risque de corrélation :** Le "Worst-case SL loss" de **26%** confirme qu'un levier de 6x est une limite prudente. Un levier supérieur (ex: 10x+) risquerait de déclencher le Kill Switch global (45%) en cas de crash synchronisé des cryptos.

## 5. Conclusion
**La version v2 est validée comme standard de production.** 
Elle offre un profil de risque plus sain et une efficience économique supérieure de 46% par rapport à la v1.
