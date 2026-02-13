# Hotfix — Monte Carlo underpowered detection

## Problème

Le test Monte Carlo block bootstrap pénalisait injustement les stratégies basse fréquence
(envelope_dca sur BTC : 28 trades OOS → 4 blocs de taille 7 → seulement 24 permutations possibles).

Résultat : BTC obtenait 0/25 pts MC → Grade D au lieu de B.

## Solution

### overfitting.py
- Seuil underpowered relevé à **< 30 trades** (avant : pas de seuil)
- Quand underpowered : retourne `p_value=0.50`, `underpowered=True`, pas de simulation
- `block_size` reste fixe à **7** (préserve la corrélation temporelle, essentielle pour DCA)
- Champ `underpowered: bool = False` ajouté à `MonteCarloResult`

### report.py
- `compute_grade()` : si `mc_underpowered=True` → **12/25 pts** (neutre, ni bonus ni pénalité)
- `FinalReport` : champ `mc_underpowered: bool` + sérialisation JSON
- `build_final_report()` : warning explicite quand MC est sous-puissant

### optimize.py
- Logs MC avec suffixe `(underpowered)` quand applicable
- `_print_report()` : marqueur `!` pour underpowered
- Tous les caractères Unicode (═, ─, ✓, ✗, ⚠) remplacés par ASCII (compatibilité cp1252 Windows)

## Tests ajoutés (6)
- `test_mc_underpowered_10_trades` : 10 trades → underpowered
- `test_mc_underpowered_25_trades` : 25 trades → underpowered
- `test_mc_30_trades_not_underpowered` : 30 trades → test MC complet
- `test_mc_block_size_default_7` : block_size=7 par défaut
- `test_grade_mc_underpowered` : underpowered → 12/25 pts

## Impact grades envelope_dca

| Asset | OOS Trades | MC | Grade |
|-------|-----------|-----|-------|
| BTC | 28 | underpowered (12/25) | **B** |
| ETH | 86 | p=0.170 (0/25) | C |
| SOL | 157 | p=0.874 (0/25) | C |
| DOGE | 231 | p=0.889 (0/25) | D |
| LINK | 110 | p=0.386 (0/25) | F |

456 tests passants, 0 régression.
