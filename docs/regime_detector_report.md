# BTC Regime Detector Calibration Report — Sprint 50a

*Genere automatiquement le 2026-02-26 00:46*

**Donnees** : 18673 candles 4h BTC/USDT

## 1. Resume executif

**Detecteur recommande** : `multi_ma_vol` (Macro F1 = 0.472, Crash F1 = 0.252, Robustesse = ROBUSTE)

Parametres optimaux : `{"h_down": 6, "h_up": 24, "vol_window": 14, "vol_percentile": 97, "vol_lookback": 365}`

## 2. Tableau comparatif (top 3 par detecteur)

| Detecteur | Rang | Macro F1 | Accuracy | Crash F1 | Delay moy | Transitions | Stable | Robuste |
|-----------|------|----------|----------|----------|-----------|-------------|--------|---------|
| sma_stress | 1 | 0.421 | 0.484 | 0.492 | 156h | 42 | 1.00 | NON |
| sma_stress | 2 | 0.418 | 0.490 | 0.465 | 156h | 46 | 1.00 | - |
| sma_stress | 3 | 0.416 | 0.480 | 0.487 | 156h | 30 | 1.00 | - |
| ema_atr | 1 | 0.378 | 0.495 | 0.275 | 140h | 27 | 1.00 | NON |
| ema_atr | 2 | 0.372 | 0.492 | 0.261 | 140h | 33 | 1.00 | - |
| ema_atr | 3 | 0.371 | 0.503 | 0.237 | 140h | 17 | 1.00 | - |
| multi_ma_vol | 1 | 0.472 | 0.528 | 0.252 | 170h | 124 | 1.00 | OUI |
| multi_ma_vol | 2 | 0.469 | 0.531 | 0.222 | 170h | 98 | 1.00 | - |
| multi_ma_vol | 3 | 0.466 | 0.520 | 0.255 | 212h | 113 | 1.00 | - |

## 3. Detecteur : sma_stress

### Meilleure configuration

```json
{
  "h_down": 6,
  "h_up": 48,
  "sma_period": 250,
  "stress_window": 5,
  "stress_threshold": -25
}
```

- Macro F1 : 0.4210
- Accuracy : 0.4838
- Candles evaluees : 17169 (apres warmup)
- Transitions : 42
- Stabilite : 1.000

**F1 par regime :**

- Bull : 0.584
- Bear : 0.608
- Range : 0.000
- Crash : 0.492

**Distribution :**

- Bull : 55.5%
- Bear : 42.9%
- Range : 0.3%
- Crash : 1.4%

### Confusion Matrix

| True \ Pred | Bull | Bear | Range | Crash |
|---|---|---|---|---|
| **Bull** | 4538 | 1472 | 0 | 5 |
| **Bear** | 815 | 3622 | 47 | 66 |
| **Range** | 4089 | 2138 | 0 | 23 |
| **Crash** | 82 | 126 | 0 | 146 |

### Timeline

![sma_stress](../docs/images/regime_sma_stress_best.png)

## 3. Detecteur : ema_atr

### Meilleure configuration

```json
{
  "h_down": 12,
  "h_up": 48,
  "ema_fast": 50,
  "ema_slow": 200,
  "atr_fast": 5,
  "atr_slow": 30,
  "atr_stress_ratio": 2.0
}
```

- Macro F1 : 0.3779
- Accuracy : 0.4946
- Candles evaluees : 17469 (apres warmup)
- Transitions : 27
- Stabilite : 1.000

**F1 par regime :**

- Bull : 0.584
- Bear : 0.652
- Range : 0.000
- Crash : 0.275

**Distribution :**

- Bull : 59.8%
- Bear : 37.7%
- Range : 0.3%
- Crash : 2.2%

### Confusion Matrix

| True \ Pred | Bull | Bear | Range | Crash |
|---|---|---|---|---|
| **Bull** | 4809 | 968 | 0 | 238 |
| **Bear** | 1073 | 3730 | 47 | 0 |
| **Range** | 4425 | 1777 | 0 | 48 |
| **Crash** | 143 | 109 | 0 | 102 |

### Timeline

![ema_atr](../docs/images/regime_ema_atr_best.png)

## 3. Detecteur : multi_ma_vol

### Meilleure configuration

```json
{
  "h_down": 6,
  "h_up": 24,
  "vol_window": 14,
  "vol_percentile": 97,
  "vol_lookback": 365
}
```

- Macro F1 : 0.4724
- Accuracy : 0.5281
- Candles evaluees : 17469 (apres warmup)
- Transitions : 124
- Stabilite : 1.000

**F1 par regime :**

- Bull : 0.574
- Bear : 0.604
- Range : 0.460
- Crash : 0.252

**Distribution :**

- Bull : 27.6%
- Bear : 27.9%
- Range : 39.6%
- Crash : 4.9%

### Confusion Matrix

| True \ Pred | Bull | Bear | Range | Crash |
|---|---|---|---|---|
| **Bull** | 3110 | 575 | 2169 | 161 |
| **Bear** | 249 | 2936 | 1587 | 78 |
| **Range** | 1464 | 1297 | 3028 | 461 |
| **Crash** | 5 | 61 | 136 | 152 |

### Timeline

![multi_ma_vol](../docs/images/regime_multi_ma_vol_best.png)

## 4. Analyse des crashs

| Detecteur | Crashs GT | Detectes | Delay moy | Delay max | Faux positifs |
|-----------|-----------|----------|-----------|-----------|---------------|
| sma_stress | 5 | 3 | 156h | 236h | 2 |
| ema_atr | 5 | 2 | 140h | 140h | 7 |
| multi_ma_vol | 5 | 4 | 170h | 260h | 11 |

**Detail pour multi_ma_vol (meilleur detecteur) :**

- Crash #1 : delai = 116h
- Crash #2 : delai = 260h
- Crash #3 : delai = NON DETECTE
- Crash #4 : delai = 140h
- Crash #5 : delai = 164h

## 5. Analyse de robustesse

### sma_stress : WARNING: PIC ISOLE

- Best F1 : 0.4210
- Avg voisins F1 : 0.3883 (delta: 7.8%)
- Voisins testes : 11
- **ISOLATED PEAK: F1 varie de 7.8% chez les voisins**

### ema_atr : WARNING: PIC ISOLE

- Best F1 : 0.3779
- Avg voisins F1 : 0.3388 (delta: 10.3%)
- Voisins testes : 10
- **ISOLATED PEAK: F1 varie de 10.3% chez les voisins**

### multi_ma_vol : ROBUSTE

- Best F1 : 0.4724
- Avg voisins F1 : 0.4506 (delta: 4.6%)
- Voisins testes : 7

## 6. Recommandation

Le detecteur `multi_ma_vol` est recommande avec les parametres ci-dessus. 
La configuration est robuste (voisins dans +-5% F1).

Limitation Detecteur 3 : sma_fast=50, sma_slow=200 fixes. Si retenu, explorer d'autres MAs en Sprint 50b.
