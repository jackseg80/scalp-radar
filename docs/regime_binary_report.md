# Regime Binary Analysis Report — Sprint 50a-bis

*Genere automatiquement le 2026-02-26 01:29*

**Donnees** : 18673 candles 4h BTC/USDT
**Seuil operationnel** : -3.0% (single-candle drop)

## 1. Resume executif

**Couche 1 (Tendance)** : `ema_atr` — F1 defensive = 0.668 (vs F1 4-classes = 0.352)
- false_defensive = 21.8% (cible <15%), missed_defensive = 24.0% (cible <20%)

**Couche 2 (Stress 4h)** : lookback=6 (24h), threshold=-15.0%
- Detecte 4/5 crashs, delai moyen 132h
- Faux positifs : 22

**Verdict Sprint 50b** : GO

## 2. Couche 1 — Tendance binaire

### 2.1 Comparaison des detecteurs

| Detecteur | F1 def | F1 norm | Accuracy | time_def% | false_def% | missed_def% | trans/an | F1 4-classes |
|-----------|--------|---------|----------|-----------|------------|-------------|----------|-------------|
| sma_stress | 0.661 | 0.822 | 0.766 | 40.4 | 24.6 | 20.2 | 4.2 | 0.392 |
| ema_atr | 0.668 | 0.830 | 0.775 | 38.0 | 21.8 | 24.0 | 1.6 | 0.352 |
| multi_ma_vol | 0.598 | 0.814 | 0.746 | 33.5 | 20.7 | 36.5 | 5.9 | 0.463 |

### 2.2 Top 3 : sma_stress

| Rang | F1 def | false_def% | missed_def% | trans/an | Params |
|------|--------|------------|-------------|----------|--------|
| 1 | 0.661 | 24.6 | 20.2 | 4.2 | `{"h_down":12,"h_up":24,"sma_period":250,"stress_window":5,"stress_threshold":-20}` |
| 2 | 0.660 | 24.0 | 21.1 | 4.0 | `{"h_down":18,"h_up":24,"sma_period":250,"stress_window":7,"stress_threshold":-20}` |
| 3 | 0.660 | 24.7 | 20.2 | 4.2 | `{"h_down":12,"h_up":24,"sma_period":250,"stress_window":7,"stress_threshold":-25}` |

### 2.2 Top 3 : ema_atr

| Rang | F1 def | false_def% | missed_def% | trans/an | Params |
|------|--------|------------|-------------|----------|--------|
| 1 | 0.668 | 21.8 | 24.0 | 1.6 | `{"h_down":6,"h_up":24,"ema_fast":50,"ema_slow":200,"atr_fast":7,"atr_slow":30,"atr_stress_ratio":2.0}` |
| 2 | 0.667 | 22.0 | 23.9 | 1.9 | `{"h_down":6,"h_up":24,"ema_fast":50,"ema_slow":200,"atr_fast":5,"atr_slow":20,"atr_stress_ratio":2.0}` |
| 3 | 0.667 | 21.8 | 24.3 | 1.6 | `{"h_down":6,"h_up":24,"ema_fast":50,"ema_slow":200,"atr_fast":5,"atr_slow":20,"atr_stress_ratio":2.5}` |

### 2.2 Top 3 : multi_ma_vol

| Rang | F1 def | false_def% | missed_def% | trans/an | Params |
|------|--------|------------|-------------|----------|--------|
| 1 | 0.598 | 20.7 | 36.5 | 5.9 | `{"h_down":12,"h_up":36,"vol_window":14,"vol_percentile":97,"vol_lookback":365}` |
| 2 | 0.597 | 20.9 | 36.5 | 5.4 | `{"h_down":12,"h_up":36,"vol_window":21,"vol_percentile":97,"vol_lookback":365}` |
| 3 | 0.597 | 22.1 | 35.2 | 5.6 | `{"h_down":6,"h_up":36,"vol_window":21,"vol_percentile":97,"vol_lookback":365}` |

## 3. Couche 2 — Stress 4h

### 3.1 Heatmap

![stress_heatmap](../docs/images/stress_4h_heatmap.png)

### 3.2 Analyse par crash

**Config** : lookback=6, threshold=-15.0%

| Crash | Delay GT (h) | Delay Op (h) | Avant bottom | Recovery (h) | Alarm 48h |
|-------|-------------|-------------|-------------|-------------|-----------|
| COVID Crash | 104 | 92 | OUI | 0 | NON |
| China Ban Crash | 236 | 220 | OUI | 0 | NON |
| LUNA Crash | N/D | N/D | NON | 0 | NON |
| FTX Crash | 108 | 20 | OUI | 0 | NON |
| Aug 2024 Crash | 80 | 8 | NON | 0 | NON |

### 3.3 Metriques globales

- Stress total : 248h (0.3%)
- Faux positifs : 22 (2.6/an)
- Duree moyenne faux positifs : 9h

### 3.4 Top 5 configurations

| Rang | Lookback | Threshold | Score | Detected | Delay moy | False events |
|------|----------|-----------|-------|----------|-----------|-------------|
| 1 | 6 (24h) | -15.0% | 48 | 4/5 | 132h | 22 |
| 2 | 18 (72h) | -20.0% | -50 | 4/5 | 130h | 32 |
| 3 | 12 (48h) | -20.0% | -90 | 2/5 | 110h | 18 |
| 4 | 6 (24h) | -20.0% | -104 | 1/5 | 104h | 10 |
| 5 | 12 (48h) | -15.0% | -154 | 5/5 | 84h | 57 |

### 3.5 Detail faux positifs

| Date | Duree (h) |
|------|-----------|
| 2017-09-14 | 8 |
| 2017-09-15 | 8 |
| 2017-11-29 | 4 |
| 2017-11-30 | 8 |
| 2017-12-10 | 4 |
| 2017-12-21 | 8 |
| 2017-12-22 | 24 |
| 2017-12-24 | 12 |
| 2017-12-28 | 4 |
| 2017-12-30 | 4 |
| 2018-01-16 | 16 |
| 2018-01-17 | 8 |
| 2018-02-05 | 20 |
| 2018-11-20 | 4 |
| 2018-11-25 | 4 |
| 2019-06-27 | 12 |
| 2021-01-11 | 4 |
| 2021-01-11 | 8 |
| 2021-02-23 | 4 |
| 2021-02-23 | 4 |

## 4. Combinaison des 2 couches

![combined](../docs/images/regime_binary_combined.png)

Verification visuelle : les 2 couches se completent-elles ?
- Couche 1 (lente) couvre les regimes tendanciels (bear markets)
- Couche 2 (rapide) reagit aux chocs ponctuels (crashs)

## 5. Recommandation finale

**Couche 1** : `ema_atr` avec params `{"h_down":6,"h_up":24,"ema_fast":50,"ema_slow":200,"atr_fast":7,"atr_slow":30,"atr_stress_ratio":2.0}`
- leverage normal=7x, defensive=4x

**Couche 2** : lookback=6, threshold=-15.0%
- DCA throttle quand stress ON (reduire taille DCA de 50%)

**Sprint 50b** : integrer ces 2 couches dans le backtester portfolio
pour mesurer l'impact sur le PnL et le drawdown.
