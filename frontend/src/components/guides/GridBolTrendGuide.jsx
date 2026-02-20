/**
 * GridBolTrendGuide — Tutoriel interactif Grid BolTrend en 8 étapes
 * Données ETH-like avec Bollinger Bands + SMA longue + breakout + DCA fixé
 * Sprint Strategy Lab V2
 */

import { useState, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ReferenceLine, ReferenceDot, ResponsiveContainer, Area, Tooltip,
} from 'recharts'

// ─── Paramètres de la stratégie ───
const ATR_SPACING_MULT = 1.0
const NUM_LEVELS = 3
const SL_PERCENT = 15

// ─── Données brutes du scénario (ETH-like, ~3200$) ───

// Phase attente (prix dans les bandes) + breakout + pullback + DCA
const BASE_DATA = [
  // Phase attente — prix oscille dans les bandes
  { t: 0,  label: '1 Jan',  price: 3180, bb_sma: 3200, bb_upper: 3350, bb_lower: 3050, long_ma: 3100, atr: 80 },
  { t: 1,  label: '2 Jan',  price: 3220, bb_sma: 3205, bb_upper: 3355, bb_lower: 3055, long_ma: 3105, atr: 78 },
  { t: 2,  label: '3 Jan',  price: 3150, bb_sma: 3195, bb_upper: 3340, bb_lower: 3050, long_ma: 3108, atr: 82 },
  { t: 3,  label: '4 Jan',  price: 3250, bb_sma: 3200, bb_upper: 3360, bb_lower: 3040, long_ma: 3115, atr: 85 },
  { t: 4,  label: '5 Jan',  price: 3280, bb_sma: 3215, bb_upper: 3370, bb_lower: 3060, long_ma: 3120, atr: 83 },
  { t: 5,  label: '6 Jan',  price: 3200, bb_sma: 3210, bb_upper: 3365, bb_lower: 3055, long_ma: 3125, atr: 80 },
  { t: 6,  label: '7 Jan',  price: 3300, bb_sma: 3220, bb_upper: 3380, bb_lower: 3060, long_ma: 3130, atr: 86 },
  { t: 7,  label: '8 Jan',  price: 3340, bb_sma: 3235, bb_upper: 3395, bb_lower: 3075, long_ma: 3140, atr: 88 },

  // Phase breakout — close > bb_upper ET close > long_ma
  // prev_close=3340 < prev_upper=3395 ✓, close=3430 > upper=3400 ✓, close=3430 > long_ma=3150 ✓
  { t: 8,  label: '9 Jan',  price: 3430, bb_sma: 3255, bb_upper: 3400, bb_lower: 3110, long_ma: 3150, atr: 95 },

  // Phase pullback + DCA (niveaux fixés au breakout)
  // Level 0 = 3430 (breakout), Level 1 = 3430 - 95*1.0 = 3335, Level 2 = 3430 - 95*2.0 = 3240
  { t: 9,  label: '10 Jan', price: 3380, bb_sma: 3270, bb_upper: 3420, bb_lower: 3120, long_ma: 3155, atr: 98 },
  { t: 10, label: '11 Jan', price: 3320, bb_sma: 3280, bb_upper: 3430, bb_lower: 3130, long_ma: 3160, atr: 100 },
  // prix=3320 < Level 1=3335 → ACHAT niveau 1
  { t: 11, label: '12 Jan', price: 3280, bb_sma: 3285, bb_upper: 3435, bb_lower: 3135, long_ma: 3165, atr: 102 },
  { t: 12, label: '13 Jan', price: 3230, bb_sma: 3280, bb_upper: 3430, bb_lower: 3130, long_ma: 3168, atr: 105 },
  // prix=3230 < Level 2=3240 → ACHAT niveau 2
]

// Scénario recovery — TP inversé (close < bb_sma)
const RECOVERY_DATA = [
  { t: 13, label: '14 Jan', price: 3350, bb_sma: 3290, bb_upper: 3440, bb_lower: 3140, long_ma: 3172, atr: 100 },
  { t: 14, label: '15 Jan', price: 3420, bb_sma: 3310, bb_upper: 3450, bb_lower: 3170, long_ma: 3178, atr: 95 },
  { t: 15, label: '16 Jan', price: 3380, bb_sma: 3320, bb_upper: 3440, bb_lower: 3200, long_ma: 3182, atr: 90 },
  { t: 16, label: '17 Jan', price: 3300, bb_sma: 3325, bb_upper: 3430, bb_lower: 3220, long_ma: 3185, atr: 88 },
  // prix=3300 < bb_sma=3325 → TP INVERSÉ
  { t: 17, label: '18 Jan', price: 3280, bb_sma: 3320, bb_upper: 3420, bb_lower: 3220, long_ma: 3185, atr: 85 },
]

// Scénario disaster — SL touché
const DISASTER_DATA = [
  { t: 13, label: '14 Jan', price: 3180, bb_sma: 3275, bb_upper: 3420, bb_lower: 3130, long_ma: 3170, atr: 110 },
  { t: 14, label: '15 Jan', price: 3100, bb_sma: 3260, bb_upper: 3410, bb_lower: 3110, long_ma: 3165, atr: 115 },
  { t: 15, label: '16 Jan', price: 3020, bb_sma: 3240, bb_upper: 3395, bb_lower: 3085, long_ma: 3158, atr: 120 },
  { t: 16, label: '17 Jan', price: 2950, bb_sma: 3215, bb_upper: 3380, bb_lower: 3050, long_ma: 3150, atr: 125 },
  { t: 17, label: '18 Jan', price: 2820, bb_sma: 3185, bb_upper: 3360, bb_lower: 3010, long_ma: 3140, atr: 130 },
  // avg ≈ 3333, SL = 3333 × 0.85 = 2833 → prix=2820 < SL=2833 → SL TOUCHÉ
]

// ─── Le breakout se produit à t=8 ───
const BREAKOUT_INDEX = 8
const BREAKOUT_PRICE = BASE_DATA[BREAKOUT_INDEX].price   // 3430
const BREAKOUT_ATR = BASE_DATA[BREAKOUT_INDEX].atr        // 95

// ─── Niveaux FIXES au breakout ───
function computeFixedLevels(breakoutPrice, atr) {
  const levels = []
  for (let i = 0; i < NUM_LEVELS; i++) {
    levels.push({
      level: i,
      price: breakoutPrice - atr * ATR_SPACING_MULT * (i + 1),
    })
  }
  return levels
}

const FIXED_LEVELS = computeFixedLevels(BREAKOUT_PRICE, BREAKOUT_ATR)
// Level 0 = breakout (3430), Level 1 = 3335, Level 2 = 3240

// ─── Points d'achat (DCA) ───
function findBuyPoints(data) {
  const buys = [{ t: BREAKOUT_INDEX, level: 0, price: BREAKOUT_PRICE }]
  const bought = new Set([0])

  for (let i = BREAKOUT_INDEX + 1; i < data.length; i++) {
    for (const lvl of FIXED_LEVELS) {
      if (!bought.has(lvl.level) && data[i].price <= lvl.price && data[i - 1].price > lvl.price) {
        buys.push({ t: data[i].t, level: lvl.level, price: data[i].price })
        bought.add(lvl.level)
      }
    }
  }
  return buys
}

// ─── Point TP inversé (premier close < bb_sma après achats) ───
function findTpPoint(data) {
  for (let i = BREAKOUT_INDEX + 2; i < data.length; i++) {
    if (data[i].price < data[i].bb_sma && data[i - 1].price >= data[i - 1].bb_sma) {
      return { t: data[i].t, price: data[i].price }
    }
  }
  return null
}

// ─── Enrichir les données avec niveaux fixes (horizontaux après breakout) ───
function enrichData(rawData) {
  return rawData.map(d => {
    const enriched = { ...d }
    if (d.t >= BREAKOUT_INDEX) {
      enriched.fixedLevel1 = FIXED_LEVELS[0].price
      enriched.fixedLevel2 = FIXED_LEVELS[1].price
      enriched.fixedLevel3 = FIXED_LEVELS[2] ? FIXED_LEVELS[2].price : undefined
    }
    return enriched
  })
}

// ─── Étapes du tutoriel ───
const STEPS = [
  {
    title: 'Le prix et les Bandes de Bollinger',
    desc: "Les Bandes de Bollinger mesurent la volatilité. La bande haute = SMA + 2 écarts-types, la bande basse = SMA - 2 écarts-types. Quand les bandes s'écartent, le marché est nerveux. Quand elles se resserrent, il est calme.",
    keyInsight: "Les bandes encadrent ~95% des mouvements de prix. Un franchissement est un événement rare et significatif.",
    show: { price: true, bb: true, bbSma: true },
  },
  {
    title: 'Le filtre de tendance (SMA longue)',
    desc: "La SMA 200 (orange) filtre la direction. On n'accepte les breakouts LONG que si le prix est AU-DESSUS de cette SMA longue. Cela évite d'acheter dans une tendance baissière.",
    keyInsight: "Filtre tendance = filet de sécurité. On ne trade que dans le sens du courant.",
    show: { price: true, bb: true, bbSma: true, longMa: true },
  },
  {
    title: 'La grille est OFF — En attente',
    desc: "Contrairement à Grid ATR qui achète en permanence, Grid BolTrend attend. Le prix oscille dans les bandes → pas de signal → pas d'action. Le bot est patient.",
    keyInsight: "Grid ATR serait déjà en train d'acheter des dips. Grid BolTrend attend un breakout confirmé.",
    show: { price: true, bb: true, bbSma: true, longMa: true },
  },
  {
    title: 'BREAKOUT ! La grille s\'active',
    desc: "Le prix franchit violemment la bande haute (close > bb_upper) alors qu'il était en-dessous (prev_close < prev_upper). Le filtre tendance est OK (close > SMA longue). Les 4 conditions sont réunies → Level 0 entre au prix du breakout. Les niveaux DCA sont FIXÉS maintenant.",
    keyInsight: "Les niveaux sont calculés UNE FOIS au breakout : Level k = breakout - k × ATR × spacing. Ils ne bougent plus.",
    show: { price: true, bb: true, bbSma: true, longMa: true, levels: true, buys: [0], breakout: true },
  },
  {
    title: 'Pullback : DCA en action',
    desc: "Après un breakout, le pullback est classique. Le prix corrige et touche les niveaux DCA fixés au breakout. Chaque franchissement ouvre une nouvelle position. Le prix moyen baisse à chaque achat.",
    keyInsight: "Les niveaux ne bougent PAS (fixés au breakout). Contrairement à Grid ATR dont les niveaux s'adaptent à chaque candle.",
    show: { price: true, bb: true, bbSma: true, longMa: true, levels: true, buys: [0, 1, 2], avgPrice: true },
  },
  {
    title: 'Happy end : TP inversé',
    desc: "Le prix remonte mais s'essouffle. Quand il redescend sous la BB SMA (SMA Bollinger), c'est le signal : le breakout est épuisé, retour à la normale. Toutes les positions sont fermées.",
    keyInsight: "Le TP est INVERSÉ vs Grid ATR. Grid ATR vend quand close > SMA. Grid BolTrend vend quand close < BB SMA.",
    show: { price: true, bb: true, bbSma: true, longMa: true, levels: true, buys: [0, 1, 2], avgPrice: true, tp: true },
    scenario: 'recovery',
  },
  {
    title: 'La catastrophe : faux breakout',
    desc: "Scénario alternatif. Le breakout était un piège haussier. Le prix s'effondre sans jamais remonter. Le SL global est touché. Les faux breakouts sont le risque principal de cette stratégie.",
    keyInsight: "Les faux breakouts sont fréquents. Le SL global protège le capital. C'est pourquoi le sl_percent est crucial.",
    show: { price: true, bb: true, bbSma: true, longMa: true, levels: true, buys: [0, 1, 2], avgPrice: true, sl: true },
    scenario: 'disaster',
  },
  {
    title: 'Grid ATR vs Grid BolTrend',
    desc: null,
    keyInsight: "Grid ATR = filet de pêche permanent. Grid BolTrend = harpon déclenché sur signal.",
    show: { comparison: true },
  },
]

// ─── Composant principal ───
export default function GridBolTrendGuide() {
  const [step, setStep] = useState(0)
  const currentStep = STEPS[step]

  // Données selon le scénario
  const fullData = useMemo(() => {
    if (currentStep.scenario === 'recovery') {
      return enrichData([...BASE_DATA, ...RECOVERY_DATA])
    }
    if (currentStep.scenario === 'disaster') {
      return enrichData([...BASE_DATA, ...DISASTER_DATA])
    }
    return enrichData(BASE_DATA)
  }, [step]) // eslint-disable-line react-hooks/exhaustive-deps

  // Points d'achat
  const buyPoints = useMemo(() => {
    const scenario = currentStep.scenario === 'disaster'
      ? [...BASE_DATA, ...DISASTER_DATA]
      : currentStep.scenario === 'recovery'
        ? [...BASE_DATA, ...RECOVERY_DATA]
        : BASE_DATA
    return findBuyPoints(scenario)
  }, [step]) // eslint-disable-line react-hooks/exhaustive-deps

  const visibleBuys = (currentStep.show.buys || [])
    .map(lvl => buyPoints.find(b => b.level === lvl))
    .filter(Boolean)

  // Prix moyen pondéré
  const avgPrice = visibleBuys.length > 0
    ? visibleBuys.reduce((s, b) => s + b.price, 0) / visibleBuys.length
    : null

  // SL price
  const slPrice = avgPrice ? avgPrice * (1 - SL_PERCENT / 100) : null

  // Point TP inversé
  const tpPoint = useMemo(() => {
    if (!currentStep.show.tp) return null
    const data = [...BASE_DATA, ...RECOVERY_DATA]
    return findTpPoint(data)
  }, [step]) // eslint-disable-line react-hooks/exhaustive-deps

  // Y domain
  const allPrices = fullData.flatMap(d => {
    const vals = [d.price, d.bb_sma, d.bb_upper, d.bb_lower]
    if (d.fixedLevel1) vals.push(d.fixedLevel1, d.fixedLevel2)
    if (d.fixedLevel3) vals.push(d.fixedLevel3)
    return vals
  })
  if (slPrice) allPrices.push(slPrice)
  const yMin = Math.floor(Math.min(...allPrices) / 100) * 100 - 100
  const yMax = Math.ceil(Math.max(...allPrices) / 100) * 100 + 100

  // Chiffres de référence
  const breakoutPoint = BASE_DATA[BREAKOUT_INDEX]

  // Gain/perte
  let resultText = null
  if (step === 5 && tpPoint && avgPrice) {
    const gain = ((tpPoint.price - avgPrice) / avgPrice * 100).toFixed(1)
    resultText = `Gain : +${gain}% (hors leverage et fees)`
  }
  if (step === 6 && slPrice && avgPrice) {
    resultText = `Perte : -${SL_PERCENT}% (SL global touché)`
  }

  // Étape comparative (pas de graphique)
  if (currentStep.show.comparison) {
    return (
      <div>
        {/* Navigation dots */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginBottom: 16 }}>
          {STEPS.map((_, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              style={{
                width: 10, height: 10, borderRadius: '50%', border: 'none',
                background: i === step ? '#4da6ff' : '#444',
                cursor: 'pointer', transition: 'background 0.2s', padding: 0,
              }}
              title={STEPS[i].title}
            />
          ))}
        </div>

        {/* Panneau d'info comparatif */}
        <div style={{
          background: '#0d1117', border: '1px solid #333',
          borderRadius: 8, padding: 16, minHeight: 290,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <h4 style={{ margin: 0, color: '#e8eaed', fontSize: 16 }}>
              {step + 1}/{STEPS.length} — {currentStep.title}
            </h4>
            <div style={{ display: 'flex', gap: 8 }}>
              <NavButton dir="prev" step={step} setStep={setStep} total={STEPS.length} />
              <NavButton dir="next" step={step} setStep={setStep} total={STEPS.length} />
            </div>
          </div>

          {/* Tableau comparatif */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <ComparisonCard
              title="Grid ATR"
              color="#4da6ff"
              items={[
                { icon: '✓', text: 'Grille toujours active', color: '#00e68a' },
                { icon: '✓', text: 'Niveaux recalculés à chaque candle', color: '#00e68a' },
                { icon: '✓', text: 'TP = retour à la SMA', color: '#00e68a' },
                { icon: '✓', text: 'Mean reversion', color: '#00e68a' },
                { icon: '✓', text: 'Excelle en crash + recovery', color: '#00e68a' },
                { icon: '✗', text: 'Inactif en range serré', color: '#ff4466' },
              ]}
            />
            <ComparisonCard
              title="Grid BolTrend"
              color="#ffc53d"
              items={[
                { icon: '✓', text: 'Grille OFF par défaut', color: '#00e68a' },
                { icon: '✓', text: 'Niveaux fixés au breakout', color: '#00e68a' },
                { icon: '✓', text: 'TP inversé (close < BB SMA)', color: '#00e68a' },
                { icon: '✓', text: 'Trend following', color: '#00e68a' },
                { icon: '✓', text: 'Excelle en breakout + pullback', color: '#00e68a' },
                { icon: '✗', text: 'Faux breakouts = piège', color: '#ff4466' },
              ]}
            />
          </div>

          {/* Concept clé */}
          <div style={{ fontSize: 13, color: '#4da6ff', fontWeight: 600 }}>
            {currentStep.keyInsight}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Graphique */}
      <div style={{ width: '100%', marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={370}>
          <LineChart data={fullData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#222" />
            <XAxis dataKey="label" tick={{ fill: '#888', fontSize: 11 }} />
            <YAxis
              domain={[yMin, yMax]}
              tick={{ fill: '#888', fontSize: 11 }}
              tickFormatter={v => `${v}`}
              width={50}
            />
            <Tooltip
              contentStyle={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 6, fontSize: 12 }}
              labelStyle={{ color: '#888' }}
              formatter={(val, name) => {
                if (val === undefined || val === null) return ['-', name]
                return [`$${val.toLocaleString()}`, name]
              }}
            />

            {/* Bandes de Bollinger (zone semi-transparente) */}
            {currentStep.show.bb && (
              <>
                <Line type="monotone" dataKey="bb_upper" stroke="rgba(138,180,248,0.5)" strokeWidth={1} strokeDasharray="4 4" dot={false} name="BB Upper" />
                <Line type="monotone" dataKey="bb_lower" stroke="rgba(138,180,248,0.5)" strokeWidth={1} strokeDasharray="4 4" dot={false} name="BB Lower" />
                {/* Zone entre les bandes */}
                {fullData.map((d, i) => (
                  <ReferenceLine
                    key={`bb-zone-${i}`}
                    segment={[
                      { x: d.label, y: d.bb_lower },
                      { x: d.label, y: d.bb_upper },
                    ]}
                    stroke="rgba(138,180,248,0.06)"
                    strokeWidth={18}
                  />
                ))}
              </>
            )}

            {/* BB SMA (SMA Bollinger) */}
            {currentStep.show.bbSma && (
              <Line type="monotone" dataKey="bb_sma" stroke="#ffc53d" strokeWidth={1.5} strokeDasharray="6 4" dot={false} name="BB SMA" />
            )}

            {/* SMA longue (filtre tendance) */}
            {currentStep.show.longMa && (
              <Line type="monotone" dataKey="long_ma" stroke="#f97316" strokeWidth={1.5} strokeDasharray="8 4" dot={false} name="SMA 200" />
            )}

            {/* Niveaux DCA fixes (horizontaux après breakout) */}
            {currentStep.show.levels && (
              <>
                <Line type="stepAfter" dataKey="fixedLevel1" stroke="#00e68a" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Niveau 1" connectNulls={false} />
                <Line type="stepAfter" dataKey="fixedLevel2" stroke="#00e68a" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Niveau 2" opacity={0.7} connectNulls={false} />
                {NUM_LEVELS >= 3 && (
                  <Line type="stepAfter" dataKey="fixedLevel3" stroke="#00e68a" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Niveau 3" opacity={0.5} connectNulls={false} />
                )}
              </>
            )}

            {/* Prix */}
            {currentStep.show.price && (
              <Line type="monotone" dataKey="price" stroke="#4da6ff" strokeWidth={2.5} dot={false} name="Prix" />
            )}

            {/* Prix moyen */}
            {currentStep.show.avgPrice && avgPrice && (
              <ReferenceLine y={avgPrice} stroke="#ff8c42" strokeWidth={1.5} strokeDasharray="6 3" label={{ value: `Avg: $${avgPrice.toFixed(0)}`, fill: '#ff8c42', fontSize: 11, position: 'right' }} />
            )}

            {/* SL */}
            {currentStep.show.sl && slPrice && (
              <ReferenceLine y={slPrice} stroke="#ff4466" strokeWidth={2} strokeDasharray="4 4" label={{ value: `SL: $${slPrice.toFixed(0)}`, fill: '#ff4466', fontSize: 11, position: 'right' }} />
            )}

            {/* TP point (quand close < bb_sma) */}
            {tpPoint && (
              <ReferenceDot x={fullData.find(d => d.t === tpPoint.t)?.label} y={tpPoint.price} r={7} fill="#00e68a" stroke="#fff" strokeWidth={2} />
            )}

            {/* Point de breakout */}
            {currentStep.show.breakout && (
              <ReferenceDot
                x={fullData.find(d => d.t === BREAKOUT_INDEX)?.label}
                y={BREAKOUT_PRICE}
                r={8}
                fill="#ffc53d"
                stroke="#fff"
                strokeWidth={2}
              />
            )}

            {/* Points d'achat DCA */}
            {visibleBuys.map(b => (
              <ReferenceDot
                key={b.level}
                x={fullData.find(d => d.t === b.t)?.label}
                y={b.price}
                r={6}
                fill={b.level === 0 ? '#ffc53d' : '#00e68a'}
                stroke="#fff"
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Navigation dots */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginBottom: 16 }}>
        {STEPS.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            style={{
              width: 10, height: 10, borderRadius: '50%', border: 'none',
              background: i === step ? '#4da6ff' : '#444',
              cursor: 'pointer', transition: 'background 0.2s', padding: 0,
            }}
            title={STEPS[i].title}
          />
        ))}
      </div>

      {/* Panneau d'info */}
      <div style={{
        background: '#0d1117', border: '1px solid #333',
        borderRadius: 8, padding: 16, minHeight: 290,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <h4 style={{ margin: 0, color: '#e8eaed', fontSize: 16 }}>
            {step + 1}/{STEPS.length} — {currentStep.title}
          </h4>
          <div style={{ display: 'flex', gap: 8 }}>
            <NavButton dir="prev" step={step} setStep={setStep} total={STEPS.length} />
            <NavButton dir="next" step={step} setStep={setStep} total={STEPS.length} />
          </div>
        </div>

        <p style={{ color: '#aaa', fontSize: 13, lineHeight: 1.6, margin: '0 0 12px 0' }}>
          {currentStep.desc}
        </p>

        {/* Encadré chiffres */}
        <div style={{
          background: '#1a1a2a',
          border: '1px solid rgba(77,166,255,0.2)',
          borderRadius: 6, padding: 10, marginBottom: 10,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
          gap: 6, fontSize: 12,
        }}>
          <span style={{ color: '#888' }}>BB SMA: <b style={{ color: '#ffc53d' }}>${breakoutPoint.bb_sma}</b></span>
          <span style={{ color: '#888' }}>BB Upper: <b style={{ color: '#8ab4f8' }}>${breakoutPoint.bb_upper}</b></span>
          <span style={{ color: '#888' }}>SMA 200: <b style={{ color: '#f97316' }}>${breakoutPoint.long_ma}</b></span>
          <span style={{ color: '#888' }}>ATR: <b style={{ color: '#ff8c42' }}>${breakoutPoint.atr}</b></span>
          {step >= 3 && (
            <>
              <span style={{ color: '#888' }}>Breakout: <b style={{ color: '#ffc53d' }}>${BREAKOUT_PRICE}</b></span>
              {FIXED_LEVELS.map(l => (
                <span key={l.level} style={{ color: '#888' }}>
                  Niv.{l.level + 1}: <b style={{ color: '#00e68a' }}>${l.price.toFixed(0)}</b>
                </span>
              ))}
            </>
          )}
          {avgPrice && step >= 4 && (
            <span style={{ color: '#888' }}>Avg: <b style={{ color: '#ff8c42' }}>${avgPrice.toFixed(0)}</b></span>
          )}
          {slPrice && step >= 6 && (
            <span style={{ color: '#888' }}>SL: <b style={{ color: '#ff4466' }}>${slPrice.toFixed(0)}</b></span>
          )}
        </div>

        {/* Résultat gain/perte */}
        {resultText && (
          <div style={{
            padding: '8px 12px', borderRadius: 4, marginBottom: 10,
            fontSize: 13, fontWeight: 600,
            background: step === 5 ? 'rgba(0,230,138,0.1)' : 'rgba(255,68,102,0.1)',
            color: step === 5 ? '#00e68a' : '#ff4466',
            border: `1px solid ${step === 5 ? 'rgba(0,230,138,0.3)' : 'rgba(255,68,102,0.3)'}`,
          }}>
            {resultText}
          </div>
        )}

        {/* Concept clé */}
        <div style={{ fontSize: 13, color: '#4da6ff', fontWeight: 600 }}>
          {currentStep.keyInsight}
        </div>
      </div>
    </div>
  )
}

// ─── Composants utilitaires ───

function NavButton({ dir, step, setStep, total }) {
  const isPrev = dir === 'prev'
  const disabled = isPrev ? step === 0 : step === total - 1
  return (
    <button
      onClick={() => setStep(isPrev ? Math.max(0, step - 1) : Math.min(total - 1, step + 1))}
      disabled={disabled}
      style={{
        padding: '5px 14px', border: '1px solid #444', borderRadius: 4,
        background: disabled ? '#1a1a1a' : '#2a2a2a',
        color: disabled ? '#555' : '#ccc',
        cursor: disabled ? 'not-allowed' : 'pointer', fontSize: 13,
      }}
    >
      {isPrev ? '← Précédent' : 'Suivant →'}
    </button>
  )
}

function ComparisonCard({ title, color, items }) {
  return (
    <div style={{
      background: '#1a1a2a', border: `1px solid ${color}33`,
      borderRadius: 8, padding: 14, borderTop: `3px solid ${color}`,
    }}>
      <h4 style={{ margin: '0 0 10px 0', color, fontSize: 14 }}>{title}</h4>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {items.map((item, i) => (
          <div key={i} style={{ fontSize: 12, color: '#ccc', display: 'flex', gap: 6 }}>
            <span style={{ color: item.color, fontWeight: 700, minWidth: 14 }}>{item.icon}</span>
            <span>{item.text}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
