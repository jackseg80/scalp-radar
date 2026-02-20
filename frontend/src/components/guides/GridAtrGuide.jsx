/**
 * GridAtrGuide — Tutoriel interactif Grid ATR en 7 étapes
 * Données BTC-like avec ATR variable, niveaux calculés dynamiquement
 * Sprint Strategy Lab
 */

import { useState, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ReferenceLine, ReferenceDot, ResponsiveContainer, Area, Tooltip,
} from 'recharts'

// ─── Paramètres de la stratégie (identiques à strategyMeta) ───
const ATR_MULT_START = 2.0
const ATR_MULT_STEP = 1.0
const NUM_LEVELS = 3
const SL_PERCENT = 20

// ─── Données brutes du scénario ───
// ATR variable : ~450 → ~640 (crash) → ~520 (recovery)
// SMA avec lag crédible sur le prix
const BASE_DATA = [
  { t: 0,  label: '1 Jan',   price: 42500, sma: 42200, atr: 450 },
  { t: 1,  label: '2 Jan',   price: 42100, sma: 42180, atr: 455 },
  { t: 2,  label: '3 Jan',   price: 41600, sma: 42100, atr: 470 },
  { t: 3,  label: '4 Jan',   price: 41000, sma: 41950, atr: 490 },
  { t: 4,  label: '5 Jan',   price: 40200, sma: 41750, atr: 520 },
  { t: 5,  label: '6 Jan',   price: 39500, sma: 41500, atr: 550 },
  { t: 6,  label: '7 Jan',   price: 39200, sma: 41200, atr: 570 },
  { t: 7,  label: '8 Jan',   price: 38400, sma: 40900, atr: 600 },
  { t: 8,  label: '9 Jan',   price: 37600, sma: 40500, atr: 620 },
  { t: 9,  label: '10 Jan',  price: 37100, sma: 40100, atr: 640 },
  { t: 10, label: '11 Jan',  price: 37400, sma: 39800, atr: 635 },
  { t: 11, label: '12 Jan',  price: 37800, sma: 39500, atr: 620 },
]

// Scénario recovery (happy end)
const RECOVERY_DATA = [
  { t: 12, label: '13 Jan', price: 38500, sma: 39300, atr: 600 },
  { t: 13, label: '14 Jan', price: 39200, sma: 39200, atr: 580 },
  { t: 14, label: '15 Jan', price: 39800, sma: 39150, atr: 560 },
  { t: 15, label: '16 Jan', price: 40200, sma: 39200, atr: 540 },
  { t: 16, label: '17 Jan', price: 40800, sma: 39400, atr: 520 },
  { t: 17, label: '18 Jan', price: 41200, sma: 39700, atr: 510 },
]

// Scénario disaster (SL touché)
const DISASTER_DATA = [
  { t: 12, label: '13 Jan', price: 37000, sma: 39300, atr: 650 },
  { t: 13, label: '14 Jan', price: 36200, sma: 39000, atr: 670 },
  { t: 14, label: '15 Jan', price: 35500, sma: 38600, atr: 690 },
  { t: 15, label: '16 Jan', price: 34800, sma: 38200, atr: 710 },
  { t: 16, label: '17 Jan', price: 33500, sma: 37700, atr: 730 },
  { t: 17, label: '18 Jan', price: 32500, sma: 37200, atr: 750 },
]

// ─── Calcul dynamique des niveaux ───
function computeLevels(dataPoint) {
  const levels = []
  for (let i = 0; i < NUM_LEVELS; i++) {
    const mult = ATR_MULT_START + i * ATR_MULT_STEP
    levels.push({
      level: i + 1,
      mult,
      price: dataPoint.sma - dataPoint.atr * mult,
    })
  }
  return levels
}

// Points d'achat (basés sur le moment où le prix croise chaque niveau)
function findBuyPoints(data) {
  const buys = []
  const bought = new Set()

  for (let i = 1; i < data.length; i++) {
    const levels = computeLevels(data[i])
    for (const lvl of levels) {
      if (!bought.has(lvl.level) && data[i].price <= lvl.price && data[i - 1].price > lvl.price) {
        buys.push({ t: data[i].t, level: lvl.level, price: data[i].price, levelPrice: lvl.price })
        bought.add(lvl.level)
      }
    }
  }
  return buys
}

// ─── Enrichir les données avec les niveaux calculés ───
function enrichData(rawData) {
  return rawData.map(d => {
    const levels = computeLevels(d)
    return {
      ...d,
      level1: levels[0].price,
      level2: levels[1].price,
      level3: levels[2].price,
    }
  })
}

// ─── Étapes du tutoriel ───
const STEPS = [
  {
    title: 'Le prix et sa moyenne',
    desc: 'La SMA (Simple Moving Average) représente le prix "juste" à moyen terme. Quand le prix s\'en écarte fortement, il a tendance à y revenir — c\'est le principe du mean reversion.',
    keyInsight: 'La SMA est notre ancre. Tout le système Grid ATR tourne autour de cet indicateur.',
    show: { price: true, sma: true },
  },
  {
    title: "L'ATR mesure la volatilité",
    desc: "L'ATR (Average True Range) mesure l'amplitude moyenne des mouvements de prix. Quand le marché est nerveux (crash), l'ATR augmente. Quand c'est calme, il diminue. C'est ce qui rend Grid ATR adaptatif.",
    keyInsight: "L'ATR passe de ~450$ en temps calme à ~640$ pendant le crash. Les niveaux d'achat s'adaptent automatiquement.",
    show: { price: true, sma: true, atr: true },
  },
  {
    title: 'La grille adaptative',
    desc: 'Les 3 niveaux d\'achat sont positionnés dynamiquement : Niveau i = SMA - ATR × (2 + i). Comme la SMA et l\'ATR bougent, la grille se déplace avec la volatilité — contrairement aux enveloppes à % fixe.',
    keyInsight: 'Quand la volatilité augmente, les niveaux s\'écartent → on achète plus bas → meilleur prix moyen.',
    show: { price: true, sma: true, levels: true },
  },
  {
    title: 'Le crash : Achat Niveau 1',
    desc: 'Le prix plonge et franchit le premier niveau. Grid ATR ouvre automatiquement une position LONG. La grille a anticipé le dip grâce à l\'ATR élevé.',
    keyInsight: 'Le Niveau 1 est à SMA - ATR × 2.0. Avec ATR=570$, ça place l\'achat ~1140$ sous la SMA.',
    show: { price: true, sma: true, levels: true, buys: [1] },
  },
  {
    title: 'DCA : Niveaux 2 et 3',
    desc: 'Le crash continue. Le prix franchit les niveaux 2 puis 3. À chaque franchissement, une nouvelle position s\'ouvre — c\'est le DCA (Dollar Cost Averaging). Le prix moyen baisse à chaque achat.',
    keyInsight: 'Prix moyen pondéré = somme des prix d\'entrée / nombre de positions. Plus on DCA bas, meilleur est le prix moyen.',
    show: { price: true, sma: true, levels: true, buys: [1, 2, 3], avgPrice: true },
  },
  {
    title: 'Happy end : le TP',
    desc: 'Le marché se retourne. Le prix remonte et croise la SMA — c\'est le signal de Take Profit. Toutes les positions sont fermées en profit.',
    keyInsight: 'TP = prix croise la SMA. Pas de prix fixe : le TP suit la SMA, donc il est aussi adaptatif.',
    show: { price: true, sma: true, levels: true, buys: [1, 2, 3], avgPrice: true, tp: true },
    scenario: 'recovery',
  },
  {
    title: 'La catastrophe : le SL',
    desc: 'Scénario alternatif. Le crash ne s\'arrête pas. Le prix continue de chuter au-delà du SL global (prix moyen - 20%). Toutes les positions sont fermées en perte.',
    keyInsight: 'Le SL global protège le capital. Perte = 20% du prix moyen × 3 positions × leverage. C\'est pourquoi le sizing est crucial.',
    show: { price: true, sma: true, levels: true, buys: [1, 2, 3], avgPrice: true, sl: true },
    scenario: 'disaster',
  },
]

// ─── Composant principal ───
export default function GridAtrGuide() {
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

  // Prix moyen pondéré des achats visibles
  const avgPrice = visibleBuys.length > 0
    ? visibleBuys.reduce((s, b) => s + b.price, 0) / visibleBuys.length
    : null

  // SL price
  const slPrice = avgPrice ? avgPrice * (1 - SL_PERCENT / 100) : null

  // Point de TP (prix croise la SMA)
  const tpPoint = useMemo(() => {
    if (!currentStep.show.tp) return null
    const data = currentStep.scenario === 'recovery'
      ? [...BASE_DATA, ...RECOVERY_DATA]
      : BASE_DATA
    for (let i = 1; i < data.length; i++) {
      if (data[i].price >= data[i].sma && data[i - 1].price < data[i - 1].sma && data[i].t >= 12) {
        return { t: data[i].t, price: data[i].sma }
      }
    }
    return null
  }, [step]) // eslint-disable-line react-hooks/exhaustive-deps

  // Y domain
  const allPrices = fullData.flatMap(d => [d.price, d.sma, d.level1, d.level2, d.level3])
  if (slPrice) allPrices.push(slPrice)
  const yMin = Math.floor(Math.min(...allPrices) / 500) * 500 - 500
  const yMax = Math.ceil(Math.max(...allPrices) / 500) * 500 + 500

  // Chiffres de l'étape courante (point milieu du crash)
  const refPoint = fullData[Math.min(7, fullData.length - 1)]
  const refLevels = computeLevels(refPoint)

  // Gain/perte pour les étapes 6 et 7
  let resultText = null
  if (step === 5 && tpPoint && avgPrice) {
    const gain = ((tpPoint.price - avgPrice) / avgPrice * 100).toFixed(1)
    resultText = `Gain : +${gain}% (hors leverage et fees)`
  }
  if (step === 6 && slPrice && avgPrice) {
    resultText = `Perte : -${SL_PERCENT}% (SL global touché)`
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
              tickFormatter={v => `${(v / 1000).toFixed(1)}k`}
              width={50}
            />
            <Tooltip
              contentStyle={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 6, fontSize: 12 }}
              labelStyle={{ color: '#888' }}
              formatter={(val, name) => [`$${val.toLocaleString()}`, name]}
            />

            {/* Niveaux d'achat (zones) */}
            {currentStep.show.levels && (
              <>
                <Line type="monotone" dataKey="level1" stroke="#00e68a" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Niveau 1" />
                <Line type="monotone" dataKey="level2" stroke="#00e68a" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Niveau 2" opacity={0.7} />
                <Line type="monotone" dataKey="level3" stroke="#00e68a" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Niveau 3" opacity={0.5} />
              </>
            )}

            {/* SMA */}
            {currentStep.show.sma && (
              <Line type="monotone" dataKey="sma" stroke="#ffc53d" strokeWidth={2} strokeDasharray="6 4" dot={false} name="SMA" />
            )}

            {/* Prix */}
            {currentStep.show.price && (
              <Line type="monotone" dataKey="price" stroke="#4da6ff" strokeWidth={2.5} dot={false} name="Prix" />
            )}

            {/* ATR zone (bande autour de la SMA) */}
            {currentStep.show.atr && (
              <>
                <Area type="monotone" dataKey="sma" stroke="none" fill="none" />
                {fullData.map((d, i) => (
                  <ReferenceLine
                    key={`atr-${i}`}
                    segment={[
                      { x: d.label, y: d.sma - d.atr },
                      { x: d.label, y: d.sma + d.atr },
                    ]}
                    stroke="rgba(255,140,66,0.2)"
                    strokeWidth={18}
                  />
                ))}
              </>
            )}

            {/* Prix moyen */}
            {currentStep.show.avgPrice && avgPrice && (
              <ReferenceLine y={avgPrice} stroke="#ff8c42" strokeWidth={1.5} strokeDasharray="6 3" label={{ value: `Avg: $${avgPrice.toFixed(0)}`, fill: '#ff8c42', fontSize: 11, position: 'right' }} />
            )}

            {/* SL */}
            {currentStep.show.sl && slPrice && (
              <ReferenceLine y={slPrice} stroke="#ff4466" strokeWidth={2} strokeDasharray="4 4" label={{ value: `SL: $${slPrice.toFixed(0)}`, fill: '#ff4466', fontSize: 11, position: 'right' }} />
            )}

            {/* TP point */}
            {tpPoint && (
              <ReferenceDot x={fullData.find(d => d.t === tpPoint.t)?.label} y={tpPoint.price} r={7} fill="#00e68a" stroke="#fff" strokeWidth={2} />
            )}

            {/* Points d'achat */}
            {visibleBuys.map(b => (
              <ReferenceDot
                key={b.level}
                x={fullData.find(d => d.t === b.t)?.label}
                y={b.price}
                r={6}
                fill="#00e68a"
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
              width: 10,
              height: 10,
              borderRadius: '50%',
              border: 'none',
              background: i === step ? '#4da6ff' : '#444',
              cursor: 'pointer',
              transition: 'background 0.2s',
              padding: 0,
            }}
            title={STEPS[i].title}
          />
        ))}
      </div>

      {/* Panneau d'info */}
      <div style={{
        background: '#0d1117',
        border: '1px solid #333',
        borderRadius: 8,
        padding: 16,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <h4 style={{ margin: 0, color: '#e8eaed', fontSize: 16 }}>
            {step + 1}/{STEPS.length} — {currentStep.title}
          </h4>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => setStep(Math.max(0, step - 1))}
              disabled={step === 0}
              style={{
                padding: '5px 14px',
                border: '1px solid #444',
                borderRadius: 4,
                background: step === 0 ? '#1a1a1a' : '#2a2a2a',
                color: step === 0 ? '#555' : '#ccc',
                cursor: step === 0 ? 'not-allowed' : 'pointer',
                fontSize: 13,
              }}
            >
              ← Précédent
            </button>
            <button
              onClick={() => setStep(Math.min(STEPS.length - 1, step + 1))}
              disabled={step === STEPS.length - 1}
              style={{
                padding: '5px 14px',
                border: '1px solid #444',
                borderRadius: 4,
                background: step === STEPS.length - 1 ? '#1a1a1a' : '#2a2a2a',
                color: step === STEPS.length - 1 ? '#555' : '#ccc',
                cursor: step === STEPS.length - 1 ? 'not-allowed' : 'pointer',
                fontSize: 13,
              }}
            >
              Suivant →
            </button>
          </div>
        </div>

        <p style={{ color: '#aaa', fontSize: 13, lineHeight: 1.6, margin: '0 0 12px 0' }}>
          {currentStep.desc}
        </p>

        {/* Encadré chiffres */}
        <div style={{
          background: '#1a1a2a',
          border: '1px solid rgba(77,166,255,0.2)',
          borderRadius: 6,
          padding: 10,
          marginBottom: 10,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
          gap: 6,
          fontSize: 12,
        }}>
          <span style={{ color: '#888' }}>SMA: <b style={{ color: '#ffc53d' }}>${refPoint.sma.toLocaleString()}</b></span>
          <span style={{ color: '#888' }}>ATR: <b style={{ color: '#ff8c42' }}>${refPoint.atr}</b></span>
          {refLevels.map(l => (
            <span key={l.level} style={{ color: '#888' }}>
              Niv.{l.level} (×{l.mult}): <b style={{ color: '#00e68a' }}>${l.price.toFixed(0)}</b>
            </span>
          ))}
          {avgPrice && (
            <span style={{ color: '#888' }}>Avg: <b style={{ color: '#ff8c42' }}>${avgPrice.toFixed(0)}</b></span>
          )}
          {slPrice && step >= 6 && (
            <span style={{ color: '#888' }}>SL: <b style={{ color: '#ff4466' }}>${slPrice.toFixed(0)}</b></span>
          )}
        </div>

        {/* Résultat gain/perte */}
        {resultText && (
          <div style={{
            padding: '8px 12px',
            borderRadius: 4,
            marginBottom: 10,
            fontSize: 13,
            fontWeight: 600,
            background: step === 5 ? 'rgba(0,230,138,0.1)' : 'rgba(255,68,102,0.1)',
            color: step === 5 ? '#00e68a' : '#ff4466',
            border: `1px solid ${step === 5 ? 'rgba(0,230,138,0.3)' : 'rgba(255,68,102,0.3)'}`,
          }}>
            {resultText}
          </div>
        )}

        {/* Concept clé */}
        <div style={{
          fontSize: 13,
          color: '#4da6ff',
          fontWeight: 600,
        }}>
          🔑 {currentStep.keyInsight}
        </div>
      </div>
    </div>
  )
}
