/**
 * GridAtrGuide — Tutoriel interactif Grid ATR en 7 étapes
 * Données BTC-like avec ATR variable, niveaux calculés dynamiquement
 * Sprint Strategy Lab + V2 (sliders de paramètres)
 */

import { useState, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ReferenceLine, ReferenceDot, ResponsiveContainer, Area, Tooltip,
} from 'recharts'

// ─── Valeurs par défaut des paramètres ───
const DEFAULTS = {
  atrMultStart: 2.0,
  atrMultStep: 1.0,
  numLevels: 3,
  slPercent: 20,
  minGridSpacingPct: 0,
}

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

// ─── Calcul dynamique des niveaux (paramétré) ───
function computeLevels(dataPoint, params) {
  const levels = []
  const floor = dataPoint.price * params.minGridSpacingPct / 100
  const effectiveAtr = Math.max(dataPoint.atr, floor)
  for (let i = 0; i < params.numLevels; i++) {
    const mult = params.atrMultStart + i * params.atrMultStep
    levels.push({
      level: i + 1,
      mult,
      price: dataPoint.sma - effectiveAtr * mult,
    })
  }
  return levels
}

// Points d'achat (basés sur le moment où le prix croise chaque niveau)
function findBuyPoints(data, params) {
  const buys = []
  const bought = new Set()

  for (let i = 1; i < data.length; i++) {
    const levels = computeLevels(data[i], params)
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
function enrichData(rawData, params) {
  return rawData.map(d => {
    const levels = computeLevels(d, params)
    const enriched = { ...d }
    for (let i = 0; i < 5; i++) {
      enriched[`level${i + 1}`] = levels[i] ? levels[i].price : undefined
    }
    return enriched
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
    desc: 'Les niveaux d\'achat sont positionnés dynamiquement : Niveau i = SMA - effective_atr × (start + i × step). En V2, un plancher (min_grid_spacing) empêche les niveaux de se rapprocher trop du prix en période calme. Essayez le slider ci-dessous !',
    keyInsight: 'Quand la volatilité augmente, les niveaux s\'écartent → on achète plus bas → meilleur prix moyen. En basse vol, le plancher prend le relais.',
    show: { price: true, sma: true, levels: true },
  },
  {
    title: 'Le crash : Achat Niveau 1',
    desc: 'Le prix plonge et franchit le premier niveau. Grid ATR ouvre automatiquement une position LONG. La grille a anticipé le dip grâce à l\'ATR élevé.',
    keyInsight: 'Le Niveau 1 est à SMA - ATR × start. Avec ATR=570$, ça place l\'achat bien sous la SMA.',
    show: { price: true, sma: true, levels: true, buys: [1] },
  },
  {
    title: 'DCA : Niveaux suivants',
    desc: 'Le crash continue. Le prix franchit les niveaux suivants. À chaque franchissement, une nouvelle position s\'ouvre — c\'est le DCA (Dollar Cost Averaging). Le prix moyen baisse à chaque achat.',
    keyInsight: 'Prix moyen pondéré = somme des prix d\'entrée / nombre de positions. Plus on DCA bas, meilleur est le prix moyen.',
    show: { price: true, sma: true, levels: true, buys: 'all', avgPrice: true },
  },
  {
    title: 'Happy end : le TP',
    desc: 'Le marché se retourne. Le prix remonte et croise la SMA — c\'est le signal de Take Profit. Toutes les positions sont fermées en profit.',
    keyInsight: 'TP = prix croise la SMA. Pas de prix fixe : le TP suit la SMA, donc il est aussi adaptatif.',
    show: { price: true, sma: true, levels: true, buys: 'all', avgPrice: true, tp: true },
    scenario: 'recovery',
  },
  {
    title: 'La catastrophe : le SL',
    desc: 'Scénario alternatif. Le crash ne s\'arrête pas. Le prix continue de chuter au-delà du SL global. Toutes les positions sont fermées en perte.',
    keyInsight: 'Le SL global protège le capital. C\'est pourquoi le sizing est crucial.',
    show: { price: true, sma: true, levels: true, buys: 'all', avgPrice: true, sl: true },
    scenario: 'disaster',
  },
]

// ─── Level colors ───
const LEVEL_COLORS = ['#00e68a', '#00e68a', '#00e68a', '#00cc7a', '#00b36b']
const LEVEL_OPACITIES = [1, 0.7, 0.5, 0.4, 0.35]

// ─── Composant principal ───
export default function GridAtrGuide() {
  const [step, setStep] = useState(0)
  const [params, setParams] = useState(DEFAULTS)
  const [slidersOpen, setSlidersOpen] = useState(false)
  const currentStep = STEPS[step]

  // Resolve buys list (handles 'all' for dynamic num_levels)
  const buyLevels = currentStep.show.buys === 'all'
    ? Array.from({ length: params.numLevels }, (_, i) => i + 1)
    : (currentStep.show.buys || [])

  // Données selon le scénario
  const fullData = useMemo(() => {
    if (currentStep.scenario === 'recovery') {
      return enrichData([...BASE_DATA, ...RECOVERY_DATA], params)
    }
    if (currentStep.scenario === 'disaster') {
      return enrichData([...BASE_DATA, ...DISASTER_DATA], params)
    }
    return enrichData(BASE_DATA, params)
  }, [step, params]) // eslint-disable-line react-hooks/exhaustive-deps

  // Points d'achat
  const buyPoints = useMemo(() => {
    const scenario = currentStep.scenario === 'disaster'
      ? [...BASE_DATA, ...DISASTER_DATA]
      : currentStep.scenario === 'recovery'
        ? [...BASE_DATA, ...RECOVERY_DATA]
        : BASE_DATA
    return findBuyPoints(scenario, params)
  }, [step, params]) // eslint-disable-line react-hooks/exhaustive-deps

  const visibleBuys = buyLevels
    .map(lvl => buyPoints.find(b => b.level === lvl))
    .filter(Boolean)

  // Prix moyen pondéré des achats visibles
  const avgPrice = visibleBuys.length > 0
    ? visibleBuys.reduce((s, b) => s + b.price, 0) / visibleBuys.length
    : null

  // SL price
  const slPrice = avgPrice ? avgPrice * (1 - params.slPercent / 100) : null

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
  const allPrices = fullData.flatMap(d => {
    const vals = [d.price, d.sma]
    for (let i = 1; i <= params.numLevels; i++) {
      if (d[`level${i}`] !== undefined) vals.push(d[`level${i}`])
    }
    return vals
  })
  if (slPrice) allPrices.push(slPrice)
  const yMin = Math.floor(Math.min(...allPrices) / 500) * 500 - 500
  const yMax = Math.ceil(Math.max(...allPrices) / 500) * 500 + 500

  // Chiffres de l'étape courante (point milieu du crash)
  const refPoint = fullData[Math.min(7, fullData.length - 1)]
  const refLevels = computeLevels(refPoint, params)

  // Gain/perte pour les étapes 6 et 7
  let resultText = null
  if (step === 5 && tpPoint && avgPrice) {
    const gain = ((tpPoint.price - avgPrice) / avgPrice * 100).toFixed(1)
    resultText = `Gain : +${gain}% (hors leverage et fees)`
  }
  if (step === 6 && slPrice && avgPrice) {
    resultText = `Perte : -${params.slPercent}% (SL global touché)`
  }

  const isDefault = params.atrMultStart === DEFAULTS.atrMultStart
    && params.atrMultStep === DEFAULTS.atrMultStep
    && params.numLevels === DEFAULTS.numLevels
    && params.slPercent === DEFAULTS.slPercent
    && params.minGridSpacingPct === DEFAULTS.minGridSpacingPct

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
              formatter={(val, name) => {
                if (val === undefined || val === null) return ['-', name]
                return [`$${val.toLocaleString()}`, name]
              }}
            />

            {/* Niveaux d'achat (dynamiques selon numLevels) */}
            {currentStep.show.levels && (
              <>
                {Array.from({ length: params.numLevels }, (_, i) => (
                  <Line
                    key={`level-${i}`}
                    type="monotone"
                    dataKey={`level${i + 1}`}
                    stroke={LEVEL_COLORS[i] || '#00e68a'}
                    strokeWidth={1}
                    strokeDasharray="4 4"
                    dot={false}
                    name={`Niveau ${i + 1}`}
                    opacity={LEVEL_OPACITIES[i] || 0.3}
                    connectNulls={false}
                  />
                ))}
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

      {/* Panneau d'info — min-height fixe pour éviter les sauts de hauteur entre étapes */}
      <div style={{
        background: '#0d1117',
        border: '1px solid #333',
        borderRadius: 8,
        padding: 16,
        minHeight: 290,
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
          {params.minGridSpacingPct > 0 && (
            <span style={{ color: '#888' }}>Eff. ATR: <b style={{ color: '#ff8c42' }}>${Math.max(refPoint.atr, refPoint.price * params.minGridSpacingPct / 100).toFixed(0)}</b></span>
          )}
          {refLevels.map(l => (
            <span key={l.level} style={{ color: '#888' }}>
              Niv.{l.level} (×{l.mult.toFixed(1)}): <b style={{ color: '#00e68a' }}>${l.price.toFixed(0)}</b>
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
          {currentStep.keyInsight}
        </div>
      </div>

      {/* Panneau sliders collapsible */}
      <div style={{
        marginTop: 12,
        border: '1px solid #333',
        borderRadius: 8,
        overflow: 'hidden',
      }}>
        <button
          onClick={() => setSlidersOpen(!slidersOpen)}
          style={{
            width: '100%',
            padding: '10px 16px',
            background: '#0d1117',
            border: 'none',
            color: '#ccc',
            fontSize: 13,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>Ajuster les paramètres</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {!isDefault && (
              <span
                onClick={(e) => { e.stopPropagation(); setParams(DEFAULTS) }}
                style={{
                  fontSize: 11, color: '#4da6ff', cursor: 'pointer',
                  padding: '2px 8px', border: '1px solid #4da6ff33',
                  borderRadius: 4, background: '#4da6ff11',
                }}
              >
                Reset
              </span>
            )}
            <span style={{ fontSize: 11, color: '#666', transition: 'transform 0.2s', transform: slidersOpen ? 'rotate(180deg)' : 'none' }}>
              ▼
            </span>
          </div>
        </button>

        {slidersOpen && (
          <div style={{
            padding: '12px 16px',
            background: '#0d1117',
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
          }}>
            <SliderRow
              label="ATR Mult. Start"
              value={params.atrMultStart}
              min={1.0} max={4.0} step={0.5}
              onChange={v => setParams(p => ({ ...p, atrMultStart: v }))}
            />
            <SliderRow
              label="ATR Mult. Step"
              value={params.atrMultStep}
              min={0.5} max={3.0} step={0.5}
              onChange={v => setParams(p => ({ ...p, atrMultStep: v }))}
            />
            <SliderRow
              label="Niveaux DCA"
              value={params.numLevels}
              min={1} max={5} step={1}
              onChange={v => setParams(p => ({ ...p, numLevels: v }))}
              format={v => `${v}`}
            />
            <SliderRow
              label="Stop Loss %"
              value={params.slPercent}
              min={5} max={40} step={5}
              onChange={v => setParams(p => ({ ...p, slPercent: v }))}
              format={v => `${v}%`}
            />
            <SliderRow
              label="Min Spacing %"
              value={params.minGridSpacingPct}
              min={0} max={3} step={0.1}
              onChange={v => setParams(p => ({ ...p, minGridSpacingPct: v }))}
              format={v => v === 0 ? 'OFF' : `${v.toFixed(1)}%`}
            />
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Composant Slider ───
function SliderRow({ label, value, min, max, step, onChange, format }) {
  const display = format ? format(value) : value.toFixed(1)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ color: '#aaa', fontSize: 12, minWidth: 110, flexShrink: 0 }}>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{
          flex: 1,
          height: 4,
          accentColor: '#4da6ff',
          cursor: 'pointer',
        }}
      />
      <span style={{ color: '#e8eaed', fontSize: 12, fontWeight: 600, minWidth: 40, textAlign: 'right' }}>
        {display}
      </span>
    </div>
  )
}
