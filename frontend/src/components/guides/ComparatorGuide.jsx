/**
 * ComparatorGuide — Comparateur interactif Envelope DCA vs Grid ATR en 6 etapes
 * Meme scenario de crash, deux strategies cote a cote
 * Sprint Strategy Lab V3
 */

import { useState, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ReferenceLine, ReferenceDot, ResponsiveContainer, Tooltip,
} from 'recharts'

// --- Donnees partagees (identiques aux deux graphiques, BTC-like) ---
// ATR variable : ~450 -> ~640 (crash) -> ~520 (recovery)
const SHARED_DATA = [
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
  // Recovery
  { t: 12, label: '13 Jan', price: 38500, sma: 39300, atr: 600 },
  { t: 13, label: '14 Jan', price: 39200, sma: 39200, atr: 580 },
  { t: 14, label: '15 Jan', price: 39800, sma: 39150, atr: 560 },
  { t: 15, label: '16 Jan', price: 40200, sma: 39200, atr: 540 },
  { t: 16, label: '17 Jan', price: 40800, sma: 39400, atr: 520 },
  { t: 17, label: '18 Jan', price: 41200, sma: 39700, atr: 510 },
]

// --- Calcul des niveaux Envelope DCA (% fixes) ---
function computeEnvelopeLevels(dataPoint, { envelopeStart = 0.05, envelopeStep = 0.05, numLevels = 3 } = {}) {
  const levels = []
  for (let i = 0; i < numLevels; i++) {
    const offset = envelopeStart + i * envelopeStep
    levels.push({
      level: i + 1,
      price: dataPoint.sma * (1 - offset),
      label: `${(offset * 100).toFixed(0)}%`,
    })
  }
  return levels
}

// --- Calcul des niveaux Grid ATR (adaptatif) ---
function computeGridAtrLevels(dataPoint, { atrMultStart = 2.0, atrMultStep = 1.0, numLevels = 3 } = {}) {
  const levels = []
  for (let i = 0; i < numLevels; i++) {
    const mult = atrMultStart + i * atrMultStep
    levels.push({
      level: i + 1,
      price: dataPoint.sma - dataPoint.atr * mult,
      label: `ATR x${mult.toFixed(1)}`,
    })
  }
  return levels
}

// --- Enrichir les donnees avec niveaux ---
function enrichWithLevels(data, computeFn, params) {
  return data.map(d => {
    const levels = computeFn(d, params)
    const enriched = { ...d }
    for (let i = 0; i < 3; i++) {
      enriched[`level${i + 1}`] = levels[i] ? levels[i].price : undefined
    }
    return enriched
  })
}

// --- Points d'achat (quand le prix croise un niveau vers le bas) ---
function findBuyPoints(data, computeFn, params) {
  const buys = []
  const bought = new Set()
  for (let i = 1; i < data.length; i++) {
    const levels = computeFn(data[i], params)
    for (const lvl of levels) {
      if (!bought.has(lvl.level) && data[i].price <= lvl.price && data[i - 1].price > lvl.price) {
        buys.push({ t: data[i].t, level: lvl.level, price: data[i].price, levelPrice: lvl.price })
        bought.add(lvl.level)
      }
    }
  }
  return buys
}

// --- Point TP (prix croise la SMA vers le haut apres le crash) ---
function findTpPoint(data, minT = 10) {
  for (let i = 1; i < data.length; i++) {
    if (data[i].t >= minT && data[i].price >= data[i].sma && data[i - 1].price < data[i - 1].sma) {
      return { t: data[i].t, price: data[i].sma }
    }
  }
  return null
}

// --- Etapes du tutoriel ---
const STEPS = [
  {
    title: 'Deux strategies, meme objectif',
    desc: "Envelope DCA et Grid ATR partagent le meme concept : acheter les dips sous la SMA par paliers (DCA) et revendre au retour a la moyenne. La difference ? Comment les niveaux d'achat sont calcules.",
    keyInsight: "Meme concept de base, mais la methode de calcul des niveaux fait toute la difference.",
    showLevels: false,
    showBuys: false,
    showAvg: false,
    showTp: false,
    dataSlice: [0, 12],
  },
  {
    title: 'Marche calme : niveaux similaires',
    desc: "En marche calme (ATR ~ 450$), les deux strategies placent leurs niveaux dans le meme ordre de grandeur. Envelope DCA a des niveaux fixes (5/10/15% sous la SMA). Grid ATR utilise ATR x 2/3/4 qui, avec un ATR bas, donne des ecarts similaires.",
    keyInsight: "En marche calme, les deux approches donnent des resultats comparables.",
    showLevels: true,
    showBuys: false,
    showAvg: false,
    showTp: false,
    dataSlice: [0, 4],
  },
  {
    title: "Le crash : l'ATR explose",
    desc: "Le prix plonge. L'ATR passe de 450$ a 640$. Les niveaux Grid ATR s'ecartent automatiquement (SMA - 640x2 au lieu de SMA - 450x2). Les niveaux Envelope DCA restent a % fixe de la SMA et ne reagissent pas a la volatilite.",
    keyInsight: "L'ATR mesure la peur du marche. Quand la peur augmente, Grid ATR recule ses niveaux. Envelope DCA ne bouge pas.",
    showLevels: true,
    showBuys: false,
    showAvg: false,
    showTp: false,
    dataSlice: [0, 12],
  },
  {
    title: 'Le DCA : qui achete ou ?',
    desc: "Les deux strategies DCA pendant le crash, mais a des prix differents. Envelope DCA achete plus tot (niveaux plus hauts, plus proches de la SMA). Grid ATR achete plus tard (niveaux plus bas grace a l'ATR eleve). Le prix moyen final est different.",
    keyInsight: null, // computed dynamically
    showLevels: true,
    showBuys: true,
    showAvg: true,
    showTp: false,
    dataSlice: [0, 12],
  },
  {
    title: 'Le TP : meme sortie, profits differents',
    desc: "Les deux strategies ferment au retour a la SMA. Mais Grid ATR a un meilleur prix moyen (achats plus bas) donc un profit superieur pour la meme sortie.",
    keyInsight: null, // computed dynamically
    showLevels: true,
    showBuys: true,
    showAvg: true,
    showTp: true,
    dataSlice: [0, 18],
  },
  {
    title: 'Verdict : pourquoi Grid ATR a remplace Envelope DCA',
    desc: null,
    keyInsight: "Grid ATR a remplace Envelope DCA dans Scalp Radar. Meme concept, mais les niveaux adaptatifs font la difference quand ca compte le plus : pendant les crashes.",
    showComparison: true,
  },
]

const LEVEL_COLORS = ['#00e68a', '#00e68a', '#00e68a']
const LEVEL_OPACITIES = [1, 0.7, 0.5]

// --- Composant principal ---
export default function ComparatorGuide() {
  const [step, setStep] = useState(0)
  const currentStep = STEPS[step]

  // Slice de donnees visible selon l'etape
  const visibleData = useMemo(() => {
    if (currentStep.showComparison) return []
    const [start, end] = currentStep.dataSlice
    return SHARED_DATA.slice(start, end)
  }, [step]) // eslint-disable-line react-hooks/exhaustive-deps

  // Donnees enrichies pour les deux strategies
  const envelopeData = useMemo(() => enrichWithLevels(visibleData, computeEnvelopeLevels, {}), [visibleData])
  const gridAtrData = useMemo(() => enrichWithLevels(visibleData, computeGridAtrLevels, {}), [visibleData])

  // Points d'achat
  const envelopeBuys = useMemo(() => findBuyPoints(visibleData, computeEnvelopeLevels, {}), [visibleData])
  const gridAtrBuys = useMemo(() => findBuyPoints(visibleData, computeGridAtrLevels, {}), [visibleData])

  // Pour le TP (etape 5), on utilise toutes les donnees
  const fullEnvelopeData = useMemo(() => enrichWithLevels(SHARED_DATA, computeEnvelopeLevels, {}), [])
  const fullGridAtrData = useMemo(() => enrichWithLevels(SHARED_DATA, computeGridAtrLevels, {}), [])
  const fullEnvelopeBuys = useMemo(() => findBuyPoints(SHARED_DATA, computeEnvelopeLevels, {}), [])
  const fullGridAtrBuys = useMemo(() => findBuyPoints(SHARED_DATA, computeGridAtrLevels, {}), [])
  const tpPoint = useMemo(() => findTpPoint(SHARED_DATA), [])

  // Pour les etapes TP, utiliser les donnees completes
  const useFullData = currentStep.showTp
  const chartEnvelopeData = useFullData ? fullEnvelopeData : envelopeData
  const chartGridAtrData = useFullData ? fullGridAtrData : gridAtrData
  const chartEnvelopeBuys = useFullData ? fullEnvelopeBuys : envelopeBuys
  const chartGridAtrBuys = useFullData ? fullGridAtrBuys : gridAtrBuys

  // Prix moyen
  const envAvg = chartEnvelopeBuys.length > 0
    ? chartEnvelopeBuys.reduce((s, b) => s + b.price, 0) / chartEnvelopeBuys.length
    : null
  const atrAvg = chartGridAtrBuys.length > 0
    ? chartGridAtrBuys.reduce((s, b) => s + b.price, 0) / chartGridAtrBuys.length
    : null

  // Key insight dynamique pour etapes 4 et 5
  let keyInsight = currentStep.keyInsight
  if (step === 3 && envAvg && atrAvg) {
    const diff = envAvg - atrAvg
    keyInsight = `Envelope DCA : avg ~$${envAvg.toFixed(0)}. Grid ATR : avg ~$${atrAvg.toFixed(0)}. $${diff.toFixed(0)} de difference sur le prix moyen.`
  }
  if (step === 4 && envAvg && atrAvg && tpPoint) {
    const envGain = ((tpPoint.price - envAvg) / envAvg * 100).toFixed(1)
    const atrGain = ((tpPoint.price - atrAvg) / atrAvg * 100).toFixed(1)
    const delta = (parseFloat(atrGain) - parseFloat(envGain)).toFixed(1)
    keyInsight = `Meme sortie a ~$${tpPoint.price.toFixed(0)}, mais Grid ATR gagne +${atrGain}% vs +${envGain}% pour Envelope DCA (+${delta}% de bonus).`
  }

  // Domaine Y partage (meme echelle pour les deux graphiques)
  const allChartData = [...chartEnvelopeData, ...chartGridAtrData]
  const allPrices = allChartData.flatMap(d => {
    const vals = [d.price, d.sma]
    for (let i = 1; i <= 3; i++) {
      if (d[`level${i}`] !== undefined) vals.push(d[`level${i}`])
    }
    return vals
  })
  const yMin = Math.floor(Math.min(...(allPrices.length > 0 ? allPrices : [35000])) / 500) * 500 - 500
  const yMax = Math.ceil(Math.max(...(allPrices.length > 0 ? allPrices : [43000])) / 500) * 500 + 500

  // Point de reference pour les chiffres
  const refIdx = Math.min(7, visibleData.length - 1)
  const refPoint = visibleData.length > 0 ? visibleData[Math.max(0, refIdx)] : SHARED_DATA[7]
  const envLevels = computeEnvelopeLevels(refPoint)
  const atrLevels = computeGridAtrLevels(refPoint)

  // Etape comparative (tableau, pas de graphique)
  if (currentStep.showComparison) {
    return (
      <div>
        {/* Navigation dots */}
        <NavDots steps={STEPS} step={step} setStep={setStep} />

        {/* Panneau comparatif */}
        <div style={{
          background: '#0d1117', border: '1px solid #333',
          borderRadius: 8, padding: 16, minHeight: 290,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <h4 style={{ margin: 0, color: '#e8eaed', fontSize: 16 }}>
              {step + 1}/{STEPS.length} — {currentStep.title}
            </h4>
            <NavButtons step={step} setStep={setStep} total={STEPS.length} />
          </div>

          {/* Tableau recapitulatif */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%', borderCollapse: 'collapse', fontSize: 12,
              color: '#ccc', marginBottom: 16,
            }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #333' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#888' }}></th>
                  <th style={{ textAlign: 'center', padding: '8px 10px', color: '#f97316' }}>Envelope DCA</th>
                  <th style={{ textAlign: 'center', padding: '8px 10px', color: '#4da6ff' }}>Grid ATR</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Niveaux', '% fixe (5/10/15%)', 'ATR x mult (adaptatif)'],
                  ['En marche calme', '~ similaire', '~ similaire'],
                  ['En crash (ATR \u2191)', 'Achete trop tot', 'Achete plus bas'],
                  ['Prix moyen', 'Plus haut', 'Plus bas'],
                  ['Profit sur le TP', 'Bon', 'Meilleur'],
                  ['Indicateurs', 'SMA seule', 'SMA + ATR'],
                  ['Grade WFO', 'B (21/23 assets)', 'B (comparable)'],
                  ['Statut', 'Remplace', 'LIVE'],
                ].map(([label, env, atr], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #222' }}>
                    <td style={{ padding: '6px 10px', color: '#888', fontWeight: 600 }}>{label}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{env}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{atr}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Concept cle */}
          <div style={{ fontSize: 13, color: '#4da6ff', fontWeight: 600 }}>
            {keyInsight}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Deux graphiques cote a cote */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <ChartPanel
          title="Envelope DCA (% fixes)"
          titleColor="#f97316"
          data={chartEnvelopeData}
          buys={currentStep.showBuys ? chartEnvelopeBuys : []}
          avgPrice={currentStep.showAvg ? envAvg : null}
          tpPoint={currentStep.showTp ? tpPoint : null}
          showLevels={currentStep.showLevels}
          yMin={yMin}
          yMax={yMax}
        />
        <ChartPanel
          title="Grid ATR (adaptatif)"
          titleColor="#4da6ff"
          data={chartGridAtrData}
          buys={currentStep.showBuys ? chartGridAtrBuys : []}
          avgPrice={currentStep.showAvg ? atrAvg : null}
          tpPoint={currentStep.showTp ? tpPoint : null}
          showLevels={currentStep.showLevels}
          yMin={yMin}
          yMax={yMax}
        />
      </div>

      {/* Navigation dots */}
      <NavDots steps={STEPS} step={step} setStep={setStep} />

      {/* Panneau d'info */}
      <div style={{
        background: '#0d1117', border: '1px solid #333',
        borderRadius: 8, padding: 16, minHeight: 290,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <h4 style={{ margin: 0, color: '#e8eaed', fontSize: 16 }}>
            {step + 1}/{STEPS.length} — {currentStep.title}
          </h4>
          <NavButtons step={step} setStep={setStep} total={STEPS.length} />
        </div>

        <p style={{ color: '#aaa', fontSize: 13, lineHeight: 1.6, margin: '0 0 12px 0' }}>
          {currentStep.desc}
        </p>

        {/* Encadre chiffres — deux colonnes */}
        <div style={{
          background: '#1a1a2a',
          border: '1px solid rgba(77,166,255,0.2)',
          borderRadius: 6, padding: 10, marginBottom: 10,
          fontSize: 12,
        }}>
          <div style={{ display: 'flex', gap: 16, marginBottom: 6 }}>
            <span style={{ color: '#888' }}>SMA: <b style={{ color: '#ffc53d' }}>${refPoint.sma.toLocaleString()}</b></span>
            <span style={{ color: '#888' }}>ATR: <b style={{ color: '#ff8c42' }}>${refPoint.atr}</b></span>
          </div>
          {currentStep.showLevels && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              <div>
                <div style={{ color: '#f97316', fontWeight: 600, fontSize: 11, marginBottom: 4 }}>Envelope DCA</div>
                {envLevels.map(l => (
                  <div key={l.level} style={{ color: '#888' }}>
                    Niv.{l.level} ({l.label}): <b style={{ color: '#00e68a' }}>${l.price.toFixed(0)}</b>
                  </div>
                ))}
                {currentStep.showAvg && envAvg && (
                  <div style={{ color: '#888', marginTop: 2 }}>Avg: <b style={{ color: '#ff8c42' }}>${envAvg.toFixed(0)}</b></div>
                )}
              </div>
              <div>
                <div style={{ color: '#4da6ff', fontWeight: 600, fontSize: 11, marginBottom: 4 }}>Grid ATR</div>
                {atrLevels.map(l => (
                  <div key={l.level} style={{ color: '#888' }}>
                    Niv.{l.level} ({l.label}): <b style={{ color: '#00e68a' }}>${l.price.toFixed(0)}</b>
                  </div>
                ))}
                {currentStep.showAvg && atrAvg && (
                  <div style={{ color: '#888', marginTop: 2 }}>Avg: <b style={{ color: '#ff8c42' }}>${atrAvg.toFixed(0)}</b></div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Resultat gain/perte a l'etape 5 */}
        {step === 4 && envAvg && atrAvg && tpPoint && (
          <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
            <div style={{
              flex: 1, padding: '8px 12px', borderRadius: 4, fontSize: 13, fontWeight: 600,
              background: 'rgba(249,115,22,0.1)', color: '#f97316',
              border: '1px solid rgba(249,115,22,0.3)',
            }}>
              Envelope : +{((tpPoint.price - envAvg) / envAvg * 100).toFixed(1)}%
            </div>
            <div style={{
              flex: 1, padding: '8px 12px', borderRadius: 4, fontSize: 13, fontWeight: 600,
              background: 'rgba(0,230,138,0.1)', color: '#00e68a',
              border: '1px solid rgba(0,230,138,0.3)',
            }}>
              Grid ATR : +{((tpPoint.price - atrAvg) / atrAvg * 100).toFixed(1)}%
            </div>
          </div>
        )}

        {/* Concept cle */}
        {keyInsight && (
          <div style={{ fontSize: 13, color: '#4da6ff', fontWeight: 600 }}>
            {keyInsight}
          </div>
        )}
      </div>
    </div>
  )
}

// --- Panneau graphique (reutilise pour les deux cotes) ---
function ChartPanel({ title, titleColor, data, buys, avgPrice, tpPoint, showLevels, yMin, yMax }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <h4 style={{ margin: '0 0 8px 0', color: titleColor, fontSize: 13, textAlign: 'center' }}>
        {title}
      </h4>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis dataKey="label" tick={{ fill: '#888', fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis
            domain={[yMin, yMax]}
            tick={{ fill: '#888', fontSize: 10 }}
            tickFormatter={v => `${(v / 1000).toFixed(1)}k`}
            width={42}
          />
          <Tooltip
            contentStyle={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 6, fontSize: 11 }}
            labelStyle={{ color: '#888' }}
            formatter={(val, name) => {
              if (val === undefined || val === null) return ['-', name]
              return [`$${val.toLocaleString()}`, name]
            }}
          />

          {/* Niveaux d'achat */}
          {showLevels && (
            <>
              {[1, 2, 3].map(i => (
                <Line
                  key={`level-${i}`}
                  type="monotone"
                  dataKey={`level${i}`}
                  stroke={LEVEL_COLORS[i - 1]}
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  dot={false}
                  name={`Niveau ${i}`}
                  opacity={LEVEL_OPACITIES[i - 1]}
                  connectNulls={false}
                />
              ))}
            </>
          )}

          {/* SMA */}
          <Line type="monotone" dataKey="sma" stroke="#ffc53d" strokeWidth={2} strokeDasharray="6 4" dot={false} name="SMA" />

          {/* Prix */}
          <Line type="monotone" dataKey="price" stroke="#4da6ff" strokeWidth={2.5} dot={false} name="Prix" />

          {/* Prix moyen */}
          {avgPrice && (
            <ReferenceLine
              y={avgPrice}
              stroke="#ff8c42"
              strokeWidth={1.5}
              strokeDasharray="6 3"
              label={{ value: `Avg: $${avgPrice.toFixed(0)}`, fill: '#ff8c42', fontSize: 10, position: 'right' }}
            />
          )}

          {/* TP point */}
          {tpPoint && (
            <ReferenceDot
              x={data.find(d => d.t === tpPoint.t)?.label}
              y={tpPoint.price}
              r={6}
              fill="#00e68a"
              stroke="#fff"
              strokeWidth={2}
            />
          )}

          {/* Points d'achat */}
          {buys.map(b => (
            <ReferenceDot
              key={b.level}
              x={data.find(d => d.t === b.t)?.label}
              y={b.price}
              r={5}
              fill="#00e68a"
              stroke="#fff"
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// --- Composants utilitaires ---

function NavDots({ steps, step, setStep }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginBottom: 16 }}>
      {steps.map((_, i) => (
        <button
          key={i}
          onClick={() => setStep(i)}
          style={{
            width: 10, height: 10, borderRadius: '50%', border: 'none',
            background: i === step ? '#4da6ff' : '#444',
            cursor: 'pointer', transition: 'background 0.2s', padding: 0,
          }}
          title={steps[i].title}
        />
      ))}
    </div>
  )
}

function NavButtons({ step, setStep, total }) {
  return (
    <div style={{ display: 'flex', gap: 8 }}>
      <button
        onClick={() => setStep(Math.max(0, step - 1))}
        disabled={step === 0}
        style={{
          padding: '5px 14px', border: '1px solid #444', borderRadius: 4,
          background: step === 0 ? '#1a1a1a' : '#2a2a2a',
          color: step === 0 ? '#555' : '#ccc',
          cursor: step === 0 ? 'not-allowed' : 'pointer', fontSize: 13,
        }}
      >
        &larr; Precedent
      </button>
      <button
        onClick={() => setStep(Math.min(total - 1, step + 1))}
        disabled={step === total - 1}
        style={{
          padding: '5px 14px', border: '1px solid #444', borderRadius: 4,
          background: step === total - 1 ? '#1a1a1a' : '#2a2a2a',
          color: step === total - 1 ? '#555' : '#ccc',
          cursor: step === total - 1 ? 'not-allowed' : 'pointer', fontSize: 13,
        }}
      >
        Suivant &rarr;
      </button>
    </div>
  )
}
