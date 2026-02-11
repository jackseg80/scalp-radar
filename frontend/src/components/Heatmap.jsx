/**
 * Heatmap — Matrice assets x stratégies avec cellules à gradient d'intensité.
 * Utilise useApi('/api/simulator/conditions', 10000).
 */
import { useApi } from '../hooks/useApi'
import Tooltip from './Tooltip'

const STRATEGY_DESCRIPTIONS = {
  vwap_rsi: 'Mean reversion : achat/vente quand le prix touche le VWAP avec RSI extrême',
  momentum: 'Breakout : trade avec la tendance sur cassure de range + volume',
  funding: 'Arbitrage funding rate : long si taux très négatif, short si très positif',
  liquidation: 'Chasse aux liquidations : trade la cascade quand le prix approche les zones de liquidation',
}

export default function Heatmap() {
  const { data, loading } = useApi('/api/simulator/conditions', 10000)

  const assetsObj = data?.assets || {}
  const assets = Object.entries(assetsObj).map(([symbol, a]) => ({ symbol, ...a }))

  const strategyNames = []
  assets.forEach(a => {
    Object.keys(a.strategies || {}).forEach(name => {
      if (!strategyNames.includes(name)) strategyNames.push(name)
    })
  })

  if (loading && assets.length === 0) {
    return (
      <div className="card">
        <h2>Heatmap</h2>
        <div className="empty-state">
          <div className="skeleton skeleton-line" style={{ width: '90%', margin: '0 auto' }} />
          <div className="skeleton skeleton-line" style={{ width: '90%', margin: '8px auto 0' }} />
          <div className="skeleton skeleton-line" style={{ width: '90%', margin: '8px auto 0' }} />
        </div>
      </div>
    )
  }

  if (assets.length === 0 || strategyNames.length === 0) {
    return (
      <div className="card">
        <h2>Heatmap</h2>
        <div className="empty-state">En attente de données...</div>
      </div>
    )
  }

  const gridTemplate = `100px repeat(${strategyNames.length}, 1fr) 50px`

  return (
    <div className="card">
      <h2>Heatmap — Conditions</h2>

      <div className="heatmap-grid" style={{ gridTemplateColumns: gridTemplate }}>
        {/* En-tête : cellule vide + noms des stratégies + score total */}
        <div className="heatmap-header" />
        {strategyNames.map(name => (
          <div key={name} className="heatmap-header">
            <Tooltip content={STRATEGY_DESCRIPTIONS[name] || name} position="bottom">
              <span>{name}</span>
            </Tooltip>
          </div>
        ))}
        <div className="heatmap-header" style={{ textAlign: 'center', fontWeight: 700 }}>
          <Tooltip content="Score agrégé : conditions remplies / total, toutes stratégies" position="bottom">
            <span>&#931;</span>
          </Tooltip>
        </div>

        {/* Lignes : un asset par ligne */}
        {assets.map(asset => (
          <AssetRow
            key={asset.symbol}
            symbol={asset.symbol}
            strategies={asset.strategies || {}}
            strategyNames={strategyNames}
          />
        ))}
      </div>
    </div>
  )
}

function AssetRow({ symbol, strategies, strategyNames }) {
  let totalMet = 0, totalConditions = 0
  strategyNames.forEach(name => {
    const s = strategies[name]
    if (s) {
      const conditions = s.conditions || []
      totalMet += conditions.filter(c => c.met).length
      totalConditions += conditions.length
    }
  })
  const globalRatio = totalConditions > 0 ? totalMet / totalConditions : 0

  return (
    <>
      <div className="heatmap-cell" style={{ fontWeight: 600, textAlign: 'left', fontSize: 12 }}>
        {symbol}
      </div>
      {strategyNames.map(name => {
        const s = strategies[name]
        if (!s) {
          return (
            <div
              key={name}
              className="heatmap-cell"
              style={{ background: 'rgba(255,255,255,0.015)', color: 'var(--text-dim)' }}
            >
              --
            </div>
          )
        }

        const conditions = s.conditions || []
        const total = conditions.length || 1
        const met = conditions.filter(c => c.met).length
        const ratio = met / total

        const { bg, fg } = getCellColor(ratio)

        const cellTooltip = (
          <div>
            <div style={{ fontWeight: 600, marginBottom: 3 }}>{symbol} — {name}</div>
            {conditions.map((c, i) => (
              <div key={i} style={{ fontSize: 10, opacity: 0.85 }}>
                {c.met ? '\u2713' : '\u2717'} {c.name}
              </div>
            ))}
          </div>
        )

        return (
          <Tooltip key={name} content={cellTooltip}>
            <div
              className="heatmap-cell"
              style={{ background: bg, color: fg, fontWeight: 700 }}
            >
              {met}/{total}
            </div>
          </Tooltip>
        )
      })}
      <div
        className="heatmap-cell"
        style={{
          textAlign: 'center',
          fontWeight: 800,
          fontSize: 12,
          color: getScoreColor(globalRatio),
          background: getScoreBg(globalRatio),
        }}
      >
        {Math.round(globalRatio * 100)}
      </div>
    </>
  )
}

function getCellColor(ratio) {
  // Intensité proportionnelle au ratio pour un vrai gradient
  if (ratio >= 0.75) return { bg: `rgba(0, 230, 138, ${0.08 + ratio * 0.22})`, fg: '#00e68a' }
  if (ratio >= 0.55) return { bg: `rgba(255, 197, 61, ${0.06 + ratio * 0.18})`, fg: '#ffc53d' }
  if (ratio >= 0.35) return { bg: `rgba(255, 140, 66, ${0.04 + ratio * 0.14})`, fg: '#ff8c42' }
  if (ratio > 0)     return { bg: `rgba(255, 68, 102, ${0.03 + ratio * 0.10})`, fg: '#ff4466' }
  return { bg: 'rgba(255, 255, 255, 0.015)', fg: 'var(--text-dim)' }
}

function getScoreColor(score) {
  if (score >= 0.75) return '#00e68a'
  if (score >= 0.55) return '#ffc53d'
  if (score >= 0.35) return '#ff8c42'
  return '#ff4466'
}

function getScoreBg(score) {
  const color = getScoreColor(score)
  return `${color}18`
}
