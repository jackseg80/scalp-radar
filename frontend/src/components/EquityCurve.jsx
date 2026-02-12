/**
 * EquityCurve — Courbe d'equity en SVG.
 * Conçu pour être wrappé par CollapsibleCard dans App.jsx.
 */
import { useApi } from '../hooks/useApi'

const SVG_W = 400
const SVG_H = 120
const PADDING = { top: 10, right: 10, bottom: 20, left: 10 }

export default function EquityCurve() {
  const { data, loading } = useApi('/api/simulator/equity', 30000)

  const points = data?.equity || []
  const initialCapital = data?.initial_capital || 10000

  if (loading && points.length === 0) {
    return (
      <div className="empty-state">
        <div className="skeleton" style={{ width: '100%', height: 80 }} />
      </div>
    )
  }

  if (points.length === 0) {
    return (
      <svg className="equity-svg" viewBox={`0 0 ${SVG_W} ${SVG_H}`} preserveAspectRatio="none">
        <line
          x1={PADDING.left}
          y1={SVG_H / 2}
          x2={SVG_W - PADDING.right}
          y2={SVG_H / 2}
          stroke="var(--border)"
          strokeWidth={1.5}
          strokeDasharray="4,4"
        />
        <text
          x={SVG_W / 2}
          y={SVG_H / 2 - 8}
          fill="var(--text-muted)"
          textAnchor="middle"
          fontSize={11}
          fontFamily="var(--font-sans)"
        >
          En attente de données...
        </text>
      </svg>
    )
  }

  return <EquityChart points={points} initialCapital={initialCapital} />
}

function EquityChart({ points, initialCapital }) {
  const values = points.map(p => typeof p === 'number' ? p : p.capital || p.equity || p.value || 0)

  if (values.length === 0) return null

  const allValues = [...values, initialCapital]
  const min = Math.min(...allValues)
  const max = Math.max(...allValues)
  const range = max - min || 1

  const chartW = SVG_W - PADDING.left - PADDING.right
  const chartH = SVG_H - PADDING.top - PADDING.bottom

  const toX = (i) => PADDING.left + (i / (values.length - 1)) * chartW
  const toY = (v) => PADDING.top + ((max - v) / range) * chartH

  const polyPoints = values.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')
  const baselineY = toY(initialCapital)

  const lastValue = values[values.length - 1]
  const isProfit = lastValue >= initialCapital
  const lineColor = isProfit ? 'var(--accent)' : 'var(--red)'
  const fillColor = isProfit ? 'rgba(0, 230, 138, 0.08)' : 'rgba(255, 68, 102, 0.08)'

  const areaPoints = `${PADDING.left},${baselineY} ${polyPoints} ${toX(values.length - 1).toFixed(1)},${baselineY}`

  const pnl = lastValue - initialCapital
  const pnlPct = ((pnl / initialCapital) * 100).toFixed(2)

  return (
    <div>
      <svg className="equity-svg" viewBox={`0 0 ${SVG_W} ${SVG_H}`} preserveAspectRatio="none">
        <polygon points={areaPoints} fill={fillColor} />
        <line
          x1={PADDING.left} y1={baselineY}
          x2={SVG_W - PADDING.right} y2={baselineY}
          stroke="var(--border)" strokeWidth={1} strokeDasharray="4,4"
        />
        <polyline
          points={polyPoints} fill="none"
          stroke={lineColor} strokeWidth={1.5}
          strokeLinejoin="round" strokeLinecap="round"
        />
        <circle cx={toX(values.length - 1)} cy={toY(lastValue)} r={3} fill={lineColor} />
        <text
          x={PADDING.left + 2} y={baselineY - 4}
          fill="var(--text-dim)" fontSize={9} fontFamily="var(--font-mono)"
        >
          {initialCapital.toLocaleString('fr-FR')}$
        </text>
      </svg>
      <div className="flex-between" style={{ marginTop: 4 }}>
        <span className="text-xs muted">Capital actuel</span>
        <span className={`mono text-xs ${isProfit ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
          {lastValue.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}$
          ({pnl >= 0 ? '+' : ''}{pnlPct}%)
        </span>
      </div>
    </div>
  )
}
