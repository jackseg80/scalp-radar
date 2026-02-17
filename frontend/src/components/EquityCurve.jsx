/**
 * EquityCurve — Courbe d'equity en SVG.
 * Conçu pour être wrappé par CollapsibleCard dans App.jsx.
 *
 * Sprint 25 : double source — snapshots journal (equity + unrealized) si disponible,
 * sinon fallback sur /api/simulator/equity (trades fermés uniquement).
 */
import { useState } from 'react'
import { useApi } from '../hooks/useApi'

const SVG_W = 400
const SVG_H = 140
const PADDING = { top: 10, right: 10, bottom: 20, left: 10 }

export default function EquityCurve() {
  const { data: equityData, loading: eqLoading } = useApi('/api/simulator/equity', 30000)
  const { data: journalData } = useApi('/api/journal/snapshots?limit=500', 60000)

  const snapshots = journalData?.snapshots || []
  const useJournal = snapshots.length >= 2

  // Source journal : equity incluant le non réalisé
  const journalPoints = useJournal
    ? snapshots.map(s => ({ equity: s.equity, unrealized: s.unrealized_pnl, margin_ratio: s.margin_ratio, n_positions: s.n_positions, timestamp: s.timestamp }))
    : []

  // Source fallback : trades fermés
  const fallbackPoints = equityData?.equity || []
  const initialCapital = equityData?.initial_capital ?? 10000
  const currentEquity = equityData?.current_equity ?? equityData?.current_capital ?? null

  if (eqLoading && fallbackPoints.length === 0 && journalPoints.length === 0) {
    return (
      <div className="empty-state">
        <div className="skeleton" style={{ width: '100%', height: 80 }} />
      </div>
    )
  }

  if (useJournal) {
    return <JournalEquityChart points={journalPoints} initialCapital={initialCapital} />
  }

  if (fallbackPoints.length === 0) {
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
          En attente de donn&eacute;es...
        </text>
      </svg>
    )
  }

  return <FallbackEquityChart points={fallbackPoints} initialCapital={initialCapital} currentEquity={currentEquity} />
}

/** Courbe depuis les snapshots journal (equity = capital + unrealized). */
function JournalEquityChart({ points, initialCapital }) {
  const [hoverIdx, setHoverIdx] = useState(null)

  const values = points.map(p => p.equity)
  if (values.length === 0) return null

  const allValues = [...values, initialCapital]
  const min = Math.min(...allValues)
  const max = Math.max(...allValues)
  const range = max - min || 1

  const chartW = SVG_W - PADDING.left - PADDING.right
  const chartH = SVG_H - PADDING.top - PADDING.bottom

  const toX = (i) => PADDING.left + (i / Math.max(values.length - 1, 1)) * chartW
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

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const svgX = (x / rect.width) * SVG_W
    const dataX = (svgX - PADDING.left) / chartW
    const idx = Math.round(dataX * (values.length - 1))
    setHoverIdx(Math.max(0, Math.min(idx, values.length - 1)))
  }

  const hoverPoint = hoverIdx !== null ? points[hoverIdx] : null

  return (
    <div>
      <svg
        className="equity-svg"
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        preserveAspectRatio="none"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
        style={{ cursor: hoverIdx !== null ? 'crosshair' : 'default' }}
      >
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

        {/* Hover crosshair */}
        {hoverIdx !== null && (
          <>
            <line
              x1={toX(hoverIdx)} x2={toX(hoverIdx)}
              y1={PADDING.top} y2={PADDING.top + chartH}
              stroke="var(--text-dim)" strokeWidth={1} strokeDasharray="2,2"
            />
            <circle cx={toX(hoverIdx)} cy={toY(values[hoverIdx])} r={3.5} fill={lineColor} stroke="var(--bg)" strokeWidth={1.5} />
          </>
        )}
      </svg>

      {/* Tooltip */}
      {hoverPoint && (
        <div className="text-xs mono" style={{ padding: '2px 0', color: 'var(--text-muted)' }}>
          Equity: {hoverPoint.equity.toLocaleString('fr-FR', { maximumFractionDigits: 0 })}$
          {' '}&middot;{' '}Unreal: {hoverPoint.unrealized >= 0 ? '+' : ''}{hoverPoint.unrealized.toFixed(2)}$
          {' '}&middot;{' '}Margin: {(hoverPoint.margin_ratio * 100).toFixed(0)}%
          {' '}&middot;{' '}Pos: {hoverPoint.n_positions}
        </div>
      )}

      <div className="flex-between" style={{ marginTop: 4 }}>
        <span className="text-xs muted">Equity (live)</span>
        <span className={`mono text-xs ${isProfit ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
          {lastValue.toLocaleString('fr-FR', { maximumFractionDigits: 0 })}$
          ({pnl >= 0 ? '+' : ''}{pnlPct}%)
        </span>
      </div>
    </div>
  )
}

/** Courbe fallback depuis /api/simulator/equity (trades fermés uniquement). */
function FallbackEquityChart({ points, initialCapital, currentEquity }) {
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

  const displayValue = currentEquity ?? lastValue
  const pnl = displayValue - initialCapital
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
        <span className="text-xs muted">Equity</span>
        <span className={`mono text-xs ${isProfit ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
          {displayValue.toLocaleString('fr-FR', { maximumFractionDigits: 0 })}$
          ({pnl >= 0 ? '+' : ''}{pnlPct}%)
        </span>
      </div>
    </div>
  )
}
