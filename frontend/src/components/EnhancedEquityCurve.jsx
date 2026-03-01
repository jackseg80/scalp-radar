/**
 * EnhancedEquityCurve — Recharts AreaChart avec gradient, tooltip, regime overlay, period selector.
 * Sprint 63a — remplace LiveEquityCurve (SVG basique).
 */
import { useState, useMemo } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  ReferenceLine, ReferenceArea, ResponsiveContainer, Tooltip,
} from 'recharts'
import { useApi } from '../hooks/useApi'
import './EnhancedEquityCurve.css'

const PERIODS = [
  { id: 7, label: '7j' },
  { id: 30, label: '30j' },
  { id: 90, label: '90j' },
  { id: null, label: 'Tout' },
]

const REGIME_COLORS = {
  BULL: 'rgba(0, 230, 138, 0.07)',
  RANGE: 'rgba(255, 197, 61, 0.05)',
  BEAR: 'rgba(255, 140, 66, 0.07)',
  CRASH: 'rgba(255, 68, 102, 0.09)',
}

function formatDateShort(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return `${d.getDate().toString().padStart(2, '0')}/${(d.getMonth() + 1).toString().padStart(2, '0')}`
}

function formatDateFull(ts) {
  if (!ts) return ''
  return new Date(ts).toLocaleDateString('fr-FR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="enhanced-equity-tooltip">
      <div className="tooltip-date">{formatDateFull(p.timestamp)}</div>
      <div className="tooltip-row">
        <span className="tooltip-label">Equity</span>
        <span className="tooltip-value">{p.equity?.toFixed(2)}$</span>
      </div>
      {p.dailyPnl != null && (
        <div className="tooltip-row">
          <span className="tooltip-label">P&L jour</span>
          <span className={`tooltip-value ${p.dailyPnl >= 0 ? 'positive' : 'negative'}`}>
            {p.dailyPnl >= 0 ? '+' : ''}{p.dailyPnl.toFixed(2)}$
          </span>
        </div>
      )}
      {p.margin_ratio != null && (
        <div className="tooltip-row">
          <span className="tooltip-label">Marge</span>
          <span className="tooltip-value">{(p.margin_ratio * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  )
}

/** Grouper les entries regime consecutives en bandes {regime, startTs, endTs} */
function buildRegimeBands(history) {
  if (!history?.length) return []
  const bands = []
  let current = { regime: history[0].regime, startTs: history[0].timestamp }
  for (let i = 1; i < history.length; i++) {
    if (history[i].regime !== current.regime) {
      bands.push({ ...current, endTs: history[i].timestamp })
      current = { regime: history[i].regime, startTs: history[i].timestamp }
    }
  }
  bands.push({ ...current, endTs: history[history.length - 1].timestamp })
  return bands
}

/** Trouver le timestamp equity le plus proche d'un timestamp cible */
function findClosestTimestamp(equityData, targetTs) {
  if (!equityData.length) return null
  const target = new Date(targetTs).getTime()
  let best = equityData[0].timestamp
  let bestDiff = Math.abs(new Date(best).getTime() - target)
  for (const p of equityData) {
    const diff = Math.abs(new Date(p.timestamp).getTime() - target)
    if (diff < bestDiff) {
      bestDiff = diff
      best = p.timestamp
    }
  }
  return best
}

export default function EnhancedEquityCurve({
  strategy = null,
  defaultDays = 30,
  height = 250,
  showRegimes = true,
  showReference = true,
}) {
  const [days, setDays] = useState(defaultDays)

  const daysParam = days || 365
  const stratQ = strategy ? `&strategy=${encodeURIComponent(strategy)}` : ''
  const { data: eqData, loading } = useApi(
    `/api/journal/live-equity?days=${daysParam}${stratQ}`, 60000,
  )
  const { data: regimeData } = useApi(
    showRegimes ? `/api/regime/history?days=${daysParam}` : null, 60000,
  )

  const points = eqData?.equity_curve || []

  // Enrichir avec dailyPnl
  const chartData = useMemo(() => {
    return points.map((p, i) => ({
      ...p,
      dailyPnl: i > 0 ? p.equity - points[i - 1].equity : 0,
    }))
  }, [points])

  const initialCapital = chartData.length > 0 ? chartData[0].equity : null

  const isPositive = chartData.length >= 2
    ? chartData[chartData.length - 1].equity >= chartData[0].equity
    : true

  // Bandes de regime mappees sur les timestamps equity
  const regimeBands = useMemo(() => {
    if (!showRegimes || !regimeData?.history?.length || chartData.length < 2) return []
    const bands = buildRegimeBands(regimeData.history)
    return bands
      .map(b => ({
        regime: b.regime,
        x1: findClosestTimestamp(chartData, b.startTs),
        x2: findClosestTimestamp(chartData, b.endTs),
      }))
      .filter(b => b.x1 && b.x2 && REGIME_COLORS[b.regime])
  }, [showRegimes, regimeData, chartData])

  if (loading && !eqData) return <div className="empty-state">Chargement...</div>
  if (chartData.length < 2) return <div className="empty-state">Pas assez de snapshots (min 2h)</div>

  const lineColor = isPositive ? 'var(--accent)' : 'var(--red)'
  const gradientId = `eqGrad-${strategy || 'all'}`

  return (
    <div className="enhanced-equity-container">
      <div className="enhanced-equity-periods">
        {PERIODS.map(p => (
          <button
            key={p.id ?? 'all'}
            className={`period-btn ${days === p.id ? 'active' : ''}`}
            onClick={() => setDays(p.id)}
          >
            {p.label}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor={isPositive ? '#00e68a' : '#ff4466'}
                stopOpacity={0.25}
              />
              <stop
                offset="95%"
                stopColor={isPositive ? '#00e68a' : '#ff4466'}
                stopOpacity={0.02}
              />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatDateShort}
            tick={{ fill: '#888', fontSize: 10 }}
            axisLine={{ stroke: 'rgba(255,255,255,0.05)' }}
            tickLine={false}
            minTickGap={40}
          />
          <YAxis
            tick={{ fill: '#888', fontSize: 10 }}
            tickFormatter={v => `${v.toFixed(0)}$`}
            axisLine={false}
            tickLine={false}
            domain={['auto', 'auto']}
            width={55}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Bandes de regime */}
          {regimeBands.map((b, i) => (
            <ReferenceArea
              key={i}
              x1={b.x1}
              x2={b.x2}
              fill={REGIME_COLORS[b.regime]}
              fillOpacity={1}
              ifOverflow="extendDomain"
            />
          ))}

          {/* Ligne de reference capital initial */}
          {showReference && initialCapital != null && (
            <ReferenceLine
              y={initialCapital}
              stroke="rgba(255,255,255,0.15)"
              strokeDasharray="4 4"
            />
          )}

          <Area
            type="monotone"
            dataKey="equity"
            stroke={lineColor}
            fill={`url(#${gradientId})`}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
