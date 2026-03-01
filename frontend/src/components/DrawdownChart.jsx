/**
 * DrawdownChart — Recharts AreaChart affichant le drawdown (%).
 * Sprint 63a — reecrit de SVG vers Recharts.
 *
 * Deux modes :
 * - Autonome : strategy + days fournis -> fetch ses propres donnees
 * - Legacy : curves fourni (PortfolioPage) -> pas de fetch
 */
import { useMemo } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  ReferenceLine, ResponsiveContainer, Tooltip,
} from 'recharts'
import { useApi } from '../hooks/useApi'
import './DrawdownChart.css'

const COLORS = ['#ff4466', '#f59e0b', '#3b82f6', '#8b5cf6', '#06b6d4', '#10b981']

function formatDateShort(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return `${d.getDate().toString().padStart(2, '0')}/${(d.getMonth() + 1).toString().padStart(2, '0')}`
}

function computeDrawdown(points) {
  let peak = -Infinity
  return points.map(p => {
    if (p.equity > peak) peak = p.equity
    const dd = peak > 0 ? ((p.equity / peak) - 1) * 100 : 0
    return { timestamp: p.timestamp, dd }
  })
}

function DDTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  const date = p.timestamp
    ? new Date(p.timestamp).toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric' })
    : ''
  // Afficher toutes les courbes si multi
  const values = payload.filter(e => e.value != null)
  return (
    <div className="dd-tooltip">
      <div className="dd-tooltip-date">{date}</div>
      {values.map((v, i) => (
        <div key={i} className="dd-tooltip-value" style={values.length > 1 ? { color: v.stroke } : undefined}>
          {v.value.toFixed(2)}%
        </div>
      ))}
    </div>
  )
}

export default function DrawdownChart({
  strategy = null,
  days = 30,
  curves = null,
  height = 120,
  killSwitchPct = 45,
}) {
  // Mode autonome : fetch equity si pas de curves
  const isAutonomous = !curves
  const stratQ = strategy ? `&strategy=${encodeURIComponent(strategy)}` : ''
  const { data: eqData } = useApi(
    isAutonomous ? `/api/journal/live-equity?days=${days}${stratQ}` : null, 60000,
  )

  // Normaliser en format unifie [{name, data: [{timestamp, dd}], color}]
  const ddCurves = useMemo(() => {
    let normalized
    if (curves) {
      // Mode legacy (PortfolioPage) : curves = [{name, points: [{equity, timestamp}], color}]
      normalized = curves.map((c, i) => ({
        name: c.name || `Curve ${i}`,
        data: computeDrawdown(c.points || []),
        color: c.color || COLORS[i % COLORS.length],
      }))
    } else {
      // Mode autonome
      const points = eqData?.equity_curve || []
      if (points.length < 2) return []
      normalized = [{
        name: 'Live',
        data: computeDrawdown(points),
        color: COLORS[0],
      }]
    }
    return normalized.filter(c => c.data.length > 0)
  }, [curves, eqData])

  // Fusionner toutes les courbes en un seul tableau pour Recharts
  const { chartData, curveKeys, yMin } = useMemo(() => {
    if (!ddCurves.length) return { chartData: [], curveKeys: [], yMin: 0 }

    // Si une seule courbe, structure simple
    if (ddCurves.length === 1) {
      const data = ddCurves[0].data.map(p => ({ timestamp: p.timestamp, dd_0: p.dd }))
      const mn = Math.min(...ddCurves[0].data.map(p => p.dd), -killSwitchPct)
      return { chartData: data, curveKeys: ['dd_0'], yMin: mn * 1.1 }
    }

    // Multi-courbes : merger par index (meme logique que l'ancien SVG)
    const maxLen = Math.max(...ddCurves.map(c => c.data.length))
    const data = []
    for (let i = 0; i < maxLen; i++) {
      const row = { timestamp: null }
      for (let ci = 0; ci < ddCurves.length; ci++) {
        const p = ddCurves[ci].data[i]
        if (p) {
          if (!row.timestamp) row.timestamp = p.timestamp
          row[`dd_${ci}`] = p.dd
        }
      }
      data.push(row)
    }
    let mn = 0
    for (const c of ddCurves) for (const p of c.data) if (p.dd < mn) mn = p.dd
    mn = Math.min(mn, -killSwitchPct)

    return {
      chartData: data,
      curveKeys: ddCurves.map((_, i) => `dd_${i}`),
      yMin: mn * 1.1,
    }
  }, [ddCurves, killSwitchPct])

  if (!chartData.length) return null

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 2, right: 10, bottom: 0, left: 10 }}>
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
          tickFormatter={v => `${v.toFixed(0)}%`}
          domain={[yMin, 0]}
          axisLine={false}
          tickLine={false}
          width={45}
        />
        <Tooltip content={<DDTooltip />} />

        {/* Kill switch */}
        <ReferenceLine
          y={-killSwitchPct}
          stroke="var(--red)"
          strokeDasharray="4 4"
          strokeOpacity={0.6}
        />

        {/* Zero line */}
        <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />

        {curveKeys.map((key, i) => (
          <Area
            key={key}
            type="monotone"
            dataKey={key}
            stroke={ddCurves[i]?.color || COLORS[i % COLORS.length]}
            fill={ddCurves[i]?.color || COLORS[i % COLORS.length]}
            fillOpacity={0.12}
            strokeWidth={1.2}
            dot={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  )
}
