/**
 * PaperLiveOverlay — Superpose les equity curves Paper vs Live (Sprint 63b).
 * Normalise les deux courbes en % de return depuis le début.
 */
import { useState, useMemo } from 'react'
import { useApi } from '../hooks/useApi'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const PERIODS = [
  { days: 7, label: '7j' },
  { days: 30, label: '30j' },
]

export default function PaperLiveOverlay({ strategy }) {
  const [days, setDays] = useState(30)

  const stratQ = strategy ? `&strategy=${encodeURIComponent(strategy)}` : ''
  const sinceISO = new Date(Date.now() - days * 86400000).toISOString()

  const { data: liveData, loading: liveLoading } = useApi(
    `/api/journal/live-equity?days=${days}${stratQ}`, 60000
  )
  const { data: paperData, loading: paperLoading } = useApi(
    `/api/simulator/equity?since=${encodeURIComponent(sinceISO)}${stratQ}`, 60000
  )

  const merged = useMemo(() => {
    const livePoints = liveData?.equity_curve || []
    const paperPoints = paperData?.equity || []
    const liveBaseline = livePoints.length > 0 ? livePoints[0].equity : null
    const paperBaseline = paperData?.initial_capital || (paperPoints.length > 0 ? paperPoints[0].capital : null)

    if (!liveBaseline && !paperBaseline) return []

    const map = new Map()

    for (const p of livePoints) {
      const ts = new Date(p.timestamp).getTime()
      if (liveBaseline) {
        map.set(ts, { timestamp: ts, live: ((p.equity - liveBaseline) / liveBaseline) * 100 })
      }
    }

    for (const p of paperPoints) {
      const ts = new Date(p.timestamp).getTime()
      const existing = map.get(ts) || { timestamp: ts }
      if (paperBaseline) {
        existing.paper = ((p.capital - paperBaseline) / paperBaseline) * 100
      }
      map.set(ts, existing)
    }

    return [...map.values()].sort((a, b) => a.timestamp - b.timestamp)
  }, [liveData, paperData])

  const loading = liveLoading || paperLoading
  const hasLive = (liveData?.equity_curve || []).length > 0
  const hasPaper = (paperData?.equity || []).length > 0

  if (loading) return <p className="text-xs muted">Chargement...</p>
  if (!hasLive && !hasPaper) return <p className="text-xs muted">Aucune donnée disponible</p>

  const fmtTick = (ts) => new Date(ts).toLocaleDateString('fr-FR', { day: '2-digit', month: 'short' })
  const fmtLabel = (ts) => new Date(ts).toLocaleString('fr-FR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
  const fmtVal = (v) => v != null ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` : '—'

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <div className="period-selector" style={{ marginBottom: 0 }}>
          {PERIODS.map(p => (
            <button key={p.days} className={days === p.days ? 'active' : ''} onClick={() => setDays(p.days)}>
              {p.label}
            </button>
          ))}
        </div>
        <span className="text-xs muted" style={{ marginLeft: 'auto' }}>
          Live = equity (incl. P&L non réalisés) · Paper = capital réalisé
        </span>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={merged} margin={{ top: 5, right: 10, bottom: 5, left: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border, #333)" />
          <XAxis
            dataKey="timestamp"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={fmtTick}
            stroke="var(--text-muted, #888)"
            fontSize={10}
          />
          <YAxis
            tickFormatter={v => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
            stroke="var(--text-muted, #888)"
            fontSize={10}
          />
          <Tooltip
            labelFormatter={fmtLabel}
            formatter={(val, name) => [fmtVal(val), name === 'live' ? 'Live' : 'Paper']}
            contentStyle={{ background: 'var(--bg-card, #1a1a2e)', border: '1px solid var(--border, #333)', fontSize: 12 }}
          />
          <ReferenceLine y={0} stroke="var(--border, #555)" strokeDasharray="4 4" />
          {hasLive && (
            <Line
              type="monotone" dataKey="live" stroke="var(--accent, #00d4aa)"
              strokeWidth={2} dot={false} connectNulls name="Live"
            />
          )}
          {hasPaper && (
            <Line
              type="monotone" dataKey="paper" stroke="#f0ad4e"
              strokeWidth={2} dot={false} connectNulls name="Paper"
              strokeDasharray="6 3"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
