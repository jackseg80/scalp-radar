import { useRef, useEffect, useState, useMemo, useCallback } from 'react'

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#06b6d4', '#10b981']
const PAD = { top: 8, right: 60, bottom: 24, left: 70 }

function formatDate(iso) {
  const d = new Date(iso)
  return `${d.getDate().toString().padStart(2, '0')}/${(d.getMonth() + 1).toString().padStart(2, '0')}`
}

function computeDrawdown(points) {
  let peak = -Infinity
  return points.map(p => {
    if (p.equity > peak) peak = p.equity
    const dd = peak > 0 ? ((p.equity / peak) - 1) * 100 : 0
    return { ...p, dd }
  })
}

export default function DrawdownChart({ curves = [], height = 120, killSwitchPct = 30 }) {
  const containerRef = useRef(null)
  const [width, setWidth] = useState(600)

  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setWidth(Math.max(400, e.contentRect.width))
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  const ddCurves = useMemo(() => {
    return curves.map((c, ci) => ({
      ...c,
      data: computeDrawdown(c.points || []),
      color: c.color || COLORS[ci % COLORS.length],
    })).filter(c => c.data.length > 0)
  }, [curves])

  const { yMin } = useMemo(() => {
    let mn = 0
    for (const c of ddCurves)
      for (const p of c.data)
        if (p.dd < mn) mn = p.dd
    const absMin = Math.min(mn, -killSwitchPct)
    return { yMin: absMin * 1.1 }
  }, [ddCurves, killSwitchPct])

  const chartW = width - PAD.left - PAD.right
  const chartH = height - PAD.top - PAD.bottom

  const toX = useCallback((i, len) => PAD.left + (i / Math.max(len - 1, 1)) * chartW, [chartW])
  const toY = useCallback((v) => PAD.top + (-v / -yMin) * chartH, [chartH, yMin])

  if (!ddCurves.length) return null

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg width={width} height={height} style={{ display: 'block' }}>
        {/* Zero line */}
        <line x1={PAD.left} x2={width - PAD.right} y1={toY(0)} y2={toY(0)} stroke="#333" strokeWidth={1} />

        {/* Kill switch threshold */}
        <line
          x1={PAD.left} x2={width - PAD.right}
          y1={toY(-killSwitchPct)} y2={toY(-killSwitchPct)}
          stroke="#ef4444" strokeWidth={1} strokeDasharray="4,4" strokeOpacity={0.5}
        />
        <text x={width - PAD.right + 4} y={toY(-killSwitchPct) + 4} fill="#ef4444" fontSize={10} fillOpacity={0.7}>
          -{killSwitchPct}%
        </text>

        {/* Drawdown areas */}
        {ddCurves.map((c, ci) => {
          const points = c.data.map((p, i) => `${toX(i, c.data.length).toFixed(1)},${toY(p.dd).toFixed(1)}`).join(' ')
          const first = `${toX(0, c.data.length).toFixed(1)},${toY(0).toFixed(1)}`
          const last = `${toX(c.data.length - 1, c.data.length).toFixed(1)},${toY(0).toFixed(1)}`
          return (
            <g key={ci}>
              <polygon points={`${first} ${points} ${last}`} fill={c.color} fillOpacity={0.15} />
              <polyline points={points} fill="none" stroke={c.color} strokeWidth={1.2} />
            </g>
          )
        })}

        {/* Y axis labels */}
        <text x={PAD.left - 8} y={toY(0) + 4} textAnchor="end" fill="#888" fontSize={10}>0%</text>
        <text x={PAD.left - 8} y={toY(yMin / 2) + 4} textAnchor="end" fill="#888" fontSize={10}>
          {(yMin / 2).toFixed(0)}%
        </text>
        <text x={PAD.left - 8} y={toY(yMin) + 4} textAnchor="end" fill="#888" fontSize={10}>
          {yMin.toFixed(0)}%
        </text>

        {/* X axis labels */}
        {ddCurves[0] && (() => {
          const d = ddCurves[0].data
          const n = Math.min(6, d.length)
          const step = Math.max(1, Math.floor((d.length - 1) / (n - 1)))
          const labels = []
          for (let i = 0; i < d.length; i += step) {
            labels.push(
              <text key={i} x={toX(i, d.length)} y={height - 4} textAnchor="middle" fill="#888" fontSize={10}>
                {formatDate(d[i].timestamp)}
              </text>
            )
          }
          return labels
        })()}
      </svg>
    </div>
  )
}
