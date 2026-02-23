import { useState, useRef, useEffect, useMemo, useCallback } from 'react'

const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

const PAD = { top: 20, right: 60, bottom: 30, left: 70 }

function formatDate(iso) {
  const d = new Date(iso)
  return `${d.getDate().toString().padStart(2, '0')}/${(d.getMonth() + 1).toString().padStart(2, '0')}`
}

function formatVal(v, pct) {
  if (pct) return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
  return `${v.toLocaleString('fr-FR', { maximumFractionDigits: 0 })} $`
}

function niceStep(range, targetTicks) {
  const rough = range / targetTicks
  const mag = Math.pow(10, Math.floor(Math.log10(rough)))
  const residual = rough / mag
  let nice
  if (residual <= 1.5) nice = 1
  else if (residual <= 3) nice = 2
  else if (residual <= 7) nice = 5
  else nice = 10
  return nice * mag
}

export default function EquityCurveSVG({ curves = [], height = 300 }) {
  const containerRef = useRef(null)
  const [width, setWidth] = useState(600)
  const [hoverIdx, setHoverIdx] = useState(null)

  // ResizeObserver
  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        setWidth(Math.max(400, e.contentRect.width))
      }
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  // Mode comparaison = normaliser en %
  const normalized = curves.length > 1

  // Transformer les points pour le rendu
  const processedCurves = useMemo(() => {
    return curves.map((curve, ci) => {
      const pts = curve.points || []
      if (!pts.length) return { ...curve, data: [], color: COLORS[ci % COLORS.length] }
      const cap = curve.initialCapital || pts[0].equity
      const data = pts.map(p => ({
        ...p,
        val: normalized ? ((p.equity / cap) - 1) * 100 : p.equity,
      }))
      return { ...curve, data, color: curve.color || COLORS[ci % COLORS.length], dashed: curve.dashed || false }
    }).filter(c => c.data.length > 0)
  }, [curves, normalized])

  // Bornes Y
  const { yMin, yMax, xLen } = useMemo(() => {
    let mn = Infinity, mx = -Infinity, maxLen = 0
    for (const c of processedCurves) {
      for (const p of c.data) {
        if (p.val < mn) mn = p.val
        if (p.val > mx) mx = p.val
      }
      if (c.data.length > maxLen) maxLen = c.data.length
    }
    if (mn === Infinity) return { yMin: 0, yMax: 100, xLen: 0 }
    const margin = (mx - mn) * 0.1 || 1
    return { yMin: mn - margin, yMax: mx + margin, xLen: maxLen }
  }, [processedCurves])

  const chartW = width - PAD.left - PAD.right
  const chartH = height - PAD.top - PAD.bottom

  const toX = useCallback((i, len) => PAD.left + (i / Math.max(len - 1, 1)) * chartW, [chartW])
  const toY = useCallback((v) => PAD.top + (1 - (v - yMin) / (yMax - yMin)) * chartH, [chartH, yMin, yMax])

  // Baseline (0% en mode compare, capital initial en mode solo)
  const baselineVal = normalized ? 0 : (processedCurves[0]?.data[0]?.val ?? 0)
  const baselineY = toY(baselineVal)

  // SVG paths
  const paths = useMemo(() => {
    return processedCurves.map(c => {
      const pts = c.data.map((p, i) => `${toX(i, c.data.length).toFixed(1)},${toY(p.val).toFixed(1)}`).join(' ')
      // Area fill: courbe + retour par la baseline
      const first = `${toX(0, c.data.length).toFixed(1)},${baselineY.toFixed(1)}`
      const last = `${toX(c.data.length - 1, c.data.length).toFixed(1)},${baselineY.toFixed(1)}`
      const area = `${first} ${pts} ${last}`
      return { line: pts, area, color: c.color, label: c.label, dashed: c.dashed }
    })
  }, [processedCurves, toX, toY, baselineY])

  // Y axis ticks
  const yTicks = useMemo(() => {
    const range = yMax - yMin
    if (range <= 0) return []
    const step = niceStep(range, 5)
    const ticks = []
    let tick = Math.ceil(yMin / step) * step
    while (tick <= yMax) {
      ticks.push(tick)
      tick += step
    }
    return ticks
  }, [yMin, yMax])

  // X axis labels (5-6 dates)
  const xLabels = useMemo(() => {
    const longest = processedCurves.reduce((best, c) => c.data.length > best.length ? c.data : best, [])
    if (!longest.length) return []
    const n = Math.min(6, longest.length)
    const step = Math.max(1, Math.floor((longest.length - 1) / (n - 1)))
    const labels = []
    for (let i = 0; i < longest.length; i += step) {
      labels.push({ i, label: formatDate(longest[i].timestamp), x: toX(i, longest.length) })
    }
    return labels
  }, [processedCurves, toX])

  // Hover
  const handleMouseMove = useCallback((e) => {
    if (!processedCurves.length) return
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    const x = e.clientX - rect.left - PAD.left
    const longest = processedCurves.reduce((best, c) => c.data.length > best.length ? c.data : best, [])
    const idx = Math.round((x / chartW) * (longest.length - 1))
    setHoverIdx(Math.max(0, Math.min(idx, longest.length - 1)))
  }, [processedCurves, chartW])

  const handleMouseLeave = useCallback(() => setHoverIdx(null), [])

  if (!processedCurves.length) {
    return (
      <div ref={containerRef} style={{ width: '100%', minHeight: height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>
        Aucune donnée à afficher
      </div>
    )
  }

  // Hover crosshair & tooltip
  const hoverData = hoverIdx !== null ? processedCurves.map(c => {
    const p = c.data[Math.min(hoverIdx, c.data.length - 1)]
    return { label: c.label, color: c.color, ...p }
  }) : null

  return (
    <div ref={containerRef} style={{ width: '100%', position: 'relative' }}>
      <svg
        width={width}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ display: 'block', cursor: hoverIdx !== null ? 'crosshair' : 'default' }}
      >
        {/* Grid lines */}
        {yTicks.map(t => (
          <line key={t} x1={PAD.left} x2={width - PAD.right} y1={toY(t)} y2={toY(t)} stroke="#222" strokeWidth={1} />
        ))}

        {/* Baseline */}
        <line
          x1={PAD.left} x2={width - PAD.right}
          y1={baselineY} y2={baselineY}
          stroke="#555" strokeWidth={1} strokeDasharray="4,4"
        />

        {/* Areas + Lines */}
        {paths.map((p, i) => (
          <g key={i}>
            {i === 0 && (
              <polygon points={p.area} fill={p.color} fillOpacity={0.08} />
            )}
            <polyline points={p.line} fill="none" stroke={p.color} strokeWidth={1.5} strokeDasharray={p.dashed ? '6,4' : undefined} />
          </g>
        ))}

        {/* Y axis labels */}
        {yTicks.map(t => (
          <text key={t} x={PAD.left - 8} y={toY(t) + 4} textAnchor="end" fill="#888" fontSize={11}>
            {formatVal(t, normalized)}
          </text>
        ))}

        {/* X axis labels */}
        {xLabels.map(l => (
          <text key={l.i} x={l.x} y={height - 6} textAnchor="middle" fill="#888" fontSize={11}>
            {l.label}
          </text>
        ))}

        {/* Hover crosshair */}
        {hoverIdx !== null && (
          <>
            <line
              x1={toX(hoverIdx, processedCurves[0].data.length)}
              x2={toX(hoverIdx, processedCurves[0].data.length)}
              y1={PAD.top} y2={PAD.top + chartH}
              stroke="#555" strokeWidth={1} strokeDasharray="3,3"
            />
            {processedCurves.map((c, ci) => {
              const idx = Math.min(hoverIdx, c.data.length - 1)
              return (
                <circle key={ci}
                  cx={toX(idx, c.data.length)}
                  cy={toY(c.data[idx].val)}
                  r={4} fill={c.color} stroke="#111" strokeWidth={2}
                />
              )
            })}
          </>
        )}

        {/* Legend (si multi-courbes) */}
        {curves.length > 1 && processedCurves.map((c, i) => (
          <g key={i} transform={`translate(${PAD.left + 10 + i * 120}, ${PAD.top - 6})`}>
            <rect width={10} height={10} fill={c.color} rx={2} />
            <text x={14} y={9} fill="#ccc" fontSize={11}>{c.label || `Run ${i + 1}`}</text>
          </g>
        ))}
      </svg>

      {/* Tooltip */}
      {hoverData && (
        <div style={{
          position: 'absolute',
          top: 8,
          right: 8,
          background: '#1a1a1a',
          border: '1px solid #444',
          borderRadius: 6,
          padding: '8px 12px',
          fontSize: 12,
          color: '#ccc',
          pointerEvents: 'none',
          zIndex: 10,
          minWidth: 160,
        }}>
          <div style={{ color: '#888', marginBottom: 4 }}>
            {hoverData[0]?.timestamp ? formatDate(hoverData[0].timestamp) : ''}
            {hoverData[0]?.timestamp && (
              <span style={{ marginLeft: 6, fontSize: 10 }}>
                {new Date(hoverData[0].timestamp).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
          </div>
          {hoverData.map((d, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: d.color, display: 'inline-block' }} />
              <span style={{ fontWeight: 600, color: '#fff' }}>{formatVal(d.val, normalized)}</span>
              {d.margin_ratio !== undefined && (
                <span style={{ color: '#888', fontSize: 11, marginLeft: 'auto' }}>
                  M:{(d.margin_ratio * 100).toFixed(0)}% P:{d.positions}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
