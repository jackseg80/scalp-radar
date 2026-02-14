/**
 * ScatterChart — Graphique SVG IS Sharpe vs OOS Sharpe
 * Sprint 14b Bloc F
 */

import { useMemo } from 'react'
import './ScatterChart.css'

export default function ScatterChart({ combos }) {
  const data = useMemo(() => {
    if (!combos || combos.length === 0) return null

    const points = combos
      .filter((c) => c.is_sharpe != null && c.oos_sharpe != null)
      .map((c) => ({
        is_sharpe: c.is_sharpe,
        oos_sharpe: c.oos_sharpe,
        consistency: c.consistency ?? 0,
        is_best: c.is_best === 1 || c.is_best === true,
        params: c.params,
      }))

    if (points.length === 0) return null

    const allIS = points.map((p) => p.is_sharpe)
    const allOOS = points.map((p) => p.oos_sharpe)

    const minIS = Math.min(...allIS)
    const maxIS = Math.max(...allIS)
    const minOOS = Math.min(...allOOS)
    const maxOOS = Math.max(...allOOS)

    const paddingX = Math.abs(maxIS - minIS) * 0.1 || 0.5
    const paddingY = Math.abs(maxOOS - minOOS) * 0.1 || 0.5

    return {
      points,
      domainX: { min: minIS - paddingX, max: maxIS + paddingX },
      domainY: { min: minOOS - paddingY, max: maxOOS + paddingY },
    }
  }, [combos])

  if (!data) {
    return (
      <div className="scatter-chart">
        <p style={{ textAlign: 'center', color: '#888', padding: '20px' }}>
          Aucune donnée disponible
        </p>
      </div>
    )
  }

  const { points, domainX, domainY } = data

  const marginLeft = 60
  const marginRight = 40
  const marginTop = 40
  const marginBottom = 60
  const chartWidth = 600 - marginLeft - marginRight
  const chartHeight = 400 - marginTop - marginBottom

  const scaleX = (val) => marginLeft + ((val - domainX.min) / (domainX.max - domainX.min)) * chartWidth
  const scaleY = (val) => marginTop + chartHeight - ((val - domainY.min) / (domainY.max - domainY.min)) * chartHeight

  // Diagonale IS = OOS
  const diagMin = Math.max(domainX.min, domainY.min)
  const diagMax = Math.min(domainX.max, domainY.max)

  // Couleur selon consistance
  const getConsistencyColor = (consistency) => {
    if (consistency < 0.5) return '#ef4444' // Rouge
    if (consistency < 0.8) return '#f59e0b' // Orange
    return '#10b981' // Vert
  }

  // Ticks axes
  const xTicks = 5
  const yTicks = 5
  const xTickValues = Array.from({ length: xTicks }, (_, i) => {
    return domainX.min + ((domainX.max - domainX.min) / (xTicks - 1)) * i
  })
  const yTickValues = Array.from({ length: yTicks }, (_, i) => {
    return domainY.min + ((domainY.max - domainY.min) / (yTicks - 1)) * i
  })

  return (
    <div className="scatter-chart">
      <h4>IS Sharpe vs OOS Sharpe</h4>
      <svg viewBox="0 0 600 400" width="100%" preserveAspectRatio="xMidYMid meet">
        {/* Grille */}
        <g className="grid">
          {xTickValues.map((val, i) => (
            <line
              key={`grid-x-${i}`}
              x1={scaleX(val)}
              y1={marginTop}
              x2={scaleX(val)}
              y2={marginTop + chartHeight}
              stroke="#333"
              strokeDasharray="2,2"
              opacity={0.3}
            />
          ))}
          {yTickValues.map((val, i) => (
            <line
              key={`grid-y-${i}`}
              x1={marginLeft}
              y1={scaleY(val)}
              x2={marginLeft + chartWidth}
              y2={scaleY(val)}
              stroke="#333"
              strokeDasharray="2,2"
              opacity={0.3}
            />
          ))}
        </g>

        {/* Axes */}
        <line
          x1={marginLeft}
          y1={marginTop}
          x2={marginLeft}
          y2={marginTop + chartHeight}
          stroke="#666"
          strokeWidth={1.5}
        />
        <line
          x1={marginLeft}
          y1={marginTop + chartHeight}
          x2={marginLeft + chartWidth}
          y2={marginTop + chartHeight}
          stroke="#666"
          strokeWidth={1.5}
        />

        {/* Ticks X */}
        {xTickValues.map((val, i) => (
          <g key={`tick-x-${i}`}>
            <line
              x1={scaleX(val)}
              y1={marginTop + chartHeight}
              x2={scaleX(val)}
              y2={marginTop + chartHeight + 5}
              stroke="#666"
              strokeWidth={1.5}
            />
            <text
              x={scaleX(val)}
              y={marginTop + chartHeight + 20}
              textAnchor="middle"
              fill="#ccc"
              fontSize={11}
            >
              {val.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Ticks Y */}
        {yTickValues.map((val, i) => (
          <g key={`tick-y-${i}`}>
            <line
              x1={marginLeft - 5}
              y1={scaleY(val)}
              x2={marginLeft}
              y2={scaleY(val)}
              stroke="#666"
              strokeWidth={1.5}
            />
            <text
              x={marginLeft - 10}
              y={scaleY(val)}
              textAnchor="end"
              alignmentBaseline="middle"
              fill="#ccc"
              fontSize={11}
            >
              {val.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Diagonale IS = OOS */}
        {diagMin < diagMax && (
          <line
            x1={scaleX(diagMin)}
            y1={scaleY(diagMin)}
            x2={scaleX(diagMax)}
            y2={scaleY(diagMax)}
            stroke="#888"
            strokeWidth={1}
            strokeDasharray="4,4"
            opacity={0.6}
          />
        )}

        {/* Points */}
        {points.map((p, i) => {
          const cx = scaleX(p.is_sharpe)
          const cy = scaleY(p.oos_sharpe)
          const color = getConsistencyColor(p.consistency)
          const radius = p.is_best ? 6 : 4

          const paramsStr = Object.entries(p.params)
            .map(([k, v]) => `${k}: ${v}`)
            .join(', ')

          return (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={radius}
              fill={color}
              stroke={p.is_best ? '#fff' : 'none'}
              strokeWidth={p.is_best ? 2 : 0}
              opacity={0.8}
              className="scatter-point"
            >
              <title>
                {`IS Sharpe: ${p.is_sharpe.toFixed(2)}\nOOS Sharpe: ${p.oos_sharpe.toFixed(2)}\nConsistance: ${(
                  p.consistency * 100
                ).toFixed(0)}%\n${paramsStr}`}
              </title>
            </circle>
          )
        })}

        {/* Label X */}
        <text
          x={marginLeft + chartWidth / 2}
          y={marginTop + chartHeight + 50}
          textAnchor="middle"
          fill="#ccc"
          fontSize={13}
          fontWeight="600"
        >
          IS Sharpe
        </text>

        {/* Label Y */}
        <text
          x={15}
          y={marginTop + chartHeight / 2}
          textAnchor="middle"
          fill="#ccc"
          fontSize={13}
          fontWeight="600"
          transform={`rotate(-90, 15, ${marginTop + chartHeight / 2})`}
        >
          OOS Sharpe
        </text>

        {/* Légende consistance */}
        <g transform={`translate(${marginLeft + chartWidth - 120}, ${marginTop})`}>
          <text x={0} y={0} fill="#aaa" fontSize={11} fontWeight="600">
            Consistance:
          </text>
          <circle cx={10} cy={18} r={4} fill="#ef4444" opacity={0.8} />
          <text x={18} y={18} alignmentBaseline="middle" fill="#ccc" fontSize={10}>
            &lt; 50%
          </text>
          <circle cx={10} cy={33} r={4} fill="#f59e0b" opacity={0.8} />
          <text x={18} y={33} alignmentBaseline="middle" fill="#ccc" fontSize={10}>
            50-80%
          </text>
          <circle cx={10} cy={48} r={4} fill="#10b981" opacity={0.8} />
          <text x={18} y={48} alignmentBaseline="middle" fill="#ccc" fontSize={10}>
            &gt; 80%
          </text>
        </g>
      </svg>
    </div>
  )
}
