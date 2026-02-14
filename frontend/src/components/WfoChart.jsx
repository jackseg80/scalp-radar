/**
 * WfoChart — Chart SVG pour equity curve IS vs OOS par fenêtre WFO
 * Sprint 13
 */

import { useState, useMemo } from 'react'
import './WfoChart.css'

export default function WfoChart({ windows }) {
  const [metric, setMetric] = useState('sharpe') // 'sharpe' ou 'return'

  const data = useMemo(() => {
    if (!windows || windows.length === 0) return null

    const values = windows.map(w => ({
      index: w.window_index,
      is: metric === 'sharpe' ? w.is_sharpe : w.is_net_return_pct,
      oos: metric === 'sharpe' ? w.oos_sharpe : w.oos_net_return_pct,
    }))

    // Calcul du domaine
    const allValues = values.flatMap(v => [v.is, v.oos]).filter(v => v != null && !isNaN(v))
    if (allValues.length === 0) return null

    const minY = Math.min(...allValues)
    const maxY = Math.max(...allValues)
    const padding = Math.abs(maxY - minY) * 0.1
    const domain = {
      minX: 0,
      maxX: values.length - 1,
      minY: minY - padding,
      maxY: maxY + padding,
    }

    return { values, domain }
  }, [windows, metric])

  if (!data) {
    return (
      <div className="wfo-chart">
        <p style={{ textAlign: 'center', color: '#888' }}>Aucune donnée WFO disponible</p>
      </div>
    )
  }

  const { values, domain } = data
  const width = 800
  const height = 300
  const marginLeft = 60
  const marginRight = 40
  const marginTop = 20
  const marginBottom = 40
  const chartWidth = width - marginLeft - marginRight
  const chartHeight = height - marginTop - marginBottom

  const scaleX = (index) => marginLeft + (index / domain.maxX) * chartWidth
  const scaleY = (val) => {
    if (val == null || isNaN(val)) return null
    return marginTop + chartHeight - ((val - domain.minY) / (domain.maxY - domain.minY)) * chartHeight
  }

  // Lignes IS et OOS
  const lineIS = values.map((v, i) => {
    const y = scaleY(v.is)
    if (y == null) return null
    return `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${y}`
  }).filter(Boolean).join(' ')

  const lineOOS = values.map((v, i) => {
    const y = scaleY(v.oos)
    if (y == null) return null
    return `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${y}`
  }).filter(Boolean).join(' ')

  // Axes
  const yTicks = 5
  const yTickValues = Array.from({ length: yTicks }, (_, i) => {
    return domain.minY + ((domain.maxY - domain.minY) / (yTicks - 1)) * i
  })

  return (
    <div className="wfo-chart">
      <div className="wfo-chart__controls">
        <button
          onClick={() => setMetric('sharpe')}
          className={metric === 'sharpe' ? 'active' : ''}
        >
          Sharpe Ratio
        </button>
        <button
          onClick={() => setMetric('return')}
          className={metric === 'return' ? 'active' : ''}
        >
          Net Return %
        </button>
      </div>

      <svg width={width} height={height}>
        {/* Grille */}
        <g className="grid">
          {yTickValues.map((val, i) => (
            <line
              key={i}
              x1={marginLeft}
              y1={scaleY(val)}
              x2={width - marginRight}
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
          y2={height - marginBottom}
          stroke="#666"
          strokeWidth={1.5}
        />
        <line
          x1={marginLeft}
          y1={height - marginBottom}
          x2={width - marginRight}
          y2={height - marginBottom}
          stroke="#666"
          strokeWidth={1.5}
        />

        {/* Ticks Y */}
        {yTickValues.map((val, i) => (
          <g key={i}>
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
              {val.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Lignes IS et OOS */}
        {lineIS && (
          <path
            d={lineIS}
            fill="none"
            stroke="#3b82f6"
            strokeWidth={2}
          />
        )}
        {lineOOS && (
          <path
            d={lineOOS}
            fill="none"
            stroke="#f97316"
            strokeWidth={2}
          />
        )}

        {/* Points */}
        {values.map((v, i) => {
          const yIS = scaleY(v.is)
          const yOOS = scaleY(v.oos)
          return (
            <g key={i}>
              {yIS != null && (
                <circle
                  cx={scaleX(i)}
                  cy={yIS}
                  r={4}
                  fill="#3b82f6"
                  className="wfo-chart__point"
                >
                  <title>Window {v.index} IS: {v.is?.toFixed(2)}</title>
                </circle>
              )}
              {yOOS != null && (
                <circle
                  cx={scaleX(i)}
                  cy={yOOS}
                  r={4}
                  fill="#f97316"
                  className="wfo-chart__point"
                >
                  <title>Window {v.index} OOS: {v.oos?.toFixed(2)}</title>
                </circle>
              )}
            </g>
          )
        })}

        {/* Légende */}
        <g transform={`translate(${width - marginRight - 120}, ${marginTop + 10})`}>
          <line x1={0} y1={0} x2={30} y2={0} stroke="#3b82f6" strokeWidth={2} />
          <text x={35} y={0} alignmentBaseline="middle" fill="#ccc" fontSize={12}>
            IS (In-Sample)
          </text>
          <line x1={0} y1={20} x2={30} y2={20} stroke="#f97316" strokeWidth={2} />
          <text x={35} y={20} alignmentBaseline="middle" fill="#ccc" fontSize={12}>
            OOS (Out-of-Sample)
          </text>
        </g>

        {/* Label X */}
        <text
          x={marginLeft + chartWidth / 2}
          y={height - 5}
          textAnchor="middle"
          fill="#ccc"
          fontSize={12}
        >
          Fenêtre WFO
        </text>

        {/* Label Y */}
        <text
          x={15}
          y={marginTop + chartHeight / 2}
          textAnchor="middle"
          fill="#ccc"
          fontSize={12}
          transform={`rotate(-90, 15, ${marginTop + chartHeight / 2})`}
        >
          {metric === 'sharpe' ? 'Sharpe Ratio' : 'Net Return %'}
        </text>
      </svg>
    </div>
  )
}
