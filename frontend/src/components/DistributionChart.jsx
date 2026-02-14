/**
 * DistributionChart — Histogramme SVG de distribution des OOS Sharpe
 * Sprint 14b Bloc F
 */

import { useMemo } from 'react'
import './DistributionChart.css'

export default function DistributionChart({ combos }) {
  const data = useMemo(() => {
    if (!combos || combos.length === 0) return null

    const values = combos
      .map((c) => c.oos_sharpe)
      .filter((v) => v != null && !isNaN(v))

    if (values.length === 0) return null

    const minVal = Math.min(...values)
    const maxVal = Math.max(...values)

    // Nombre de bins : sqrt(n)
    const nBins = Math.max(5, Math.ceil(Math.sqrt(values.length)))

    // Largeur d'un bin
    const binWidth = (maxVal - minVal) / nBins
    if (binWidth === 0) {
      // Tous les combos ont le même OOS Sharpe
      return {
        bins: [{ start: minVal - 0.5, end: minVal + 0.5, count: values.length, negative: minVal < 0 }],
        maxCount: values.length,
        bestOOSSharpe: null,
      }
    }

    // Créer les bins
    const bins = []
    for (let i = 0; i < nBins; i++) {
      const start = minVal + i * binWidth
      const end = start + binWidth
      bins.push({
        start,
        end,
        count: 0,
        negative: end <= 0,
      })
    }

    // Répartir les valeurs dans les bins
    values.forEach((v) => {
      const binIdx = Math.min(Math.floor((v - minVal) / binWidth), nBins - 1)
      bins[binIdx].count++
    })

    const maxCount = Math.max(...bins.map((b) => b.count))

    // Trouver le best combo OOS Sharpe
    const bestCombo = combos.find((c) => c.is_best === 1 || c.is_best === true)
    const bestOOSSharpe = bestCombo?.oos_sharpe ?? null

    return { bins, maxCount, bestOOSSharpe }
  }, [combos])

  if (!data) {
    return (
      <div className="distribution-chart">
        <p style={{ textAlign: 'center', color: '#888', padding: '20px' }}>
          Aucune donnée disponible
        </p>
      </div>
    )
  }

  const { bins, maxCount, bestOOSSharpe } = data

  const marginLeft = 60
  const marginRight = 40
  const marginTop = 40
  const marginBottom = 60
  const chartWidth = 600 - marginLeft - marginRight
  const chartHeight = 300 - marginTop - marginBottom

  const minX = bins[0].start
  const maxX = bins[bins.length - 1].end

  const scaleX = (val) => marginLeft + ((val - minX) / (maxX - minX)) * chartWidth
  const scaleY = (count) => marginTop + chartHeight - (count / maxCount) * chartHeight

  // Barres
  const barWidth = chartWidth / bins.length

  // Ligne verticale à OOS Sharpe = 0
  const zeroX = minX <= 0 && maxX >= 0 ? scaleX(0) : null

  // Marqueur best combo
  const bestX = bestOOSSharpe != null ? scaleX(bestOOSSharpe) : null

  // Ticks Y
  const yTicks = 5
  const yTickValues = Array.from({ length: yTicks }, (_, i) => {
    return Math.round((maxCount / (yTicks - 1)) * i)
  })

  return (
    <div className="distribution-chart">
      <h4>Distribution OOS Sharpe</h4>
      <svg viewBox="0 0 600 300" width="100%" preserveAspectRatio="xMidYMid meet">
        {/* Grille */}
        <g className="grid">
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
              {val}
            </text>
          </g>
        ))}

        {/* Barres */}
        {bins.map((bin, i) => {
          const x = marginLeft + (i / bins.length) * chartWidth
          const y = scaleY(bin.count)
          const height = marginTop + chartHeight - y
          const color = bin.negative ? '#ef4444' : '#10b981'

          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barWidth - 2}
              height={height}
              fill={color}
              opacity={0.7}
              stroke="#1a1f2e"
              strokeWidth={1}
            >
              <title>
                {`${bin.start.toFixed(2)} à ${bin.end.toFixed(2)}\nNombre: ${bin.count}`}
              </title>
            </rect>
          )
        })}

        {/* Ligne verticale à OOS Sharpe = 0 */}
        {zeroX !== null && (
          <line
            x1={zeroX}
            y1={marginTop}
            x2={zeroX}
            y2={marginTop + chartHeight}
            stroke="#fff"
            strokeWidth={1.5}
            strokeDasharray="4,4"
            opacity={0.8}
          />
        )}

        {/* Marqueur best combo (triangle en haut) */}
        {bestX !== null && (
          <g>
            <polygon
              points={`${bestX},${marginTop - 10} ${bestX - 6},${marginTop - 2} ${bestX + 6},${marginTop - 2}`}
              fill="#f59e0b"
              stroke="#fff"
              strokeWidth={1}
            >
              <title>{`Best combo: OOS Sharpe ${bestOOSSharpe.toFixed(2)}`}</title>
            </polygon>
          </g>
        )}

        {/* Label X */}
        <text
          x={marginLeft + chartWidth / 2}
          y={marginTop + chartHeight + 50}
          textAnchor="middle"
          fill="#ccc"
          fontSize={13}
          fontWeight="600"
        >
          OOS Sharpe
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
          Nombre de combos
        </text>

        {/* Ticks X (min, 0 si présent, max) */}
        <g>
          <text
            x={scaleX(minX)}
            y={marginTop + chartHeight + 20}
            textAnchor="middle"
            fill="#ccc"
            fontSize={11}
          >
            {minX.toFixed(1)}
          </text>
          {zeroX !== null && (
            <text
              x={zeroX}
              y={marginTop + chartHeight + 20}
              textAnchor="middle"
              fill="#fff"
              fontSize={11}
              fontWeight="600"
            >
              0
            </text>
          )}
          <text
            x={scaleX(maxX)}
            y={marginTop + chartHeight + 20}
            textAnchor="middle"
            fill="#ccc"
            fontSize={11}
          >
            {maxX.toFixed(1)}
          </text>
        </g>
      </svg>
    </div>
  )
}
