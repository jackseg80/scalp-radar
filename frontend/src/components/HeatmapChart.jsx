/**
 * HeatmapChart — Grille 2D interactive des résultats WFO
 * Sprint 14 Bloc E
 */

import { useMemo } from 'react'

const GRADE_COLORS = {
  A: '#10b981',
  B: '#3b82f6',
  C: '#f59e0b',
  D: '#f97316',
  F: '#ef4444',
}

// Échelle couleur pour métrique numérique (rouge → jaune → vert)
function getColorForValue(value, min, max) {
  if (value == null) return '#374151' // Cellule vide = gris

  // Normaliser 0-1
  const t = max === min ? 0.5 : (value - min) / (max - min)

  // Rouge (0) → Jaune (0.5) → Vert (1)
  let r, g, b
  if (t < 0.5) {
    // Rouge → Jaune
    r = 255
    g = Math.round(255 * (t * 2))
    b = 0
  } else {
    // Jaune → Vert
    r = Math.round(255 * (1 - (t - 0.5) * 2))
    g = 255
    b = 0
  }

  return `rgb(${r}, ${g}, ${b})`
}

export default function HeatmapChart({ data }) {
  const { x_values, y_values, cells } = data

  // Calculer min/max des valeurs pour l'échelle
  const { minVal, maxVal } = useMemo(() => {
    let min = Infinity
    let max = -Infinity

    cells.forEach(row => {
      row.forEach(cell => {
        if (cell.value != null) {
          min = Math.min(min, cell.value)
          max = Math.max(max, cell.value)
        }
      })
    })

    return { minVal: min === Infinity ? 0 : min, maxVal: max === -Infinity ? 100 : max }
  }, [cells])

  // Dimensions
  const cellSize = 60
  const marginLeft = 80
  const marginTop = 60
  const marginBottom = 40
  const marginRight = 20

  const chartWidth = x_values.length * cellSize
  const chartHeight = y_values.length * cellSize
  const svgWidth = chartWidth + marginLeft + marginRight
  const svgHeight = chartHeight + marginTop + marginBottom

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg
        width={svgWidth}
        height={svgHeight}
        style={{ background: '#111827', fontFamily: 'monospace' }}
      >
        {/* Cellules */}
        {cells.map((row, rowIdx) => {
          const y = marginTop + rowIdx * cellSize

          return row.map((cell, colIdx) => {
            const x = marginLeft + colIdx * cellSize
            const color = cell.grade
              ? GRADE_COLORS[cell.grade]
              : getColorForValue(cell.value, minVal, maxVal)

            return (
              <g key={`${rowIdx}-${colIdx}`}>
                {/* Rectangle */}
                <rect
                  x={x}
                  y={y}
                  width={cellSize}
                  height={cellSize}
                  fill={color}
                  stroke="#1f2937"
                  strokeWidth={2}
                  style={{ cursor: cell.value != null ? 'pointer' : 'default' }}
                />

                {/* Texte valeur */}
                {cell.value != null && (
                  <>
                    <text
                      x={x + cellSize / 2}
                      y={y + cellSize / 2 - 6}
                      textAnchor="middle"
                      fill="#fff"
                      fontSize="12"
                      fontWeight="bold"
                    >
                      {cell.value.toFixed(1)}
                    </text>
                    {cell.grade && (
                      <text
                        x={x + cellSize / 2}
                        y={y + cellSize / 2 + 10}
                        textAnchor="middle"
                        fill="#fff"
                        fontSize="10"
                      >
                        Grade {cell.grade}
                      </text>
                    )}
                  </>
                )}
              </g>
            )
          })
        })}

        {/* Axes X (en haut) */}
        {x_values.map((xVal, idx) => {
          const x = marginLeft + idx * cellSize + cellSize / 2
          const y = marginTop - 10

          return (
            <text
              key={`x-${idx}`}
              x={x}
              y={y}
              textAnchor="middle"
              fill="#9ca3af"
              fontSize="12"
            >
              {xVal}
            </text>
          )
        })}

        {/* Label axe X */}
        <text
          x={marginLeft + chartWidth / 2}
          y={marginTop - 35}
          textAnchor="middle"
          fill="#e5e7eb"
          fontSize="14"
          fontWeight="bold"
        >
          {data.x_param}
        </text>

        {/* Axes Y (à gauche) */}
        {y_values.map((yVal, idx) => {
          const x = marginLeft - 10
          const y = marginTop + idx * cellSize + cellSize / 2

          return (
            <text
              key={`y-${idx}`}
              x={x}
              y={y}
              textAnchor="end"
              dominantBaseline="middle"
              fill="#9ca3af"
              fontSize="12"
            >
              {yVal}
            </text>
          )
        })}

        {/* Label axe Y */}
        <text
          x={20}
          y={marginTop + chartHeight / 2}
          textAnchor="middle"
          fill="#e5e7eb"
          fontSize="14"
          fontWeight="bold"
          transform={`rotate(-90 20 ${marginTop + chartHeight / 2})`}
        >
          {data.y_param}
        </text>

        {/* Légende échelle couleur (en bas) */}
        <g transform={`translate(${marginLeft}, ${svgHeight - 30})`}>
          <text x={0} y={0} fill="#9ca3af" fontSize="12">
            Min: {minVal.toFixed(1)}
          </text>
          {/* Gradient */}
          <defs>
            <linearGradient id="heatmap-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style={{ stopColor: 'rgb(255, 0, 0)', stopOpacity: 1 }} />
              <stop offset="50%" style={{ stopColor: 'rgb(255, 255, 0)', stopOpacity: 1 }} />
              <stop offset="100%" style={{ stopColor: 'rgb(0, 255, 0)', stopOpacity: 1 }} />
            </linearGradient>
          </defs>
          <rect
            x={80}
            y={-10}
            width={200}
            height={15}
            fill="url(#heatmap-gradient)"
            stroke="#4b5563"
            strokeWidth={1}
          />
          <text x={290} y={0} fill="#9ca3af" fontSize="12">
            Max: {maxVal.toFixed(1)}
          </text>
        </g>
      </svg>
    </div>
  )
}
