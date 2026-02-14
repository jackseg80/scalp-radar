/**
 * HeatmapChart — Grille 2D interactive des résultats WFO (cellules grandes, responsive)
 * Sprint 14 Bloc E
 */

import { useMemo, useState, useRef, useEffect } from 'react'

const GRADE_COLORS = {
  A: '#10b981',
  B: '#3b82f6',
  C: '#f59e0b',
  D: '#f97316',
  F: '#ef4444',
}

// Échelle couleur pour métrique numérique (rouge → jaune → vert)
function getColorForValue(value, min, max) {
  if (value == null) return '#2a2a2a' // Cellule vide = gris foncé

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
  const { x_values, y_values, cells, x_param, y_param, metric } = data
  const containerRef = useRef(null)
  const [containerWidth, setContainerWidth] = useState(800)

  // Observer le conteneur pour adapter la taille
  useEffect(() => {
    if (!containerRef.current) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width)
      }
    })

    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  // Calculer min/max des valeurs pour l'échelle
  const { minVal, maxVal } = useMemo(() => {
    let min = Infinity
    let max = -Infinity

    cells.forEach((row) => {
      row.forEach((cell) => {
        if (cell.value != null) {
          min = Math.min(min, cell.value)
          max = Math.max(max, cell.value)
        }
      })
    })

    return { minVal: min === Infinity ? 0 : min, maxVal: max === -Infinity ? 100 : max }
  }, [cells])

  // Calculer la taille des cellules en fonction de l'espace disponible
  const cellSize = useMemo(() => {
    const marginLeft = 100
    const marginRight = 40
    const availableWidth = containerWidth - marginLeft - marginRight
    const calculatedSize = Math.floor(availableWidth / x_values.length)

    // Min 60px, max 300px (pour remplir l'espace)
    return Math.max(60, Math.min(300, calculatedSize))
  }, [containerWidth, x_values.length])

  // Dimensions
  const marginLeft = 100
  const marginTop = 80
  const marginBottom = 80
  const marginRight = 40

  const chartWidth = x_values.length * cellSize
  const chartHeight = y_values.length * cellSize
  const svgWidth = chartWidth + marginLeft + marginRight
  const svgHeight = chartHeight + marginTop + marginBottom

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', overflowX: 'auto', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <svg
        width={svgWidth}
        height={svgHeight}
        style={{ background: '#0d1117', fontFamily: 'monospace', minWidth: '400px' }}
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
                  stroke="#1a1f2e"
                  strokeWidth={2}
                  style={{ cursor: cell.value != null ? 'pointer' : 'default' }}
                >
                  {cell.value != null && (
                    <title>
                      {x_param}: {x_values[colIdx]}
                      {'\n'}
                      {y_param}: {y_values[rowIdx]}
                      {'\n'}
                      {metric}: {cell.value.toFixed(2)}
                      {cell.grade && `\nGrade: ${cell.grade}`}
                      {cell.result_id && `\nID: ${cell.result_id}`}
                    </title>
                  )}
                </rect>

                {/* Texte valeur (Sprint 14b : mode dense, pas de grade) */}
                {cell.value != null && (
                  <text
                    x={x + cellSize / 2}
                    y={y + cellSize / 2}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="#fff"
                    fontSize={cellSize > 80 ? '14' : '12'}
                    fontWeight="bold"
                  >
                    {cell.value.toFixed(1)}
                  </text>
                )}
              </g>
            )
          })
        })}

        {/* Axes X (en haut) */}
        {x_values.map((xVal, idx) => {
          const x = marginLeft + idx * cellSize + cellSize / 2
          const y = marginTop - 15

          return (
            <text
              key={`x-${idx}`}
              x={x}
              y={y}
              textAnchor="middle"
              fill="#9ca3af"
              fontSize="13"
              fontWeight="500"
            >
              {xVal}
            </text>
          )
        })}

        {/* Label axe X */}
        <text
          x={marginLeft + chartWidth / 2}
          y={marginTop - 50}
          textAnchor="middle"
          fill="#e5e7eb"
          fontSize="16"
          fontWeight="bold"
        >
          {x_param}
        </text>

        {/* Axes Y (à gauche) */}
        {y_values.map((yVal, idx) => {
          const x = marginLeft - 15
          const y = marginTop + idx * cellSize + cellSize / 2

          return (
            <text
              key={`y-${idx}`}
              x={x}
              y={y}
              textAnchor="end"
              dominantBaseline="middle"
              fill="#9ca3af"
              fontSize="13"
              fontWeight="500"
            >
              {yVal}
            </text>
          )
        })}

        {/* Label axe Y */}
        <text
          x={25}
          y={marginTop + chartHeight / 2}
          textAnchor="middle"
          fill="#e5e7eb"
          fontSize="16"
          fontWeight="bold"
          transform={`rotate(-90 25 ${marginTop + chartHeight / 2})`}
        >
          {y_param}
        </text>

        {/* Légende échelle couleur (en bas, horizontale) */}
        <g transform={`translate(${marginLeft}, ${svgHeight - marginBottom + 20})`}>
          <text x={0} y={0} fill="#9ca3af" fontSize="13" fontWeight="500">
            Min: {minVal.toFixed(1)}
          </text>

          {/* Gradient horizontal */}
          <defs>
            <linearGradient id="heatmap-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style={{ stopColor: 'rgb(255, 0, 0)', stopOpacity: 1 }} />
              <stop offset="50%" style={{ stopColor: 'rgb(255, 255, 0)', stopOpacity: 1 }} />
              <stop offset="100%" style={{ stopColor: 'rgb(0, 255, 0)', stopOpacity: 1 }} />
            </linearGradient>
          </defs>

          <rect
            x={100}
            y={-12}
            width={Math.max(0, chartWidth - 200)}
            height={20}
            fill="url(#heatmap-gradient)"
            stroke="#4b5563"
            strokeWidth={1}
            rx={4}
          />

          <text
            x={chartWidth - 100}
            y={0}
            fill="#9ca3af"
            fontSize="13"
            fontWeight="500"
            textAnchor="end"
          >
            Max: {maxVal.toFixed(1)}
          </text>
        </g>

        {/* Légende métrique */}
        <text
          x={marginLeft + chartWidth / 2}
          y={svgHeight - marginBottom + 50}
          textAnchor="middle"
          fill="#aaa"
          fontSize="14"
          fontWeight="500"
        >
          Métrique: {metric}
        </text>
      </svg>
    </div>
  )
}
