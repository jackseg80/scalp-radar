/**
 * Spark — Sparkline SVG avec gradient fill et dot animé.
 * Props : data (array de nombres), w (largeur, default 110), h (hauteur, default 32)
 * Vert si dernier > premier, rouge sinon.
 */
import { useMemo } from 'react'

export default function Spark({ data = [], w = 110, h = 32, stroke = 1.5 }) {
  const id = useMemo(() => `sg${Math.random().toString(36).slice(2, 8)}`, [])

  if (!data || data.length < 3) {
    return (
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
        <line
          x1={0} y1={h / 2} x2={w} y2={h / 2}
          stroke="var(--border)" strokeWidth={1.5}
        />
      </svg>
    )
  }

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const padding = 2

  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * w
    const y = padding + ((max - val) / range) * (h - padding * 2)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  const trending = data[data.length - 1] > data[0]
  const color = trending ? 'var(--accent)' : 'var(--red)'

  // Position du dernier point pour le dot animé
  const lastPt = points.split(' ').pop().split(',')
  const lastX = parseFloat(lastPt[0])
  const lastY = parseFloat(lastPt[1])

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block', overflow: 'visible' }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon
        fill={`url(#${id})`}
        points={`0,${h} ${points} ${w},${h}`}
      />
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx={lastX} cy={lastY} r="2.5" fill={color}>
        <animate attributeName="opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite" />
      </circle>
    </svg>
  )
}
