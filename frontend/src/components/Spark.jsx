/**
 * Spark — Sparkline SVG minimaliste.
 * Props : data (array de nombres), w (largeur, default 110), h (hauteur, default 32)
 * Vert si dernier > premier, rouge sinon.
 */
export default function Spark({ data = [], w = 110, h = 32 }) {
  if (!data || data.length < 2) {
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

  // Construire les points du polyline
  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * w
    const y = padding + ((max - val) / range) * (h - padding * 2)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')

  const trending = data[data.length - 1] > data[0]
  const color = trending ? 'var(--accent)' : 'var(--red)'

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}
