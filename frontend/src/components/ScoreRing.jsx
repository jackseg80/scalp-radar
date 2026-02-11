/**
 * ScoreRing — Anneau SVG de score.
 * Props : score (0-1), size (default 72)
 * Couleur basée sur le score : vert >= 0.75, jaune >= 0.55, orange >= 0.35, rouge < 0.35
 */
export default function ScoreRing({ score = 0, size = 72 }) {
  const clampedScore = Math.max(0, Math.min(1, score))
  const pct = Math.round(clampedScore * 100)

  // Dimensions du cercle
  const strokeWidth = 5
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - clampedScore)

  // Couleur selon le score
  const color = clampedScore >= 0.75
    ? 'var(--accent)'
    : clampedScore >= 0.55
      ? 'var(--yellow)'
      : clampedScore >= 0.35
        ? 'var(--orange)'
        : 'var(--red)'

  return (
    <div className="score-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Cercle de fond */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--border)"
          strokeWidth={strokeWidth}
        />
        {/* Arc de progression */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 0.4s ease, stroke 0.3s ease' }}
        />
      </svg>
      <span className="score-ring__value" style={{ color, fontSize: size * 0.19 }}>
        {pct}%
      </span>
    </div>
  )
}
