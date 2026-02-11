/**
 * SignalDots — Grille de mini points colorés par stratégie.
 * Props : strategies (objet { stratName: { conditions_met, conditions_total } })
 * Couleur par ratio : high >= 0.75, medium >= 0.5, low >= 0.25, none < 0.25
 */
export default function SignalDots({ strategies = {} }) {
  const entries = Object.entries(strategies)

  if (entries.length === 0) {
    return <div className="signal-dots"><span className="dim text-xs">--</span></div>
  }

  return (
    <div className="signal-dots">
      {entries.map(([name, s]) => {
        const conditions = s.conditions || []
        const total = conditions.length || 1
        const met = conditions.filter(c => c.met).length
        const ratio = met / total

        const level = ratio >= 0.75
          ? 'high'
          : ratio >= 0.5
            ? 'medium'
            : ratio >= 0.25
              ? 'low'
              : 'none'

        return (
          <span
            key={name}
            className={`signal-dot signal-dot--${level}`}
            title={`${name}: ${met}/${total}`}
          />
        )
      })}
    </div>
  )
}
