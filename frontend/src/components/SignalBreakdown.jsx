/**
 * SignalBreakdown — Barres de progression horizontales par stratégie.
 * Props : strategies (objet { stratName: { conditions_met, conditions_total, direction } })
 * Barre remplie selon le ratio conditions_met / conditions_total.
 */
export default function SignalBreakdown({ strategies = {} }) {
  const entries = Object.entries(strategies)

  if (entries.length === 0) {
    return <div className="empty-state">Aucune stratégie</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {entries.map(([name, s]) => {
        const conditions = s.conditions || []
        const total = conditions.length || 1
        const met = conditions.filter(c => c.met).length
        const ratio = met / total
        const pct = Math.round(ratio * 100)

        // Couleur de la barre
        const color = ratio >= 0.75
          ? 'var(--accent)'
          : ratio >= 0.55
            ? 'var(--yellow)'
            : ratio >= 0.35
              ? 'var(--orange)'
              : 'var(--red)'

        return (
          <div key={name}>
            <div className="flex-between" style={{ marginBottom: 3 }}>
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{name}</span>
              <span className="mono text-xs" style={{ color }}>
                {met}/{total} ({pct}%)
                {s.direction && (
                  <span className={`badge ${s.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}
                    style={{ marginLeft: 6 }}>
                    {s.direction}
                  </span>
                )}
              </span>
            </div>
            <div className="breakdown-bar">
              <div
                className="breakdown-bar__fill"
                style={{ width: `${pct}%`, background: color }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}
