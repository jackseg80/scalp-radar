/**
 * SignalDots — Pastilles colorées par stratégie avec initiales.
 * Props : strategies (objet { stratName: { conditions } })
 * Taille 22x22, initiale visible, couleur par ratio conditions met/total.
 */

const STRATEGY_ICONS = {
  vwap_rsi: 'V',
  momentum: 'M',
  funding: 'F',
  liquidation: 'L',
}

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

        const icon = STRATEGY_ICONS[name] || name.charAt(0).toUpperCase()

        return (
          <span
            key={name}
            className={`signal-dot signal-dot--${level}`}
            title={`${name}: ${met}/${total}`}
          >
            {icon}
          </span>
        )
      })}
    </div>
  )
}
