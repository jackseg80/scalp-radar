/**
 * SignalDots — Pastilles colorées par stratégie avec initiales et tooltips détaillés.
 * Props : strategies (objet { stratName: { conditions } })
 * Taille 22x22, initiale visible, couleur par ratio conditions met/total.
 */
import Tooltip from './Tooltip'

const STRATEGY_ICONS = {
  vwap_rsi: 'V',
  momentum: 'M',
  funding: 'F',
  liquidation: 'L',
}

const STRATEGY_NAMES = {
  vwap_rsi: 'VWAP+RSI',
  momentum: 'Momentum',
  funding: 'Funding',
  liquidation: 'Liquidation',
}

function buildTooltip(name, conditions) {
  const label = STRATEGY_NAMES[name] || name
  const met = conditions.filter(c => c.met).length
  const total = conditions.length
  const lines = conditions.map(c => {
    const check = c.met ? '\u2713' : '\u2717'
    const val = c.value != null ? ` (${typeof c.value === 'number' ? c.value.toFixed(2) : c.value})` : ''
    return `${check} ${c.name}${val}`
  })
  return (
    <div>
      <div style={{ fontWeight: 600, marginBottom: 3 }}>{label} : {met}/{total}</div>
      {lines.map((line, i) => (
        <div key={i} style={{ fontSize: 10, opacity: 0.85 }}>{line}</div>
      ))}
    </div>
  )
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
          <Tooltip key={name} content={buildTooltip(name, conditions)}>
            <span className={`signal-dot signal-dot--${level}`}>
              {icon}
            </span>
          </Tooltip>
        )
      })}
    </div>
  )
}
