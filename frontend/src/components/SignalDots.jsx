/**
 * SignalDots — Pastilles colorées par stratégie avec initiales et tooltips détaillés.
 * Props : strategies (objet { stratName: { conditions } })
 * Taille 22x22, initiale visible, couleur par ratio conditions met/total.
 */
import { memo } from 'react'
import Tooltip from './Tooltip'

const STRATEGY_ICONS = {
  vwap_rsi: 'V',
  momentum: 'M',
  funding: 'F',
  liquidation: 'L',
  bollinger_mr: 'B',
  donchian_breakout: 'D',
  supertrend: 'S',
  boltrend: 'T',
  envelope_dca: 'E',
  envelope_dca_short: 'Es',
  grid_atr: 'GA',
  grid_range_atr: 'GR',
  grid_multi_tf: 'GM',
  grid_funding: 'GF',
  grid_trend: 'GT',
  grid_boltrend: 'GB',
}

const STRATEGY_NAMES = {
  vwap_rsi: 'VWAP + RSI',
  momentum: 'Momentum',
  funding: 'Funding',
  liquidation: 'Liquidation',
  bollinger_mr: 'Bollinger MR',
  donchian_breakout: 'Donchian Breakout',
  supertrend: 'Supertrend',
  boltrend: 'Bollinger Trend',
  envelope_dca: 'Envelope DCA',
  envelope_dca_short: 'Envelope DCA Short',
  grid_atr: 'Grid ATR',
  grid_range_atr: 'Grid Range ATR',
  grid_multi_tf: 'Grid Multi-TF',
  grid_funding: 'Grid Funding',
  grid_trend: 'Grid Trend',
  grid_boltrend: 'Grid BolTrend',
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

const SignalDots = memo(function SignalDots({ strategies = {} }) {
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
})

export default SignalDots
