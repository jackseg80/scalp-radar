/**
 * SessionStats — Stats du simulateur.
 * Props : wsData
 * Conçu pour être wrappé par CollapsibleCard dans App.jsx.
 */
import Tooltip from './Tooltip'

export default function SessionStats({ wsData }) {
  const strategies = wsData?.strategies || {}

  let totalPnl = 0
  let totalTrades = 0
  let totalWins = 0
  let totalLosses = 0
  let totalCapital = 0

  Object.values(strategies).forEach(s => {
    totalPnl += s.net_pnl || 0
    totalTrades += s.total_trades || 0
    totalWins += s.wins || 0
    totalLosses += s.losses || 0
    totalCapital += s.capital || 0
  })

  const winRate = totalTrades > 0 ? (totalWins / totalTrades * 100) : 0

  if (!wsData) {
    return <div className="empty-state">En attente de données...</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <StatRow
        label="P&L Net"
        value={`${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}$`}
        color={totalPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
        tooltip="Profit/perte total de toutes les stratégies en simulation (net de frais)"
      />
      <StatRow
        label="Capital (virtuel)"
        value={`${totalCapital.toFixed(0)}$`}
        tooltip="Capital de simulation : 10 000$ par stratégie active"
      />
      <StatRow label="Trades" value={totalTrades} />
      <StatRow
        label="Win Rate"
        value={`${winRate.toFixed(1)}%`}
        color={winRate >= 50 ? 'var(--accent)' : winRate > 0 ? 'var(--red)' : undefined}
        tooltip="Pourcentage de trades gagnants sur le total"
      />
      <StatRow
        label="W / L"
        value={`${totalWins} / ${totalLosses}`}
        tooltip="Nombre de trades gagnants / perdants"
      />

      {wsData?.kill_switch && (
        <Tooltip content="Trading stoppé : perte session ≥ 5% du capital" inline={false}>
          <div className="badge badge-stopped" style={{ textAlign: 'center', marginTop: 4, padding: '6px 8px' }}>
            KILL SWITCH ACTIF
          </div>
        </Tooltip>
      )}
    </div>
  )
}

// Exposer le summary pour CollapsibleCard
SessionStats.getSummary = function(wsData) {
  const strategies = wsData?.strategies || {}
  let totalPnl = 0
  Object.values(strategies).forEach(s => { totalPnl += s.net_pnl || 0 })
  return `${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(0)}$`
}

function StatRow({ label, value, color, tooltip }) {
  const row = (
    <div className="flex-between" style={{ fontSize: 12 }}>
      <span className="muted">{label}</span>
      <span className="mono" style={{ fontWeight: 600, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
  if (!tooltip) return row
  return <Tooltip content={tooltip} inline={false}>{row}</Tooltip>
}
