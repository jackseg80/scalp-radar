export default function SessionStats({ wsData }) {
  const strategies = wsData?.strategies || {}
  const killSwitch = wsData?.kill_switch || false

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
  const longCount = 0 // TODO: compter depuis les trades
  const shortCount = 0

  if (!wsData) {
    return (
      <div className="card">
        <h2>Simulator <span className="dim text-xs" style={{ textTransform: 'none', letterSpacing: 0 }}>(Paper)</span></h2>
        <div className="empty-state">En attente de données...</div>
      </div>
    )
  }

  return (
    <div className="card">
      <h2>Simulator <span className="dim text-xs" style={{ textTransform: 'none', letterSpacing: 0 }}>(Paper)</span></h2>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <StatRow
          label="P&L Net"
          value={`${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}$`}
          color={totalPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
        />
        <StatRow label="Capital (virtuel)" value={`${totalCapital.toFixed(0)}$`} />
        <StatRow label="Trades" value={totalTrades} />
        <StatRow
          label="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          color={winRate >= 50 ? 'var(--accent)' : winRate > 0 ? 'var(--red)' : undefined}
        />
        <StatRow label="W / L" value={`${totalWins} / ${totalLosses}`} />

        {killSwitch && (
          <div className="badge badge-stopped" style={{ textAlign: 'center', marginTop: 4, padding: '6px 8px' }}>
            KILL SWITCH ACTIF
          </div>
        )}
      </div>
    </div>
  )
}

function StatRow({ label, value, color }) {
  return (
    <div className="flex-between" style={{ fontSize: 12 }}>
      <span className="muted">{label}</span>
      <span className="mono" style={{ fontWeight: 600, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
}
