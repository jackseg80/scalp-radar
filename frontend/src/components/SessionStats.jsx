import { useApi } from '../hooks/useApi'

export default function SessionStats() {
  const { data } = useApi('/api/simulator/status', 3000)

  const strategies = data?.strategies || {}
  const killSwitch = data?.kill_switch_triggered || false

  // Agréger les stats de toutes les stratégies
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

  return (
    <div className="card">
      <h2>Session</h2>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <StatRow label="P&L Net" value={`${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}$`}
          color={totalPnl >= 0 ? 'var(--accent)' : 'var(--red)'} />
        <StatRow label="Capital Total" value={`${totalCapital.toFixed(2)}$`} />
        <StatRow label="Trades" value={totalTrades} />
        <StatRow label="Win Rate" value={`${winRate.toFixed(1)}%`}
          color={winRate >= 50 ? 'var(--accent)' : 'var(--red)'} />
        <StatRow label="Wins / Losses" value={`${totalWins} / ${totalLosses}`} />

        {killSwitch && (
          <div style={{
            marginTop: 8,
            padding: '8px 12px',
            background: 'var(--red-dim)',
            borderRadius: 6,
            color: 'var(--red)',
            fontSize: 12,
            fontWeight: 600,
            textAlign: 'center',
          }}>
            KILL SWITCH ACTIF
          </div>
        )}
      </div>
    </div>
  )
}

function StatRow({ label, value, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
}
