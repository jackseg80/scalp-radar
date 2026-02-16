/**
 * SessionStats — Stats du simulateur avec P&L réalisé + non réalisé.
 * Props : wsData
 * Conçu pour être wrappé par CollapsibleCard dans App.jsx.
 */
import Tooltip from './Tooltip'

export default function SessionStats({ wsData }) {
  const strategies = wsData?.strategies || {}

  let totalRealizedPnl = 0
  let totalUnrealizedPnl = 0
  let totalTrades = 0
  let totalWins = 0
  let totalLosses = 0
  let totalEquity = 0
  let totalMarginUsed = 0
  let totalOpenPositions = 0
  let totalAssetsWithPositions = 0
  let initialCapital = 0
  let runnerCount = 0

  Object.values(strategies).forEach(s => {
    totalRealizedPnl += s.net_pnl || 0
    totalUnrealizedPnl += s.unrealized_pnl || 0
    totalTrades += s.total_trades || 0
    totalWins += s.wins || 0
    totalLosses += s.losses || 0
    totalEquity += s.equity || s.capital || 0
    totalMarginUsed += s.margin_used || 0
    totalOpenPositions += s.open_positions || 0
    totalAssetsWithPositions += s.assets_with_positions || 0
    initialCapital += s.capital ? (s.capital - (s.net_pnl || 0)) : 0
    runnerCount++
  })

  // Fallback initial capital si pas de runners
  if (initialCapital <= 0) initialCapital = 10000

  const totalPnl = totalRealizedPnl + totalUnrealizedPnl
  const equityPct = ((totalEquity / initialCapital) - 1) * 100
  const marginPct = initialCapital > 0 ? (totalMarginUsed / initialCapital) * 100 : 0
  const available = totalEquity - totalMarginUsed
  const winRate = totalTrades > 0 ? (totalWins / totalTrades * 100) : 0

  if (!wsData) {
    return <div className="empty-state">En attente de données...</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {/* Section P&L */}
      <div className="sim-section">
        <StatRow
          label="P&L Total"
          value={`${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}$`}
          color={totalPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
          tooltip="P&L total (réalisé + non réalisé, net de frais)"
          primary
        />
        <StatRow
          label="Réalisé"
          value={`${totalRealizedPnl >= 0 ? '+' : ''}${totalRealizedPnl.toFixed(2)}$`}
          color={totalRealizedPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
          tooltip="Profit/perte des trades clôturés"
          small
        />
        <StatRow
          label="Non réalisé"
          value={`${totalUnrealizedPnl >= 0 ? '+' : ''}${totalUnrealizedPnl.toFixed(2)}$`}
          color={totalUnrealizedPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
          tooltip="P&L latent des positions ouvertes"
          small
        />
      </div>

      {/* Section Capital */}
      <div className="sim-section">
        <StatRow
          label="Equity"
          value={`${totalEquity.toLocaleString('fr-FR', { maximumFractionDigits: 0 })}$ (${equityPct >= 0 ? '+' : ''}${equityPct.toFixed(1)}%)`}
          color={equityPct >= 0 ? 'var(--accent)' : 'var(--red)'}
          tooltip="Capital + P&L non réalisé"
          primary
        />
        <StatRow
          label="Marge"
          value={`${totalMarginUsed.toLocaleString('fr-FR', { maximumFractionDigits: 0 })}$ (${marginPct.toFixed(1)}%)`}
          tooltip="Marge utilisée par les positions ouvertes"
          small
        />
        <StatRow
          label="Disponible"
          value={`${available.toLocaleString('fr-FR', { maximumFractionDigits: 0 })}$`}
          tooltip="Equity - marge utilisée"
          small
        />
      </div>

      {/* Section Stats compacte */}
      <div className="sim-stats-row">
        <span>Trades: {totalTrades}</span>
        <span>W/L: {totalWins}/{totalLosses}</span>
        <span>WR: {winRate.toFixed(0)}%</span>
      </div>

      {/* Section Grids */}
      {totalOpenPositions > 0 && (
        <div className="sim-stats-row">
          <span>Grids: {totalOpenPositions} pos. sur {totalAssetsWithPositions} asset{totalAssetsWithPositions > 1 ? 's' : ''}</span>
        </div>
      )}

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

// Exposer le summary pour CollapsibleCard — maintenant avec P&L total
SessionStats.getSummary = function(wsData) {
  const strategies = wsData?.strategies || {}
  let totalPnl = 0
  Object.values(strategies).forEach(s => {
    totalPnl += (s.net_pnl || 0) + (s.unrealized_pnl || 0)
  })
  return `${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(0)}$`
}

function StatRow({ label, value, color, tooltip, primary, small }) {
  const fontSize = primary ? 13 : small ? 11 : 12
  const fontWeight = primary ? 700 : 600
  const labelColor = small ? 'var(--text-dim)' : 'var(--text-muted)'
  const row = (
    <div className="flex-between" style={{ fontSize, padding: small ? '1px 0' : '2px 0' }}>
      <span style={{ color: labelColor }}>{label}</span>
      <span className="mono" style={{ fontWeight, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
  if (!tooltip) return row
  return <Tooltip content={tooltip} inline={false}>{row}</Tooltip>
}
