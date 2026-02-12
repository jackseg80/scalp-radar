/**
 * ActivityFeed — Timeline d'activité avec cartes riches (remplace AlertFeed).
 * Props : wsData
 * Utilise /api/simulator/trades pour les trades fermés.
 * Positions ouvertes depuis wsData.simulator_positions.
 */
import { useApi } from '../hooks/useApi'

const EXIT_REASONS = {
  sl: 'fermé par SL',
  tp: 'TP atteint',
  signal_exit: 'sortie signal',
  regime_change: 'changement régime',
  end_of_data: 'fin de données',
}

function timeAgo(isoString) {
  if (!isoString) return ''
  const diff = Date.now() - new Date(isoString).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "à l'instant"
  if (mins < 60) return `il y a ${mins}min`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `il y a ${hours}h`
  const days = Math.floor(hours / 24)
  return `il y a ${days}j`
}

export default function ActivityFeed({ wsData }) {
  const { data } = useApi('/api/simulator/trades?limit=20', 10000)

  const trades = data?.trades || []
  const openPositions = wsData?.simulator_positions || []

  return (
    <>
      {openPositions.length === 0 && trades.length === 0 && (
        <div className="empty-state">Aucune activité pour le moment</div>
      )}

      {/* Positions ouvertes en premier */}
      {openPositions.map((pos, i) => (
        <OpenPositionCard key={`open-${i}`} pos={pos} wsData={wsData} />
      ))}

      {/* Trades fermés */}
      {trades.map((trade, i) => (
        <ClosedTradeCard key={`trade-${trade.exit_time}-${i}`} trade={trade} />
      ))}
    </>
  )
}

function OpenPositionCard({ pos, wsData }) {
  const isLong = pos.direction === 'LONG'
  const emoji = isLong ? '\u{1F7E2}' : '\u{1F534}'
  const prices = wsData?.prices || {}
  const currentPrice = prices[pos.symbol]?.last

  let unrealizedPnl = null
  if (currentPrice != null && pos.entry_price != null && pos.quantity != null) {
    const diff = isLong
      ? (currentPrice - pos.entry_price) * pos.quantity
      : (pos.entry_price - currentPrice) * pos.quantity
    unrealizedPnl = diff
  }

  return (
    <div className="activity-card activity-card--open">
      <div className="flex-between" style={{ marginBottom: 4 }}>
        <span style={{ fontWeight: 600, fontSize: 12 }}>
          {emoji} {pos.direction} {pos.symbol}
        </span>
        <span className="text-xs muted">{timeAgo(pos.entry_time)}</span>
      </div>
      <div className="text-xs muted">
        {pos.strategy} &middot; Entry {Number(pos.entry_price).toFixed(2)}
        {pos.sl_price ? ` → SL ${Number(pos.sl_price).toFixed(2)}` : ''}
      </div>
      <div className="text-xs" style={{ marginTop: 2 }}>
        <span style={{ color: 'var(--blue)' }}>En cours</span>
        {unrealizedPnl != null && (
          <span className={`mono ${unrealizedPnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ marginLeft: 8 }}>
            P&L: {unrealizedPnl >= 0 ? '+' : ''}{unrealizedPnl.toFixed(2)}$
          </span>
        )}
      </div>
    </div>
  )
}

function ClosedTradeCard({ trade }) {
  const isLong = trade.direction === 'LONG'
  const emoji = isLong ? '\u{1F7E2}' : '\u{1F534}'
  const exitLabel = EXIT_REASONS[trade.exit_reason] || trade.exit_reason

  return (
    <div className="activity-card">
      <div className="flex-between" style={{ marginBottom: 4 }}>
        <span style={{ fontWeight: 600, fontSize: 12 }}>
          {emoji} {trade.direction} {trade.symbol || '--'}
        </span>
        <span className="text-xs muted">{timeAgo(trade.exit_time)}</span>
      </div>
      <div className="text-xs muted">
        {trade.strategy} &middot; Entry {Number(trade.entry_price).toFixed(2)}
        {trade.exit_price ? ` → Exit ${Number(trade.exit_price).toFixed(2)}` : ''}
      </div>
      <div className="text-xs" style={{ marginTop: 2 }}>
        <span className={`mono ${trade.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
          {trade.net_pnl >= 0 ? '+' : ''}{Number(trade.net_pnl).toFixed(2)}$
        </span>
        <span className="muted" style={{ marginLeft: 8 }}>({exitLabel})</span>
      </div>
    </div>
  )
}
