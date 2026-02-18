/**
 * ActivityFeed — Timeline d'activité avec cartes riches (remplace AlertFeed).
 * Props : wsData
 * Utilise /api/simulator/trades pour les trades fermés.
 * Positions ouvertes depuis wsData.simulator_positions.
 * Sprint 25 : événements journal (ouvertures/fermetures DCA) depuis /api/journal/events.
 */
import { useApi } from '../hooks/useApi'
import { formatPrice } from '../utils/format'

const EXIT_REASONS = {
  sl: 'SL',
  tp: 'TP',
  tp_global: 'TP global',
  sl_global: 'SL global',
  force_close: 'Force close',
  signal_exit: 'Signal',
  regime_change: 'Regime',
  end_of_data: 'Fin',
}

function timeAgo(isoString) {
  if (!isoString) return ''
  const diff = Date.now() - new Date(isoString).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "a l'instant"
  if (mins < 60) return `il y a ${mins}min`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `il y a ${hours}h`
  const days = Math.floor(hours / 24)
  return `il y a ${days}j`
}

function formatTime(isoString) {
  if (!isoString) return ''
  const d = new Date(isoString)
  return d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
}

export default function ActivityFeed({ wsData }) {
  const { data } = useApi('/api/simulator/trades?limit=20', 10000)
  const { data: journalData } = useApi('/api/journal/events?limit=20', 30000)

  const trades = data?.trades || []
  const openPositions = wsData?.simulator_positions || []
  const journalEvents = journalData?.events || []

  const hasActivity = openPositions.length > 0 || journalEvents.length > 0 || trades.length > 0

  return (
    <>
      {!hasActivity && (
        <div className="empty-state">Aucune activite pour le moment</div>
      )}

      {/* Positions ouvertes en premier */}
      {openPositions.map((pos, i) => (
        <OpenPositionCard key={`open-${i}`} pos={pos} wsData={wsData} />
      ))}

      {/* Evenements journal (ouvertures/fermetures DCA) */}
      {journalEvents.length > 0 && (
        <div style={{ marginTop: openPositions.length > 0 ? 6 : 0 }}>
          {journalEvents.slice(0, 15).map((event, i) => (
            <JournalEventCard key={`journal-${event.id || i}`} event={event} />
          ))}
        </div>
      )}

      {/* Trades fermes */}
      {trades.map((trade, i) => (
        <ClosedTradeCard key={`trade-${trade.exit_time}-${i}`} trade={trade} />
      ))}
    </>
  )
}

function JournalEventCard({ event }) {
  const isOpen = event.event_type === 'OPEN'
  const isProfit = event.unrealized_pnl != null && event.unrealized_pnl >= 0
  const meta = event.metadata || {}

  return (
    <div className={`activity-card ${isOpen ? 'activity-card--journal-open' : ''}`}
         style={{ borderLeft: `3px solid ${isOpen ? 'var(--accent)' : (isProfit ? 'var(--accent)' : 'var(--red)')}`, marginBottom: 4 }}>
      <div className="flex-between" style={{ marginBottom: 2 }}>
        <span style={{ fontSize: 11, fontWeight: 600 }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', marginRight: 4,
            background: isOpen ? 'var(--accent)' : (isProfit ? 'var(--accent)' : 'var(--red)') }} />
          {isOpen ? 'OPEN' : 'CLOSE'} {event.symbol}
        </span>
        <span className="text-xs muted">{formatTime(event.timestamp)}</span>
      </div>
      <div className="text-xs muted">
        {event.strategy_name}
        {event.level != null && ` Lvl ${event.level}`}
        {' '}&middot;{' '}{event.direction}
        {' '}&middot;{' '}{formatPrice(event.price)}$
      </div>
      {!isOpen && meta.net_pnl != null && (
        <div className="text-xs" style={{ marginTop: 1 }}>
          <span className={`mono ${meta.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
            {meta.net_pnl >= 0 ? '+' : ''}{Number(meta.net_pnl).toFixed(2)}$
          </span>
          {meta.exit_reason && (
            <span className="muted" style={{ marginLeft: 6 }}>
              ({EXIT_REASONS[meta.exit_reason] || meta.exit_reason})
            </span>
          )}
        </div>
      )}
      {isOpen && meta.levels_open != null && (
        <div className="text-xs muted" style={{ marginTop: 1 }}>
          {meta.levels_open}/{meta.levels_max} niveaux
        </div>
      )}
    </div>
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
        {pos.strategy} &middot; Entry {formatPrice(pos.entry_price)}
        {pos.sl_price ? ` → SL ${formatPrice(pos.sl_price)}` : ''}
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
        {trade.strategy} &middot; Entry {formatPrice(trade.entry_price)}
        {trade.exit_price ? ` → Exit ${formatPrice(trade.exit_price)}` : ''}
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
