/**
 * ActivePositions — Bandeau des positions ouvertes au-dessus du Scanner.
 * Props : wsData
 * Affiche positions simulator (paper) + executor (live).
 */

export default function ActivePositions({ wsData }) {
  const simPositions = wsData?.simulator_positions || []
  const execPositions = wsData?.executor?.positions || (wsData?.executor?.position ? [wsData.executor.position] : [])
  const killSwitch = wsData?.kill_switch || false
  const prices = wsData?.prices || {}

  const hasPositions = simPositions.length > 0 || execPositions.length > 0

  return (
    <div className="active-positions-banner">
      <h3>Positions actives</h3>
      {!hasPositions && (
        <div className="text-xs muted" style={{ padding: '2px 0' }}>
          {killSwitch
            ? 'Aucune position · Kill switch actif'
            : 'Aucune position · En attente de signal'}
        </div>
      )}
      {execPositions.map((pos, i) => (
        <PositionRow key={`live-${i}`} pos={pos} source="LIVE" currentPrice={prices[pos.symbol]?.last} />
      ))}
      {simPositions.map((pos, i) => (
        <PositionRow key={`sim-${i}`} pos={pos} source="PAPER" currentPrice={prices[pos.symbol]?.last} />
      ))}
    </div>
  )
}

function formatPnl(value) {
  if (value == null) return null
  const abs = Math.abs(value)
  if (abs >= 10000) return `${(value / 1000).toFixed(1)}K$`
  return `${value.toFixed(2)}$`
}

function PositionRow({ pos, source, currentPrice }) {
  const isLong = pos.direction === 'LONG'
  const isGrid = pos.type === 'grid'

  // P&L non réalisé
  let unrealizedPnl = null
  if (currentPrice != null && pos.entry_price != null && pos.quantity != null) {
    const diff = isLong
      ? (currentPrice - pos.entry_price) * pos.quantity
      : (pos.entry_price - currentPrice) * pos.quantity
    unrealizedPnl = diff
  }

  return (
    <div className="active-position-item">
      <span>{isLong ? '\u{1F7E2}' : '\u{1F534}'}</span>
      <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`}>
        {pos.direction}
      </span>
      <span style={{ fontWeight: 600 }}>{pos.symbol || '--'}</span>
      <span className="text-xs muted">{pos.strategy_name || pos.strategy || ''}</span>
      {isGrid && pos.level != null && (
        <span className="badge" style={{ fontSize: 9, background: 'var(--surface)', color: 'var(--text-dim)' }}>
          Niv.{pos.level}
        </span>
      )}
      <span className="mono text-xs">@ {Number(pos.entry_price).toFixed(2)}</span>
      {unrealizedPnl != null && (
        <span className={`mono text-xs ${unrealizedPnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
          {unrealizedPnl >= 0 ? '+' : ''}{formatPnl(unrealizedPnl)}
        </span>
      )}
      <span className={`badge ${source === 'LIVE' ? 'badge-active' : 'badge-simulation'}`} style={{ fontSize: 9, marginLeft: 'auto' }}>
        {source}
      </span>
    </div>
  )
}
