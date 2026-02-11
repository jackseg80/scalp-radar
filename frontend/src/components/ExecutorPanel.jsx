/**
 * ExecutorPanel — Panneau de statut de l'executor.
 * Props : wsData
 * Affiche le statut executor depuis wsData.executor.
 * Badge "SIMULATION ONLY" si pas d'executor actif.
 */
export default function ExecutorPanel({ wsData }) {
  const executor = wsData?.executor

  // Pas d'executor = mode simulation
  if (!executor) {
    return (
      <div className="card">
        <h2>Executor</h2>
        <div style={{ textAlign: 'center', padding: '12px 0' }}>
          <span className="badge badge-simulation" style={{ fontSize: 12, padding: '4px 12px' }}>
            SIMULATION ONLY
          </span>
          <p className="text-xs muted" style={{ marginTop: 8 }}>
            Aucun executor actif. Les trades sont simulés.
          </p>
        </div>
      </div>
    )
  }

  const { status, position, mode, daily_pnl, daily_trades } = executor

  const isLive = mode === 'live'
  const hasPosition = position != null

  return (
    <div className="card">
      <div className="flex-between" style={{ marginBottom: 10 }}>
        <h2 style={{ marginBottom: 0 }}>Executor</h2>
        <span className={`badge ${isLive ? 'badge-active' : 'badge-simulation'}`}>
          {isLive ? 'LIVE' : 'PAPER'}
        </span>
      </div>

      {/* Status global */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: hasPosition ? 12 : 0 }}>
        <StatusRow label="Status" value={status || 'idle'} />
        {daily_pnl != null && (
          <StatusRow
            label="P&L Jour"
            value={`${daily_pnl >= 0 ? '+' : ''}${Number(daily_pnl).toFixed(2)}$`}
            color={daily_pnl >= 0 ? 'var(--accent)' : 'var(--red)'}
          />
        )}
        {daily_trades != null && (
          <StatusRow label="Trades Jour" value={daily_trades} />
        )}
      </div>

      {/* Position ouverte */}
      {hasPosition && (
        <div className="executor-position">
          <div className="flex-between" style={{ marginBottom: 6 }}>
            <span style={{ fontWeight: 600, fontSize: 12 }}>{position.symbol}</span>
            <span className={`badge ${position.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
              {position.direction}
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <PositionRow label="Entrée" value={Number(position.entry_price).toFixed(2)} />
            {position.current_price != null && (
              <PositionRow label="Actuel" value={Number(position.current_price).toFixed(2)} />
            )}
            {position.unrealized_pnl != null && (
              <PositionRow
                label="P&L non réalisé"
                value={`${position.unrealized_pnl >= 0 ? '+' : ''}${Number(position.unrealized_pnl).toFixed(2)}$`}
                color={position.unrealized_pnl >= 0 ? 'var(--accent)' : 'var(--red)'}
              />
            )}
            {position.stop_loss != null && (
              <PositionRow label="Stop Loss" value={Number(position.stop_loss).toFixed(2)} color="var(--red)" />
            )}
            {position.take_profit != null && (
              <PositionRow label="Take Profit" value={Number(position.take_profit).toFixed(2)} color="var(--accent)" />
            )}
            {position.leverage != null && (
              <PositionRow label="Levier" value={`x${position.leverage}`} />
            )}
          </div>
        </div>
      )}

      {/* Pas de position */}
      {!hasPosition && executor.status !== 'idle' && (
        <div className="text-xs muted text-center" style={{ paddingTop: 4 }}>
          Aucune position ouverte
        </div>
      )}
    </div>
  )
}

function StatusRow({ label, value, color }) {
  return (
    <div className="flex-between" style={{ fontSize: 12 }}>
      <span className="muted">{label}</span>
      <span className="mono" style={{ fontWeight: 500, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
}

function PositionRow({ label, value, color }) {
  return (
    <div className="flex-between" style={{ fontSize: 11 }}>
      <span className="muted">{label}</span>
      <span className="mono" style={{ fontWeight: 500, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
}
