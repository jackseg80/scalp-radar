/**
 * ExecutorPanel — Panneau de statut de l'executor.
 * Props : wsData
 * Affiche le statut executor depuis wsData.executor.
 * Badge "SIMULATION ONLY" si pas d'executor actif.
 *
 * Format backend (executor.get_status()):
 *   { enabled, connected, sandbox, position, risk_manager: { session_pnl, kill_switch, initial_capital, total_orders, open_positions_count } }
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

  const { enabled, connected, sandbox, position, risk_manager: rm } = executor

  const isLive = enabled && !sandbox
  const hasPosition = position != null
  const sessionPnl = rm?.session_pnl ?? 0
  const balance = rm?.initial_capital

  return (
    <div className="card">
      <div className="flex-between" style={{ marginBottom: 10 }}>
        <h2 style={{ marginBottom: 0 }}>Executor</h2>
        <span className={`badge ${isLive ? 'badge-active' : sandbox ? 'badge-simulation' : 'badge-stopped'}`}>
          {isLive ? 'LIVE' : sandbox ? 'SANDBOX' : 'OFF'}
        </span>
      </div>

      {/* Status global */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: hasPosition ? 12 : 0 }}>
        <StatusRow
          label="Connexion"
          value={connected ? 'connecté' : 'déconnecté'}
          color={connected ? 'var(--accent)' : 'var(--red)'}
        />
        {balance != null && (
          <StatusRow label="Solde Bitget" value={`${Number(balance).toFixed(2)} USDT`} />
        )}
        <StatusRow
          label="P&L Session"
          value={`${sessionPnl >= 0 ? '+' : ''}${Number(sessionPnl).toFixed(2)}$`}
          color={sessionPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
        />
        {rm?.total_orders != null && rm.total_orders > 0 && (
          <StatusRow label="Ordres passés" value={rm.total_orders} />
        )}
        {rm?.kill_switch && (
          <div className="badge badge-stopped" style={{ textAlign: 'center', marginTop: 4, padding: '6px 8px' }}>
            KILL SWITCH LIVE ACTIF
          </div>
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
            {position.sl_price != null && (
              <PositionRow label="Stop Loss" value={Number(position.sl_price).toFixed(2)} color="var(--red)" />
            )}
            {position.tp_price != null && (
              <PositionRow label="Take Profit" value={Number(position.tp_price).toFixed(2)} color="var(--accent)" />
            )}
            {position.quantity != null && (
              <PositionRow label="Quantité" value={position.quantity} />
            )}
          </div>
        </div>
      )}

      {/* Pas de position */}
      {!hasPosition && (
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
