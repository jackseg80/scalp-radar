/**
 * ActivePositions — Bandeau des positions ouvertes au-dessus du Scanner.
 * Props : wsData
 * Affiche positions grid (agrégées par asset) + positions mono (paper/live).
 */
import { useState } from 'react'
import { formatPrice } from '../utils/format'

function formatPnl(value) {
  if (value == null) return null
  const abs = Math.abs(value)
  if (abs >= 10000) return `${(value / 1000).toFixed(1)}K$`
  return `${value.toFixed(2)}$`
}

export default function ActivePositions({ wsData }) {
  const simPositions = wsData?.simulator_positions || []
  const execPositions = wsData?.executor?.positions || (wsData?.executor?.position ? [wsData.executor.position] : [])
  const killSwitch = wsData?.kill_switch || false
  const prices = wsData?.prices || {}
  const gridState = wsData?.grid_state || null

  const [expandedGrid, setExpandedGrid] = useState(null)

  // Positions mono (filtrer les grid qui sont affichées séparément)
  const monoSimPositions = simPositions.filter(p => p.type !== 'grid')

  const hasGrids = gridState?.summary?.total_positions > 0
  const hasPaper = monoSimPositions.length > 0 || hasGrids
  const hasLive = execPositions.length > 0
  const hasPositions = hasPaper || hasLive

  return (
    <div className="active-positions-banner">
      {!hasPositions && (
        <div className="text-xs muted" style={{ padding: '2px 0' }}>
          {killSwitch
            ? 'Aucune position \u00b7 Kill switch actif'
            : gridState === null
              ? 'En attente de données prix...'
              : 'Aucune position \u00b7 En attente de signal'}
        </div>
      )}

      {hasPositions && (
        <div className="positions-columns">
          {/* Colonne PAPER */}
          {hasPaper && (
            <div className="positions-col">
              <div className="positions-col-header">PAPER</div>
              {hasGrids && (
                <GridSummary
                  gridState={gridState}
                  expandedGrid={expandedGrid}
                  onToggle={(symbol) => setExpandedGrid(prev => prev === symbol ? null : symbol)}
                />
              )}
              {monoSimPositions.map((pos, i) => (
                <PositionRow key={`sim-${i}`} pos={pos} source="PAPER" currentPrice={prices[pos.symbol]?.last} />
              ))}
            </div>
          )}

          {/* Colonne LIVE */}
          {hasLive && (
            <div className="positions-col">
              <div className="positions-col-header positions-col-header--live">LIVE</div>
              {execPositions.map((pos, i) => (
                <PositionRow key={`live-${i}`} pos={pos} source="LIVE" currentPrice={prices[pos.symbol]?.last} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function GridSummary({ gridState, expandedGrid, onToggle }) {
  const { grid_positions, summary } = gridState
  const grids = Object.values(grid_positions || {})

  if (!grids.length) return null

  return (
    <div style={{ marginBottom: 4 }}>
      {/* Bandeau résumé */}
      <div className="grid-summary-banner">
        <span>{summary.total_positions} grids sur {summary.total_assets} assets</span>
        <span className="mono">
          Marge: {summary.total_margin_used?.toFixed(0) || 0}$
        </span>
        <span className={`mono ${summary.total_unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
          P&L: {summary.total_unrealized_pnl >= 0 ? '+' : ''}
          {formatPnl(summary.total_unrealized_pnl)}
        </span>
      </div>

      {grids.map(g => (
        <div key={`${g.strategy}-${g.symbol}`}>
          <div
            className="active-position-item"
            style={{ cursor: 'pointer' }}
            onClick={() => onToggle(g.symbol)}
          >
            <span>{g.direction === 'LONG' ? '\u{1F7E2}' : '\u{1F534}'}</span>
            <span className={`badge ${g.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
              {g.direction}
            </span>
            <span style={{ fontWeight: 600 }}>{g.symbol}</span>
            <span className="text-xs muted">{g.strategy}</span>
            <span className="badge" style={{ fontSize: 9, background: 'var(--surface)', color: 'var(--text-dim)' }}>
              {g.levels_open}/{g.levels_max}
            </span>
            <span className="mono text-xs">avg@ {formatPrice(g.avg_entry)}</span>
            <span className={`mono text-xs ${g.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
              {g.unrealized_pnl >= 0 ? '+' : ''}{formatPnl(g.unrealized_pnl)}
              <span className="text-xs muted"> ({g.unrealized_pnl_pct >= 0 ? '+' : ''}{g.unrealized_pnl_pct?.toFixed(1)}%)</span>
            </span>
            {g.tp_distance_pct != null && (
              <span className="text-xs" style={{ color: 'var(--accent)' }}>
                TP {g.tp_distance_pct >= 0 ? '+' : ''}{g.tp_distance_pct.toFixed(1)}%
              </span>
            )}
            {g.sl_distance_pct != null && (
              <span className="text-xs" style={{ color: 'var(--red)' }}>
                SL {g.sl_distance_pct.toFixed(1)}%
              </span>
            )}
            <span className="badge badge-simulation" style={{ fontSize: 9, marginLeft: 'auto' }}>
              PAPER
            </span>
          </div>

          {/* Détail déplié : positions individuelles */}
          {expandedGrid === g.symbol && (
            <div className="grid-detail-row">
              {(g.positions || []).map(p => (
                <div key={p.level} className="active-position-item">
                  <span className="badge" style={{ fontSize: 9, background: 'var(--surface)', color: 'var(--text-dim)' }}>
                    Niv.{p.level + 1}
                  </span>
                  <span className="mono text-xs">@ {formatPrice(p.entry_price)}</span>
                  <span className="text-xs muted">qty: {p.quantity}</span>
                  <span className="text-xs muted">{new Date(p.entry_time).toLocaleString()}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function PositionRow({ pos, source, currentPrice }) {
  const isLong = pos.direction === 'LONG'

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
      <span className="mono text-xs">@ {formatPrice(pos.entry_price)}</span>
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
