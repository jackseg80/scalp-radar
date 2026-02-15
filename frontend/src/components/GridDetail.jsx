/**
 * GridDetail — Détail d'un asset grid au clic dans le Scanner.
 * Utilise grid_state du WebSocket (positions, TP/SL, indicateurs).
 * Props : symbol, gridInfo (grid_positions[symbol]), indicators, regime, price
 */
import Tooltip from './Tooltip'

export default function GridDetail({ symbol, gridInfo, indicators = {}, regime, price }) {
  const maxLevels = gridInfo?.levels_max || 3
  const positions = gridInfo?.positions || []
  const filledSet = new Set(positions.map(p => p.level))

  // Construire tous les niveaux (remplis + en attente)
  const allLevels = []
  for (let i = 0; i < maxLevels; i++) {
    const filled = positions.find(p => p.level === i)
    allLevels.push({
      index: i,
      filled: !!filled,
      entry_price: filled?.entry_price || null,
      quantity: filled?.quantity || null,
      direction: filled?.direction || gridInfo?.direction || 'long',
    })
  }

  // Indicateurs
  const rsi = indicators?.rsi_14
  const adx = indicators?.adx
  const atrPct = indicators?.atr_pct

  // Dist SMA (TP = SMA pour grid strategies)
  const distSma = (gridInfo?.tp_price && gridInfo?.current_price && gridInfo.tp_price > 0)
    ? ((gridInfo.current_price - gridInfo.tp_price) / gridInfo.tp_price * 100)
    : null

  return (
    <div className="scanner-expand">
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        {/* Résumé grid */}
        <div style={{ flexShrink: 0, textAlign: 'center', minWidth: 90 }}>
          <div className="grid-detail-summary">
            <div className="grid-detail-ratio">
              <span className={filledSet.size > 0 ? 'pnl-pos' : 'muted'}>{filledSet.size}</span>
              <span className="muted">/{maxLevels}</span>
            </div>
            <div className="text-xs muted" style={{ marginTop: 2 }}>{gridInfo?.strategy || 'grid_atr'}</div>
            {gridInfo?.unrealized_pnl != null && (
              <div className={`mono text-xs ${gridInfo.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ marginTop: 4 }}>
                {gridInfo.unrealized_pnl >= 0 ? '+' : ''}{gridInfo.unrealized_pnl.toFixed(2)}$
              </div>
            )}
          </div>
        </div>

        {/* Niveaux grid */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="text-xs dim" style={{ marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Niveaux grid
          </div>
          <div className="grid-detail-levels">
            {allLevels.map(lvl => (
              <div key={lvl.index} className={`grid-level ${lvl.filled ? 'grid-level--filled' : 'grid-level--pending'}`}>
                <span className="grid-level-name">
                  Lvl {lvl.index + 1}
                </span>
                <div className="grid-level-bar">
                  <div className={`grid-level-fill ${lvl.filled ? 'grid-level-fill--green' : 'grid-level-fill--red'}`} />
                </div>
                <span className="mono" style={{ minWidth: 70, textAlign: 'right' }}>
                  {lvl.entry_price ? Number(lvl.entry_price).toFixed(4) : '--'}
                </span>
                <span className={`text-xs ${lvl.filled ? '' : 'muted'}`} style={{ minWidth: 60 }}>
                  {lvl.filled ? lvl.direction.toUpperCase() : 'attente'}
                </span>
              </div>
            ))}
          </div>

          {/* TP / SL */}
          <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
            {gridInfo?.tp_price != null && (
              <span className="text-xs">
                <span className="muted">TP </span>
                <span className="mono" style={{ color: 'var(--accent)' }}>
                  {Number(gridInfo.tp_price).toFixed(4)}
                </span>
                {gridInfo.tp_distance_pct != null && (
                  <span className="muted"> ({gridInfo.tp_distance_pct > 0 ? '+' : ''}{gridInfo.tp_distance_pct.toFixed(1)}%)</span>
                )}
              </span>
            )}
            {gridInfo?.sl_price != null && (
              <span className="text-xs">
                <span className="muted">SL </span>
                <span className="mono" style={{ color: 'var(--red)' }}>
                  {Number(gridInfo.sl_price).toFixed(4)}
                </span>
                {gridInfo.sl_distance_pct != null && (
                  <span className="muted"> ({gridInfo.sl_distance_pct > 0 ? '+' : ''}{gridInfo.sl_distance_pct.toFixed(1)}%)</span>
                )}
              </span>
            )}
            {gridInfo?.margin_used != null && (
              <span className="text-xs">
                <span className="muted">Marge </span>
                <span className="mono">{gridInfo.margin_used.toFixed(1)}$</span>
              </span>
            )}
          </div>
        </div>

        {/* Indicateurs */}
        <div style={{ flexShrink: 0, minWidth: 130 }}>
          <div className="text-xs dim" style={{ marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Indicateurs
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {price != null && (
              <IndicatorRow label="Prix" value={Number(price).toFixed(2)} />
            )}
            {gridInfo?.avg_entry != null && gridInfo.avg_entry > 0 && (
              <IndicatorRow
                label="Avg entry"
                value={Number(gridInfo.avg_entry).toFixed(4)}
                tooltip="Prix d'entrée moyen pondéré"
              />
            )}
            {distSma != null && (
              <IndicatorRow
                label="Dist.SMA"
                value={`${distSma >= 0 ? '+' : ''}${distSma.toFixed(1)}%`}
                color={distSma >= 0 ? 'var(--accent)' : 'var(--red)'}
                tooltip="Distance du prix à la SMA (TP grid). 0% = proche du TP"
              />
            )}
            {rsi != null && (
              <IndicatorRow
                label="RSI"
                value={Number(rsi).toFixed(1)}
                color={rsi < 30 ? 'var(--accent)' : rsi > 70 ? 'var(--red)' : null}
                tooltip="RSI (14 périodes)"
              />
            )}
            {adx != null && (
              <IndicatorRow
                label="ADX"
                value={Number(adx).toFixed(1)}
                tooltip="Force de la tendance"
              />
            )}
            {atrPct != null && (
              <IndicatorRow
                label="ATR %"
                value={`${Number(atrPct).toFixed(2)}%`}
                tooltip="Volatilité moyenne (ATR / prix)"
              />
            )}
            {regime && (
              <IndicatorRow
                label="Régime"
                value={regime}
                badge
                tooltip="Régime de marché détecté"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function IndicatorRow({ label, value, color, badge, tooltip }) {
  const row = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
      <span className="muted" style={{ width: 55, flexShrink: 0 }}>{label}</span>
      {badge ? (
        <span className={`badge ${
          value === 'RANGING' || value === 'ranging' ? 'badge-ranging' : 'badge-trending'
        }`}>
          {value}
        </span>
      ) : (
        <span className="mono" style={{ color: color || 'var(--text-primary)', fontWeight: 500, marginLeft: 'auto' }}>
          {value}
        </span>
      )}
    </div>
  )
  if (!tooltip) return row
  return <Tooltip content={tooltip} inline={false}>{row}</Tooltip>
}
