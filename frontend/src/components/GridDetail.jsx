/**
 * GridDetail — Détail d'un asset grid au clic dans le Scanner.
 *
 * Deux modes :
 * 1. Mode position (gridInfo non null) : affiche positions remplies, P&L, TP/SL, marge
 * 2. Mode conditions (gridInfo null, conditions fourni) : affiche niveaux calculés + gates
 *
 * Props : symbol, gridInfo, indicators, regime, price, conditions, strategyName
 */
import Tooltip from './Tooltip'
import { formatPrice } from '../utils/format'

const PROXIMITY_COLORS = {
  imminent: 'var(--accent)',
  close: 'var(--yellow)',
  medium: 'var(--orange)',
  far: 'var(--red)',
}

const STRATEGY_LABELS = {
  grid_atr: 'Grid ATR',
  grid_boltrend: 'Grid BolTrend',
  grid_multi_tf: 'Grid Multi-TF',
  grid_range_atr: 'Grid Range ATR',
  grid_trend: 'Grid Trend',
  grid_funding: 'Grid Funding',
  grid_momentum: 'Grid Momentum',
  envelope_dca: 'Envelope DCA',
  envelope_dca_short: 'Envelope DCA Short',
}

export default function GridDetail({ symbol, gridInfo, indicators = {}, regime, price, conditions = [], strategyName }) {
  const hasPosition = !!gridInfo
  const condLevels = (conditions || []).filter(c => !c.gate)
  const gates = (conditions || []).filter(c => c.gate)

  // Mode position : niveaux depuis gridInfo
  const maxLevels = gridInfo?.levels_max || condLevels.length || 3
  const positions = gridInfo?.positions || []
  const filledSet = new Set(positions.map(p => p.level))

  // Construire niveaux (remplis depuis gridInfo + enrichis depuis conditions)
  const allLevels = []
  for (let i = 0; i < maxLevels; i++) {
    const filled = positions.find(p => p.level === i)
    const cond = condLevels[i]
    allLevels.push({
      index: i,
      filled: !!filled,
      entry_price: filled?.entry_price || (cond?.value != null ? Number(cond.value) : null),
      quantity: filled?.quantity || null,
      direction: filled?.direction || gridInfo?.direction || (cond?.name?.includes('short') ? 'short' : 'long'),
      distance_pct: cond?.distance_pct ?? null,
      proximity: cond?.proximity || 'far',
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

  const stratLabel = gridInfo?.strategy || strategyName || 'grid'

  return (
    <div className="scanner-expand">
      {/* Gates en haut */}
      {gates.length > 0 && (
        <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
          {gates.map((gate, i) => (
            <div key={i} style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '4px 10px', borderRadius: 6, fontSize: 12, fontWeight: 600,
              background: gate.met ? 'rgba(0, 230, 138, 0.12)' : 'rgba(255, 255, 255, 0.04)',
              color: gate.met ? 'var(--accent)' : 'var(--muted)',
              border: `1px solid ${gate.met ? 'rgba(0, 230, 138, 0.25)' : 'rgba(255, 255, 255, 0.08)'}`,
            }}>
              <span>{gate.name}</span>
              <span style={{ fontWeight: 700 }}>
                {gate.value === 'UP' && '\u2191'}
                {gate.value === 'DOWN' && '\u2193'}
                {gate.value !== 'UP' && gate.value !== 'DOWN' && ''} {gate.value}
              </span>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        {/* Résumé grid */}
        <div style={{ flexShrink: 0, textAlign: 'center', minWidth: 90 }}>
          <div className="grid-detail-summary">
            {hasPosition ? (
              <>
                <div className="grid-detail-ratio">
                  <span className={filledSet.size > 0 ? 'pnl-pos' : 'muted'}>{filledSet.size}</span>
                  <span className="muted">/{maxLevels}</span>
                </div>
                {gridInfo?.unrealized_pnl != null && (
                  <div className={`mono text-xs ${gridInfo.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ marginTop: 4 }}>
                    {gridInfo.unrealized_pnl >= 0 ? '+' : ''}{gridInfo.unrealized_pnl.toFixed(2)}$
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="grid-detail-ratio">
                  <span className="muted">0</span>
                  <span className="muted">/{maxLevels}</span>
                </div>
                <div className="text-xs muted" style={{ marginTop: 4 }}>pas de position</div>
              </>
            )}
            <div className="text-xs muted" style={{ marginTop: 2 }}>
              {STRATEGY_LABELS[stratLabel] || stratLabel}
            </div>
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
                <span className="mono" style={{ minWidth: 80, textAlign: 'right' }}>
                  {lvl.entry_price ? formatPrice(lvl.entry_price) : '--'}
                </span>
                <span className={`text-xs ${lvl.filled ? '' : 'muted'}`} style={{ minWidth: 50 }}>
                  {lvl.filled ? lvl.direction.toUpperCase() : (lvl.entry_price ? lvl.direction.toUpperCase() : 'attente')}
                </span>
                {lvl.distance_pct != null && (
                  <span className="mono text-xs" style={{
                    minWidth: 50, textAlign: 'right',
                    color: PROXIMITY_COLORS[lvl.proximity] || 'var(--muted)',
                  }}>
                    {lvl.distance_pct >= 0 ? '+' : ''}{lvl.distance_pct.toFixed(1)}%
                  </span>
                )}
              </div>
            ))}
          </div>

          {/* TP / SL (mode position uniquement) */}
          {hasPosition && (
            <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
              {gridInfo?.tp_price != null && (
                <span className="text-xs">
                  <span className="muted">TP </span>
                  <span className="mono" style={{ color: 'var(--accent)' }}>
                    {formatPrice(gridInfo.tp_price)}
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
                    {formatPrice(gridInfo.sl_price)}
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
          )}
        </div>

        {/* Indicateurs */}
        <div style={{ flexShrink: 0, minWidth: 130 }}>
          <div className="text-xs dim" style={{ marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Indicateurs
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {price != null && (
              <IndicatorRow label="Prix" value={formatPrice(price)} />
            )}
            {gridInfo?.avg_entry != null && gridInfo.avg_entry > 0 && (
              <IndicatorRow
                label="Avg entry"
                value={formatPrice(gridInfo.avg_entry)}
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
