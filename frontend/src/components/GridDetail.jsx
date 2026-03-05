/**
 * GridDetail — Détail d'un asset grid au clic dans le Scanner.
 * Aligné strictement sur les colonnes du tableau parent via une table interne.
 */
import Tooltip from './Tooltip'
import { formatPrice } from '../utils/format'
import GridChart from './GridChart'

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

export default function GridDetail({ symbol, gridInfo, indicators = {}, regime, price, conditions = [], strategyName, sparkline = [], hasMono, hasGrid }) {
  const hasPosition = !!gridInfo
  const condLevels = (conditions || []).filter(c => !c.gate)
  const gates = (conditions || []).filter(c => c.gate)

  const maxLevels = gridInfo?.levels_max || condLevels.length || 3
  const positions = gridInfo?.positions || []
  const filledSet = new Set(positions.map(p => p.level))

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

  const rsi = indicators?.rsi_14
  const adx = indicators?.adx
  const atrPct = indicators?.atr_pct
  const distSma = (gridInfo?.tp_price && gridInfo?.current_price && gridInfo.tp_price > 0)
    ? ((gridInfo.current_price - gridInfo.tp_price) / gridInfo.tp_price * 100)
    : null

  const stratLabel = gridInfo?.strategy || strategyName || 'grid'

  const chartLevels = allLevels.map(l => ({
    price: l.entry_price,
    filled: l.filled,
    direction: l.direction
  }))

  return (
    <div className="scanner-expand" style={{ padding: '12px 0' }}>
      {/* Gates */}
      {gates.length > 0 && (
        <div style={{ display: 'flex', gap: 8, marginBottom: 12, paddingLeft: 16 }}>
          {gates.map((gate, i) => (
            <div key={i} style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
              background: gate.met ? 'rgba(0, 230, 138, 0.12)' : 'rgba(255, 255, 255, 0.04)',
              color: gate.met ? 'var(--accent)' : 'var(--muted)',
              border: `1px solid ${gate.met ? 'rgba(0, 230, 138, 0.25)' : 'rgba(255, 255, 255, 0.08)'}`,
            }}>
              <span>{gate.name}</span>
              <span style={{ fontWeight: 700 }}>{gate.value}</span>
            </div>
          ))}
        </div>
      )}

      {/* Table interne pour alignement parfait avec le header du Scanner */}
      <table style={{ tableLayout: 'fixed', width: '100%', borderCollapse: 'collapse', border: 'none' }}>
        <tbody>
          <tr>
            {/* Colonne Actif (14%) - Résumé */}
            <td style={{ width: '14%', verticalAlign: 'top', padding: '0 8px', border: 'none' }}>
              <div className="grid-detail-summary">
                <div className="grid-detail-ratio">
                  <span className={filledSet.size > 0 ? 'pnl-pos' : 'muted'}>{filledSet.size}</span>
                  <span className="muted">/{maxLevels}</span>
                </div>
                {hasPosition && gridInfo?.unrealized_pnl != null && (
                  <div className={`mono text-xs ${gridInfo.unrealized_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ marginTop: 4 }}>
                    {gridInfo.unrealized_pnl >= 0 ? '+' : ''}{gridInfo.unrealized_pnl.toFixed(2)}$
                  </div>
                )}
                <div className="text-xs muted" style={{ marginTop: 2 }}>
                  {STRATEGY_LABELS[stratLabel] || stratLabel}
                </div>
              </div>
            </td>

            {/* Colonnes Prix(10), Var(8), Dir(7) - Niveaux Grid */}
            <td colSpan="3" style={{ width: '25%', verticalAlign: 'top', padding: '0 8px', border: 'none' }}>
              <div className="grid-detail-levels">
                {allLevels.map(lvl => (
                  <div key={lvl.index} className={`grid-level ${lvl.filled ? 'grid-level--filled' : 'grid-level--pending'}`}>
                    <span className="grid-level-name">Lvl {lvl.index + 1}</span>
                    <div className="grid-level-bar">
                      <div className={`grid-level-fill ${lvl.filled ? 'grid-level-fill--green' : 'grid-level-fill--red'}`} 
                           style={{ width: lvl.filled ? '100%' : '0%' }} />
                    </div>
                    <span className="mono" style={{ minWidth: 60, textAlign: 'right', fontSize: '11px' }}>
                      {lvl.entry_price ? formatPrice(lvl.entry_price) : '--'}
                    </span>
                    {lvl.distance_pct != null && (
                      <span className="mono text-xs" style={{ 
                        marginLeft: 'auto', color: PROXIMITY_COLORS[lvl.proximity] || 'var(--muted)'
                      }}>
                        {lvl.distance_pct >= 0 ? '+' : ''}{lvl.distance_pct.toFixed(1)}%
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </td>

            {/* Colonne Trend (12%) - Graphique GridChart */}
            <td style={{ width: '12%', verticalAlign: 'top', padding: '0 8px', border: 'none' }}>
              <div style={{ height: 80, minWidth: 160 }}>
                <GridChart
                  symbol={symbol}
                  data={sparkline}
                  levels={chartLevels}
                  currentPrice={price}
                  tpPrice={gridInfo?.tp_price}
                  slPrice={gridInfo?.sl_price}
                  width="100%"
                  height="100%"
                />
              </div>
            </td>

            {/* Colonne Score (7%) - SI hasMono */}
            {hasMono && <td style={{ width: '7%', border: 'none' }} />}

            {/* Colonne Grade (7%) */}
            <td style={{ width: '7%', border: 'none' }} />

            {/* Colonne Signaux (20%) - SI hasMono */}
            {hasMono && <td style={{ width: '20%', border: 'none' }} />}

            {/* Colonne Dist.SMA (9%) - Targets TP/SL */}
            <td style={{ width: '9%', verticalAlign: 'top', padding: '0 8px', border: 'none' }}>
              {hasPosition && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {gridInfo?.tp_price != null && (
                    <div className="text-xs">
                      <span className="muted">TP: </span>
                      <span className="mono pnl-pos">{formatPrice(gridInfo.tp_price)}</span>
                    </div>
                  )}
                  {gridInfo?.sl_price != null && (
                    <div className="text-xs">
                      <span className="muted">SL: </span>
                      <span className="mono pnl-neg">{formatPrice(gridInfo.sl_price)}</span>
                    </div>
                  )}
                  {gridInfo?.margin_used != null && (
                    <div className="text-xs">
                      <span className="muted">Margin: </span>
                      <span className="mono">{gridInfo.margin_used.toFixed(1)}$</span>
                    </div>
                  )}
                </div>
              )}
            </td>

            {/* Colonne Grid (7%) - Indicateurs */}
            <td style={{ width: '7%', verticalAlign: 'top', padding: '0 8px', border: 'none' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {price != null && <IndicatorRow label="Prix" value={formatPrice(price)} />}
                {rsi != null && <IndicatorRow label="RSI" value={Number(rsi).toFixed(1)} color={rsi < 30 ? 'var(--accent)' : rsi > 70 ? 'var(--red)' : null} />}
                {adx != null && <IndicatorRow label="ADX" value={Number(adx).toFixed(1)} />}
                {atrPct != null && <IndicatorRow label="ATR%" value={`${Number(atrPct).toFixed(2)}%`} />}
                {regime && <IndicatorRow label="Rég." value={regime} badge />}
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function IndicatorRow({ label, value, color, badge }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10 }}>
      <span className="muted" style={{ width: 32, flexShrink: 0 }}>{label}</span>
      {badge ? (
        <span className="badge badge-ranging" style={{ padding: '0 4px', fontSize: '9px' }}>{value}</span>
      ) : (
        <span className="mono" style={{ color: color || 'var(--text-primary)', marginLeft: 'auto' }}>{value}</span>
      )}
    </div>
  )
}
