/**
 * OverviewPage — Dashboard d'accueil enrichi.
 * Sprint 63a — KPI cards, equity curve, drawdown, top/bottom assets, regime, activite.
 */
import { useMemo } from 'react'
import { useApi } from '../hooks/useApi'
import { useStrategyContext } from '../contexts/StrategyContext'
import ActivePositions from './ActivePositions'
import CollapsibleCard from './CollapsibleCard'
import EnhancedEquityCurve from './EnhancedEquityCurve'
import DrawdownChart from './DrawdownChart'
import './OverviewPage.css'

const REGIME_COLORS = {
  BULL: 'var(--accent)',
  BEAR: 'var(--red)',
  RANGE: 'var(--yellow)',
  CRASH: 'var(--red)',
}

const REGIME_LABELS = {
  BULL: 'Haussier',
  BEAR: 'Baissier',
  RANGE: 'Range',
  CRASH: 'Crash',
}

function formatPnl(value) {
  if (value == null) return '--'
  const abs = Math.abs(value)
  if (abs >= 10000) return `${(value / 1000).toFixed(1)}K$`
  return `${value.toFixed(2)}$`
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

function KpiCard({ label, value, format, colorize, invert }) {
  let displayValue = '--'
  if (value != null) {
    if (format === '$') displayValue = formatPnl(value)
    else if (format === '%') displayValue = `${value.toFixed(1)}%`
    else displayValue = `${value}`
  }

  let colorClass = ''
  if (colorize && value != null) {
    if (invert) {
      colorClass = value < -10 ? 'negative' : ''
    } else {
      colorClass = value >= 0 ? 'positive' : 'negative'
    }
  }

  return (
    <div className="overview-kpi-card">
      <div className="overview-kpi-label">{label}</div>
      <div className={`overview-kpi-value ${colorClass}`}>{displayValue}</div>
    </div>
  )
}

export default function OverviewPage({ wsData }) {
  const { setActiveStrategy } = useStrategyContext()

  // Data fetching
  const { data: pnlData } = useApi('/api/journal/daily-pnl-summary', 60000)
  const { data: statsData } = useApi('/api/journal/live-stats?period=all', 30000)
  const { data: perAssetData } = useApi('/api/journal/live-per-asset?period=all', 30000)
  const { data: regimeData } = useApi('/api/regime/snapshot', 60000)
  const { data: eventsData } = useApi('/api/journal/events?limit=3', 30000)

  const stats = statsData?.stats
  const exchangeBalance = wsData?.executor?.exchange_balance
  const snap = regimeData?.snapshot
  const events = eventsData?.events || []

  // Top 3 / Bottom 3
  const { top3, bottom3 } = useMemo(() => {
    const assets = perAssetData?.per_asset || []
    if (!assets.length) return { top3: [], bottom3: [] }
    const sorted = [...assets].sort((a, b) => b.total_pnl - a.total_pnl)
    return {
      top3: sorted.slice(0, 3),
      bottom3: sorted.slice(-3).reverse(),
    }
  }, [perAssetData])

  // Strategies table
  const allowedLive = wsData?.executor?.selector?.allowed_strategies || []
  const rows = useMemo(() => {
    const strategies = wsData?.strategies || {}
    const gridPositions = wsData?.grid_state?.grid_positions || {}
    return Object.keys(strategies).map(name => {
      const grids = Object.values(gridPositions).filter(g => g.strategy === name)
      const assetsCount = grids.length
      const totalLevelsOpen = grids.reduce((s, g) => s + (g.levels_open || 0), 0)
      const totalUpnl = grids.reduce((s, g) => s + (g.unrealized_pnl || 0), 0)
      const totalMargin = grids.reduce((s, g) => s + (g.margin_used || 0), 0)
      const strat = strategies[name] || {}
      const realizedPnl = strat.net_pnl || 0
      const totalPnl = realizedPnl + totalUpnl
      const isLive = allowedLive.includes(name)
      return { name, assetsCount, totalLevelsOpen, totalPnl, totalMargin, isLive }
    })
  }, [wsData?.strategies, wsData?.grid_state?.grid_positions, allowedLive])

  return (
    <>
      {/* KPI Cards */}
      <div className="overview-kpi-grid">
        <KpiCard label="Capital" value={exchangeBalance} format="$" />
        <KpiCard label="P&L Total" value={pnlData?.total_pnl} format="$" colorize />
        <KpiCard label="Win Rate" value={stats?.win_rate} format="%" />
        <KpiCard label="Max DD" value={stats?.max_drawdown_pct != null ? -Math.abs(stats.max_drawdown_pct) : null} format="%" colorize invert />
      </div>

      {/* Equity Curve */}
      <CollapsibleCard title="Equity Curve" defaultOpen={true} storageKey="overview-equity">
        <EnhancedEquityCurve defaultDays={30} height={250} />
        <div style={{ marginTop: 8 }}>
          <DrawdownChart days={30} killSwitchPct={45} height={120} />
        </div>
      </CollapsibleCard>

      {/* Top/Bottom Assets */}
      {(top3.length > 0 || bottom3.length > 0) && (
        <CollapsibleCard title="Performance par Asset" defaultOpen={true} storageKey="overview-assets">
          <div className="overview-assets-grid">
            <div className="overview-asset-section">
              <h3>Top gagnants</h3>
              {top3.map(a => (
                <div key={a.symbol} className="overview-asset-row">
                  <span className="overview-asset-symbol">{a.symbol}</span>
                  <span className="overview-asset-stat">{a.total_trades} trades</span>
                  <span className="overview-asset-stat">{a.win_rate}%</span>
                  <span className="overview-asset-pnl pnl-pos">+{a.total_pnl.toFixed(2)}$</span>
                </div>
              ))}
            </div>
            <div className="overview-asset-section">
              <h3>Top perdants</h3>
              {bottom3.map(a => (
                <div key={a.symbol} className="overview-asset-row">
                  <span className="overview-asset-symbol">{a.symbol}</span>
                  <span className="overview-asset-stat">{a.total_trades} trades</span>
                  <span className="overview-asset-stat">{a.win_rate}%</span>
                  <span className="overview-asset-pnl pnl-neg">{a.total_pnl.toFixed(2)}$</span>
                </div>
              ))}
            </div>
          </div>
          <span
            className="overview-see-all"
            onClick={() => setActiveStrategy('overview')}
            role="button"
            tabIndex={0}
          >
            Voir tout dans Journal &rarr;
          </span>
        </CollapsibleCard>
      )}

      {/* Regime + Activite */}
      <div className="overview-bottom-grid">
        <CollapsibleCard title="Regime BTC" defaultOpen={true} storageKey="overview-regime">
          {snap ? (
            <div>
              <div className="overview-regime-inline">
                <span className="overview-regime-dot" style={{ background: REGIME_COLORS[snap.regime] || 'var(--text-muted)' }} />
                <span className="overview-regime-label" style={{ color: REGIME_COLORS[snap.regime] || 'var(--text-muted)' }}>
                  {REGIME_LABELS[snap.regime] || snap.regime}
                </span>
                <span className="overview-regime-days">{snap.regime_days}j</span>
              </div>
              <div className="overview-regime-metrics">
                <span>BTC 30j : <span className={snap.btc_change_30d_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                  {snap.btc_change_30d_pct >= 0 ? '+' : ''}{snap.btc_change_30d_pct?.toFixed(1)}%
                </span></span>
                <span>ATR : {snap.btc_atr_14d_pct?.toFixed(2)}%</span>
                <span>Lev : x{snap.suggested_leverage}</span>
              </div>
            </div>
          ) : (
            <div className="empty-state">Regime non disponible</div>
          )}
        </CollapsibleCard>

        <CollapsibleCard title="Activite recente" defaultOpen={true} storageKey="overview-activity">
          {events.length === 0 ? (
            <div className="empty-state">Aucun evenement</div>
          ) : (
            <div>
              {events.map((e, i) => {
                const isOpen = e.event_type === 'OPEN'
                return (
                  <div key={i} className="overview-event-card">
                    <div className="overview-event-left">
                      <span
                        className="overview-event-type"
                        style={{
                          background: isOpen ? 'var(--accent-dim)' : 'var(--red-dim)',
                          color: isOpen ? 'var(--accent)' : 'var(--red)',
                        }}
                      >
                        {isOpen ? 'OPEN' : 'CLOSE'}
                      </span>
                      <span className="overview-event-symbol">{e.symbol}</span>
                      {e.direction && (
                        <span className={`badge ${e.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
                          {e.direction}
                        </span>
                      )}
                    </div>
                    <span className="overview-event-time">{timeAgo(e.timestamp)}</span>
                  </div>
                )
              })}
            </div>
          )}
        </CollapsibleCard>
      </div>

      {/* Strategies (collapsed) */}
      <CollapsibleCard title="Strategies actives" defaultOpen={false} storageKey="overview-strategies">
        {rows.length === 0 ? (
          <div className="empty-state">En attente de donnees...</div>
        ) : (
          <table className="scanner-table">
            <thead>
              <tr>
                <th>Strategie</th>
                <th>Mode</th>
                <th style={{ textAlign: 'right' }}>Assets</th>
                <th style={{ textAlign: 'right' }}>Grids</th>
                <th style={{ textAlign: 'right' }}>P&L</th>
                <th style={{ textAlign: 'right' }}>Marge</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr
                  key={r.name}
                  className="scanner-row"
                  style={{ cursor: 'pointer' }}
                  onClick={() => setActiveStrategy(r.name)}
                >
                  <td style={{ fontWeight: 700 }}>{r.name}</td>
                  <td>
                    <span className={`badge ${r.isLive ? 'badge-active' : 'badge-simulation'}`}>
                      {r.isLive ? 'LIVE' : 'PAPER'}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right' }} className="mono">{r.assetsCount}</td>
                  <td style={{ textAlign: 'right' }} className="mono">{r.totalLevelsOpen}</td>
                  <td style={{ textAlign: 'right' }}>
                    <span className={`mono ${r.totalPnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
                      {r.totalPnl >= 0 ? '+' : ''}{formatPnl(r.totalPnl)}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right' }} className="mono">{r.totalMargin.toFixed(0)}$</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </CollapsibleCard>

      {/* Positions */}
      <CollapsibleCard title="Toutes les positions" defaultOpen={false} storageKey="overview-positions">
        <ActivePositions wsData={wsData} />
      </CollapsibleCard>
    </>
  )
}
