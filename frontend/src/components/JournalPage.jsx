/**
 * JournalPage — Journal de Trading avec 2 onglets Live / Paper (Sprint 45).
 * Chaque onglet a ses propres stats, historique, equity curve et performance par asset.
 */
import { useState, useMemo, useEffect } from 'react'
import { useApi } from '../hooks/useApi'
import { useStrategyContext } from '../contexts/StrategyContext'
import { formatPrice } from '../utils/format'
import CollapsibleCard from './CollapsibleCard'
import './JournalPage.css'

const PERIODS = [
  { id: 'today', label: "Aujourd'hui" },
  { id: '7d', label: '7j' },
  { id: '30d', label: '30j' },
  { id: 'all', label: 'Tout' },
]

const EXIT_LABELS = {
  sl: 'SL', tp: 'TP', tp_global: 'TP global', sl_global: 'SL global',
  force_close: 'Force close', signal_exit: 'Signal', regime_change: 'Regime', end_of_data: 'Fin',
  tp_close: 'TP', sl_close: 'SL', cycle_close: 'Cycle',
}

const EXIT_BADGE_CLASS = {
  sl: 'badge-stopped', sl_global: 'badge-stopped', sl_close: 'badge-stopped',
  tp: 'badge-active', tp_global: 'badge-active', tp_close: 'badge-active',
  signal_exit: 'badge-simulation', regime_change: 'badge-trending',
  cycle_close: 'badge-active',
}

function formatDuration(entryTime, exitTime) {
  if (!entryTime || !exitTime) return '--'
  const ms = new Date(exitTime) - new Date(entryTime)
  const mins = Math.floor(ms / 60000)
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  const remMins = mins % 60
  if (hours < 24) return `${hours}h${remMins > 0 ? remMins + 'm' : ''}`
  const days = Math.floor(hours / 24)
  return `${days}j ${hours % 24}h`
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

export default function JournalPage({ wsData, onTabChange }) {
  const [period, setPeriod] = useState('all')
  const [source, setSource] = useState('live') // 'live' ou 'paper'

  // Detecter si des trades live existent
  const { data: liveCountData } = useApi('/api/journal/live-count', 60000)
  const hasLiveTrades = (liveCountData?.count || 0) > 0

  // Si pas de trades live, basculer sur paper
  useEffect(() => {
    if (liveCountData && !hasLiveTrades && source === 'live') {
      setSource('paper')
    }
  }, [liveCountData, hasLiveTrades]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="journal-page">
      <div className="journal-header">
        <h2>Journal de Trading</h2>
        <div className="journal-header-controls">
          <div className="source-toggle">
            <button
              className={`source-btn ${source === 'live' ? 'active live' : ''}`}
              onClick={() => setSource('live')}
              disabled={!hasLiveTrades}
              title={!hasLiveTrades ? 'Aucun trade live' : ''}
            >
              Live
            </button>
            <button
              className={`source-btn ${source === 'paper' ? 'active paper' : ''}`}
              onClick={() => setSource('paper')}
            >
              Paper
            </button>
          </div>
          <div className="period-selector">
            {PERIODS.map(p => (
              <button
                key={p.id}
                className={`period-btn ${period === p.id ? 'active' : ''}`}
                onClick={() => setPeriod(p.id)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {source === 'live' ? (
        <LiveJournal period={period} wsData={wsData} />
      ) : (
        <PaperJournal period={period} wsData={wsData} />
      )}
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════════════
 * LIVE JOURNAL (Sprint 45)
 * ════════════════════════════════════════════════════════════════════════ */

function LiveJournal({ period, wsData }) {
  const [strategy, setStrategy] = useState('')

  // Sprint 46b : contexte stratégie global (StrategyBar dans le header)
  const { strategyFilter } = useStrategyContext()
  // Le contexte global prime sur le dropdown local
  const effectiveStrategy = strategyFilter || strategy || null

  // Détecter les stratégies live disponibles
  const { data: tradesData } = useApi(`/api/journal/live-trades?period=all&limit=500`, 60000)
  const liveStrategies = useMemo(() => {
    const trades = tradesData?.trades || []
    return [...new Set(trades.map(t => t.strategy_name).filter(Boolean))].sort()
  }, [tradesData])

  const stratParam = effectiveStrategy

  return (
    <>
      {/* Filtre stratégie local — masqué si une stratégie globale est sélectionnée */}
      {liveStrategies.length > 1 && !strategyFilter && (
        <div className="journal-strategy-filter">
          <select value={strategy} onChange={e => setStrategy(e.target.value)}>
            <option value="">Toutes les strategies</option>
            {liveStrategies.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      )}

      <CollapsibleCard title="Stats Live" defaultOpen={true} storageKey="journal-live-stats">
        <LiveStatsOverview period={period} strategy={stratParam} />
      </CollapsibleCard>

      <CollapsibleCard title="Positions & Trades" defaultOpen={true} storageKey="journal-live-positions">
        <LivePositionsAndTrades wsData={wsData} period={period} />
      </CollapsibleCard>

      <CollapsibleCard title="P&L Journalier" defaultOpen={true} storageKey="journal-live-daily">
        <LiveDailyPnl strategy={stratParam} />
      </CollapsibleCard>

      <CollapsibleCard title="Equity Curve Live" defaultOpen={true} storageKey="journal-live-equity">
        <LiveEquityCurve strategy={stratParam} />
      </CollapsibleCard>

      <CollapsibleCard title="Ordres Bitget" defaultOpen={true} storageKey="journal-orders">
        <BitgetOrders />
      </CollapsibleCard>

      <CollapsibleCard title="Performance par Asset" defaultOpen={false} storageKey="journal-live-per-asset">
        <LivePerAssetSummary period={period} strategy={stratParam} />
      </CollapsibleCard>
    </>
  )
}

function LiveStatsOverview({ period, strategy }) {
  const stratQ = strategy ? `&strategy=${encodeURIComponent(strategy)}` : ''
  const { data, loading } = useApi(`/api/journal/live-stats?period=${period}${stratQ}`, 30000)
  const stats = data?.stats

  if (loading && !stats) {
    return <div className="empty-state">Chargement des stats live...</div>
  }

  if (!stats || stats.total_trades === 0) {
    return <div className="empty-state">Aucun trade live pour cette periode</div>
  }

  const pnlClass = stats.total_pnl >= 0 ? 'positive' : 'negative'
  const pfClass = stats.profit_factor >= 1.5 ? 'positive' : stats.profit_factor >= 1 ? '' : 'negative'
  const streakIcon = stats.current_streak?.type === 'win' ? '\u{1F525}' : '\u{2744}\u{FE0F}'

  return (
    <div>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">P&L Net</div>
          <div className={`stat-value ${pnlClass}`}>
            {stats.total_pnl >= 0 ? '+' : ''}{stats.total_pnl.toFixed(2)}$
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Win Rate</div>
          <div className={`stat-value ${stats.win_rate >= 50 ? 'positive' : 'negative'}`}>
            {stats.win_rate}%
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Profit Factor</div>
          <div className={`stat-value ${pfClass}`}>
            {stats.profit_factor}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Max Drawdown</div>
          <div className={`stat-value ${stats.max_drawdown_pct != null ? 'negative' : ''}`}>
            {stats.max_drawdown_pct != null ? `${stats.max_drawdown_pct}%` : 'N/A'}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Trades</div>
          <div className="stat-value">{stats.total_trades}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Streak</div>
          <div className="stat-value">
            {streakIcon} {stats.current_streak?.count || 0}
          </div>
        </div>
      </div>

      <div className="stats-secondary">
        {stats.best_trade && (
          <span>Meilleur : <span className="pnl-pos">+{stats.best_trade.pnl.toFixed(2)}$</span> ({stats.best_trade.symbol})</span>
        )}
        {stats.worst_trade && (
          <span>Pire : <span className="pnl-neg">{stats.worst_trade.pnl.toFixed(2)}$</span> ({stats.worst_trade.symbol})</span>
        )}
        <span>Trades/jour : {stats.trades_per_day}</span>
        <span>W/L : {stats.wins}/{stats.losses}</span>
        <span className="live-badge-inline">LIVE</span>
      </div>
    </div>
  )
}

function LivePositionsAndTrades({ wsData, period }) {
  const [subTab, setSubTab] = useState('ouvertes')

  return (
    <div>
      <div className="sub-tabs">
        <button className={`sub-tab ${subTab === 'ouvertes' ? 'active' : ''}`} onClick={() => setSubTab('ouvertes')}>
          Ouvertes
        </button>
        <button className={`sub-tab ${subTab === 'historique' ? 'active' : ''}`} onClick={() => setSubTab('historique')}>
          Historique
        </button>
      </div>

      {subTab === 'ouvertes' && <OpenPositions wsData={wsData} filterSource="LIVE" />}
      {subTab === 'historique' && <LiveTradeHistory period={period} />}
    </div>
  )
}

function LiveTradeHistory({ period }) {
  const [limit, setLimit] = useState(50)
  const [filterSymbol, setFilterSymbol] = useState('')
  const [filterStrategy, setFilterStrategy] = useState('')
  const [sortBy, setSortBy] = useState('timestamp')
  const [sortAsc, setSortAsc] = useState(false)

  const { data, loading } = useApi(
    `/api/journal/live-trades?period=${period}&limit=${limit}`, 15000,
  )
  const allTrades = data?.trades || []

  // Filtrer uniquement les closes (qui ont un P&L)
  const closeTrades = useMemo(() =>
    allTrades.filter(t => t.pnl != null),
    [allTrades],
  )

  const filtered = useMemo(() => {
    let t = closeTrades
    if (filterSymbol) t = t.filter(tr => tr.symbol === filterSymbol)
    if (filterStrategy) t = t.filter(tr => tr.strategy_name === filterStrategy)
    t = [...t].sort((a, b) => {
      const va = a[sortBy] ?? ''
      const vb = b[sortBy] ?? ''
      if (typeof va === 'number') return sortAsc ? va - vb : vb - va
      return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va))
    })
    return t
  }, [closeTrades, filterSymbol, filterStrategy, sortBy, sortAsc])

  const symbols = useMemo(() => [...new Set(closeTrades.map(t => t.symbol).filter(Boolean))].sort(), [closeTrades])
  const strategies = useMemo(() => [...new Set(closeTrades.map(t => t.strategy_name).filter(Boolean))].sort(), [closeTrades])

  const handleSort = (col) => {
    if (sortBy === col) setSortAsc(!sortAsc)
    else { setSortBy(col); setSortAsc(false) }
  }

  if (loading && !data) return <div className="empty-state">Chargement...</div>
  if (closeTrades.length === 0) return <div className="empty-state">Aucun trade live ferme</div>

  return (
    <div>
      <div className="journal-filters">
        <select value={filterSymbol} onChange={e => setFilterSymbol(e.target.value)}>
          <option value="">Tous les symbols</option>
          {symbols.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <select value={filterStrategy} onChange={e => setFilterStrategy(e.target.value)}>
          <option value="">Toutes les strategies</option>
          {strategies.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="journal-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('timestamp')}>Date {sortBy === 'timestamp' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}</th>
              <th>Dir</th>
              <th onClick={() => handleSort('symbol')}>Symbol {sortBy === 'symbol' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}</th>
              <th>Strategie</th>
              <th>Prix</th>
              <th onClick={() => handleSort('pnl')}>P&L Net {sortBy === 'pnl' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}</th>
              <th>P&L %</th>
              <th>Niveaux</th>
              <th>Raison</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((t, i) => {
              const isCycle = t.trade_type === 'cycle_close'
              const rowTitle = isCycle
                ? `Cycle ${t.grid_level || '?'} niveaux — Exit: ${formatPrice(t.price)}`
                : undefined
              return (
                <tr key={i} className={(t.pnl || 0) >= 0 ? 'row-positive' : 'row-negative'} title={rowTitle}>
                  <td className="text-xs muted">
                    {t.timestamp ? new Date(t.timestamp).toLocaleDateString('fr-FR') : '--'}
                  </td>
                  <td><span className={`badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>{t.direction}</span></td>
                  <td style={{ fontWeight: 600, fontSize: 11 }}>{t.symbol || '--'}</td>
                  <td className="text-xs">{t.strategy_name}</td>
                  <td className="mono text-xs">
                    {isCycle ? <span className="muted text-xs">exit </span> : null}{formatPrice(t.price)}
                  </td>
                  <td className={`mono ${(t.pnl || 0) >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
                    {(t.pnl || 0) >= 0 ? '+' : ''}{Number(t.pnl || 0).toFixed(2)}$
                  </td>
                  <td className={`mono text-xs ${(t.pnl_pct || 0) >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
                    {t.pnl_pct != null ? `${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(1)}%` : '--'}
                  </td>
                  <td className="text-xs">{t.grid_level != null ? `${t.grid_level}` : '--'}</td>
                  <td>
                    <span className={`badge ${EXIT_BADGE_CLASS[t.trade_type] || ''}`}>
                      {EXIT_LABELS[t.trade_type] || t.trade_type}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {closeTrades.length >= limit && (
        <button className="load-more-btn" onClick={() => setLimit(prev => prev + 50)}>
          Charger plus de trades...
        </button>
      )}
    </div>
  )
}

function LiveDailyPnl({ strategy }) {
  const stratQ = strategy ? `&strategy=${encodeURIComponent(strategy)}` : ''
  const { data, loading } = useApi(`/api/journal/live-daily-pnl?days=30${stratQ}`, 60000)
  const days = data?.daily_pnl || []

  if (loading && !data) return <div className="empty-state">Chargement...</div>
  if (days.length === 0) return <div className="empty-state">Aucun P&L journalier</div>

  const maxPnl = Math.max(...days.map(d => Math.abs(d.pnl)), 1)

  return (
    <div className="daily-pnl-chart">
      {days.map((d, i) => {
        const pct = (d.pnl / maxPnl) * 100
        const isPos = d.pnl >= 0
        return (
          <div key={i} className="daily-bar-row" title={`${d.day}: ${d.pnl >= 0 ? '+' : ''}${d.pnl.toFixed(2)}$ (${d.trades} trades)`}>
            <span className="daily-bar-label">{d.day.slice(5)}</span>
            <div className="daily-bar-track">
              <div
                className={`daily-bar-fill ${isPos ? 'positive' : 'negative'}`}
                style={{ width: `${Math.abs(pct)}%` }}
              />
            </div>
            <span className={`daily-bar-value mono ${isPos ? 'pnl-pos' : 'pnl-neg'}`}>
              {d.pnl >= 0 ? '+' : ''}{d.pnl.toFixed(2)}$
            </span>
          </div>
        )
      })}
    </div>
  )
}

function LiveEquityCurve({ strategy }) {
  const stratQ = strategy ? `&strategy=${encodeURIComponent(strategy)}` : ''
  const { data, loading } = useApi(`/api/journal/live-equity?days=30${stratQ}`, 60000)
  const points = data?.equity_curve || []

  if (loading && !data) return <div className="empty-state">Chargement...</div>
  if (points.length < 2) return <div className="empty-state">Pas assez de snapshots (min 2h de donnees)</div>

  const equities = points.map(p => p.equity)
  const minEq = Math.min(...equities)
  const maxEq = Math.max(...equities)
  const range = maxEq - minEq || 1

  const width = 600
  const height = 180
  const padX = 0
  const padY = 10

  const pathPoints = points.map((p, i) => {
    const x = padX + (i / (points.length - 1)) * (width - 2 * padX)
    const y = padY + (1 - (p.equity - minEq) / range) * (height - 2 * padY)
    return `${x},${y}`
  })
  const linePath = `M${pathPoints.join(' L')}`

  // Colorer selon tendance
  const first = equities[0]
  const last = equities[equities.length - 1]
  const color = last >= first ? 'var(--accent)' : 'var(--red)'

  // Dates pour axes
  const firstDate = points[0].timestamp ? new Date(points[0].timestamp).toLocaleDateString('fr-FR') : ''
  const lastDate = points[points.length - 1].timestamp ? new Date(points[points.length - 1].timestamp).toLocaleDateString('fr-FR') : ''

  return (
    <div>
      <div className="equity-svg-container">
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height: 180 }}>
          <path d={linePath} fill="none" stroke={color} strokeWidth="2" />
        </svg>
      </div>
      <div className="equity-labels">
        <span className="text-xs muted">{firstDate}</span>
        <span className="text-xs muted">
          {last.toFixed(2)} USDT ({last >= first ? '+' : ''}{(last - first).toFixed(2)}$)
        </span>
        <span className="text-xs muted">{lastDate}</span>
      </div>
    </div>
  )
}

function LivePerAssetSummary({ period, strategy }) {
  const stratQ = strategy ? `&strategy=${encodeURIComponent(strategy)}` : ''
  const { data, loading } = useApi(`/api/journal/live-per-asset?period=${period}${stratQ}`, 30000)
  const [sortBy, setSortBy] = useState('total_pnl')
  const [sortAsc, setSortAsc] = useState(false)

  const assets = data?.per_asset || []

  const sorted = useMemo(() => {
    return [...assets].sort((a, b) => {
      const va = a[sortBy] ?? 0
      const vb = b[sortBy] ?? 0
      return sortAsc ? va - vb : vb - va
    })
  }, [assets, sortBy, sortAsc])

  const handleSort = (col) => {
    if (sortBy === col) setSortAsc(!sortAsc)
    else { setSortBy(col); setSortAsc(false) }
  }

  if (loading && !data) return <div className="empty-state">Chargement...</div>
  if (assets.length === 0) return <div className="empty-state">Aucun trade live</div>

  return (
    <div style={{ overflowX: 'auto' }}>
      <div className="text-xs" style={{ marginBottom: 4, color: 'var(--accent)' }}>
        Donnees live trading
      </div>
      <table className="journal-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('symbol')} style={{ cursor: 'pointer' }}>
              Symbol {sortBy === 'symbol' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th onClick={() => handleSort('total_trades')} style={{ cursor: 'pointer' }}>
              Trades {sortBy === 'total_trades' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th onClick={() => handleSort('win_rate')} style={{ cursor: 'pointer' }}>
              Win Rate {sortBy === 'win_rate' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th onClick={() => handleSort('total_pnl')} style={{ cursor: 'pointer' }}>
              P&L Net {sortBy === 'total_pnl' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th onClick={() => handleSort('avg_pnl')} style={{ cursor: 'pointer' }}>
              P&L Moyen {sortBy === 'avg_pnl' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th>W/L</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((a) => (
            <tr key={a.symbol} className={a.total_pnl >= 0 ? 'row-positive' : 'row-negative'}>
              <td style={{ fontWeight: 600 }}>{a.symbol}</td>
              <td className="mono text-xs">{a.total_trades}</td>
              <td className={`mono text-xs ${a.win_rate >= 50 ? 'pnl-pos' : 'pnl-neg'}`}>
                {a.win_rate}%
              </td>
              <td className={`mono ${a.total_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
                {a.total_pnl >= 0 ? '+' : ''}{a.total_pnl.toFixed(2)}$
              </td>
              <td className={`mono text-xs ${a.avg_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
                {a.avg_pnl >= 0 ? '+' : ''}{a.avg_pnl.toFixed(2)}$
              </td>
              <td className="text-xs">{a.wins}/{a.losses}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════════════
 * PAPER JOURNAL (existant, inchange)
 * ════════════════════════════════════════════════════════════════════════ */

function PaperJournal({ period, wsData }) {
  return (
    <>
      <CollapsibleCard title="Stats Paper" defaultOpen={true} storageKey="journal-stats">
        <StatsOverview period={period} wsData={wsData} />
      </CollapsibleCard>

      <CollapsibleCard title="Positions & Trades" defaultOpen={true} storageKey="journal-positions">
        <PaperPositionsAndTrades wsData={wsData} />
      </CollapsibleCard>

      <CollapsibleCard title="Equity Curve" defaultOpen={true} storageKey="journal-equity">
        <AnnotatedEquityCurve period={period} />
      </CollapsibleCard>

      <CollapsibleCard title="Ordres Bitget" defaultOpen={true} storageKey="journal-orders">
        <BitgetOrders />
      </CollapsibleCard>

      <CollapsibleCard title="Performance par Asset" defaultOpen={false} storageKey="journal-per-asset">
        <PerAssetSummary period={period} />
      </CollapsibleCard>
    </>
  )
}

/* ────────────────────────────────────────────────────────────────────────────
 * Section 1 — Stats Overview (Paper)
 * ──────────────────────────────────────────────────────────────────────── */

function StatsOverview({ period, wsData }) {
  const { data, loading } = useApi(`/api/journal/stats?period=${period}`, 30000)
  const stats = data?.stats

  // Funding costs depuis les strategies WS
  const totalFunding = Object.values(wsData?.strategies || {})
    .reduce((sum, s) => sum + (s.funding_cost || 0), 0)

  if (loading && !stats) {
    return <div className="empty-state">Chargement des stats...</div>
  }

  if (!stats || stats.total_trades === 0) {
    return <div className="empty-state">Aucun trade pour cette periode</div>
  }

  const pnlClass = stats.total_pnl >= 0 ? 'positive' : 'negative'
  const pfClass = stats.profit_factor >= 1.5 ? 'positive' : stats.profit_factor >= 1 ? '' : 'negative'
  const streakIcon = stats.current_streak?.type === 'win' ? '\u{1F525}' : '\u{2744}\u{FE0F}'

  return (
    <div>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">P&L Net</div>
          <div className={`stat-value ${pnlClass}`}>
            {stats.total_pnl >= 0 ? '+' : ''}{stats.total_pnl.toFixed(2)}$
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Win Rate</div>
          <div className={`stat-value ${stats.win_rate >= 50 ? 'positive' : 'negative'}`}>
            {stats.win_rate}%
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Profit Factor</div>
          <div className={`stat-value ${pfClass}`}>
            {stats.profit_factor}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Max Drawdown</div>
          <div className={`stat-value ${stats.max_drawdown_pct != null ? 'negative' : ''}`}>
            {stats.max_drawdown_pct != null ? `${stats.max_drawdown_pct}%` : 'N/A'}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Trades</div>
          <div className="stat-value">{stats.total_trades}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Streak</div>
          <div className="stat-value">
            {streakIcon} {stats.current_streak?.count || 0}
          </div>
        </div>
      </div>

      <div className="stats-secondary">
        {stats.best_trade && (
          <span>Meilleur : <span className="pnl-pos">+{stats.best_trade.pnl.toFixed(2)}$</span> ({stats.best_trade.symbol})</span>
        )}
        {stats.worst_trade && (
          <span>Pire : <span className="pnl-neg">{stats.worst_trade.pnl.toFixed(2)}$</span> ({stats.worst_trade.symbol})</span>
        )}
        <span>Duree moy. : {stats.avg_duration_hours}h</span>
        <span>Trades/jour : {stats.trades_per_day}</span>
        <span>W/L : {stats.wins}/{stats.losses}</span>
        {totalFunding !== 0 && (
          <span>Funding : <span className={totalFunding >= 0 ? 'pnl-pos' : 'pnl-neg'}>{totalFunding >= 0 ? '+' : ''}{totalFunding.toFixed(2)}$</span></span>
        )}
        <span className="paper-badge-inline">PAPER</span>
      </div>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────────
 * Section 2 — Positions & Trades (Paper)
 * ──────────────────────────────────────────────────────────────────────── */

function PaperPositionsAndTrades({ wsData }) {
  const [subTab, setSubTab] = useState('ouvertes')

  return (
    <div>
      <div className="sub-tabs">
        <button className={`sub-tab ${subTab === 'ouvertes' ? 'active' : ''}`} onClick={() => setSubTab('ouvertes')}>
          Ouvertes
        </button>
        <button className={`sub-tab ${subTab === 'historique' ? 'active' : ''}`} onClick={() => setSubTab('historique')}>
          Historique
        </button>
      </div>

      {subTab === 'ouvertes' && <OpenPositions wsData={wsData} />}
      {subTab === 'historique' && <TradeHistorySection />}
    </div>
  )
}

function OpenPositions({ wsData, filterSource }) {
  const [expandedSymbol, setExpandedSymbol] = useState(null)
  const simPositions = wsData?.simulator_positions || []
  const execPositions = wsData?.executor?.positions || (wsData?.executor?.position ? [wsData.executor.position] : [])
  const execGrids = wsData?.executor?.grid_states || {}
  const gridState = wsData?.grid_state || null
  const prices = wsData?.prices || {}

  const grids = gridState?.grid_positions ? Object.values(gridState.grid_positions) : []
  // LIVE grids depuis executor get_status()
  const liveGrids = Object.values(execGrids).map(gs => ({ ...gs, _source: 'LIVE', _type: 'grid' }))
  const monoSim = simPositions.filter(p => p.type !== 'grid')
  // Filtrer les exec mono (exclure ceux qui sont dans les grids LIVE)
  const monoExec = execPositions.filter(p => p.type !== 'grid')

  let allPositions = [
    ...liveGrids,
    ...grids.map(g => ({ ...g, _source: 'PAPER', _type: 'grid' })),
    ...monoExec.map(p => ({ ...p, _source: 'LIVE', _type: 'mono' })),
    ...monoSim.map(p => ({ ...p, _source: 'PAPER', _type: 'mono' })),
  ]

  // Filtre par source si specifie
  if (filterSource) {
    allPositions = allPositions.filter(p => p._source === filterSource)
  }

  const hasPositions = allPositions.length > 0

  if (!hasPositions) {
    return <div className="empty-state">Aucune position ouverte{filterSource ? ` (${filterSource})` : ''}</div>
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="journal-table">
        <thead>
          <tr>
            <th></th>
            <th>Dir</th>
            <th>Symbol</th>
            <th>Strategie</th>
            <th>Entry</th>
            <th>Qty</th>
            <th>Notionnel</th>
            <th>P&L latent</th>
            <th>P&L %</th>
            <th>Duree</th>
            <th>Niveaux</th>
          </tr>
        </thead>
        <tbody>
          {allPositions.map((pos, i) => {
            const isGrid = pos._type === 'grid'
            const symbol = pos.symbol || '--'
            const direction = pos.direction || 'LONG'
            const isLong = direction === 'LONG'
            const entryPrice = isGrid ? pos.avg_entry : pos.entry_price
            // total_quantity absent du grid_state -> calculer depuis les niveaux
            const qty = isGrid
              ? (pos.positions || []).reduce((s, p) => s + (p.quantity || 0), 0)
              : pos.quantity
            const notional = entryPrice && qty ? (entryPrice * qty).toFixed(0) : '--'
            const strategy = isGrid ? pos.strategy : (pos.strategy_name || pos.strategy || '')
            const entryTime = isGrid
              ? (pos.entry_time || pos.positions?.[0]?.entry_time)
              : pos.entry_time

            // Prix courant : les prix WS sont en format spot (BTC/USDT),
            // mais les positions live sont en format futures (BTC/USDT:USDT)
            const spotSymbol = symbol.replace(/:USDT$/, '')
            let pnl = null
            if (isGrid) {
              pnl = pos.unrealized_pnl
            } else {
              const currentPrice = prices[spotSymbol]?.last
              if (currentPrice != null && entryPrice != null && qty != null) {
                pnl = isLong
                  ? (currentPrice - entryPrice) * qty
                  : (entryPrice - currentPrice) * qty
              }
            }

            // P&L % — leverage depuis la position ou config par defaut (3)
            const leverage = pos.leverage || 3
            const margin = entryPrice && qty && leverage ? (entryPrice * qty / leverage) : 0
            const pnlPct = pnl != null && margin > 0 ? (pnl / margin * 100) : null

            const pnlClass = pnl != null ? (pnl >= 0 ? 'row-positive' : 'row-negative') : ''
            const levelsStr = isGrid ? `${pos.levels_open || 0}/${pos.levels_max || '?'}` : '--'
            const expanded = expandedSymbol === `${pos._source}-${symbol}-${i}`

            return [
              <tr key={`pos-${i}`} className={pnlClass}
                  style={{ cursor: isGrid ? 'pointer' : 'default' }}
                  onClick={() => isGrid && setExpandedSymbol(expanded ? null : `${pos._source}-${symbol}-${i}`)}>
                <td>
                  <span className={`badge ${pos._source === 'LIVE' ? 'badge-active' : 'badge-simulation'}`} style={{ fontSize: 9 }}>
                    {pos._source}
                  </span>
                </td>
                <td>
                  <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`}>{direction}</span>
                </td>
                <td style={{ fontWeight: 600 }}>{symbol}</td>
                <td className="text-xs">{strategy}</td>
                <td className="mono text-xs">{formatPrice(entryPrice)}</td>
                <td className="mono text-xs">{qty != null ? Number(qty).toFixed(4) : '--'}</td>
                <td className="mono text-xs">{notional}$</td>
                <td className={`mono text-xs ${pnl != null ? (pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : ''}`}>
                  {pnl != null ? `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}$` : '--'}
                </td>
                <td className={`mono text-xs ${pnlPct != null ? (pnlPct >= 0 ? 'pnl-pos' : 'pnl-neg') : ''}`}>
                  {pnlPct != null ? `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(1)}%` : '--'}
                </td>
                <td className="text-xs muted">{timeAgo(entryTime)}</td>
                <td className="text-xs">{levelsStr}</td>
              </tr>,
              ...(expanded && isGrid && pos.positions ? pos.positions.map((lv, li) => (
                <tr key={`pos-${i}-lv-${li}`} className="grid-expand-row">
                  <td colSpan={3}></td>
                  <td className="text-xs muted">Niv.{lv.level + 1}</td>
                  <td className="mono text-xs">{formatPrice(lv.entry_price)}</td>
                  <td className="mono text-xs">{Number(lv.quantity).toFixed(4)}</td>
                  <td colSpan={5} className="text-xs muted">{lv.entry_time ? new Date(lv.entry_time).toLocaleString('fr-FR') : ''}</td>
                </tr>
              )) : []),
            ]
          }).flat()}
        </tbody>
      </table>
    </div>
  )
}

function TradeHistorySection() {
  const [limit, setLimit] = useState(50)
  const [filterSymbol, setFilterSymbol] = useState('')
  const [filterStrategy, setFilterStrategy] = useState('')
  const [sortBy, setSortBy] = useState('exit_time')
  const [sortAsc, setSortAsc] = useState(false)

  const { data, loading } = useApi(`/api/simulator/trades?limit=${limit}`, 10000)
  const allTrades = data?.trades || []

  const filtered = useMemo(() => {
    let t = allTrades
    if (filterSymbol) t = t.filter(tr => tr.symbol === filterSymbol)
    if (filterStrategy) t = t.filter(tr => tr.strategy === filterStrategy)
    // Sort
    t = [...t].sort((a, b) => {
      const va = a[sortBy] ?? ''
      const vb = b[sortBy] ?? ''
      if (typeof va === 'number') return sortAsc ? va - vb : vb - va
      return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va))
    })
    return t
  }, [allTrades, filterSymbol, filterStrategy, sortBy, sortAsc])

  const symbols = useMemo(() => [...new Set(allTrades.map(t => t.symbol).filter(Boolean))].sort(), [allTrades])
  const strategies = useMemo(() => [...new Set(allTrades.map(t => t.strategy).filter(Boolean))].sort(), [allTrades])

  const handleSort = (col) => {
    if (sortBy === col) setSortAsc(!sortAsc)
    else { setSortBy(col); setSortAsc(false) }
  }

  if (loading && !data) return <div className="empty-state">Chargement...</div>
  if (allTrades.length === 0) return <div className="empty-state">Aucun trade ferme</div>

  return (
    <div>
      <div className="journal-filters">
        <select value={filterSymbol} onChange={e => setFilterSymbol(e.target.value)}>
          <option value="">Tous les symbols</option>
          {symbols.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <select value={filterStrategy} onChange={e => setFilterStrategy(e.target.value)}>
          <option value="">Toutes les strategies</option>
          {strategies.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="journal-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('exit_time')}>Date {sortBy === 'exit_time' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}</th>
              <th>Dir</th>
              <th onClick={() => handleSort('symbol')}>Symbol {sortBy === 'symbol' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}</th>
              <th>Strategie</th>
              <th>Entry &rarr; Exit</th>
              <th onClick={() => handleSort('net_pnl')}>P&L Net {sortBy === 'net_pnl' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}</th>
              <th>P&L %</th>
              <th>Duree</th>
              <th>Raison</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((t, i) => {
              const pnlPct = t.entry_price ? ((t.net_pnl / (t.entry_price * t.quantity)) * 100).toFixed(1) : '--'
              return (
                <tr key={i} className={t.net_pnl >= 0 ? 'row-positive' : 'row-negative'}>
                  <td className="text-xs muted">{t.exit_time ? new Date(t.exit_time).toLocaleDateString('fr-FR') : '--'}</td>
                  <td><span className={`badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>{t.direction}</span></td>
                  <td style={{ fontWeight: 600, fontSize: 11 }}>{t.symbol || '--'}</td>
                  <td className="text-xs">{t.strategy}</td>
                  <td className="mono text-xs">{formatPrice(t.entry_price)} &rarr; {formatPrice(t.exit_price)}</td>
                  <td className={`mono ${t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
                    {t.net_pnl >= 0 ? '+' : ''}{Number(t.net_pnl).toFixed(2)}$
                  </td>
                  <td className={`mono text-xs ${t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>{pnlPct}%</td>
                  <td className="text-xs muted">{formatDuration(t.entry_time, t.exit_time)}</td>
                  <td>
                    <span className={`badge ${EXIT_BADGE_CLASS[t.exit_reason] || ''}`}>
                      {EXIT_LABELS[t.exit_reason] || t.exit_reason}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {allTrades.length >= limit && (
        <button className="load-more-btn" onClick={() => setLimit(prev => prev + 50)}>
          Charger plus de trades...
        </button>
      )}
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────────
 * Section 3 — Equity Curve Annotee (Paper)
 * ──────────────────────────────────────────────────────────────────────── */

const EQ_W = 800
const EQ_H = 300
const EQ_PAD = { top: 15, right: 15, bottom: 25, left: 15 }

function AnnotatedEquityCurve({ period }) {
  const [hoverIdx, setHoverIdx] = useState(null)
  const sinceParam = period !== 'all' ? `&since=${getPeriodSince(period)}` : ''
  const { data: snapData } = useApi(`/api/journal/snapshots?limit=2000${sinceParam}`, 60000)
  const { data: eventsData } = useApi('/api/journal/events?limit=200', 60000)

  const snapshots = snapData?.snapshots || []
  const events = eventsData?.events || []

  const values = snapshots.length >= 2 ? snapshots.map(s => s.equity) : []
  const initialCapital = values.length > 0 ? values[0] : 0

  const allVals = values.length > 0 ? [...values, initialCapital] : [0]
  const min = Math.min(...allVals)
  const max = Math.max(...allVals)
  const range = max - min || 1

  const chartW = EQ_W - EQ_PAD.left - EQ_PAD.right
  const chartH = EQ_H - EQ_PAD.top - EQ_PAD.bottom

  const toX = (i) => EQ_PAD.left + (i / Math.max(values.length - 1, 1)) * chartW
  const toY = (v) => EQ_PAD.top + ((max - v) / range) * chartH

  const polyPoints = values.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')
  const baselineY = toY(initialCapital)
  const lastValue = values.length > 0 ? values[values.length - 1] : 0
  const isProfit = lastValue >= initialCapital
  const lineColor = isProfit ? 'var(--accent)' : 'var(--red)'
  const fillColor = isProfit ? 'rgba(0, 230, 138, 0.08)' : 'rgba(255, 68, 102, 0.08)'
  const areaPoints = values.length > 0
    ? `${EQ_PAD.left},${baselineY} ${polyPoints} ${toX(values.length - 1).toFixed(1)},${baselineY}`
    : ''

  // Map events to X positions — hook toujours appele (regle des hooks)
  const eventMarkers = useMemo(() => {
    if (!events.length || !snapshots.length || values.length < 2) return []
    const snapTimes = snapshots.map(s => new Date(s.timestamp).getTime())
    const firstTime = snapTimes[0]
    const lastTime = snapTimes[snapTimes.length - 1]
    const timeRange = lastTime - firstTime || 1

    return events
      .filter(e => e.event_type === 'OPEN' || e.event_type === 'CLOSE')
      .map(e => {
        const t = new Date(e.timestamp).getTime()
        if (t < firstTime || t > lastTime) return null
        const ratio = (t - firstTime) / timeRange
        const idx = Math.round(ratio * (values.length - 1))
        const clampIdx = Math.max(0, Math.min(idx, values.length - 1))
        return {
          x: toX(clampIdx),
          y: toY(values[clampIdx]),
          type: e.event_type,
          symbol: e.symbol,
          strategy: e.strategy_name,
          price: e.price,
          metadata: e.metadata,
        }
      })
      .filter(Boolean)
  }, [events, snapshots, values])

  const pnl = lastValue - initialCapital
  const pnlPct = initialCapital > 0 ? ((pnl / initialCapital) * 100).toFixed(2) : '0.00'

  const handleMouseMove = (e) => {
    if (values.length < 2) return
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const svgX = (x / rect.width) * EQ_W
    const dataX = (svgX - EQ_PAD.left) / chartW
    const idx = Math.round(dataX * (values.length - 1))
    setHoverIdx(Math.max(0, Math.min(idx, values.length - 1)))
  }

  const hoverPoint = hoverIdx !== null && hoverIdx < snapshots.length ? snapshots[hoverIdx] : null

  if (snapshots.length < 2) {
    return <div className="empty-state">Pas assez de snapshots pour la courbe (min. 2)</div>
  }

  return (
    <div className="equity-annotated">
      <svg
        viewBox={`0 0 ${EQ_W} ${EQ_H}`}
        preserveAspectRatio="none"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
        style={{ cursor: hoverIdx !== null ? 'crosshair' : 'default' }}
      >
        {/* Area fill */}
        <polygon points={areaPoints} fill={fillColor} />

        {/* Baseline */}
        <line
          x1={EQ_PAD.left} y1={baselineY}
          x2={EQ_W - EQ_PAD.right} y2={baselineY}
          stroke="var(--border)" strokeWidth={1} strokeDasharray="4,4"
        />

        {/* Line */}
        <polyline
          points={polyPoints} fill="none"
          stroke={lineColor} strokeWidth={1.8}
          strokeLinejoin="round" strokeLinecap="round"
        />

        {/* Current point */}
        <circle cx={toX(values.length - 1)} cy={toY(lastValue)} r={4} fill={lineColor} />

        {/* Baseline label */}
        <text x={EQ_PAD.left + 2} y={baselineY - 5}
          fill="var(--text-dim)" fontSize={10} fontFamily="var(--font-mono)">
          {initialCapital.toLocaleString('fr-FR')}$
        </text>

        {/* Event markers */}
        {eventMarkers.map((m, i) => (
          <g key={`marker-${i}`}>
            {m.type === 'OPEN' ? (
              <polygon
                points={`${m.x},${m.y - 7} ${m.x - 5},${m.y + 3} ${m.x + 5},${m.y + 3}`}
                fill="var(--accent)" opacity={0.8}
              />
            ) : (
              <polygon
                points={`${m.x},${m.y + 7} ${m.x - 5},${m.y - 3} ${m.x + 5},${m.y - 3}`}
                fill="var(--red)" opacity={0.8}
              />
            )}
          </g>
        ))}

        {/* Hover crosshair */}
        {hoverIdx !== null && (
          <>
            <line
              x1={toX(hoverIdx)} x2={toX(hoverIdx)}
              y1={EQ_PAD.top} y2={EQ_PAD.top + chartH}
              stroke="var(--text-dim)" strokeWidth={1} strokeDasharray="2,2"
            />
            <circle cx={toX(hoverIdx)} cy={toY(values[hoverIdx])} r={4}
              fill={lineColor} stroke="var(--bg)" strokeWidth={1.5} />
          </>
        )}
      </svg>

      {/* Hover tooltip */}
      {hoverPoint && (
        <div className="text-xs mono" style={{ padding: '2px 0', color: 'var(--text-muted)' }}>
          {new Date(hoverPoint.timestamp).toLocaleString('fr-FR')}
          {' '}&middot;{' '}Equity: {hoverPoint.equity.toLocaleString('fr-FR', { maximumFractionDigits: 0 })}$
          {' '}&middot;{' '}Unreal: {hoverPoint.unrealized_pnl >= 0 ? '+' : ''}{hoverPoint.unrealized_pnl?.toFixed(2)}$
          {' '}&middot;{' '}Margin: {(hoverPoint.margin_ratio * 100).toFixed(0)}%
          {' '}&middot;{' '}Pos: {hoverPoint.n_positions}
        </div>
      )}

      <div className="equity-period-info">
        <span className="text-xs muted">Equity (live + unrealized)</span>
        <span className={`mono text-xs ${isProfit ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
          {lastValue.toLocaleString('fr-FR', { maximumFractionDigits: 0 })}$
          ({pnl >= 0 ? '+' : ''}{pnlPct}%)
        </span>
      </div>
    </div>
  )
}

function getPeriodSince(period) {
  const now = new Date()
  if (period === 'today') {
    const d = new Date(now)
    d.setHours(0, 0, 0, 0)
    return d.toISOString()
  }
  if (period === '7d') return new Date(now - 7 * 86400000).toISOString()
  if (period === '30d') return new Date(now - 30 * 86400000).toISOString()
  return ''
}

/* ────────────────────────────────────────────────────────────────────────────
 * Section 4 — Ordres Bitget (partage entre Live et Paper)
 * ──────────────────────────────────────────────────────────────────────── */

function BitgetOrders() {
  const { data, loading } = useApi('/api/executor/orders?limit=50', 15000)
  const { data: slippageData } = useApi('/api/journal/slippage', 30000)
  const orders = data?.orders || []
  const slipSummary = slippageData?.slippage

  if (loading && !data) return <div className="empty-state">Chargement...</div>
  if (orders.length === 0) {
    return <div className="empty-state">Executor inactif ou aucun ordre</div>
  }

  return (
    <div>
      {slipSummary && slipSummary.orders_analyzed > 0 && (
        <div className="stats-secondary" style={{ marginBottom: 8 }}>
          <span>Slippage moyen : <span className={slipSummary.avg_slippage_pct > 0 ? 'pnl-neg' : 'pnl-pos'}>
            {slipSummary.avg_slippage_pct > 0 ? '+' : ''}{slipSummary.avg_slippage_pct.toFixed(3)}%
          </span></span>
          <span>Cout total : <span className={slipSummary.total_slippage_cost > 0 ? 'pnl-neg' : 'pnl-pos'}>
            {slipSummary.total_slippage_cost > 0 ? '+' : ''}{slipSummary.total_slippage_cost.toFixed(2)}$
          </span></span>
          <span className="text-xs muted">({slipSummary.orders_analyzed} ordres analyses)</span>
        </div>
      )}

      <div style={{ overflowX: 'auto' }}>
        <table className="journal-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Type</th>
              <th>Symbol</th>
              <th>Side</th>
              <th>Qty</th>
              <th>Prix moyen</th>
              <th>Slippage</th>
              <th>Order ID</th>
              <th>Strategie</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((o, i) => {
              const avgP = o.average_price || 0
              const paperP = o.paper_price || 0
              const hasSlip = avgP > 0 && paperP > 0
              const slipPct = hasSlip ? ((avgP - paperP) / paperP * 100) : null
              const slipClass = slipPct != null
                ? (slipPct > 0.01 ? 'pnl-neg' : slipPct < -0.01 ? 'pnl-pos' : 'muted')
                : ''

              return (
                <tr key={i}>
                  <td className="text-xs muted">
                    {o.timestamp ? new Date(o.timestamp).toLocaleString('fr-FR') : '--'}
                  </td>
                  <td>
                    <span className={`order-badge ${o.order_type || ''}`}>
                      {o.order_type || '--'}
                    </span>
                  </td>
                  <td style={{ fontWeight: 600, fontSize: 11 }}>{o.symbol || '--'}</td>
                  <td>
                    <span className={`badge ${o.side === 'buy' ? 'badge-long' : 'badge-short'}`}>
                      {o.side === 'buy' ? 'BUY' : 'SELL'}
                    </span>
                  </td>
                  <td className="mono text-xs">{o.filled || o.quantity || '--'}</td>
                  <td className="mono text-xs">{avgP > 0 ? formatPrice(avgP) : '--'}</td>
                  <td className={`mono text-xs ${slipClass}`}>
                    {slipPct != null ? `${slipPct >= 0 ? '+' : ''}${slipPct.toFixed(3)}%` : '--'}
                  </td>
                  <td>
                    <span className="order-id" title={o.order_id || ''} onClick={() => {
                      if (o.order_id) navigator.clipboard?.writeText(o.order_id)
                    }}>
                      {o.order_id ? o.order_id.slice(0, 12) + '...' : '--'}
                    </span>
                  </td>
                  <td className="text-xs">{o.strategy_name || '--'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}


/* ────────────────────────────────────────────────────────────────────────────
 * Section 5 — Performance par Asset (Paper)
 * ──────────────────────────────────────────────────────────────────────── */

function PerAssetSummary({ period }) {
  const { data, loading } = useApi(`/api/journal/per-asset?period=${period}`, 30000)
  const [sortBy, setSortBy] = useState('total_pnl')
  const [sortAsc, setSortAsc] = useState(false)

  const assets = data?.per_asset || []

  const sorted = useMemo(() => {
    return [...assets].sort((a, b) => {
      const va = a[sortBy] ?? 0
      const vb = b[sortBy] ?? 0
      return sortAsc ? va - vb : vb - va
    })
  }, [assets, sortBy, sortAsc])

  const handleSort = (col) => {
    if (sortBy === col) setSortAsc(!sortAsc)
    else { setSortBy(col); setSortAsc(false) }
  }

  if (loading && !data) return <div className="empty-state">Chargement...</div>
  if (assets.length === 0) return <div className="empty-state">Aucun trade (paper)</div>

  return (
    <div style={{ overflowX: 'auto' }}>
      <div className="text-xs muted" style={{ marginBottom: 4 }}>
        Donnees paper trading (simulation_trades)
      </div>
      <table className="journal-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('symbol')} style={{ cursor: 'pointer' }}>
              Symbol {sortBy === 'symbol' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th onClick={() => handleSort('total_trades')} style={{ cursor: 'pointer' }}>
              Trades {sortBy === 'total_trades' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th onClick={() => handleSort('win_rate')} style={{ cursor: 'pointer' }}>
              Win Rate {sortBy === 'win_rate' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th onClick={() => handleSort('total_pnl')} style={{ cursor: 'pointer' }}>
              P&L Net {sortBy === 'total_pnl' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th onClick={() => handleSort('avg_pnl')} style={{ cursor: 'pointer' }}>
              P&L Moyen {sortBy === 'avg_pnl' ? (sortAsc ? '\u25B2' : '\u25BC') : ''}
            </th>
            <th>W/L</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((a) => (
            <tr key={a.symbol} className={a.total_pnl >= 0 ? 'row-positive' : 'row-negative'}>
              <td style={{ fontWeight: 600 }}>{a.symbol}</td>
              <td className="mono text-xs">{a.total_trades}</td>
              <td className={`mono text-xs ${a.win_rate >= 50 ? 'pnl-pos' : 'pnl-neg'}`}>
                {a.win_rate}%
              </td>
              <td className={`mono ${a.total_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
                {a.total_pnl >= 0 ? '+' : ''}{a.total_pnl.toFixed(2)}$
              </td>
              <td className={`mono text-xs ${a.avg_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
                {a.avg_pnl >= 0 ? '+' : ''}{a.avg_pnl.toFixed(2)}$
              </td>
              <td className="text-xs">{a.wins}/{a.losses}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
