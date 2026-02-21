import { useMemo } from 'react'
import { useStrategyContext } from '../contexts/StrategyContext'
import ActivePositions from './ActivePositions'
import CandleStatus from './CandleStatus'
import CollapsibleCard from './CollapsibleCard'

function formatPnl(value) {
  if (value == null) return '0$'
  const abs = Math.abs(value)
  if (abs >= 10000) return `${(value / 1000).toFixed(1)}K$`
  return `${value.toFixed(2)}$`
}

export default function OverviewPage({ wsData }) {
  const { setActiveStrategy } = useStrategyContext()
  const allowedLive = wsData?.executor?.selector?.allowed_strategies || []

  const rows = useMemo(() => {
    const strategies = wsData?.strategies || {}
    const gridPositions = wsData?.grid_state?.grid_positions || {}

    return Object.keys(strategies).map(name => {
      // Grids de cette stratégie
      const grids = Object.values(gridPositions).filter(g => g.strategy === name)
      const assetsCount = grids.length
      const totalLevelsOpen = grids.reduce((s, g) => s + (g.levels_open || 0), 0)
      const totalUpnl = grids.reduce((s, g) => s + (g.unrealized_pnl || 0), 0)
      const totalMargin = grids.reduce((s, g) => s + (g.margin_used || 0), 0)

      // P&L réalisé depuis strategies
      const strat = strategies[name] || {}
      const realizedPnl = strat.net_pnl || 0
      const totalPnl = realizedPnl + totalUpnl

      const isLive = allowedLive.includes(name)

      return { name, assetsCount, totalLevelsOpen, totalPnl, totalMargin, isLive }
    })
  }, [wsData?.strategies, wsData?.grid_state?.grid_positions, allowedLive])

  return (
    <>
      <div className="card">
        <h2>Strategies actives</h2>
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
        <div className="text-xs muted" style={{ marginTop: 8, textAlign: 'center' }}>
          Cliquer sur une strategie pour voir le detail
        </div>
      </div>

      <CollapsibleCard
        title="Toutes les positions"
        defaultOpen={true}
        storageKey="overview-positions"
      >
        <ActivePositions wsData={wsData} />
      </CollapsibleCard>

      <CollapsibleCard
        title="Donnees historiques"
        defaultOpen={false}
        storageKey="overview-candles"
      >
        <CandleStatus wsData={wsData} />
      </CollapsibleCard>
    </>
  )
}
