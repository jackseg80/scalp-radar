/**
 * ArenaRankingMini — Classement compact de l'arena (sidebar).
 * Props : wsData
 * Conçu pour être wrappé par CollapsibleCard dans App.jsx.
 */
import Tooltip from './Tooltip'

export default function ArenaRankingMini({ wsData }) {
  const ranking = wsData?.ranking || []
  const topStrategies = ranking.slice(0, 4)

  if (topStrategies.length === 0) {
    return <div className="empty-state">En attente de données...</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {topStrategies.map((s, i) => {
        const pnl = s.net_pnl || 0
        const isProfit = pnl >= 0
        const isActive = s.is_active !== false

        return (
          <div key={s.name || i} className="flex-between" style={{ gap: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: 1 }}>
              <span className="mono dim text-xs" style={{ width: 16, textAlign: 'right', flexShrink: 0 }}>
                #{i + 1}
              </span>
              <span className="text-sm" style={{
                fontWeight: 600,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}>
                {s.name}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
              <Tooltip content={`P&L net de ${s.name} en simulation`}>
                <span className={`mono text-xs ${isProfit ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
                  {isProfit ? '+' : ''}{pnl.toFixed(2)}$
                </span>
              </Tooltip>
              <Tooltip content={isActive ? 'Stratégie en simulation active' : 'Stratégie arrêtée (kill switch ou performance insuffisante)'}>
                <span className={`badge ${isActive ? 'badge-active' : 'badge-stopped'}`}>
                  {isActive ? 'ACTIF' : 'STOP'}
                </span>
              </Tooltip>
            </div>
          </div>
        )
      })}
      {ranking.length > 4 && (
        <div className="text-xs muted text-center" style={{ paddingTop: 4 }}>
          +{ranking.length - 4} stratégies...
        </div>
      )}
    </div>
  )
}

ArenaRankingMini.getSummary = function(wsData) {
  const ranking = wsData?.ranking || []
  if (ranking.length === 0) return null
  const top = ranking[0]
  const pnl = top.net_pnl || 0
  return `#1 ${top.name} ${pnl >= 0 ? '+' : ''}${pnl.toFixed(0)}$`
}
