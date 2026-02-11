/**
 * ArenaRankingMini — Classement compact de l'arena (sidebar).
 * Props : wsData
 * Affiche le ranking depuis wsData.ranking.
 * 4 lignes max avec nom de stratégie + PnL + badge status.
 */
export default function ArenaRankingMini({ wsData }) {
  const ranking = wsData?.ranking || []

  // Limiter à 4 stratégies pour la version compacte
  const topStrategies = ranking.slice(0, 4)

  return (
    <div className="card">
      <h2>Arena</h2>

      {topStrategies.length === 0 && (
        <div className="empty-state">En attente de données...</div>
      )}

      {topStrategies.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {topStrategies.map((s, i) => {
            const pnl = s.net_pnl || 0
            const isProfit = pnl >= 0
            const isActive = s.is_active !== false

            return (
              <div key={s.name || i} className="flex-between" style={{ gap: 8 }}>
                {/* Rang + Nom */}
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

                {/* PnL + Badge */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                  <span className={`mono text-xs ${isProfit ? 'pnl-pos' : 'pnl-neg'}`} style={{ fontWeight: 600 }}>
                    {isProfit ? '+' : ''}{pnl.toFixed(2)}$
                  </span>
                  <span className={`badge ${isActive ? 'badge-active' : 'badge-stopped'}`}>
                    {isActive ? 'ACTIF' : 'STOP'}
                  </span>
                </div>
              </div>
            )
          })}

          {/* Lien vers vue complète si plus de 4 stratégies */}
          {ranking.length > 4 && (
            <div className="text-xs muted text-center" style={{ paddingTop: 4 }}>
              +{ranking.length - 4} stratégies...
            </div>
          )}
        </div>
      )}
    </div>
  )
}
