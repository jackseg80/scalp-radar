import Tooltip from './Tooltip'

/**
 * KillSwitchBar — Barre de progression visuelle pour le Drawdown vs Kill Switch.
 * @param {number} currentDD - Drawdown actuel en % (positif)
 * @param {number} limit - Seuil du Kill Switch en %
 * @param {string} label - Libellé affiché
 */
export default function KillSwitchBar({ currentDD, limit, label }) {
  const ddRatio = Math.min(100, Math.max(0, (currentDD / limit) * 100))
  const isNearKill = ddRatio > 70

  return (
    <div className="overview-kill-bar-container" style={{ padding: '8px 12px', marginBottom: 10 }}>
      <div className="flex-between" style={{ marginBottom: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 9, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
            {label}
          </span>
          {isNearKill && <span className="badge badge-stopped" style={{ fontSize: 8, padding: '0 4px' }}>CRITIQUE</span>}
        </div>
        <span className={`mono ${isNearKill ? 'pnl-neg' : 'muted'}`} style={{ fontSize: 10 }}>
          DD : <b>{currentDD.toFixed(2)}%</b> / {limit}%
        </span>
      </div>
      <div className="overview-kill-track" style={{ height: 6 }}>
        <div 
          className={`overview-kill-fill ${isNearKill ? 'critical' : ''}`} 
          style={{ width: `${ddRatio}%` }}
        />
        <div className="overview-kill-limit" />
      </div>
    </div>
  )
}
