/**
 * TradeHistory — Tableau enrichi des trades récents.
 * Colonnes : Asset, Stratégie, Dir, Entry/Exit, P&L, Exit reason, Durée.
 */
import { useState } from 'react'
import { useApi } from '../hooks/useApi'

const EXIT_LABELS = {
  sl: 'SL',
  tp: 'TP',
  signal_exit: 'Signal',
  regime_change: 'Régime',
  end_of_data: 'Fin',
}

const EXIT_BADGE_CLASS = {
  sl: 'badge-stopped',
  tp: 'badge-active',
  signal_exit: 'badge-simulation',
  regime_change: 'badge-trending',
  end_of_data: '',
}

function formatDuration(entryTime, exitTime) {
  if (!entryTime || !exitTime) return '--'
  const ms = new Date(exitTime) - new Date(entryTime)
  const mins = Math.floor(ms / 60000)
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  const remMins = mins % 60
  return `${hours}h${remMins > 0 ? remMins + 'm' : ''}`
}

export default function TradeHistory() {
  const [expanded, setExpanded] = useState(false)
  const { data, loading } = useApi('/api/simulator/trades?limit=20', 10000)

  const trades = data?.trades || []
  const displayTrades = expanded ? trades : trades.slice(0, 5)

  return (
    <>
      {loading && !data && (
        <div className="empty-state">Chargement...</div>
      )}

      {!loading && trades.length === 0 && (
        <div className="empty-state">Aucun trade pour le moment</div>
      )}

      {displayTrades.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Asset</th>
                <th>Strat.</th>
                <th>Dir</th>
                <th>Entry &rarr; Exit</th>
                <th>P&L</th>
                <th>Raison</th>
                <th>Durée</th>
              </tr>
            </thead>
            <tbody>
              {displayTrades.map((t, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600, fontSize: 11 }}>
                    {t.symbol || '--'}
                  </td>
                  <td className="text-xs">{t.strategy}</td>
                  <td>
                    <span className={`badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
                      {t.direction}
                    </span>
                  </td>
                  <td className="text-xs">
                    {Number(t.entry_price).toFixed(1)} &rarr; {Number(t.exit_price).toFixed(1)}
                  </td>
                  <td className={t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                    {t.net_pnl >= 0 ? '+' : ''}{Number(t.net_pnl).toFixed(2)}$
                  </td>
                  <td>
                    <span className={`badge ${EXIT_BADGE_CLASS[t.exit_reason] || ''}`}>
                      {EXIT_LABELS[t.exit_reason] || t.exit_reason}
                    </span>
                  </td>
                  <td className="dim text-xs">
                    {formatDuration(t.entry_time, t.exit_time)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!expanded && trades.length > 5 && (
        <div
          className="text-xs muted text-center"
          style={{ marginTop: 6, cursor: 'pointer' }}
          onClick={() => setExpanded(true)}
        >
          +{trades.length - 5} trades...
        </div>
      )}
    </>
  )
}
