import { useState } from 'react'
import { useApi } from '../hooks/useApi'

export default function TradeHistory() {
  const [expanded, setExpanded] = useState(false)
  const { data, loading } = useApi('/api/simulator/trades?limit=20', 10000)

  const trades = data?.trades || []
  const displayTrades = expanded ? trades : trades.slice(0, 5)

  return (
    <div className="card">
      <div className="collapsible-header" onClick={() => setExpanded(!expanded)}>
        <h2>Trades Récents</h2>
        <span className={`collapsible-arrow ${expanded ? 'open' : ''}`}>▼</span>
      </div>

      {loading && !data && (
        <div className="empty-state">Chargement...</div>
      )}

      {!loading && trades.length === 0 && (
        <div className="empty-state">Aucun trade pour le moment</div>
      )}

      {displayTrades.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Stratégie</th>
              <th>Dir</th>
              <th>P&L</th>
              <th>Heure</th>
            </tr>
          </thead>
          <tbody>
            {displayTrades.map((t, i) => (
              <tr key={i}>
                <td className="text-xs">{t.strategy}</td>
                <td>
                  <span className={`badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
                    {t.direction}
                  </span>
                </td>
                <td className={t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                  {t.net_pnl >= 0 ? '+' : ''}{Number(t.net_pnl).toFixed(2)}$
                </td>
                <td className="dim text-xs">
                  {new Date(t.exit_time).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {!expanded && trades.length > 5 && (
        <div className="text-xs muted text-center" style={{ marginTop: 6 }}>
          +{trades.length - 5} trades...
        </div>
      )}
    </div>
  )
}
