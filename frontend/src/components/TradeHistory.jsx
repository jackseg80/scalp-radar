import { useApi } from '../hooks/useApi'

export default function TradeHistory() {
  const { data, loading } = useApi('/api/simulator/trades?limit=20', 10000)

  const trades = data?.trades || []

  return (
    <div className="card">
      <h2>Trades Récents</h2>
      {loading && <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Chargement...</p>}
      {trades.length === 0 && !loading && (
        <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Aucun trade pour le moment</p>
      )}
      {trades.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Stratégie</th>
                <th>Dir</th>
                <th>Entrée</th>
                <th>Sortie</th>
                <th>P&L Net</th>
                <th>Raison</th>
                <th>Heure</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i}>
                  <td>{t.strategy}</td>
                  <td>
                    <span className={`badge ${t.direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
                      {t.direction}
                    </span>
                  </td>
                  <td>{Number(t.entry_price).toFixed(2)}</td>
                  <td>{Number(t.exit_price).toFixed(2)}</td>
                  <td className={t.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                    {t.net_pnl >= 0 ? '+' : ''}{Number(t.net_pnl).toFixed(2)}$
                  </td>
                  <td style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{t.exit_reason}</td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {new Date(t.exit_time).toLocaleTimeString('fr-FR')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
