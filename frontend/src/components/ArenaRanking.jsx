import { useApi } from '../hooks/useApi'

export default function ArenaRanking() {
  const { data, loading } = useApi('/api/arena/ranking', 30000)

  const ranking = data?.ranking || []

  return (
    <div className="card">
      <h2>Arena — Classement</h2>
      {loading && <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Chargement...</p>}
      {ranking.length === 0 && !loading && (
        <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>Aucune stratégie active</p>
      )}
      {ranking.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Stratégie</th>
              <th>P&L Net</th>
              <th>Return %</th>
              <th>Trades</th>
              <th>Win Rate</th>
              <th>PF</th>
              <th>Max DD</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {ranking.map((s, i) => (
              <tr key={s.name}>
                <td>{i + 1}</td>
                <td style={{ fontWeight: 600 }}>{s.name}</td>
                <td className={s.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                  {s.net_pnl >= 0 ? '+' : ''}{s.net_pnl.toFixed(2)}$
                </td>
                <td className={s.net_return_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                  {s.net_return_pct >= 0 ? '+' : ''}{s.net_return_pct.toFixed(2)}%
                </td>
                <td>{s.total_trades}</td>
                <td>{s.win_rate.toFixed(1)}%</td>
                <td>{s.profit_factor === Infinity ? '∞' : s.profit_factor.toFixed(2)}</td>
                <td className="pnl-neg">{s.max_drawdown_pct.toFixed(2)}%</td>
                <td>
                  <span className={`badge ${s.is_active ? 'badge-active' : 'badge-stopped'}`}>
                    {s.is_active ? 'ACTIF' : 'STOP'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
