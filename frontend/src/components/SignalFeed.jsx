export default function SignalFeed({ wsData }) {
  const strategies = wsData?.strategies || {}

  const entries = Object.entries(strategies)

  return (
    <div className="card">
      <h2>Stratégies Live</h2>
      {entries.length === 0 && (
        <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>En attente de données...</p>
      )}
      {entries.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Stratégie</th>
              <th>Capital</th>
              <th>P&L</th>
              <th>Trades</th>
              <th>W/L</th>
              <th>Position</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, s]) => (
              <tr key={name}>
                <td style={{ fontWeight: 600 }}>{name}</td>
                <td>{s.capital?.toFixed(2)}$</td>
                <td className={s.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}>
                  {s.net_pnl >= 0 ? '+' : ''}{s.net_pnl?.toFixed(2)}$
                </td>
                <td>{s.total_trades}</td>
                <td>{s.wins}/{s.losses}</td>
                <td>
                  {s.has_position ? (
                    <span className="badge badge-active">OPEN</span>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
