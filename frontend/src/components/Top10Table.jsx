/**
 * Top10Table — Tableau HTML des top 10 combos par métrique sélectionnée
 * Sprint 14b Bloc F
 */

import InfoTooltip from './InfoTooltip'
import './Top10Table.css'

export default function Top10Table({ combos, paramNames, metric }) {
  if (!combos || combos.length === 0) {
    return (
      <div className="top10-table">
        <p style={{ textAlign: 'center', color: '#888', padding: '20px' }}>
          Aucune combo disponible
        </p>
      </div>
    )
  }

  // Trier par métrique décroissante et prendre le top 10
  const sorted = [...combos].sort((a, b) => {
    const valA = a[metric] ?? -Infinity
    const valB = b[metric] ?? -Infinity
    return valB - valA
  })
  const top10 = sorted.slice(0, 10)

  return (
    <div className="top10-table">
      <h4>Top 10 Combos ({metric})</h4>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              {paramNames.map((pName) => (
                <th key={pName}>{pName}</th>
              ))}
              <th>
                OOS Sharpe <InfoTooltip term="oos_sharpe" />
              </th>
              <th>
                IS Sharpe <InfoTooltip term="is_sharpe" />
              </th>
              <th>
                OOS/IS <InfoTooltip term="oos_is_ratio" />
              </th>
              <th>
                Consist. <InfoTooltip term="consistency" />
              </th>
              <th>Trades</th>
            </tr>
          </thead>
          <tbody>
            {top10.map((combo, idx) => {
              const isBest = combo.is_best === 1 || combo.is_best === true

              return (
                <tr key={idx} className={isBest ? 'best-row' : ''}>
                  <td className="rank-cell">{idx + 1}</td>
                  {paramNames.map((pName) => (
                    <td key={pName} className="param-cell">
                      {combo.params[pName] ?? '—'}
                    </td>
                  ))}
                  <td className="metric-cell">
                    {combo.oos_sharpe != null ? combo.oos_sharpe.toFixed(2) : '—'}
                  </td>
                  <td className="metric-cell">
                    {combo.is_sharpe != null ? combo.is_sharpe.toFixed(2) : '—'}
                  </td>
                  <td className="metric-cell">
                    {combo.oos_is_ratio != null ? combo.oos_is_ratio.toFixed(2) : '—'}
                  </td>
                  <td className="metric-cell">
                    {combo.consistency != null ? (combo.consistency * 100).toFixed(0) + '%' : '—'}
                  </td>
                  <td className="metric-cell">{combo.oos_trades ?? '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
