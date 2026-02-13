/**
 * ResearchPage — Visualisation des résultats WFO en DB
 * Sprint 13
 */

import { useState, useMemo, useEffect, useCallback } from 'react'
import WfoChart from './WfoChart'

const GRADE_COLORS = {
  A: '#10b981',
  B: '#3b82f6',
  C: '#f59e0b',
  D: '#f97316',
  F: '#ef4444',
}

export default function ResearchPage() {
  const [view, setView] = useState('table') // 'table' | 'detail'
  const [selectedId, setSelectedId] = useState(null)
  const [filters, setFilters] = useState({
    strategy: '',
    asset: '',
    minGrade: '',
  })
  const [sortBy, setSortBy] = useState('total_score')
  const [sortDir, setSortDir] = useState('desc')

  // API: liste des résultats (fetch au montage uniquement, pas de polling)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchResults = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await fetch('/api/optimization/results')
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const json = await resp.json()
      setResults(json)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchResults()
  }, [])

  // API: détail d'un résultat (pas de polling, fetch manuel)
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)

  // Fetch détail au clic
  const fetchDetail = async (id) => {
    setDetailLoading(true)
    try {
      const resp = await fetch(`/api/optimization/${id}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const json = await resp.json()
      setDetail(json)
      setSelectedId(id)
      setView('detail')
    } catch (err) {
      alert(`Erreur: ${err.message}`)
    } finally {
      setDetailLoading(false)
    }
  }

  // Filtrage + tri
  const filteredResults = useMemo(() => {
    if (!results || !results.results) return []

    let filtered = results.results.filter(r => {
      if (filters.strategy && r.strategy_name !== filters.strategy) return false
      if (filters.asset && r.asset !== filters.asset) return false
      if (filters.minGrade) {
        const gradeOrder = { A: 4, B: 3, C: 2, D: 1, F: 0 }
        if (gradeOrder[r.grade] < gradeOrder[filters.minGrade]) return false
      }
      return true
    })

    // Tri
    filtered.sort((a, b) => {
      let valA = a[sortBy]
      let valB = b[sortBy]

      // Cas spécial pour les grades (ordre alphabétique inversé)
      if (sortBy === 'grade') {
        const gradeOrder = { A: 4, B: 3, C: 2, D: 1, F: 0 }
        valA = gradeOrder[a.grade]
        valB = gradeOrder[b.grade]
      }

      if (valA == null) return 1
      if (valB == null) return -1
      if (valA === valB) return 0

      const diff = sortDir === 'asc' ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1)
      return diff
    })

    return filtered
  }, [results, filters, sortBy, sortDir])

  // Listes uniques pour les filtres
  const strategies = useMemo(() => {
    if (!results || !results.results) return []
    return [...new Set(results.results.map(r => r.strategy_name))].sort()
  }, [results])

  const assets = useMemo(() => {
    if (!results || !results.results) return []
    return [...new Set(results.results.map(r => r.asset))].sort()
  }, [results])

  // Toggle tri
  const handleSort = (column) => {
    if (sortBy === column) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(column)
      setSortDir('desc')
    }
  }

  if (loading) {
    return (
      <div className="research-page">
        <p style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
          Chargement des résultats WFO...
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="research-page">
        <p style={{ textAlign: 'center', padding: '40px', color: '#ef4444' }}>
          Erreur: {error}
        </p>
      </div>
    )
  }

  // Vue détail
  if (view === 'detail') {
    if (detailLoading) {
      return (
        <div className="research-page">
          <button onClick={() => setView('table')} className="btn-back">← Retour</button>
          <p style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
            Chargement du détail...
          </p>
        </div>
      )
    }

    if (!detail) {
      return (
        <div className="research-page">
          <button onClick={() => setView('table')} className="btn-back">← Retour</button>
          <p style={{ textAlign: 'center', padding: '40px', color: '#ef4444' }}>
            Aucun détail disponible
          </p>
        </div>
      )
    }

    return (
      <div className="research-page">
        <button onClick={() => setView('table')} className="btn-back">← Retour au tableau</button>

        <div className="detail-header">
          <h2>{detail.strategy_name} × {detail.asset}</h2>
          <div className="detail-meta">
            <span className={`grade-badge grade-${detail.grade}`}>{detail.grade}</span>
            <span className="score">Score: {detail.total_score}/100</span>
            <span className="date">{new Date(detail.created_at).toLocaleString('fr-FR')}</span>
          </div>
        </div>

        {/* Paramètres retenus */}
        <section className="detail-section">
          <h3>Paramètres retenus</h3>
          <div className="params-grid">
            {Object.entries(detail.best_params || {}).map(([key, value]) => (
              <div key={key} className="param-item">
                <span className="param-key">{key}:</span>
                <span className="param-value">{typeof value === 'number' ? value.toFixed(3) : value}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Scores */}
        <section className="detail-section">
          <h3>Critères de notation</h3>
          <div className="scores-list">
            <ScoreBar label="OOS/IS Ratio" value={detail.oos_is_ratio} max={1} />
            <ScoreBar label="Monte Carlo p-value" value={detail.monte_carlo_summary?.p_value} max={1} invert />
            <ScoreBar label="DSR" value={detail.dsr} max={1} />
            <ScoreBar label="Stabilité Params" value={detail.param_stability} max={1} />
            <ScoreBar label="Transfer Bitget" value={detail.validation_summary?.transfer_ratio} max={1} />
          </div>
        </section>

        {/* Equity Curve WFO */}
        {detail.wfo_windows && detail.wfo_windows.length > 0 && (
          <section className="detail-section">
            <h3>Equity Curve IS vs OOS</h3>
            <WfoChart windows={detail.wfo_windows} />
          </section>
        )}

        {/* Validation Bitget */}
        <section className="detail-section">
          <h3>Validation Bitget (OOS réel)</h3>
          <div className="validation-grid">
            <div className="validation-item">
              <span className="label">Sharpe:</span>
              <span className="value">{detail.validation_summary?.bitget_sharpe?.toFixed(2) ?? 'N/A'}</span>
            </div>
            <div className="validation-item">
              <span className="label">Net Return:</span>
              <span className="value">{detail.validation_summary?.bitget_net_return_pct?.toFixed(2) ?? 'N/A'}%</span>
            </div>
            <div className="validation-item">
              <span className="label">Trades:</span>
              <span className="value">{detail.validation_summary?.bitget_trades ?? 'N/A'}</span>
            </div>
            <div className="validation-item">
              <span className="label">CI Sharpe:</span>
              <span className="value">
                [{detail.validation_summary?.bitget_sharpe_ci_low?.toFixed(2) ?? '?'}, {detail.validation_summary?.bitget_sharpe_ci_high?.toFixed(2) ?? '?'}]
              </span>
            </div>
            <div className="validation-item">
              <span className="label">Transfer Ratio:</span>
              <span className="value">{(detail.validation_summary?.transfer_ratio * 100)?.toFixed(0) ?? 'N/A'}%</span>
            </div>
            <div className="validation-item">
              <span className="label">Significatif:</span>
              <span className="value">{detail.validation_summary?.transfer_significant ? 'Oui' : 'Non'}</span>
            </div>
          </div>
        </section>

        {/* Monte Carlo */}
        <section className="detail-section">
          <h3>Monte Carlo</h3>
          <div className="monte-carlo-grid">
            <div className="mc-item">
              <span className="label">p-value:</span>
              <span className="value">{detail.monte_carlo_summary?.p_value?.toFixed(4) ?? 'N/A'}</span>
            </div>
            <div className="mc-item">
              <span className="label">Significatif:</span>
              <span className="value">{detail.monte_carlo_summary?.significant ? 'Oui (p < 0.05)' : 'Non'}</span>
            </div>
            <div className="mc-item">
              <span className="label">Underpowered:</span>
              <span className="value">{detail.monte_carlo_summary?.underpowered ? 'Oui (< 30 trades)' : 'Non'}</span>
            </div>
          </div>
        </section>

        {/* Warnings */}
        {detail.warnings && detail.warnings.length > 0 && (
          <section className="detail-section">
            <h3>Avertissements</h3>
            <ul className="warnings-list">
              {detail.warnings.map((w, i) => (
                <li key={i} className="warning-item">{w}</li>
              ))}
            </ul>
          </section>
        )}

        <style jsx>{`
          .btn-back {
            padding: 8px 16px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #ccc;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 20px;
            transition: all 0.2s;
          }
          .btn-back:hover {
            background: #333;
            border-color: #666;
          }
          .detail-header {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
          }
          .detail-header h2 {
            margin: 0 0 12px 0;
            color: #fff;
            font-size: 24px;
          }
          .detail-meta {
            display: flex;
            gap: 16px;
            align-items: center;
          }
          .grade-badge {
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 14px;
          }
          .grade-A { background: ${GRADE_COLORS.A}; color: white; }
          .grade-B { background: ${GRADE_COLORS.B}; color: white; }
          .grade-C { background: ${GRADE_COLORS.C}; color: white; }
          .grade-D { background: ${GRADE_COLORS.D}; color: white; }
          .grade-F { background: ${GRADE_COLORS.F}; color: white; }
          .score {
            color: #ccc;
            font-size: 14px;
          }
          .date {
            color: #888;
            font-size: 13px;
          }
          .detail-section {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
          }
          .detail-section h3 {
            margin: 0 0 16px 0;
            color: #fff;
            font-size: 18px;
            border-bottom: 1px solid #333;
            padding-bottom: 8px;
          }
          .params-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
          }
          .param-item {
            display: flex;
            gap: 8px;
            background: #0a0a0a;
            padding: 8px 12px;
            border-radius: 4px;
          }
          .param-key {
            color: #888;
            font-size: 13px;
          }
          .param-value {
            color: #ccc;
            font-size: 13px;
            font-weight: 500;
          }
          .scores-list {
            display: flex;
            flex-direction: column;
            gap: 16px;
          }
          .validation-grid, .monte-carlo-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
          }
          .validation-item, .mc-item {
            display: flex;
            justify-content: space-between;
            background: #0a0a0a;
            padding: 10px 14px;
            border-radius: 4px;
          }
          .validation-item .label, .mc-item .label {
            color: #888;
            font-size: 13px;
          }
          .validation-item .value, .mc-item .value {
            color: #ccc;
            font-size: 13px;
            font-weight: 500;
          }
          .warnings-list {
            list-style: none;
            padding: 0;
            margin: 0;
          }
          .warning-item {
            background: #2a1a1a;
            border-left: 3px solid #ef4444;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 4px;
            color: #ffb3b3;
            font-size: 13px;
          }
        `}</style>
      </div>
    )
  }

  // Vue tableau
  return (
    <div className="research-page">
      <div className="research-header">
        <h2>Résultats WFO — Recherche</h2>
        <div className="filters">
          <select
            value={filters.strategy}
            onChange={(e) => setFilters({ ...filters, strategy: e.target.value })}
            className="filter-select"
          >
            <option value="">Toutes les stratégies</option>
            {strategies.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>

          <select
            value={filters.asset}
            onChange={(e) => setFilters({ ...filters, asset: e.target.value })}
            className="filter-select"
          >
            <option value="">Tous les assets</option>
            {assets.map(a => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>

          <select
            value={filters.minGrade}
            onChange={(e) => setFilters({ ...filters, minGrade: e.target.value })}
            className="filter-select"
          >
            <option value="">Grade minimum: Tous</option>
            <option value="D">D et plus</option>
            <option value="C">C et plus</option>
            <option value="B">B et plus</option>
            <option value="A">A uniquement</option>
          </select>
        </div>
      </div>

      <div className="results-count">
        {filteredResults.length} résultat(s) sur {results?.total || 0} total
      </div>

      <div className="table-container">
        <table className="results-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('strategy_name')}>
                Stratégie {sortBy === 'strategy_name' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('asset')}>
                Asset {sortBy === 'asset' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('grade')}>
                Grade {sortBy === 'grade' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('total_score')}>
                Score {sortBy === 'total_score' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('oos_sharpe')}>
                OOS Sharpe {sortBy === 'oos_sharpe' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('consistency')}>
                Consistance {sortBy === 'consistency' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('oos_is_ratio')}>
                OOS/IS {sortBy === 'oos_is_ratio' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('dsr')}>
                DSR {sortBy === 'dsr' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('param_stability')}>
                Stabilité {sortBy === 'param_stability' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('created_at')}>
                Date {sortBy === 'created_at' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredResults.length === 0 ? (
              <tr>
                <td colSpan={10} style={{ textAlign: 'center', color: '#888', padding: '40px' }}>
                  Aucun résultat trouvé
                </td>
              </tr>
            ) : (
              filteredResults.map(r => (
                <tr key={r.id} onClick={() => fetchDetail(r.id)} className="clickable-row">
                  <td>{r.strategy_name}</td>
                  <td>{r.asset}</td>
                  <td>
                    <span className={`grade-badge grade-${r.grade}`}>{r.grade}</span>
                  </td>
                  <td>{r.total_score}</td>
                  <td>{r.oos_sharpe?.toFixed(2) ?? 'N/A'}</td>
                  <td>{((r.consistency || 0) * 100).toFixed(0)}%</td>
                  <td>{r.oos_is_ratio?.toFixed(2) ?? 'N/A'}</td>
                  <td>{r.dsr?.toFixed(2) ?? 'N/A'}</td>
                  <td>{r.param_stability?.toFixed(2) ?? 'N/A'}</td>
                  <td>{new Date(r.created_at).toLocaleDateString('fr-FR')}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <style jsx>{`
        .research-page {
          padding: 20px;
          max-width: 1600px;
          margin: 0 auto;
        }
        .research-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          flex-wrap: wrap;
          gap: 16px;
        }
        .research-header h2 {
          margin: 0;
          color: #fff;
          font-size: 24px;
        }
        .filters {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
        }
        .filter-select {
          padding: 8px 12px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 4px;
          color: #ccc;
          font-size: 14px;
          cursor: pointer;
        }
        .filter-select:hover {
          border-color: #666;
        }
        .results-count {
          color: #888;
          font-size: 13px;
          margin-bottom: 12px;
        }
        .table-container {
          overflow-x: auto;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
        }
        .results-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }
        .results-table th {
          background: #0a0a0a;
          color: #ccc;
          font-weight: 600;
          text-align: left;
          padding: 12px 16px;
          border-bottom: 2px solid #333;
          cursor: pointer;
          user-select: none;
          white-space: nowrap;
        }
        .results-table th:hover {
          background: #1a1a1a;
        }
        .results-table td {
          padding: 12px 16px;
          border-bottom: 1px solid #222;
          color: #ccc;
          white-space: nowrap;
        }
        .clickable-row {
          cursor: pointer;
          transition: background 0.2s;
        }
        .clickable-row:hover {
          background: #252525;
        }
        .grade-badge {
          padding: 3px 8px;
          border-radius: 3px;
          font-weight: 600;
          font-size: 12px;
          display: inline-block;
        }
        .grade-A { background: ${GRADE_COLORS.A}; color: white; }
        .grade-B { background: ${GRADE_COLORS.B}; color: white; }
        .grade-C { background: ${GRADE_COLORS.C}; color: white; }
        .grade-D { background: ${GRADE_COLORS.D}; color: white; }
        .grade-F { background: ${GRADE_COLORS.F}; color: white; }
      `}</style>
    </div>
  )
}

// ScoreBar: Barre de progression pour les critères
function ScoreBar({ label, value, max = 1, invert = false }) {
  const pct = value != null ? (value / max) * 100 : 0
  // Pour p-value : plus bas = mieux (invert)
  const displayPct = invert && value != null ? 100 - pct : pct
  const color = displayPct >= 80 ? '#10b981' : displayPct >= 60 ? '#3b82f6' : displayPct >= 40 ? '#f59e0b' : '#ef4444'

  return (
    <div className="score-bar">
      <div className="score-bar__label">
        <span>{label}</span>
        <span className="score-bar__value">{value != null ? value.toFixed(3) : 'N/A'}</span>
      </div>
      <div className="score-bar__track">
        <div className="score-bar__fill" style={{ width: `${displayPct}%`, background: color }} />
      </div>

      <style jsx>{`
        .score-bar {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .score-bar__label {
          display: flex;
          justify-content: space-between;
          font-size: 13px;
          color: #ccc;
        }
        .score-bar__value {
          font-weight: 600;
        }
        .score-bar__track {
          height: 8px;
          background: #0a0a0a;
          border-radius: 4px;
          overflow: hidden;
        }
        .score-bar__fill {
          height: 100%;
          transition: width 0.3s, background 0.3s;
          border-radius: 4px;
        }
      `}</style>
    </div>
  )
}
