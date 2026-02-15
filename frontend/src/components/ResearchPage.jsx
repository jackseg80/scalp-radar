/**
 * ResearchPage — Visualisation des résultats WFO en DB
 * Sprint 13
 */

import { useState, useMemo, useEffect, useCallback } from 'react'
import WfoChart from './WfoChart'
import InfoTooltip from './InfoTooltip'
import './ResearchPage.css'

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

  // Apply params
  const [applyResult, setApplyResult] = useState(null)
  const [applying, setApplying] = useState(false)

  const handleApply = async () => {
    const stratLabel = filters.strategy || 'toutes les stratégies'
    const ok = window.confirm(
      `Appliquer les params Grade A/B de "${stratLabel}" dans strategies.yaml ?`
    )
    if (!ok) return

    setApplying(true)
    setApplyResult(null)
    try {
      const qs = filters.strategy ? `?strategy_name=${encodeURIComponent(filters.strategy)}` : ''
      const resp = await fetch(`/api/optimization/apply${qs}`, { method: 'POST' })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }
      const json = await resp.json()
      setApplyResult(json)
      // Auto-dismiss après 15s
      setTimeout(() => setApplyResult(null), 15000)
    } catch (err) {
      alert(`Erreur apply: ${err.message}`)
    } finally {
      setApplying(false)
    }
  }

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
            <ScoreBar
              label={
                <>
                  OOS/IS Ratio <InfoTooltip term="oos_is_ratio" />
                </>
              }
              value={detail.oos_is_ratio}
              max={1}
            />
            <ScoreBar
              label={
                <>
                  Monte Carlo p-value <InfoTooltip term="monte_carlo_pvalue" />
                </>
              }
              value={detail.monte_carlo_summary?.p_value}
              max={1}
              invert
            />
            <ScoreBar
              label={
                <>
                  DSR <InfoTooltip term="dsr" />
                </>
              }
              value={detail.dsr}
              max={1}
            />
            <ScoreBar
              label={
                <>
                  Stabilité Params <InfoTooltip term="param_stability" />
                </>
              }
              value={detail.param_stability}
              max={1}
            />
            <ScoreBar
              label={
                <>
                  Transfer Bitget <InfoTooltip term="transfer_ratio" />
                </>
              }
              value={detail.validation_summary?.transfer_ratio}
              max={1}
            />
          </div>
        </section>

        {/* Equity Curve WFO */}
        {detail.wfo_windows && detail.wfo_windows.length > 0 && (
          <section className="detail-section">
            <h3>
              Equity Curve IS vs OOS <InfoTooltip term="is_vs_oos_chart" />
            </h3>
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

          <button
            className="btn-apply"
            onClick={handleApply}
            disabled={applying}
            title="Appliquer les params Grade A/B dans strategies.yaml"
          >
            {applying ? 'Application...' : 'Appliquer A/B'}
          </button>
        </div>
      </div>

      {applyResult && (
        <div
          className={`apply-result ${applyResult.changed ? 'apply-changed' : 'apply-noop'}`}
          onClick={() => setApplyResult(null)}
        >
          {applyResult.changed ? (
            <>
              <strong>strategies.yaml mis à jour</strong>
              {applyResult.applied.length > 0 && (
                <span className="apply-applied">
                  {' '}Appliqués : {applyResult.applied.join(', ')}
                </span>
              )}
              {applyResult.removed.length > 0 && (
                <span className="apply-removed">
                  {' '}Retirés : {applyResult.removed.join(', ')}
                </span>
              )}
              {applyResult.assets_added?.length > 0 && (
                <span className="apply-assets-added">
                  {' '}Ajoutés dans assets.yaml : {applyResult.assets_added.join(', ')}
                </span>
              )}
              <span className="apply-dismiss"> (cliquer pour fermer)</span>
            </>
          ) : (
            <span>Aucun changement — strategies.yaml inchangé</span>
          )}
        </div>
      )}

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
                Score <InfoTooltip term="total_score" /> {sortBy === 'total_score' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('oos_sharpe')}>
                OOS Sharpe <InfoTooltip term="oos_sharpe" /> {sortBy === 'oos_sharpe' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('consistency')}>
                Consistance <InfoTooltip term="consistency" /> {sortBy === 'consistency' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('oos_is_ratio')}>
                OOS/IS <InfoTooltip term="oos_is_ratio" /> {sortBy === 'oos_is_ratio' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('dsr')}>
                DSR <InfoTooltip term="dsr" /> {sortBy === 'dsr' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('param_stability')}>
                Stabilité <InfoTooltip term="param_stability" /> {sortBy === 'param_stability' && (sortDir === 'asc' ? '↑' : '↓')}
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
    </div>
  )
}
