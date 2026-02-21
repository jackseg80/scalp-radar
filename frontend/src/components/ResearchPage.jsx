/**
 * ResearchPage — Visualisation des résultats WFO en DB
 * Sprint 13
 */

import { useState, useMemo, useEffect, useCallback } from 'react'
import WfoChart from './WfoChart'
import InfoTooltip from './InfoTooltip'
import StrategySummaryPanel from './StrategySummaryPanel'
import { usePersistedObject } from '../hooks/usePersistedState'
import './ResearchPage.css'

const GRADE_COLORS = {
  A: '#10b981',
  B: '#3b82f6',
  C: '#f59e0b',
  D: '#f97316',
  F: '#ef4444',
}

export default function ResearchPage({ onTabChange, evalStrategy, setEvalStrategy }) {
  // États persistés dans localStorage
  const [persistedState, updatePersistedState] = usePersistedObject('research-page', {
    view: 'table',
    selectedId: null,
    filters: {
      strategy: '',
      asset: '',
      minGrade: '',
    },
    sortBy: 'total_score',
    sortDir: 'desc',
  })

  // Raccourcis locaux pour faciliter l'accès
  const { view, selectedId, filters, sortBy, sortDir } = persistedState
  const setView = (v) => updatePersistedState({ view: v })
  const setSelectedId = (id) => updatePersistedState({ selectedId: id })
  const setFilters = (f) => updatePersistedState({ filters: f })
  const setSortBy = (s) => updatePersistedState({ sortBy: s })
  const setSortDir = (d) => updatePersistedState({ sortDir: d })

  // Sync prop evalStrategy → filters.strategy au montage (navigation depuis StrategyDetail)
  // L'événement eval-strategy-change est dispatché AVANT le changement de tab, donc avant
  // le montage de ResearchPage — ce useEffect rattrape la valeur via la prop React.
  useEffect(() => {
    if (evalStrategy && evalStrategy !== filters.strategy) {
      setFilters({ ...filters, strategy: evalStrategy })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evalStrategy])

  // Sync via CustomEvent (EvalBar -> ResearchPage et inversement)
  useEffect(() => {
    const handler = (e) => {
      const val = e.detail || ''
      if (val !== filters.strategy) {
        setFilters({ ...filters, strategy: val })
      }
    }
    window.addEventListener('eval-strategy-change', handler)
    return () => window.removeEventListener('eval-strategy-change', handler)
  }, [filters.strategy]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleStrategyFilter = useCallback((value) => {
    setFilters({ ...filters, strategy: value })
    if (setEvalStrategy) setEvalStrategy(value)
    window.dispatchEvent(new CustomEvent('eval-strategy-change', { detail: value }))
  }, [filters, setEvalStrategy]) // eslint-disable-line react-hooks/exhaustive-deps

  // API: liste des résultats (fetch au montage uniquement, pas de polling)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchResults = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ limit: '500' })
      if (filters.strategy) params.append('strategy', filters.strategy)
      const resp = await fetch(`/api/optimization/results?${params}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const json = await resp.json()
      setResults(json)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [filters.strategy]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    fetchResults()
  }, [fetchResults])

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

  // Stratégies : fetch depuis endpoint dédié (pas paginé)
  const [strategies, setStrategies] = useState([])
  useEffect(() => {
    fetch('/api/optimization/strategies')
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data?.strategies) setStrategies(data.strategies) })
      .catch(() => {})
  }, [])

  const assets = useMemo(() => {
    if (!results || !results.results) return []
    return [...new Set(results.results.map(r => r.asset))].sort()
  }, [results])

  // Apply params
  const [applyResult, setApplyResult] = useState(null)
  const [applying, setApplying] = useState(false)
  const [tfConflict, setTfConflict] = useState(null)

  const handleApply = async () => {
    const stratLabel = filters.strategy || 'toutes les stratégies'
    const ok = window.confirm(
      `Appliquer les params Grade A/B de "${stratLabel}" dans strategies.yaml ?`
    )
    if (!ok) return

    setApplying(true)
    setApplyResult(null)
    setTfConflict(null)
    try {
      const qs = new URLSearchParams()
      if (filters.strategy) qs.set('strategy_name', filters.strategy)
      const resp = await fetch(`/api/optimization/apply?${qs}`, { method: 'POST' })
      if (resp.status === 409) {
        const err = await resp.json().catch(() => ({}))
        const detail = err.detail || {}
        setTfConflict({
          majority_tf: detail.majority_tf || '?',
          tf_outliers: detail.tf_outliers || [],
          strategy_name: filters.strategy || 'toutes',
        })
        return
      }
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

        {/* Trades OOS par fenêtre */}
        {detail.wfo_windows && (() => {
          let windows = detail.wfo_windows
          if (typeof windows === 'string') try { windows = JSON.parse(windows) } catch { return null }
          if (typeof windows === 'string') try { windows = JSON.parse(windows) } catch { return null }
          if (!Array.isArray(windows)) windows = windows?.windows || []
          if (!Array.isArray(windows) || windows.length === 0) return null

          const trades = windows.map(w => w.oos_trades || 0)
          const total = trades.reduce((a, b) => a + b, 0)
          const avg = total / trades.length
          const min = Math.min(...trades)
          const max = Math.max(...trades)

          return (
            <section className="detail-section">
              <h3>Trades OOS par fenetre</h3>
              <div className="oos-summary">
                <div className="metric-card"><div className="metric-label">Total OOS</div><div className="metric-value">{total}</div></div>
                <div className="metric-card"><div className="metric-label">Moyenne</div><div className="metric-value">{avg.toFixed(1)}</div></div>
                <div className="metric-card"><div className="metric-label">Min</div><div className="metric-value">{min}</div></div>
                <div className="metric-card"><div className="metric-label">Max</div><div className="metric-value">{max}</div></div>
              </div>
              {total < 50 && (
                <p style={{color: '#f59e0b', fontSize: '13px', marginTop: '8px', marginBottom: '12px'}}>
                  {'\u26A0\uFE0F'} Moins de 50 trades OOS — resultats statistiquement fragiles
                </p>
              )}
              <div className="oos-bars">
                {windows.map((w, i) => {
                  const t = w.oos_trades || 0
                  const pct = max > 0 ? (t / max) * 100 : 0
                  return (
                    <div key={i} className="oos-bar-row" title={`Fenetre ${i+1}: ${t} trades`}>
                      <span className="oos-bar-label">W{i+1}</span>
                      <div className="oos-bar-bg">
                        <div className="oos-bar-fill" style={{width: `${pct}%`, background: t < 10 ? '#ef4444' : t < 30 ? '#f59e0b' : '#10b981'}} />
                      </div>
                      <span className="oos-bar-value">{t}</span>
                    </div>
                  )
                })}
              </div>
            </section>
          )
        })()}

        {/* Lien Explorer */}
        <div className="detail-explorer-link">
          <span className="muted">Pour l'analyse detaillee (heatmap, scatter, distribution) {'\u2192'}</span>
          <button className="btn-link" onClick={() => {
            localStorage.setItem('explorer-strategy', detail.strategy_name)
            localStorage.setItem('explorer-asset', detail.asset)
            onTabChange?.('explorer')
          }}>
            Ouvrir dans Explorer
          </button>
        </div>

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
            onChange={(e) => handleStrategyFilter(e.target.value)}
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

      {tfConflict && (
        <div className="tf-conflict-modal">
          <div className="tf-conflict-content">
            <h3>❌ Conflit de timeframe — apply bloqué</h3>
            <p>
              Timeframe majoritaire : <strong>{tfConflict.majority_tf}</strong><br />
              Outliers ({tfConflict.tf_outliers.length} asset{tfConflict.tf_outliers.length > 1 ? 's' : ''}) :
            </p>
            <ul>
              {tfConflict.tf_outliers.map(s => <li key={s}><code>{s}</code></li>)}
            </ul>
            <p>Actions :</p>
            <ol>
              <li>Re-tester en {tfConflict.majority_tf} :
                <code className="cmd">
                  uv run python -m scripts.optimize --strategy {tfConflict.strategy_name}{' '}
                  --symbols {tfConflict.tf_outliers.join(',')} --force-timeframe {tfConflict.majority_tf}
                </code>
              </li>
              <li>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    const qs = new URLSearchParams()
                    if (filters.strategy) qs.set('strategy_name', filters.strategy)
                    qs.set('exclude', tfConflict.tf_outliers.join(','))
                    fetch(`/api/optimization/apply?${qs}`, { method: 'POST' })
                      .then(r => r.json()).then(j => { setApplyResult(j); setTfConflict(null) })
                  }}
                >
                  Exclure les outliers et appliquer
                </button>
              </li>
              <li>
                <button
                  className="btn-secondary"
                  onClick={() => {
                    const qs = new URLSearchParams()
                    if (filters.strategy) qs.set('strategy_name', filters.strategy)
                    qs.set('ignore_tf_conflicts', 'true')
                    fetch(`/api/optimization/apply?${qs}`, { method: 'POST' })
                      .then(r => r.json()).then(j => { setApplyResult(j); setTfConflict(null) })
                  }}
                >
                  Forcer (exclure silencieusement)
                </button>
              </li>
            </ol>
            <button className="btn-close" onClick={() => setTfConflict(null)}>Fermer</button>
          </div>
        </div>
      )}

      {filters.strategy && (
        <StrategySummaryPanel
          strategyName={filters.strategy}
          onNavigatePortfolio={() => onTabChange?.('portfolio')}
        />
      )}

      {filters.strategy && filteredResults.length === 0 && (
        <div className="info-banner">
          <span>Aucun resultat pour {filters.strategy} avec les filtres actuels</span>
          {filters.minGrade && (
            <button className="btn-link" onClick={() => setFilters({...filters, minGrade: ''})}>
              Retirer le filtre grade ({filters.minGrade})
            </button>
          )}
          {filters.asset && (
            <button className="btn-link" onClick={() => setFilters({...filters, asset: ''})}>
              Retirer le filtre asset
            </button>
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
              <th style={{ width: '14%' }} onClick={() => handleSort('strategy_name')}>
                Stratégie {sortBy === 'strategy_name' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '12%' }} onClick={() => handleSort('asset')}>
                Asset {sortBy === 'asset' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '5%' }}>TF</th>
              <th style={{ width: '7%' }} onClick={() => handleSort('grade')}>
                Grade {sortBy === 'grade' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '8%' }} onClick={() => handleSort('total_score')}>
                Score <InfoTooltip term="total_score" /> {sortBy === 'total_score' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '11%' }} onClick={() => handleSort('oos_sharpe')}>
                Sharpe <InfoTooltip term="oos_sharpe" /> {sortBy === 'oos_sharpe' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '11%' }} onClick={() => handleSort('consistency')}>
                Consist. <InfoTooltip term="consistency" /> {sortBy === 'consistency' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '9%' }} onClick={() => handleSort('oos_is_ratio')}>
                OOS/IS <InfoTooltip term="oos_is_ratio" /> {sortBy === 'oos_is_ratio' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '7%' }} onClick={() => handleSort('dsr')}>
                DSR <InfoTooltip term="dsr" /> {sortBy === 'dsr' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '10%' }} onClick={() => handleSort('param_stability')}>
                Stab. <InfoTooltip term="param_stability" /> {sortBy === 'param_stability' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '11%' }} onClick={() => handleSort('created_at')}>
                Date {sortBy === 'created_at' && (sortDir === 'asc' ? '↑' : '↓')}
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredResults.length === 0 ? (
              <tr>
                <td colSpan={11} style={{ textAlign: 'center', color: '#888', padding: '40px' }}>
                  Aucun résultat trouvé
                </td>
              </tr>
            ) : (
              filteredResults.map(r => (
                <tr key={r.id} onClick={() => fetchDetail(r.id)} className="clickable-row">
                  <td>{r.strategy_name}</td>
                  <td>{r.asset}</td>
                  <td>
                    <span
                      className={`timeframe-badge${r.timeframe && r.timeframe !== '1h' ? ' timeframe-badge--warn' : ''}`}
                      title={r.timeframe && r.timeframe !== '1h'
                        ? `Optimisé en ${r.timeframe}. Incompatible avec paper/live (1h). Re-testez avec --force-timeframe 1h.`
                        : `Timeframe : ${r.timeframe || '1h'}`}
                    >
                      {r.timeframe || '1h'}
                    </span>
                  </td>
                  <td>
                    <span className={`grade-badge grade-${r.grade}`}>{r.grade}</span>
                  </td>
                  <td>{r.total_score}</td>
                  <td>{r.oos_sharpe?.toFixed(2) ?? 'N/A'}</td>
                  <td>{((r.consistency || 0) * 100).toFixed(0)}%</td>
                  <td>{r.oos_is_ratio?.toFixed(2) ?? 'N/A'}</td>
                  <td>{r.dsr?.toFixed(2) ?? 'N/A'}</td>
                  <td>{r.param_stability?.toFixed(2) ?? 'N/A'}</td>
                  <td>
                    {new Date(r.created_at).toLocaleDateString('fr-FR')}
                    <span style={{ fontSize: '0.75em', color: '#888', marginLeft: '4px' }}>
                      {new Date(r.created_at).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </td>
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
