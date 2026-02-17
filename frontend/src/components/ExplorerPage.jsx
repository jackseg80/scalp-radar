/**
 * ExplorerPage — Explorateur de Paramètres avec layout grid professionnel
 * Sprint 14
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import HeatmapChart from './HeatmapChart'
import Top10Table from './Top10Table'
import ScatterChart from './ScatterChart'
import DistributionChart from './DistributionChart'
import InfoTooltip from './InfoTooltip'
import DiagnosticPanel from './DiagnosticPanel'
import ExportButton from './ExportDiagnostic'
import { usePersistedState } from '../hooks/usePersistedState'
import './ExplorerPage.css'

const STATUS_COLORS = {
  pending: '#f59e0b',
  running: '#3b82f6',
  completed: '#10b981',
  failed: '#ef4444',
  cancelled: '#6b7280',
}

export default function ExplorerPage({ wsData }) {
  // États persistés (sélections utilisateur)
  const [strategy, setStrategy] = usePersistedState('explorer-strategy', '')
  const [asset, setAsset] = usePersistedState('explorer-asset', '')
  const [paramsOverride, setParamsOverride] = usePersistedState('explorer-params', {})
  const [activeParamsArray, setActiveParamsArray] = usePersistedState('explorer-active-params', [])
  const activeParams = useMemo(() => new Set(activeParamsArray), [activeParamsArray])
  const setActiveParams = useCallback((setOrUpdater) => {
    if (typeof setOrUpdater === 'function') {
      setActiveParamsArray(prev => Array.from(setOrUpdater(new Set(prev))))
    } else {
      setActiveParamsArray(Array.from(setOrUpdater))
    }
  }, [setActiveParamsArray])

  const [axisX, setAxisX] = usePersistedState('explorer-axis-x', '')
  const [axisY, setAxisY] = usePersistedState('explorer-axis-y', '')
  const [metric, setMetric] = usePersistedState('explorer-metric', 'oos_sharpe')
  const [paramsExpanded, setParamsExpanded] = usePersistedState('explorer-params-expanded', false)
  const [selectedRunId, setSelectedRunId] = usePersistedState('explorer-run-id', null)

  // États non-persistés (données temporaires)
  const [paramGrid, setParamGrid] = useState(null)
  const [heatmapData, setHeatmapData] = useState(null)
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [availableRuns, setAvailableRuns] = useState([])
  const [comboResults, setComboResults] = useState(null)

  // Stratégies disponibles (chargées dynamiquement depuis le backend)
  const [strategies, setStrategies] = useState([])

  // Assets disponibles (chargés dynamiquement depuis les résultats d'optimisation)
  const [assets, setAssets] = useState([])

  // Métriques disponibles (Sprint 14b : nouvelles métriques combo_results)
  const metrics = [
    { value: 'oos_sharpe', label: 'OOS Sharpe' },
    { value: 'oos_return_pct', label: 'OOS Return %' },
    { value: 'consistency', label: 'Consistance' },
    { value: 'oos_is_ratio', label: 'Ratio OOS/IS' },
    { value: 'is_sharpe', label: 'IS Sharpe' },
    { value: 'total_score', label: 'Score Total (legacy)' },
  ]

  // Charger la liste des stratégies optimisables au montage
  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        const resp = await fetch('/api/optimization/strategies')
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json()
        setStrategies(data.strategies || [])
      } catch (err) {
        console.error('Erreur fetch strategies:', err)
        setStrategies([])
      }
    }
    fetchStrategies()
  }, [])

  // Charger les assets dynamiquement depuis les résultats d'optimisation
  useEffect(() => {
    const fetchAssets = async () => {
      try {
        const resp = await fetch('/api/optimization/results?limit=500&latest_only=false')
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json()
        const uniqueAssets = [...new Set((data.results || []).map(r => r.asset))].sort()
        setAssets(uniqueAssets)
        // Reset sélection si l'asset actuel n'est plus dans la liste
        if (asset && uniqueAssets.length > 0 && !uniqueAssets.includes(asset)) {
          setAsset('')
        }
      } catch (err) {
        console.error('Erreur fetch assets:', err)
      }
    }
    fetchAssets()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Charger la grille de paramètres quand la stratégie change
  useEffect(() => {
    if (!strategy) {
      setParamGrid(null)
      setParamsOverride({})
      setAxisX('')
      setAxisY('')
      return
    }

    const fetchParamGrid = async () => {
      try {
        const resp = await fetch(`/api/optimization/param-grid/${strategy}`)
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json()
        setParamGrid(data)

        // Ne PAS pré-remplir params_override — rester vide jusqu'à ce que l'utilisateur touche un slider
        // (sinon ça écrase le grid entier avec 1 seule combo au lieu de 324)
        // setParamsOverride({}) déjà fait ligne 68, on ne refait rien ici

        // Sélectionner les premiers params pour les axes (si disponibles)
        const paramNames = Object.keys(data.params)
        if (paramNames.length >= 2) {
          setAxisX(paramNames[0])
          setAxisY(paramNames[1])
        }
      } catch (err) {
        console.error('Erreur fetch param-grid:', err)
      }
    }

    fetchParamGrid()
  }, [strategy])

  // Fonction réutilisable pour charger les runs disponibles
  const fetchAvailableRuns = async (autoSelect = true) => {
    if (!strategy || !asset) {
      setAvailableRuns([])
      setSelectedRunId(null)
      setComboResults(null)
      return
    }

    try {
      const params = new URLSearchParams({
        strategy,
        asset,
        latest_only: 'false',
        limit: '20',
      })
      const resp = await fetch(`/api/optimization/results?${params}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()

      setAvailableRuns(data.results || [])

      if (!autoSelect) return

      // Auto-sélectionner le meilleur run : préférer le run avec le plus de combos
      // (un run à 1-5 combos = params verrouillés, pas de heatmap utile)
      const runs = data.results || []
      const MIN_COMBOS_FOR_HEATMAP = 10
      const fullRuns = runs.filter((r) => (r.combo_count || 0) >= MIN_COMBOS_FOR_HEATMAP)

      if (fullRuns.length > 0) {
        // Parmi les runs complets, prendre le plus récent
        const bestRun = fullRuns.reduce((a, b) =>
          new Date(b.created_at) > new Date(a.created_at) ? b : a
        )
        setSelectedRunId(bestRun.id)
      } else if (runs.length > 0) {
        // Fallback : prendre le premier run (trié par score desc)
        setSelectedRunId(runs[0].id)
      }
    } catch (err) {
      console.error('Erreur fetch available runs:', err)
      setAvailableRuns([])
    }
  }

  // Sprint 14b : Charger les runs disponibles quand strategy/asset changent
  useEffect(() => {
    fetchAvailableRuns()
  }, [strategy, asset])

  // Charger la heatmap quand strategy/asset/axisX/axisY/metric/selectedRunId changent
  useEffect(() => {
    if (!strategy || !asset || !axisX || !axisY) {
      setHeatmapData(null)
      return
    }

    const fetchHeatmap = async () => {
      try {
        const params = new URLSearchParams({
          strategy,
          asset,
          param_x: axisX,
          param_y: axisY,
          metric,
        })

        // Sprint 14b : passer result_id si sélectionné
        if (selectedRunId) {
          params.append('result_id', selectedRunId)
        }

        const resp = await fetch(`/api/optimization/heatmap?${params}`)
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json()
        setHeatmapData(data)
      } catch (err) {
        console.error('Erreur fetch heatmap:', err)
        setHeatmapData(null)
      }
    }

    fetchHeatmap()
  }, [strategy, asset, axisX, axisY, metric, selectedRunId])

  // Sprint 14b : Charger les combo results quand selectedRunId change
  useEffect(() => {
    if (!selectedRunId) {
      setComboResults(null)
      return
    }

    const fetchComboResults = async () => {
      try {
        const resp = await fetch(`/api/optimization/combo-results/${selectedRunId}`)
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json()
        setComboResults(data)
      } catch (err) {
        console.error('Erreur fetch combo-results:', err)
        setComboResults(null)
      }
    }

    fetchComboResults()
  }, [selectedRunId])

  // Charger les jobs au montage
  useEffect(() => {
    fetchJobs()
  }, [])

  // Écouter les updates WebSocket pour mettre à jour les jobs en temps réel
  useEffect(() => {
    if (wsData?.type === 'optimization_progress') {
      setJobs((prev) =>
        prev.map((j) =>
          j.id === wsData.job_id
            ? {
                ...j,
                status: wsData.status,
                progress_pct: wsData.progress_pct,
                current_phase: wsData.current_phase,
              }
            : j
        )
      )

      // Si le job est completed, recharger les runs + jobs + heatmap
      if (wsData.status === 'completed') {
        fetchJobs()
        fetchAvailableRuns()  // Rafraîchir la liste des runs et auto-sélectionner le nouveau
      }
    }
  }, [wsData])

  const fetchJobs = async () => {
    try {
      const resp = await fetch('/api/optimization/jobs')
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setJobs(data.jobs || [])
    } catch (err) {
      console.error('Erreur fetch jobs:', err)
    }
  }

  const handleSubmitJob = async () => {
    if (!strategy || !asset) {
      alert('Sélectionner une stratégie et un asset')
      return
    }

    // Vérifier qu'il n'y a pas déjà un job running/pending pour cette combo
    const alreadyRunning = jobs.some(
      (j) =>
        j.strategy_name === strategy &&
        j.asset === asset &&
        (j.status === 'pending' || j.status === 'running')
    )
    if (alreadyRunning) {
      alert(`Un job pour ${strategy} × ${asset} est déjà en cours`)
      return
    }

    setLoading(true)
    setError(null)
    try {
      // Construire params_override : seulement les params actifs
      const override = {}
      activeParams.forEach((paramName) => {
        if (paramsOverride[paramName] !== undefined) {
          override[paramName] = paramsOverride[paramName]
        }
      })

      // Si aucun param actif → envoyer null (grille complète)
      const finalOverride = Object.keys(override).length > 0 ? override : null

      const resp = await fetch('/api/optimization/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_name: strategy,
          asset,
          params_override: finalOverride,
        }),
      })

      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }

      // Recharger les jobs
      await fetchJobs()
    } catch (err) {
      setError(err.message)
      alert(`Erreur: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleCancelJob = async (jobId) => {
    try {
      const resp = await fetch(`/api/optimization/jobs/${jobId}`, {
        method: 'DELETE',
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      await fetchJobs()
    } catch (err) {
      alert(`Erreur annulation: ${err.message}`)
    }
  }

  const handleReset = () => {
    // Vider params_override et désactiver tous les sliders → grille complète
    setParamsOverride({})
    setActiveParams(new Set())
  }

  // Toggle activation d'un paramètre
  const toggleParamActive = (paramName) => {
    setActiveParams((prev) => {
      const next = new Set(prev)
      if (next.has(paramName)) {
        // Désactiver : retirer du set ET de paramsOverride
        next.delete(paramName)
        setParamsOverride((prevOverride) => {
          const { [paramName]: _, ...rest } = prevOverride
          return rest
        })
      } else {
        // Activer : ajouter au set ET fixer à la valeur default
        next.add(paramName)
        if (paramGrid) {
          const defaultVal = paramGrid.params[paramName]?.default
          if (defaultVal !== undefined) {
            setParamsOverride((prevOverride) => ({
              ...prevOverride,
              [paramName]: defaultVal,
            }))
          }
        }
      }
      return next
    })
  }

  // Calculer le nombre de combos selon les params actifs
  const comboCount = useMemo(() => {
    if (!paramGrid) return 0
    let count = 1
    Object.entries(paramGrid.params).forEach(([pName, pConfig]) => {
      if (activeParams.has(pName)) {
        count *= 1 // Param fixé → 1 seule valeur
      } else {
        count *= pConfig.values.length // Toutes les valeurs
      }
    })
    return count
  }, [paramGrid, activeParams])

  // Nombre de jobs running/pending
  const activeJobsCount = useMemo(() => {
    return jobs.filter((j) => j.status === 'pending' || j.status === 'running').length
  }, [jobs])

  // Sprint 14c : Run sélectionné (pour DiagnosticPanel)
  const selectedRun = useMemo(() => {
    return availableRuns.find((r) => r.id === selectedRunId) || null
  }, [availableRuns, selectedRunId])

  return (
    <div className="explorer-page">
      {/* Layout Grid : Config (gauche) + Heatmap (centre) */}
      <div className="explorer-main">
        {/* Panel de configuration (gauche) */}
        <aside className="config-panel">
          <h3>Configuration</h3>

          <div className="form-group">
            <label>Stratégie</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="select-input"
            >
              <option value="">Sélectionner...</option>
              {strategies.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Asset</label>
            <select
              value={asset}
              onChange={(e) => setAsset(e.target.value)}
              className="select-input"
            >
              <option value="">Sélectionner...</option>
              {assets.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </div>

          {availableRuns.length > 0 && (
            <div className="form-group">
              <label>Run WFO (historique)</label>
              <select
                value={selectedRunId || ''}
                onChange={(e) => setSelectedRunId(e.target.value ? parseInt(e.target.value, 10) : null)}
                className="select-input"
              >
                {availableRuns.map((run) => {
                  const date = new Date(run.created_at).toLocaleDateString('fr-FR')
                  const time = new Date(run.created_at).toLocaleTimeString('fr-FR', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })
                  const combos = run.combo_count || 0
                  const comboInfo = combos > 0 ? ` ${combos}c` : ''
                  const label = `${date} ${time} — Grade ${run.grade} (${run.total_score?.toFixed(0) || 0})${comboInfo}${
                    run.is_latest ? ' [latest]' : ''
                  }`
                  return (
                    <option key={run.id} value={run.id}>
                      {label}
                    </option>
                  )
                })}
              </select>
            </div>
          )}

          {paramGrid && (
            <>
              <div className="divider" />
              <h4
                className="collapsible-header"
                onClick={() => setParamsExpanded((v) => !v)}
              >
                <span className={`collapse-arrow ${paramsExpanded ? 'expanded' : ''}`}>▶</span>
                Sous-grille de paramètres
                <span className="combo-badge">{comboCount} combos</span>
              </h4>
              {paramsExpanded && (
                <>
                  <div className="params-list">
                    {Object.entries(paramGrid.params).map(([pName, pConfig]) => {
                      const { values, default: defaultVal } = pConfig
                      const isActive = activeParams.has(pName)
                      const currentVal = isActive ? (paramsOverride[pName] ?? defaultVal) : null
                      const idx = isActive && currentVal !== null ? values.indexOf(currentVal) : 0

                      return (
                        <div key={pName} className={`param-item ${isActive ? 'active' : 'inactive'}`}>
                          <div className="param-header">
                            <input
                              type="checkbox"
                              checked={isActive}
                              onChange={() => toggleParamActive(pName)}
                              className="param-checkbox"
                            />
                            <label onClick={() => toggleParamActive(pName)} className="param-label">
                              {pName}
                              {isActive ? ` : ${currentVal}` : ` : toutes (${values.join(', ')})`}
                            </label>
                          </div>
                          <div className="param-controls">
                            <input
                              type="range"
                              min={0}
                              max={values.length - 1}
                              step={1}
                              value={idx >= 0 ? idx : 0}
                              onChange={(e) => {
                                if (!isActive) return
                                const newIdx = parseInt(e.target.value, 10)
                                setParamsOverride((prev) => ({
                                  ...prev,
                                  [pName]: values[newIdx],
                                }))
                              }}
                              className="range-input"
                              disabled={!isActive}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  <div className="combo-count-summary">
                    <strong>Grille :</strong> {comboCount} combo{comboCount > 1 ? 's' : ''}
                    {comboCount > 1 && paramGrid && (
                      <span className="combo-breakdown">
                        {' '}
                        (
                        {Object.entries(paramGrid.params)
                          .map(([pName, pConfig]) => (activeParams.has(pName) ? '1' : pConfig.values.length))
                          .join('×')}
                        )
                      </span>
                    )}
                  </div>
                </>
              )}

              <div className="divider" />
              <h4>Axes de la heatmap</h4>
              <div className="form-group">
                <label>Axe X</label>
                <select
                  value={axisX}
                  onChange={(e) => setAxisX(e.target.value)}
                  className="select-input"
                >
                  {Object.keys(paramGrid.params).map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Axe Y</label>
                <select
                  value={axisY}
                  onChange={(e) => setAxisY(e.target.value)}
                  className="select-input"
                >
                  {Object.keys(paramGrid.params).map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Métrique</label>
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                  className="select-input"
                >
                  {metrics.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>
            </>
          )}

          <div className="divider" />
          <div className="action-buttons">
            <button
              onClick={handleSubmitJob}
              disabled={!strategy || !asset || loading || activeJobsCount >= 5}
              className="btn btn-primary"
            >
              {loading ? 'Lancement...' : 'Lancer WFO'}
            </button>
            <button onClick={handleReset} disabled={!paramGrid} className="btn btn-secondary">
              Réinitialiser (grille complète)
            </button>
          </div>
          {activeJobsCount >= 5 && (
            <p className="warning-text">Queue pleine (5 jobs max simultanés)</p>
          )}
        </aside>

        {/* Zone centrale : Heatmap */}
        <div className="heatmap-container">
          {!strategy || !asset ? (
            <div className="empty-state">
              <p>Sélectionnez une stratégie et un asset pour commencer</p>
            </div>
          ) : !heatmapData || heatmapData.x_values.length === 0 ? (
            <div className="empty-state">
              <p>Aucun résultat existant pour cette combinaison.</p>
              <p>Lancez un WFO pour remplir la heatmap.</p>
            </div>
          ) : (
            <>
              <div className="heatmap-header">
                <h3>
                  Heatmap {strategy} × {asset}
                </h3>
                <p className="heatmap-subtitle">
                  Axe X: {axisX} | Axe Y: {axisY} | Métrique: {metric}
                  <InfoTooltip term={metric} />
                </p>
              </div>
              <HeatmapChart data={heatmapData} />
            </>
          )}
        </div>
      </div>

      {/* Sprint 14b : Section Analyse (charts analytiques) */}
      {comboResults && comboResults.combos && comboResults.combos.length > 0 && (
        <div className="analysis-section">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <h3 style={{ margin: 0 }}>Analyse des combos ({comboResults.combos.length} testées)</h3>
            {selectedRun && (
              <ExportButton
                strategy={strategy}
                asset={asset}
                selectedRun={selectedRun}
                combos={comboResults.combos}
                regimeAnalysis={comboResults.regime_analysis || null}
              />
            )}
          </div>

          {/* Sprint 14c : Diagnostic — première chose visible */}
          <DiagnosticPanel
            combos={comboResults.combos}
            grade={selectedRun?.grade || '?'}
            totalScore={selectedRun?.total_score || 0}
            nWindows={Math.max(...comboResults.combos.map((c) => c.n_windows_evaluated || 0))}
            regimeAnalysis={comboResults.regime_analysis || null}
          />

          {/* Top 10 pleine largeur */}
          <div className="analysis-top10">
            <Top10Table
              combos={comboResults.combos}
              paramNames={paramGrid ? Object.keys(paramGrid.params) : []}
              metric={metric}
            />
          </div>

          {/* Scatter + Distribution en grid 2 colonnes */}
          <div className="analysis-charts">
            <ScatterChart combos={comboResults.combos} />
            <DistributionChart combos={comboResults.combos} />
          </div>
        </div>
      )}

      {/* Jobs d'optimisation (bas, pleine largeur) */}
      <div className="jobs-section">
        <div className="jobs-header">
          <h3>Jobs d'optimisation</h3>
          <button onClick={fetchJobs} className="btn-refresh">
            ↻ Rafraîchir
          </button>
        </div>

        <div className="jobs-table-container">
          <table className="jobs-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Stratégie</th>
                <th>Asset</th>
                <th>Status</th>
                <th>Progression</th>
                <th>Durée</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', color: '#888', padding: '20px' }}>
                    Aucun job en cours ou terminé
                  </td>
                </tr>
              ) : (
                jobs.map((job) => (
                  <tr key={job.id}>
                    <td>
                      <span className="job-id">{job.id.substring(0, 8)}</span>
                    </td>
                    <td>{job.strategy_name}</td>
                    <td>{job.asset}</td>
                    <td>
                      <span
                        className="status-badge"
                        style={{ background: STATUS_COLORS[job.status] }}
                      >
                        {job.status}
                      </span>
                    </td>
                    <td>
                      <div className="progress-cell">
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{ width: `${job.progress_pct || 0}%` }}
                          />
                        </div>
                        <span className="progress-text">
                          {job.progress_pct ? `${job.progress_pct.toFixed(0)}%` : '—'}
                        </span>
                        {job.current_phase && (
                          <span className="progress-phase">{job.current_phase}</span>
                        )}
                      </div>
                    </td>
                    <td>
                      {job.duration_seconds
                        ? `${(job.duration_seconds / 60).toFixed(1)} min`
                        : '—'}
                    </td>
                    <td>
                      {job.status === 'pending' || job.status === 'running' ? (
                        <button
                          onClick={() => handleCancelJob(job.id)}
                          className="btn-cancel"
                        >
                          Annuler
                        </button>
                      ) : null}
                      {job.status === 'failed' && job.error_message && (
                        <span className="error-hint" title={job.error_message}>
                          ⚠
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
