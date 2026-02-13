/**
 * ExplorerPage — Explorateur de Paramètres avec layout grid professionnel
 * Sprint 14
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import HeatmapChart from './HeatmapChart'

const STATUS_COLORS = {
  pending: '#f59e0b',
  running: '#3b82f6',
  completed: '#10b981',
  failed: '#ef4444',
  cancelled: '#6b7280',
}

export default function ExplorerPage({ wsData }) {
  const [strategy, setStrategy] = useState('')
  const [asset, setAsset] = useState('')
  const [paramGrid, setParamGrid] = useState(null)
  const [paramsOverride, setParamsOverride] = useState({})
  const [axisX, setAxisX] = useState('')
  const [axisY, setAxisY] = useState('')
  const [metric, setMetric] = useState('total_score')
  const [heatmapData, setHeatmapData] = useState(null)
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Stratégies disponibles
  const strategies = [
    'vwap_rsi',
    'momentum',
    'funding',
    'liquidation',
    'bollinger_mr',
    'donchian_breakout',
    'supertrend',
    'envelope_dca',
  ]

  // Assets disponibles
  const assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'LINK/USDT']

  // Métriques disponibles
  const metrics = [
    { value: 'total_score', label: 'Score Total' },
    { value: 'oos_sharpe', label: 'OOS Sharpe' },
    { value: 'consistency', label: 'Consistance' },
    { value: 'dsr', label: 'DSR' },
  ]

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

        // Init params_override avec les valeurs par défaut
        const defaults = {}
        Object.entries(data.params).forEach(([key, config]) => {
          defaults[key] = config.default
        })
        setParamsOverride(defaults)

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

  // Charger la heatmap quand strategy/asset/axisX/axisY/metric changent
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
  }, [strategy, asset, axisX, axisY, metric])

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

      // Si le job est completed, recharger la heatmap et les jobs
      if (wsData.status === 'completed') {
        fetchJobs()
        // Trigger heatmap refresh (la heatmap se met à jour via useEffect déjà)
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
      const resp = await fetch('/api/optimization/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy_name: strategy,
          asset,
          params_override: paramsOverride,
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
    if (!paramGrid) return
    const defaults = {}
    Object.entries(paramGrid.params).forEach(([key, config]) => {
      defaults[key] = config.default
    })
    setParamsOverride(defaults)
  }

  // Nombre de jobs running/pending
  const activeJobsCount = useMemo(() => {
    return jobs.filter((j) => j.status === 'pending' || j.status === 'running').length
  }, [jobs])

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

          {paramGrid && (
            <>
              <div className="divider" />
              <h4>Sous-grille de paramètres</h4>
              <div className="params-list">
                {Object.entries(paramGrid.params).map(([pName, pConfig]) => {
                  const { values, default: defaultVal } = pConfig
                  const currentVal = paramsOverride[pName] ?? defaultVal
                  const idx = values.indexOf(currentVal)

                  return (
                    <div key={pName} className="param-item">
                      <label>{pName}</label>
                      <div className="param-controls">
                        <input
                          type="range"
                          min={0}
                          max={values.length - 1}
                          step={1}
                          value={idx >= 0 ? idx : 0}
                          onChange={(e) => {
                            const newIdx = parseInt(e.target.value, 10)
                            setParamsOverride((prev) => ({
                              ...prev,
                              [pName]: values[newIdx],
                            }))
                          }}
                          className="range-input"
                        />
                        <span className="param-value">{currentVal}</span>
                      </div>
                    </div>
                  )
                })}
              </div>

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
              Réinitialiser
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
                </p>
              </div>
              <HeatmapChart data={heatmapData} />
            </>
          )}
        </div>
      </div>

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

      <style jsx>{`
        .explorer-page {
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          gap: 16px;
          padding: 16px;
          overflow: hidden;
        }

        /* Layout principal : Config + Heatmap */
        .explorer-main {
          display: grid;
          grid-template-columns: 320px 1fr;
          gap: 16px;
          flex: 1;
          overflow: hidden;
        }

        /* Panel de configuration (gauche) */
        .config-panel {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
          overflow-y: auto;
          max-height: calc(100vh - 200px);
        }

        .config-panel h3 {
          margin: 0 0 20px 0;
          color: #fff;
          font-size: 18px;
          border-bottom: 1px solid #333;
          padding-bottom: 10px;
        }

        .config-panel h4 {
          margin: 16px 0 12px 0;
          color: #ccc;
          font-size: 14px;
          font-weight: 600;
        }

        .divider {
          height: 1px;
          background: #333;
          margin: 20px 0;
        }

        .form-group {
          margin-bottom: 16px;
        }

        .form-group label {
          display: block;
          margin-bottom: 6px;
          color: #aaa;
          font-size: 13px;
          font-weight: 500;
        }

        .select-input {
          width: 100%;
          padding: 8px 10px;
          background: #0d1117;
          border: 1px solid #444;
          border-radius: 4px;
          color: #ccc;
          font-size: 14px;
          cursor: pointer;
          transition: border-color 0.2s;
        }

        .select-input:hover {
          border-color: #666;
        }

        .select-input:focus {
          border-color: #3b82f6;
          outline: none;
        }

        .params-list {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .param-item {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .param-item label {
          color: #aaa;
          font-size: 12px;
          font-weight: 500;
        }

        .param-controls {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .range-input {
          flex: 1;
          height: 6px;
          background: #333;
          border-radius: 3px;
          outline: none;
          cursor: pointer;
        }

        .range-input::-webkit-slider-thumb {
          appearance: none;
          width: 14px;
          height: 14px;
          background: #3b82f6;
          border-radius: 50%;
          cursor: pointer;
        }

        .param-value {
          min-width: 50px;
          text-align: right;
          color: #fff;
          font-size: 13px;
          font-weight: 600;
        }

        .action-buttons {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .btn {
          padding: 10px 16px;
          border: none;
          border-radius: 4px;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-primary {
          background: #3b82f6;
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          background: #2563eb;
        }

        .btn-secondary {
          background: #2a2a2a;
          color: #ccc;
          border: 1px solid #444;
        }

        .btn-secondary:hover:not(:disabled) {
          background: #333;
        }

        .warning-text {
          margin-top: 8px;
          color: #f59e0b;
          font-size: 12px;
          text-align: center;
        }

        /* Zone centrale Heatmap */
        .heatmap-container {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 20px;
          overflow: auto;
          display: flex;
          flex-direction: column;
        }

        .heatmap-header {
          margin-bottom: 20px;
        }

        .heatmap-header h3 {
          margin: 0 0 8px 0;
          color: #fff;
          font-size: 18px;
        }

        .heatmap-subtitle {
          margin: 0;
          color: #888;
          font-size: 13px;
        }

        .empty-state {
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          color: #888;
          text-align: center;
        }

        .empty-state p {
          margin: 8px 0;
          font-size: 14px;
        }

        /* Jobs section (bas) */
        .jobs-section {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 16px;
          max-height: 200px;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .jobs-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }

        .jobs-header h3 {
          margin: 0;
          color: #fff;
          font-size: 16px;
        }

        .btn-refresh {
          padding: 6px 12px;
          background: #2a2a2a;
          border: 1px solid #444;
          border-radius: 4px;
          color: #ccc;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-refresh:hover {
          background: #333;
        }

        .jobs-table-container {
          flex: 1;
          overflow-y: auto;
        }

        .jobs-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }

        .jobs-table thead {
          position: sticky;
          top: 0;
          background: #0d1117;
          z-index: 1;
        }

        .jobs-table th {
          text-align: left;
          padding: 8px 12px;
          color: #aaa;
          font-weight: 600;
          border-bottom: 1px solid #333;
        }

        .jobs-table td {
          padding: 10px 12px;
          color: #ccc;
          border-bottom: 1px solid #222;
        }

        .jobs-table tbody tr:hover {
          background: #1a1f2e;
        }

        .job-id {
          font-family: monospace;
          color: #888;
          font-size: 11px;
        }

        .status-badge {
          display: inline-block;
          padding: 3px 8px;
          border-radius: 3px;
          font-size: 11px;
          font-weight: 600;
          color: white;
          text-transform: uppercase;
        }

        .progress-cell {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .progress-bar {
          width: 120px;
          height: 8px;
          background: #333;
          border-radius: 4px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          background: #10b981;
          transition: width 0.3s;
        }

        .progress-text {
          font-size: 11px;
          color: #aaa;
        }

        .progress-phase {
          font-size: 10px;
          color: #666;
        }

        .btn-cancel {
          padding: 4px 10px;
          background: #ef4444;
          border: none;
          border-radius: 3px;
          color: white;
          font-size: 11px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-cancel:hover {
          background: #dc2626;
        }

        .error-hint {
          color: #ef4444;
          font-size: 16px;
          cursor: help;
        }

        /* Responsive : empiler config au-dessus de heatmap en dessous de 1024px */
        @media (max-width: 1024px) {
          .explorer-main {
            grid-template-columns: 1fr;
            grid-template-rows: auto 1fr;
          }

          .config-panel {
            max-height: none;
          }
        }
      `}</style>
    </div>
  )
}
