/**
 * ExplorerPage — Explorateur de Paramètres WFO interactif
 * Sprint 14 Bloc E
 */

import { useState, useEffect, useMemo, useCallback } from 'react'
import HeatmapChart from './HeatmapChart'

const GRADE_COLORS = {
  A: '#10b981',
  B: '#3b82f6',
  C: '#f59e0b',
  D: '#f97316',
  F: '#ef4444',
}

const STATUS_LABELS = {
  pending: 'En attente',
  running: 'En cours',
  completed: 'Terminé',
  failed: 'Échec',
  cancelled: 'Annulé',
}

const STATUS_COLORS = {
  pending: '#6b7280',
  running: '#3b82f6',
  completed: '#10b981',
  failed: '#ef4444',
  cancelled: '#f97316',
}

export default function ExplorerPage({ wsData }) {
  const [strategy, setStrategy] = useState('envelope_dca')
  const [asset, setAsset] = useState('BTC/USDT')
  const [paramGrid, setParamGrid] = useState(null)
  const [paramsOverride, setParamsOverride] = useState({})
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [heatmapData, setHeatmapData] = useState(null)
  const [heatmapParams, setHeatmapParams] = useState({ paramX: null, paramY: null })
  const [heatmapMetric, setHeatmapMetric] = useState('total_score')

  // Fetch param grid pour la stratégie sélectionnée
  const fetchParamGrid = useCallback(async () => {
    try {
      const resp = await fetch(`/api/optimization/param-grid/${strategy}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const json = await resp.json()
      setParamGrid(json)
      // Reset params override quand on change de stratégie
      setParamsOverride({})
      // Auto-sélectionner les 2 premiers params pour la heatmap
      const paramNames = Object.keys(json.params || {})
      if (paramNames.length >= 2) {
        setHeatmapParams({ paramX: paramNames[0], paramY: paramNames[1] })
      }
    } catch (err) {
      console.error('Erreur fetch param-grid:', err)
      setParamGrid(null)
    }
  }, [strategy])

  useEffect(() => {
    fetchParamGrid()
  }, [fetchParamGrid])

  // Fetch jobs
  const fetchJobs = useCallback(async () => {
    try {
      const resp = await fetch('/api/optimization/jobs?limit=20')
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const json = await resp.json()
      setJobs(json.jobs || [])
    } catch (err) {
      console.error('Erreur fetch jobs:', err)
    }
  }, [])

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 5000) // Poll toutes les 5s
    return () => clearInterval(interval)
  }, [fetchJobs])

  // Fetch heatmap
  const fetchHeatmap = useCallback(async () => {
    if (!heatmapParams.paramX || !heatmapParams.paramY) return

    try {
      const params = new URLSearchParams({
        strategy,
        asset,
        param_x: heatmapParams.paramX,
        param_y: heatmapParams.paramY,
        metric: heatmapMetric,
      })
      const resp = await fetch(`/api/optimization/heatmap?${params}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const json = await resp.json()
      setHeatmapData(json)
    } catch (err) {
      console.error('Erreur fetch heatmap:', err)
      setHeatmapData(null)
    }
  }, [strategy, asset, heatmapParams, heatmapMetric])

  useEffect(() => {
    fetchHeatmap()
  }, [fetchHeatmap])

  // WS: mettre à jour jobs en temps réel
  useEffect(() => {
    if (wsData?.type === 'optimization_progress') {
      setJobs(prev =>
        prev.map(j =>
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
      // Si un job passe en completed, re-fetch heatmap
      if (wsData.status === 'completed') {
        fetchHeatmap()
        fetchJobs() // Pour avoir result_id et duration
      }
    }
  }, [wsData, fetchHeatmap, fetchJobs])

  // Submit job
  const handleRun = async () => {
    const payload = { strategy_name: strategy, asset }
    // Convertir paramsOverride en format API (obj de listes)
    const override = {}
    for (const [key, val] of Object.entries(paramsOverride)) {
      override[key] = [val]
    }
    if (Object.keys(override).length > 0) {
      payload.params_override = override
    }

    try {
      const resp = await fetch('/api/optimization/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }
      const json = await resp.json()
      // Re-fetch jobs pour afficher le nouveau
      fetchJobs()
      alert(`Job ${json.job_id.slice(0, 8)} soumis`)
    } catch (err) {
      alert(`Erreur: ${err.message}`)
    }
  }

  // Cancel job
  const handleCancel = async (jobId) => {
    try {
      const resp = await fetch(`/api/optimization/jobs/${jobId}`, {
        method: 'DELETE',
      })
      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || `HTTP ${resp.status}`)
      }
      fetchJobs()
    } catch (err) {
      alert(`Erreur cancel: ${err.message}`)
    }
  }

  // Liste des paramètres disponibles
  const paramNames = useMemo(() => {
    if (!paramGrid || !paramGrid.params) return []
    return Object.keys(paramGrid.params)
  }, [paramGrid])

  return (
    <div style={{ padding: '24px', maxWidth: '1600px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '24px', color: '#e5e7eb' }}>
        Explorateur de Paramètres WFO
      </h1>

      {/* Panneau de contrôle */}
      <div
        style={{
          background: '#1f2937',
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '24px',
        }}
      >
        <h2 style={{ marginTop: 0, marginBottom: '16px', color: '#e5e7eb' }}>
          Configuration
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          {/* Stratégie */}
          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: '#9ca3af' }}>
              Stratégie
            </label>
            <select
              value={strategy}
              onChange={e => setStrategy(e.target.value)}
              style={{
                width: '100%',
                padding: '8px',
                background: '#374151',
                color: '#e5e7eb',
                border: '1px solid #4b5563',
                borderRadius: '4px',
              }}
            >
              <option value="envelope_dca">Envelope DCA</option>
              <option value="vwap_rsi">VWAP + RSI</option>
              <option value="momentum">Momentum</option>
              <option value="bollinger_mr">Bollinger MR</option>
              <option value="donchian_breakout">Donchian Breakout</option>
              <option value="supertrend">SuperTrend</option>
            </select>
          </div>

          {/* Asset */}
          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: '#9ca3af' }}>
              Asset
            </label>
            <select
              value={asset}
              onChange={e => setAsset(e.target.value)}
              style={{
                width: '100%',
                padding: '8px',
                background: '#374151',
                color: '#e5e7eb',
                border: '1px solid #4b5563',
                borderRadius: '4px',
              }}
            >
              <option value="BTC/USDT">BTC/USDT</option>
              <option value="ETH/USDT">ETH/USDT</option>
              <option value="SOL/USDT">SOL/USDT</option>
              <option value="DOGE/USDT">DOGE/USDT</option>
              <option value="LINK/USDT">LINK/USDT</option>
            </select>
          </div>
        </div>

        {/* Sliders params override */}
        {paramGrid && paramGrid.params && (
          <div style={{ marginTop: '20px' }}>
            <h3 style={{ marginBottom: '12px', color: '#e5e7eb', fontSize: '16px' }}>
              Sous-grille personnalisée (optionnel)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
              {paramNames.map(pName => {
                const pData = paramGrid.params[pName]
                const values = pData.values || []
                const defaultVal = pData.default ?? values[0]
                const currentVal = paramsOverride[pName] ?? defaultVal

                return (
                  <div key={pName}>
                    <label style={{ display: 'block', marginBottom: '4px', color: '#9ca3af', fontSize: '14px' }}>
                      {pName}: <span style={{ color: '#3b82f6' }}>{currentVal}</span>
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={values.length - 1}
                      step={1}
                      value={values.indexOf(currentVal)}
                      onChange={e => {
                        const idx = parseInt(e.target.value, 10)
                        setParamsOverride(prev => ({ ...prev, [pName]: values[idx] }))
                      }}
                      style={{ width: '100%' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#6b7280', marginTop: '4px' }}>
                      <span>{values[0]}</span>
                      <span>{values[values.length - 1]}</span>
                    </div>
                  </div>
                )
              })}
            </div>
            <button
              onClick={() => setParamsOverride({})}
              style={{
                marginTop: '12px',
                padding: '6px 12px',
                background: '#374151',
                color: '#e5e7eb',
                border: '1px solid #4b5563',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Réinitialiser
            </button>
          </div>
        )}

        {/* Bouton Run */}
        <button
          onClick={handleRun}
          style={{
            marginTop: '20px',
            padding: '12px 24px',
            background: '#3b82f6',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '16px',
          }}
        >
          Lancer WFO
        </button>
      </div>

      {/* Jobs en cours */}
      <div
        style={{
          background: '#1f2937',
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '24px',
        }}
      >
        <h2 style={{ marginTop: 0, marginBottom: '16px', color: '#e5e7eb' }}>
          Jobs d'optimisation
        </h2>

        {jobs.length === 0 ? (
          <p style={{ color: '#9ca3af' }}>Aucun job en cours ou récent</p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #374151' }}>
                  <th style={{ padding: '8px', textAlign: 'left', color: '#9ca3af', fontSize: '14px' }}>ID</th>
                  <th style={{ padding: '8px', textAlign: 'left', color: '#9ca3af', fontSize: '14px' }}>Stratégie</th>
                  <th style={{ padding: '8px', textAlign: 'left', color: '#9ca3af', fontSize: '14px' }}>Asset</th>
                  <th style={{ padding: '8px', textAlign: 'left', color: '#9ca3af', fontSize: '14px' }}>Status</th>
                  <th style={{ padding: '8px', textAlign: 'left', color: '#9ca3af', fontSize: '14px' }}>Progression</th>
                  <th style={{ padding: '8px', textAlign: 'left', color: '#9ca3af', fontSize: '14px' }}>Durée</th>
                  <th style={{ padding: '8px', textAlign: 'left', color: '#9ca3af', fontSize: '14px' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map(job => (
                  <tr key={job.id} style={{ borderBottom: '1px solid #374151' }}>
                    <td style={{ padding: '8px', color: '#e5e7eb', fontSize: '13px', fontFamily: 'monospace' }}>
                      {job.id.slice(0, 8)}
                    </td>
                    <td style={{ padding: '8px', color: '#e5e7eb', fontSize: '13px' }}>
                      {job.strategy_name}
                    </td>
                    <td style={{ padding: '8px', color: '#e5e7eb', fontSize: '13px' }}>
                      {job.asset}
                    </td>
                    <td style={{ padding: '8px', fontSize: '13px' }}>
                      <span
                        style={{
                          background: STATUS_COLORS[job.status] || '#6b7280',
                          color: '#fff',
                          padding: '2px 8px',
                          borderRadius: '4px',
                          fontSize: '12px',
                        }}
                      >
                        {STATUS_LABELS[job.status] || job.status}
                      </span>
                    </td>
                    <td style={{ padding: '8px', fontSize: '13px' }}>
                      {job.status === 'running' ? (
                        <div>
                          <div style={{ background: '#374151', height: '20px', borderRadius: '4px', overflow: 'hidden' }}>
                            <div
                              style={{
                                background: '#3b82f6',
                                height: '100%',
                                width: `${job.progress_pct || 0}%`,
                                transition: 'width 0.3s',
                              }}
                            />
                          </div>
                          <div style={{ fontSize: '12px', color: '#9ca3af', marginTop: '4px' }}>
                            {Math.round(job.progress_pct || 0)}% — {job.current_phase}
                          </div>
                        </div>
                      ) : (
                        <span style={{ color: '#9ca3af' }}>—</span>
                      )}
                    </td>
                    <td style={{ padding: '8px', color: '#e5e7eb', fontSize: '13px' }}>
                      {job.duration_seconds ? `${Math.round(job.duration_seconds)}s` : '—'}
                    </td>
                    <td style={{ padding: '8px' }}>
                      {(job.status === 'pending' || job.status === 'running') && (
                        <button
                          onClick={() => handleCancel(job.id)}
                          style={{
                            padding: '4px 8px',
                            background: '#ef4444',
                            color: '#fff',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '12px',
                          }}
                        >
                          Annuler
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Heatmap */}
      <div
        style={{
          background: '#1f2937',
          borderRadius: '8px',
          padding: '20px',
        }}
      >
        <h2 style={{ marginTop: 0, marginBottom: '16px', color: '#e5e7eb' }}>
          Heatmap 2D
        </h2>

        {/* Sélecteurs axes */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', marginBottom: '16px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: '#9ca3af', fontSize: '14px' }}>
              Axe X
            </label>
            <select
              value={heatmapParams.paramX || ''}
              onChange={e => setHeatmapParams(prev => ({ ...prev, paramX: e.target.value }))}
              style={{
                width: '100%',
                padding: '8px',
                background: '#374151',
                color: '#e5e7eb',
                border: '1px solid #4b5563',
                borderRadius: '4px',
              }}
            >
              <option value="">—</option>
              {paramNames.map(p => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: '#9ca3af', fontSize: '14px' }}>
              Axe Y
            </label>
            <select
              value={heatmapParams.paramY || ''}
              onChange={e => setHeatmapParams(prev => ({ ...prev, paramY: e.target.value }))}
              style={{
                width: '100%',
                padding: '8px',
                background: '#374151',
                color: '#e5e7eb',
                border: '1px solid #4b5563',
                borderRadius: '4px',
              }}
            >
              <option value="">—</option>
              {paramNames.map(p => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', color: '#9ca3af', fontSize: '14px' }}>
              Métrique
            </label>
            <select
              value={heatmapMetric}
              onChange={e => setHeatmapMetric(e.target.value)}
              style={{
                width: '100%',
                padding: '8px',
                background: '#374151',
                color: '#e5e7eb',
                border: '1px solid #4b5563',
                borderRadius: '4px',
              }}
            >
              <option value="total_score">Score Total</option>
              <option value="oos_sharpe">OOS Sharpe</option>
              <option value="consistency">Consistance</option>
              <option value="dsr">DSR</option>
            </select>
          </div>
        </div>

        {/* Chart */}
        {heatmapData && heatmapData.x_values.length > 0 ? (
          <HeatmapChart data={heatmapData} />
        ) : (
          <p style={{ color: '#9ca3af', textAlign: 'center', padding: '40px 0' }}>
            Aucune donnée disponible pour cette combinaison de paramètres
          </p>
        )}
      </div>
    </div>
  )
}
