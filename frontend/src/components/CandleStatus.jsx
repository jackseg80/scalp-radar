import { useState, useEffect, useCallback } from 'react'
import { useApi } from '../hooks/useApi'
import './CandleStatus.css'

function formatDate(iso) {
  if (!iso) return '\u2014'
  const d = new Date(iso)
  const pad = n => String(n).padStart(2, '0')
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`
}

function formatDays(n) {
  if (n == null || n === 0) return '\u2014'
  return `${n}j`
}

export default function CandleStatus({ wsData }) {
  const { data, loading } = useApi('/api/data/candle-status', 30000)
  const [backfillRunning, setBackfillRunning] = useState(false)
  const [progress, setProgress] = useState(null)

  // Ecouter le WS pour la progression du backfill
  useEffect(() => {
    if (!wsData) return
    if (wsData.type === 'backfill_progress' && wsData.data) {
      setProgress(wsData.data)
      setBackfillRunning(wsData.data.running)
    }
  }, [wsData])

  // Sync running state depuis API
  useEffect(() => {
    if (data?.running) {
      setBackfillRunning(true)
    }
  }, [data?.running])

  const handleBackfill = useCallback(async () => {
    try {
      const resp = await fetch('/api/data/backfill', { method: 'POST' })
      if (resp.status === 202) {
        setBackfillRunning(true)
        setProgress({ progress_pct: 0, done: 0, total: 0, current: '', running: true })
      } else if (resp.status === 409) {
        setBackfillRunning(true)
      }
    } catch {
      // ignore
    }
  }, [])

  const assets = data?.assets || {}
  const symbols = Object.keys(assets).sort()

  if (loading && !data) {
    return <div className="text-xs muted">Chargement...</div>
  }

  return (
    <div>
      <div className="candle-status-header">
        <span className="text-xs muted">{symbols.length} assets</span>
        <button
          className="btn"
          disabled={backfillRunning}
          onClick={handleBackfill}
        >
          {backfillRunning ? 'En cours...' : 'Mettre a jour'}
        </button>
      </div>

      {backfillRunning && progress && (
        <div className="candle-progress">
          <div className="candle-progress-bar">
            <div
              className="candle-progress-fill"
              style={{ width: `${progress.progress_pct || 0}%` }}
            />
          </div>
          <div className="candle-progress-text">
            <span>{progress.current || ''}</span>
            <span>{progress.done}/{progress.total} ({progress.progress_pct}%)</span>
          </div>
        </div>
      )}

      {symbols.length === 0 ? (
        <div className="text-xs muted" style={{ textAlign: 'center', padding: 16 }}>
          Aucune donnee disponible
        </div>
      ) : (
        <table className="candle-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th style={{ textAlign: 'right' }}>Binance (jours)</th>
              <th style={{ textAlign: 'right' }}>Derniere</th>
              <th style={{ textAlign: 'right' }}>Bitget (jours)</th>
              <th style={{ textAlign: 'right' }}>Derniere</th>
            </tr>
          </thead>
          <tbody>
            {symbols.map(sym => {
              const bin = assets[sym]?.binance || {}
              const bit = assets[sym]?.bitget || {}
              return (
                <tr key={sym}>
                  <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                    {sym.replace('/USDT', '')}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <StatusDot stats={bin} />
                    {formatDays(bin.days_available)}
                  </td>
                  <td style={{ textAlign: 'right' }}>{formatDate(bin.last_candle)}</td>
                  <td style={{ textAlign: 'right' }}>
                    <StatusDot stats={bit} />
                    {formatDays(bit.days_available)}
                  </td>
                  <td style={{ textAlign: 'right' }}>{formatDate(bit.last_candle)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}

function StatusDot({ stats }) {
  if (!stats || stats.candle_count === 0) {
    return <span className="candle-dot candle-dot--empty" />
  }
  return (
    <span className={`candle-dot ${stats.is_stale ? 'candle-dot--stale' : 'candle-dot--ok'}`} />
  )
}
