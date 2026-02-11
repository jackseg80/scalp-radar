/**
 * AlertFeed — Timeline de signaux en ordre chronologique inversé.
 * Props : wsData
 * Utilise useApi('/api/signals/recent', 30000) pour les signaux.
 * Fusionne les signaux API et les signaux WS temps réel.
 */
import { useState, useEffect } from 'react'
import { useApi } from '../hooks/useApi'

export default function AlertFeed({ wsData }) {
  const { data, loading } = useApi('/api/signals/recent', 30000)
  const [signals, setSignals] = useState([])

  // Mettre à jour les signaux depuis l'API
  useEffect(() => {
    if (data?.signals) {
      setSignals(data.signals)
    }
  }, [data])

  // Ajouter les signaux WS en temps réel
  useEffect(() => {
    if (wsData?.signal) {
      setSignals(prev => {
        const updated = [wsData.signal, ...prev]
        // Garder max 50 signaux
        return updated.slice(0, 50)
      })
    }
  }, [wsData?.signal])

  return (
    <div className="card">
      <h2>Signaux</h2>

      {loading && signals.length === 0 && (
        <div className="empty-state">
          <div className="skeleton skeleton-line" style={{ width: '90%' }} />
          <div className="skeleton skeleton-line" style={{ width: '70%' }} />
        </div>
      )}

      {!loading && signals.length === 0 && (
        <div className="empty-state">En attente de données...</div>
      )}

      {signals.length > 0 && (
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {signals.map((sig, i) => (
            <SignalItem key={`${sig.timestamp || sig.time}-${i}`} signal={sig} />
          ))}
        </div>
      )}
    </div>
  )
}

function SignalItem({ signal }) {
  const direction = signal.direction || ''
  const isLong = direction === 'LONG'

  // Couleur du point selon la direction
  const dotColor = isLong ? 'var(--accent)' : direction === 'SHORT' ? 'var(--red)' : 'var(--yellow)'

  // Temps formaté
  const time = signal.timestamp || signal.time
  const formattedTime = time
    ? new Date(time).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '--:--:--'

  // Score formaté
  const score = signal.score != null ? `${Math.round(signal.score * 100)}%` : null

  return (
    <div className="alert-item">
      <span className="alert-dot" style={{ background: dotColor }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="flex-between">
          <span style={{ fontWeight: 600, fontSize: 12 }}>
            {signal.symbol || '--'}
            <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`} style={{ marginLeft: 6 }}>
              {direction || '--'}
            </span>
          </span>
          <span className="text-xs dim mono">{formattedTime}</span>
        </div>
        <div className="text-xs muted" style={{ marginTop: 2 }}>
          {signal.strategy || 'Stratégie inconnue'}
          {score && <span className="mono" style={{ marginLeft: 8, color: 'var(--text-secondary)' }}>{score}</span>}
          {signal.entry_price && (
            <span className="mono" style={{ marginLeft: 8 }}>@ {Number(signal.entry_price).toFixed(2)}</span>
          )}
        </div>
      </div>
    </div>
  )
}
