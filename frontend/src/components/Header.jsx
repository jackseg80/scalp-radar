import { useState, useEffect, useRef } from 'react'
import { useApi } from '../hooks/useApi'
import Tooltip from './Tooltip'
import StrategyBar from './StrategyBar'

export default function Header({ wsConnected, tabs, activeTab, onTabChange, unseenLogErrors = 0, wsData }) {
  const { data } = useApi('/health', 10000)

  const engineOk = data?.data_engine?.connected
  const dbOk = data?.database?.connected
  const status = data?.status || 'loading'

  return (
    <header className="header">
      <div className="header-top">
        <span className="header-logo">SCALP RADAR</span>
        <span className="header-version">v1.0.0</span>
        <StrategyBar wsData={wsData} />
        <div className="header-right">
          <DataFreshness wsData={wsData} />
          <div style={{ width: 1, height: 16, background: 'var(--border)', margin: '0 8px' }} />
          <StatusDot label="Engine" ok={engineOk} tooltip="DataEngine : connexion WebSocket Bitget" />
          <StatusDot label="DB" ok={dbOk} tooltip="Base de données SQLite" />
          <StatusDot label="WS" ok={wsConnected} tooltip="WebSocket frontend ↔ backend (/api/ws/live)" />
          <Tooltip content="État de santé global du système">
            <span className={`status-badge ${status === 'ok' ? 'status-badge--ok' : 'status-badge--error'}`}>
              {status.toUpperCase()}
            </span>
          </Tooltip>
        </div>
      </div>

      <div className="tabs">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`tab ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => onTabChange(t.id)}
            style={{ position: 'relative' }}
          >
            {t.label}
            {t.id === 'logs' && unseenLogErrors > 0 && (
              <span style={{
                position: 'absolute',
                top: -2,
                right: -4,
                minWidth: 14,
                height: 14,
                borderRadius: 7,
                background: 'var(--red)',
                color: '#fff',
                fontSize: 9,
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '0 3px',
                lineHeight: 1,
              }}>
                {unseenLogErrors > 99 ? '99+' : unseenLogErrors}
              </span>
            )}
          </button>
        ))}
      </div>
    </header>
  )
}

function DataFreshness({ wsData }) {
  const [seconds, setSeconds] = useState(0)
  const lastUpdateRef = useRef(Date.now())

  useEffect(() => {
    lastUpdateRef.current = Date.now()
    setSeconds(0)
  }, [wsData])

  useEffect(() => {
    const id = setInterval(() => {
      setSeconds(Math.floor((Date.now() - lastUpdateRef.current) / 1000))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const isStale = seconds > 10
  const color = isStale ? 'var(--red)' : 'var(--accent)'

  return (
    <Tooltip content="Temps depuis la dernière mise à jour WebSocket">
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: 'var(--text-dim)', padding: '0 4px' }}>
        <span style={{ 
          width: 6, height: 6, borderRadius: '50%', background: color, 
          boxShadow: isStale ? 'none' : `0 0 8px ${color}`,
          transition: 'all 0.3s'
        }} />
        <span className="mono" style={{ color: seconds > 5 ? 'var(--text-primary)' : 'inherit', minWidth: '20px' }}>
          {seconds}s
        </span>
      </div>
    </Tooltip>
  )
}

function StatusDot({ label, ok, tooltip }) {
  const cls = ok ? 'status-dot__indicator--ok'
    : ok === false ? 'status-dot__indicator--error'
    : 'status-dot__indicator--loading'

  const dot = (
    <span className="status-dot">
      <span className={`status-dot__indicator ${cls}`} />
      {label}
    </span>
  )
  if (!tooltip) return dot
  return <Tooltip content={tooltip}>{dot}</Tooltip>
}
