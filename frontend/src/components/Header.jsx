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
          <StatusDot label="Engine" ok={engineOk} tooltip="DataEngine : connexion WebSocket Bitget" />
          <StatusDot label="DB" ok={dbOk} tooltip="Base de données SQLite" />
          <StatusDot label="WS" ok={wsConnected} tooltip="WebSocket frontend ↔ backend (/ws/live)" />
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
