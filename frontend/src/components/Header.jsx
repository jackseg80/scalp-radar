import { useApi } from '../hooks/useApi'

export default function Header({ wsConnected, tabs, activeTab, onTabChange }) {
  const { data } = useApi('/health', 10000)

  const engineOk = data?.data_engine?.connected
  const dbOk = data?.database?.connected
  const status = data?.status || 'loading'

  return (
    <header className="header">
      <div className="header-left">
        <span className="header-logo">SCALP RADAR</span>
        <span className="header-version">v0.6.0</span>

        <div className="tabs">
          {tabs.map(t => (
            <button
              key={t.id}
              className={`tab ${activeTab === t.id ? 'active' : ''}`}
              onClick={() => onTabChange(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="header-right">
        <StatusDot label="Engine" ok={engineOk} />
        <StatusDot label="DB" ok={dbOk} />
        <StatusDot label="WS" ok={wsConnected} />
        <span className={`status-badge ${status === 'ok' ? 'status-badge--ok' : 'status-badge--error'}`}>
          {status.toUpperCase()}
        </span>
      </div>
    </header>
  )
}

function StatusDot({ label, ok }) {
  const cls = ok ? 'status-dot__indicator--ok'
    : ok === false ? 'status-dot__indicator--error'
    : 'status-dot__indicator--loading'

  return (
    <span className="status-dot">
      <span className={`status-dot__indicator ${cls}`} />
      {label}
    </span>
  )
}
