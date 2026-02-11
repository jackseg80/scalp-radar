import { useApi } from '../hooks/useApi'

export default function Header({ wsConnected }) {
  const { data } = useApi('/health', 10000)

  const engineOk = data?.data_engine?.connected
  const dbOk = data?.database?.connected
  const status = data?.status || 'loading'

  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 20px',
      borderBottom: '1px solid var(--border)',
      background: 'var(--bg-secondary)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <h1 style={{ fontSize: 18, fontWeight: 800, color: 'var(--accent)', letterSpacing: -0.5 }}>
          SCALP RADAR
        </h1>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          v0.3.0
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: 12 }}>
        <StatusDot label="Engine" ok={engineOk} />
        <StatusDot label="DB" ok={dbOk} />
        <StatusDot label="WS" ok={wsConnected} />
        <span style={{
          padding: '3px 10px',
          borderRadius: 4,
          fontSize: 11,
          fontWeight: 600,
          background: status === 'ok' ? 'var(--accent-dim)' : 'var(--red-dim)',
          color: status === 'ok' ? 'var(--accent)' : 'var(--red)',
        }}>
          {status.toUpperCase()}
        </span>
      </div>
    </header>
  )
}

function StatusDot({ label, ok }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-secondary)' }}>
      <span style={{
        width: 7, height: 7, borderRadius: '50%',
        background: ok ? 'var(--accent)' : ok === false ? 'var(--red)' : 'var(--yellow)',
      }} />
      {label}
    </span>
  )
}
