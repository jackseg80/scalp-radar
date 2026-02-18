/**
 * LogMini — Mini-feed temps réel des WARNING/ERROR dans la sidebar.
 * Sprint 31
 * Props : logAlerts (array), onTabChange (callback)
 */

function formatTime(isoString) {
  if (!isoString) return ''
  const d = new Date(isoString)
  return d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const LEVEL_COLORS = {
  WARNING: 'var(--orange)',
  ERROR: 'var(--red)',
  CRITICAL: '#ff1744',
}

export default function LogMini({ logAlerts = [], onTabChange }) {
  if (logAlerts.length === 0) {
    return (
      <div style={{ color: 'var(--accent)', fontSize: 12, textAlign: 'center', padding: '8px 0' }}>
        Aucune alerte
      </div>
    )
  }

  const displayed = logAlerts.slice(0, 20)

  return (
    <div>
      {displayed.map((entry, i) => (
        <div
          key={`${entry.timestamp}-${i}`}
          className="log-mini-entry"
          onClick={() => onTabChange?.('logs')}
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 6,
            padding: '4px 0',
            borderBottom: i < displayed.length - 1 ? '1px solid var(--border)' : 'none',
            cursor: 'pointer',
            animation: i === 0 ? 'slideIn 0.3s ease-out' : 'none',
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: LEVEL_COLORS[entry.level] || 'var(--orange)',
              flexShrink: 0,
              marginTop: 4,
            }}
          />
          <span className="text-xs mono muted" style={{ flexShrink: 0, whiteSpace: 'nowrap' }}>
            {formatTime(entry.timestamp)}
          </span>
          <span
            className="text-xs"
            style={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              color: LEVEL_COLORS[entry.level] || 'var(--text-secondary)',
            }}
          >
            {entry.message}
          </span>
        </div>
      ))}
    </div>
  )
}

LogMini.getSummary = function getSummary(logAlerts) {
  if (!logAlerts || logAlerts.length === 0) return 'OK'
  const errorCount = logAlerts.filter(a => a.level === 'ERROR' || a.level === 'CRITICAL').length
  if (errorCount > 0) return `${errorCount} err`
  return `${logAlerts.length} warn`
}
