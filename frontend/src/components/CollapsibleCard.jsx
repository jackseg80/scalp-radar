/**
 * CollapsibleCard — Wrapper card collapsible pour la sidebar.
 * Props : title, summary (affiché quand fermé), defaultOpen, storageKey, children
 */
import { useState, useEffect } from 'react'

export default function CollapsibleCard({ title, summary, defaultOpen = true, storageKey, cardClassName, children }) {
  const [open, setOpen] = useState(() => {
    if (storageKey) {
      const saved = localStorage.getItem(`scalp-radar-collapse-${storageKey}`)
      if (saved !== null) return saved === 'true'
    }
    return defaultOpen
  })

  useEffect(() => {
    if (storageKey) {
      localStorage.setItem(`scalp-radar-collapse-${storageKey}`, String(open))
    }
  }, [open, storageKey])

  return (
    <div className={`card ${cardClassName || ''}`}>
      <div className="collapsible-header" onClick={() => setOpen(!open)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          <h2 style={{ marginBottom: 0 }}>{title}</h2>
          {!open && summary && (
            <span className="text-xs mono muted" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {summary}
            </span>
          )}
        </div>
        <span className={`collapsible-arrow ${open ? 'open' : ''}`}>&#9660;</span>
      </div>
      {open && <div style={{ marginTop: 10 }}>{children}</div>}
    </div>
  )
}
