/**
 * LogViewer — Onglet complet style terminal Linux pour le debug approfondi.
 * Sprint 31
 * Polling HTTP GET /api/logs avec refresh incrémental via `since`.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import './LogViewer.css'

const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
const DEFAULT_ACTIVE = new Set(['INFO', 'WARNING', 'ERROR', 'CRITICAL'])
const MODULES = ['Tous', 'executor', 'simulator', 'watchdog', 'selector', 'data_engine', 'notifier', 'state_manager']
const MAX_LINES = 500
const POLL_INTERVAL = 5000
const INITIAL_LIMIT = 200

function formatTimestamp(ts) {
  if (!ts) return '??:??:??.???'
  try {
    const d = new Date(ts)
    const h = String(d.getHours()).padStart(2, '0')
    const m = String(d.getMinutes()).padStart(2, '0')
    const s = String(d.getSeconds()).padStart(2, '0')
    const ms = String(d.getMilliseconds()).padStart(3, '0')
    return `${h}:${m}:${s}.${ms}`
  } catch {
    return ts.slice(11, 23) || '??:??:??.???'
  }
}

function padLevel(level) {
  return (level || '').padEnd(8, ' ')
}

export default function LogViewer() {
  const [logs, setLogs] = useState([])
  const [activeLevels, setActiveLevels] = useState(DEFAULT_ACTIVE)
  const [search, setSearch] = useState('')
  const [searchDebounced, setSearchDebounced] = useState('')
  const [module, setModule] = useState('Tous')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [autoScroll, setAutoScroll] = useState(true)
  const [expandedIdx, setExpandedIdx] = useState(null)
  const [loading, setLoading] = useState(false)
  const [newLineIds, setNewLineIds] = useState(new Set())
  const terminalRef = useRef(null)
  const sinceRef = useRef(null)
  const fetchingRef = useRef(false)

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => setSearchDebounced(search), 300)
    return () => clearTimeout(timer)
  }, [search])

  // Fetch logs from API
  const fetchLogs = useCallback(async (incremental = false) => {
    if (fetchingRef.current) return
    fetchingRef.current = true
    setLoading(true)

    try {
      const params = new URLSearchParams()
      params.set('limit', incremental ? '100' : String(INITIAL_LIMIT))
      if (incremental && sinceRef.current) {
        params.set('since', sinceRef.current)
      }

      const resp = await fetch(`/api/logs?${params}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      const entries = data.logs || []

      if (entries.length > 0) {
        // Track le timestamp le plus récent pour le refresh incrémental
        sinceRef.current = entries[entries.length - 1].timestamp

        if (incremental) {
          // Marquer les nouvelles lignes pour l'animation flash
          const newIds = new Set(entries.map((_, i) => `new-${Date.now()}-${i}`))
          setNewLineIds(newIds)
          setTimeout(() => setNewLineIds(new Set()), 1200)

          setLogs(prev => {
            const combined = [...prev, ...entries]
            return combined.length > MAX_LINES ? combined.slice(combined.length - MAX_LINES) : combined
          })
        } else {
          setLogs(entries)
        }
      }
    } catch { /* silently ignore fetch errors */ }
    finally {
      setLoading(false)
      fetchingRef.current = false
    }
  }, [])

  // Initial fetch
  useEffect(() => {
    fetchLogs(false)
  }, [fetchLogs])

  // Auto-refresh polling
  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(() => fetchLogs(true), POLL_INTERVAL)
    return () => clearInterval(id)
  }, [autoRefresh, fetchLogs])

  // Auto-scroll
  useEffect(() => {
    if (autoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  // Detect manual scroll (pause auto-scroll if user scrolls up)
  const handleScroll = useCallback(() => {
    if (!terminalRef.current) return
    const el = terminalRef.current
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30
    setAutoScroll(isAtBottom)
  }, [])

  // Toggle level filter
  const toggleLevel = useCallback((level) => {
    setActiveLevels(prev => {
      const next = new Set(prev)
      if (next.has(level)) next.delete(level)
      else next.add(level)
      return next
    })
  }, [])

  // Load older logs
  const loadMore = useCallback(async () => {
    if (logs.length === 0) return
    try {
      const params = new URLSearchParams()
      params.set('limit', '200')
      const resp = await fetch(`/api/logs?${params}`)
      if (!resp.ok) return
      const data = await resp.json()
      const entries = data.logs || []
      if (entries.length > 0) {
        setLogs(entries)
        sinceRef.current = entries[entries.length - 1].timestamp
      }
    } catch { /* ignore */ }
  }, [logs.length])

  // Filter logs
  const filtered = logs.filter(entry => {
    if (!activeLevels.has(entry.level)) return false
    if (module !== 'Tous' && !entry.module.toLowerCase().includes(module.toLowerCase())) return false
    if (searchDebounced && !entry.message.toLowerCase().includes(searchDebounced.toLowerCase())) return false
    return true
  })

  return (
    <div className="log-viewer">
      {/* Toolbar */}
      <div className="log-toolbar">
        <div className="log-toolbar-group">
          {LEVELS.map(lvl => (
            <button
              key={lvl}
              className={`log-level-btn ${activeLevels.has(lvl) ? 'active' : ''}`}
              data-level={lvl}
              onClick={() => toggleLevel(lvl)}
            >
              {lvl}
            </button>
          ))}
        </div>

        <input
          className="log-search"
          type="text"
          placeholder="grep..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />

        <select
          className="log-module-select"
          value={module}
          onChange={e => setModule(e.target.value)}
        >
          {MODULES.map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        <button
          className={`log-toolbar-btn ${autoRefresh ? 'active' : ''}`}
          onClick={() => setAutoRefresh(!autoRefresh)}
        >
          Auto-refresh
        </button>

        <button
          className="log-toolbar-btn"
          onClick={() => fetchLogs(true)}
          disabled={loading}
        >
          Refresh
        </button>

        <div className="log-live-indicator">
          <span className={`log-live-dot ${autoRefresh ? '' : 'log-live-dot--off'}`} />
          {autoRefresh ? 'live' : 'pause'}
        </div>
      </div>

      {/* Terminal */}
      <div
        className="log-terminal"
        ref={terminalRef}
        onScroll={handleScroll}
        style={{ position: 'relative' }}
      >
        {filtered.length === 0 && (
          <div className="log-empty">
            {logs.length === 0 ? 'Aucun log disponible' : 'Aucun log correspondant aux filtres'}
          </div>
        )}

        {filtered.length > 0 && (
          <div className="log-load-more" onClick={loadMore}>
            Charger plus anciens
          </div>
        )}

        {filtered.map((entry, i) => {
          const isNew = newLineIds.size > 0 && i >= filtered.length - newLineIds.size
          const isExpanded = expandedIdx === i

          return (
            <div key={`${entry.timestamp}-${i}`}>
              <div
                className={`log-line ${isNew ? 'log-line--new' : ''} ${isExpanded ? 'log-line--expanded' : ''}`}
                onClick={() => setExpandedIdx(isExpanded ? null : i)}
              >
                <span className="log-timestamp">{formatTimestamp(entry.timestamp)}</span>
                {' | '}
                <span className={`log-level--${entry.level}`}>{padLevel(entry.level)}</span>
                {' | '}
                <span className="log-module">{entry.module}:{entry.function}:{entry.line}</span>
                {' | '}
                <span className="log-message">{entry.message}</span>
              </div>
              {isExpanded && (
                <div className="log-detail">
                  module: {entry.module}{'\n'}
                  function: {entry.function}{'\n'}
                  line: {entry.line}{'\n'}
                  level: {entry.level}{'\n'}
                  time: {entry.timestamp}{'\n'}
                  message: {entry.message}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Scroll pause indicator */}
      {!autoScroll && (
        <button
          className="log-scroll-pause"
          onClick={() => {
            setAutoScroll(true)
            if (terminalRef.current) {
              terminalRef.current.scrollTop = terminalRef.current.scrollHeight
            }
          }}
        >
          Scroll auto
        </button>
      )}
    </div>
  )
}
