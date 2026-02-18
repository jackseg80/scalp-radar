import { useState, useEffect, useRef } from 'react'

const MAX_LOG_ALERTS = 50

/**
 * Hook WebSocket avec reconnexion automatique (backoff exponentiel).
 * Sprint 31 : dispatch par type de message (update / log_alert / event).
 *
 * @param {string} url - URL WebSocket
 * @returns {{ lastUpdate: object|null, lastEvent: object|null, logAlerts: array, connected: boolean }}
 */
export function useWebSocket(url) {
  const [lastUpdate, setLastUpdate] = useState(null)
  const [lastEvent, setLastEvent] = useState(null)
  const [logAlerts, setLogAlerts] = useState([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const retryRef = useRef(0)
  const maxRetryDelay = 30000

  useEffect(() => {
    let unmounted = false
    let timeoutId = null

    function connect() {
      if (unmounted) return

      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        retryRef.current = 0
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'update') {
            // Payload standard (strategies, prices, executor, etc.)
            setLastUpdate(data)
          } else if (data.type === 'log_alert') {
            // WARNING/ERROR temps réel
            setLogAlerts(prev => {
              const next = [data.entry, ...prev]
              return next.length > MAX_LOG_ALERTS ? next.slice(0, MAX_LOG_ALERTS) : next
            })
          } else {
            // optimization_progress, portfolio_progress, portfolio_completed, etc.
            setLastEvent(data)
          }
        } catch { /* ignore parse errors */ }
      }

      ws.onclose = () => {
        setConnected(false)
        if (!unmounted) {
          const delay = Math.min(1000 * Math.pow(2, retryRef.current), maxRetryDelay)
          retryRef.current++
          timeoutId = setTimeout(connect, delay)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      unmounted = true
      if (timeoutId) clearTimeout(timeoutId)
      if (wsRef.current) {
        // Nullify handlers to avoid StrictMode double-mount warnings
        wsRef.current.onopen = null
        wsRef.current.onmessage = null
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        wsRef.current.close()
      }
    }
  }, [url])

  return { lastUpdate, lastEvent, logAlerts, connected }
}
