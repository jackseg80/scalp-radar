import { useState, useEffect, useRef } from 'react'

/**
 * Hook WebSocket avec reconnexion automatique (backoff exponentiel).
 * @param {string} url - URL WebSocket
 */
export function useWebSocket(url) {
  const [lastMessage, setLastMessage] = useState(null)
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
          setLastMessage(data)
        } catch { /* ignore */ }
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

  return { lastMessage, connected }
}
