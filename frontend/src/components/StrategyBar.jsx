import { useMemo, useState, useEffect, useRef } from 'react'
import { useStrategyContext } from '../contexts/StrategyContext'

const STORAGE_KEY = 'strategy-bar-order'

export default function StrategyBar({ wsData }) {
  const { activeStrategy, setActiveStrategy } = useStrategyContext()
  const [order, setOrder] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [] } catch { return [] }
  })
  const [dragOver, setDragOver] = useState(null)
  const dragSrc = useRef(null)

  const allowedLive = wsData?.executor?.selector?.allowed_strategies || []

  const rawStrategies = useMemo(() => {
    const names = Object.keys(wsData?.strategies || {})
    for (const g of Object.values(wsData?.grid_state?.grid_positions || {})) {
      if (g.strategy && !names.includes(g.strategy)) names.push(g.strategy)
    }
    return names
  }, [wsData?.strategies, wsData?.grid_state?.grid_positions])

  // Synchronise l'ordre : conserve l'ordre sauvegardé, ajoute les nouvelles à la fin
  useEffect(() => {
    if (rawStrategies.length === 0) return
    setOrder(prev => {
      const existing = prev.filter(n => rawStrategies.includes(n))
      const newOnes = rawStrategies.filter(n => !prev.includes(n)).sort()
      const next = [...existing, ...newOnes]
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      return next
    })
  }, [rawStrategies])

  if (order.length === 0 && rawStrategies.length === 0) return null

  const handleDragStart = (e, name) => {
    dragSrc.current = name
    e.dataTransfer.effectAllowed = 'move'
  }

  const handleDragEnd = () => {
    dragSrc.current = null
    setDragOver(null)
  }

  const handleDragOver = (e, name) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    if (dragOver !== name) setDragOver(name)
  }

  const handleDragLeave = (e) => {
    // Ignorer les dragLeave vers les enfants du bouton (ex: le span du dot)
    if (e.currentTarget.contains(e.relatedTarget)) return
    setDragOver(null)
  }

  const handleDrop = (e, targetName) => {
    e.preventDefault()
    setDragOver(null)
    const src = dragSrc.current
    if (!src || src === targetName) return
    setOrder(prev => {
      const next = [...prev]
      const from = next.indexOf(src)
      const to = next.indexOf(targetName)
      if (from === -1 || to === -1) return prev
      next.splice(from, 1)
      next.splice(to, 0, src)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      return next
    })
    dragSrc.current = null
  }

  return (
    <div className="strategy-bar">
      <button
        className={`strategy-btn ${activeStrategy === 'overview' ? 'active' : ''}`}
        onClick={() => setActiveStrategy('overview')}
      >
        Overview
      </button>
      {order.map(name => {
        const isLive = allowedLive.includes(name)
        return (
          <button
            key={name}
            draggable
            className={`strategy-btn strategy-btn--draggable ${activeStrategy === name ? 'active' : ''} ${dragOver === name ? 'drag-over' : ''}`}
            onClick={() => setActiveStrategy(name)}
            onDragStart={e => handleDragStart(e, name)}
            onDragEnd={handleDragEnd}
            onDragOver={e => handleDragOver(e, name)}
            onDragLeave={handleDragLeave}
            onDrop={e => handleDrop(e, name)}
          >
            {name}
            <span className={isLive ? 'strategy-dot--live' : 'strategy-dot--paper'}>
              {isLive ? '\u25CF' : '\u25CB'}
            </span>
          </button>
        )
      })}
    </div>
  )
}
