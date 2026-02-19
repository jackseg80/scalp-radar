import { useMemo } from 'react'
import { useStrategyContext } from '../contexts/StrategyContext'

export default function StrategyBar({ wsData }) {
  const { activeStrategy, setActiveStrategy } = useStrategyContext()

  const allowedLive = wsData?.executor?.selector?.allowed_strategies || []

  const strategies = useMemo(() => {
    const names = Object.keys(wsData?.strategies || {})
    // Dédupliquer avec les noms issus de grid_state
    for (const g of Object.values(wsData?.grid_state?.grid_positions || {})) {
      if (g.strategy && !names.includes(g.strategy)) {
        names.push(g.strategy)
      }
    }
    return names.sort()
  }, [wsData?.strategies, wsData?.grid_state?.grid_positions])

  if (strategies.length === 0) return null

  return (
    <div className="strategy-bar">
      <button
        className={`strategy-btn ${activeStrategy === 'overview' ? 'active' : ''}`}
        onClick={() => setActiveStrategy('overview')}
      >
        Overview
      </button>
      {strategies.map(name => {
        const isLive = allowedLive.includes(name)
        return (
          <button
            key={name}
            className={`strategy-btn ${activeStrategy === name ? 'active' : ''}`}
            onClick={() => setActiveStrategy(name)}
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
