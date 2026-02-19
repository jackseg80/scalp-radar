import { createContext, useContext, useState, useMemo, useCallback } from 'react'

const STORAGE_KEY = 'scalp-radar-strategy'

const StrategyContext = createContext({
  activeStrategy: 'overview',
  setActiveStrategy: () => {},
  strategyFilter: null,
})

function loadStrategy() {
  return localStorage.getItem(STORAGE_KEY) || 'overview'
}

export function StrategyProvider({ children }) {
  const [activeStrategy, setRaw] = useState(loadStrategy)

  const setActiveStrategy = useCallback((s) => {
    setRaw(s)
    localStorage.setItem(STORAGE_KEY, s)
  }, [])

  const strategyFilter = useMemo(
    () => (activeStrategy === 'overview' ? null : activeStrategy),
    [activeStrategy],
  )

  const value = useMemo(
    () => ({ activeStrategy, setActiveStrategy, strategyFilter }),
    [activeStrategy, setActiveStrategy, strategyFilter],
  )

  return (
    <StrategyContext.Provider value={value}>
      {children}
    </StrategyContext.Provider>
  )
}

export function useStrategyContext() {
  return useContext(StrategyContext)
}
