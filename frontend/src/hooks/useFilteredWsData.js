import { useMemo } from 'react'

/**
 * Filtre les données WebSocket selon la stratégie sélectionnée.
 * Si strategyFilter est null (overview), retourne les données brutes.
 */
export default function useFilteredWsData(wsData, strategyFilter) {
  return useMemo(() => {
    if (!strategyFilter || !wsData) return wsData

    // 1. Filtrer grid_state.grid_positions (clé strategy:symbol)
    const filteredGridPositions = {}
    for (const [key, g] of Object.entries(wsData.grid_state?.grid_positions || {})) {
      if (g.strategy === strategyFilter) {
        filteredGridPositions[key] = g
      }
    }
    const grids = Object.values(filteredGridPositions)
    const filteredGridState = {
      grid_positions: filteredGridPositions,
      summary: {
        total_positions: grids.reduce((s, g) => s + (g.levels_open || 0), 0),
        total_assets: grids.length,
        total_margin_used: grids.reduce((s, g) => s + (g.margin_used || 0), 0),
        total_unrealized_pnl: grids.reduce((s, g) => s + (g.unrealized_pnl || 0), 0),
        capital_available: wsData.grid_state?.summary?.capital_available ?? 0,
      },
    }

    // 2. Filtrer strategies (dict keyed par runner name)
    const filteredStrategies = {}
    for (const [name, s] of Object.entries(wsData.strategies || {})) {
      if (name === strategyFilter) {
        filteredStrategies[name] = s
      }
    }

    // 3. Filtrer executor.positions par strategy_name
    const filteredExecutor = wsData.executor ? {
      ...wsData.executor,
      positions: (wsData.executor.positions || [])
        .filter(p => p.strategy_name === strategyFilter),
    } : wsData.executor

    // 4. Filtrer simulator_positions par strategy
    const filteredSimPositions = (wsData.simulator_positions || [])
      .filter(p => (p.strategy_name || p.strategy) === strategyFilter)

    return {
      ...wsData,
      grid_state: filteredGridState,
      strategies: filteredStrategies,
      executor: filteredExecutor,
      simulator_positions: filteredSimPositions,
    }
  }, [wsData, strategyFilter])
}

/**
 * Construit un lookup grid_positions keyed par symbol (au lieu de strategy:symbol).
 * Utilisé par Scanner et ActivePositions pour accéder aux grids par symbol.
 */
export function buildGridLookupBySymbol(gridPositions) {
  const lookup = {}
  for (const g of Object.values(gridPositions || {})) {
    if (!lookup[g.symbol]) lookup[g.symbol] = g
  }
  return lookup
}
