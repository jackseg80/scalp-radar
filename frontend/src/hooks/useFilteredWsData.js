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

    // 3. Filtrer executor par stratégie
    const filteredPositions = (wsData.executor?.positions || [])
      .filter(p => p.strategy_name === strategyFilter)

    // Déterminer si la stratégie sélectionnée est réellement live
    const isStrategyLive = wsData.executor?.enabled &&
      (wsData.executor?.selector?.allowed_strategies || []).includes(strategyFilter)

    // Données du runner paper pour enrichir l'affichage
    const strategyData = filteredStrategies[strategyFilter]
    const filteredExecutor = isStrategyLive
      // Stratégie live : garder l'executor mais filtrer les positions
      ? { ...wsData.executor, positions: filteredPositions }
      // Stratégie paper : objet enrichi avec infos du runner (watched_symbols, positions, warming_up)
      : {
          enabled: false,
          mode: 'paper',
          positions: [],
          connected: false,
          selector: { allowed_strategies: [] },
          paper_assets: strategyData?.watched_symbols || [],
          paper_num_positions: strategyData?.open_positions ?? 0,
          paper_is_warming_up: strategyData?.is_warming_up ?? false,
        }

    // 4. Filtrer simulator_positions par strategy
    const filteredSimPositions = (wsData.simulator_positions || [])
      .filter(p => (p.strategy_name || p.strategy) === strategyFilter)

    // 5. kill_switch : utiliser celui du runner individuel si disponible
    const stratRunner = filteredStrategies[strategyFilter]
    const filteredKillSwitch = stratRunner?.kill_switch ?? wsData.kill_switch ?? false

    return {
      ...wsData,
      grid_state: filteredGridState,
      strategies: filteredStrategies,
      executor: filteredExecutor,
      simulator_positions: filteredSimPositions,
      kill_switch: filteredKillSwitch,
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
