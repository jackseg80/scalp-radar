/**
 * Scanner — Table principale du scanner avec lignes cliquables et détail extensible.
 * Props : wsData
 * Utilise useApi('/api/simulator/conditions', 10000) pour les données de conditions.
 * Colonnes dynamiques : Score/Signaux masqués si aucune stratégie mono, Dist.SMA affiché si grid actif.
 */
import { useState, useMemo, Fragment } from 'react'
import { useApi } from '../hooks/useApi'
import SignalDots from './SignalDots'
import SignalDetail from './SignalDetail'
import { formatPrice } from '../utils/format'
import GridDetail from './GridDetail'
import Spark from './Spark'
import Tooltip from './Tooltip'
import ActivePositions from './ActivePositions'
import CollapsibleCard from './CollapsibleCard'
import { buildGridLookupBySymbol } from '../hooks/useFilteredWsData'
import { useStrategyContext } from '../contexts/StrategyContext'

function getAssetScore(asset) {
  const strats = asset.strategies || {}
  let bestRatio = 0
  Object.values(strats).forEach(s => {
    const conditions = s.conditions || []
    const total = conditions.length || 1
    const met = conditions.filter(c => c.met).length
    const ratio = met / total
    if (ratio > bestRatio) bestRatio = ratio
  })
  return bestRatio
}

function scoreColor(score) {
  if (score >= 0.75) return 'var(--accent)'
  if (score >= 0.55) return 'var(--yellow)'
  if (score >= 0.35) return 'var(--orange)'
  return 'var(--red)'
}

function getDirection(indicators, gridInfo) {
  // Si grid actif avec positions ouvertes, montrer la direction des positions
  if (gridInfo && gridInfo.levels_open > 0) {
    return gridInfo.direction === 'long' ? 'LONG' : 'SHORT'
  }
  // Fallback mono : RSI/VWAP
  if (!indicators) return null
  const rsi = indicators.rsi_14
  const vwap = indicators.vwap_distance_pct
  if (rsi == null) return null
  if (rsi < 30) return 'LONG'
  if (rsi > 70) return 'SHORT'
  if (vwap != null) {
    if (vwap < -0.3) return 'LONG'
    if (vwap > 0.3) return 'SHORT'
  }
  return null
}

const GRADE_ORDER = { A: 5, B: 4, C: 3, D: 2, F: 1 }

export default function Scanner({ wsData }) {
  const { data, loading } = useApi('/api/simulator/conditions', 10000)
  const { data: gradesData } = useApi('/api/optimization/results?latest_only=true&limit=500', 60000)
  const [selectedAsset, setSelectedAsset] = useState(null)
  const { strategyFilter } = useStrategyContext()

  // Backend renvoie assets comme dict {symbol: data}, convertir en tableau
  const assetsObj = data?.assets || {}
  const assets = Object.entries(assetsObj).map(([symbol, a]) => ({ symbol, ...a }))

  // Fusionner les prix WS si disponibles (temps réel)
  const enrichedAssets = assets.map(a => {
    const wsPrice = wsData?.prices?.[a.symbol]
    return {
      ...a,
      price: wsPrice?.last ?? a.price,
    }
  })

  // Lookup grades : pour chaque asset, meilleur grade toutes stratégies confondues
  const gradesLookup = useMemo(() => {
    const lookup = {}
    if (!gradesData?.results) return lookup
    for (const r of gradesData.results) {
      const existing = lookup[r.asset]
      if (!existing || (GRADE_ORDER[r.grade] || 0) > (GRADE_ORDER[existing.grade] || 0)) {
        lookup[r.asset] = { grade: r.grade, strategy: r.strategy_name, score: r.total_score }
      }
    }
    return lookup
  }, [gradesData])

  // Lookup grid state par symbol (clés backend = strategy:symbol)
  const gridLookup = useMemo(
    () => buildGridLookupBySymbol(wsData?.grid_state?.grid_positions),
    [wsData?.grid_state?.grid_positions],
  )

  // Symbols qui ont une position grid ouverte
  const inPositionSymbols = useMemo(() => {
    const symbols = new Set()
    for (const g of Object.values(wsData?.grid_state?.grid_positions || {})) {
      symbols.add(g.symbol)
    }
    return symbols
  }, [wsData?.grid_state?.grid_positions])

  // Symbols surveillés par la stratégie filtrée (whitelist per_asset)
  const watchedSymbols = useMemo(() => {
    if (!strategyFilter) return null
    const ws = wsData?.strategies?.[strategyFilter]?.watched_symbols
    if (!ws || ws.length === 0) return null
    return new Set(ws)
  }, [strategyFilter, wsData?.strategies])

  // Filtrer les assets : watched > in-position > tous
  const filteredAssets = useMemo(() => {
    if (!strategyFilter) return enrichedAssets
    if (watchedSymbols) return enrichedAssets.filter(a => watchedSymbols.has(a.symbol))
    if (inPositionSymbols.size > 0) return enrichedAssets.filter(a => inPositionSymbols.has(a.symbol))
    return enrichedAssets
  }, [enrichedAssets, strategyFilter, watchedSymbols, inPositionSymbols])

  // Détecter quels types de stratégies sont actives
  const hasGridStrategies = Object.keys(gridLookup).length > 0
  const hasMonoStrategies = filteredAssets.some(a => {
    const strats = a.strategies || {}
    return Object.values(strats).some(s => {
      const conditions = s.conditions || []
      return conditions.length > 0 && conditions.some(c => c.name && !c.name.startsWith('Level'))
    })
  })

  // Nombre de colonnes dynamique pour colSpan du détail
  // Base: Actif, Prix, Var, Dir, Trend, Grade, Grid = 7
  const colCount = 7 + (hasMonoStrategies ? 2 : 0) + (hasGridStrategies ? 1 : 0)

  // Trier : positions grid ouvertes en premier, puis par grade décroissant
  const sortedAssets = [...filteredAssets].sort((a, b) => {
    const aHasGrid = gridLookup[a.symbol] ? 1 : 0
    const bHasGrid = gridLookup[b.symbol] ? 1 : 0
    if (aHasGrid !== bHasGrid) return bHasGrid - aHasGrid
    // Au sein des grids ouvertes, trier par P&L non réalisé desc
    if (aHasGrid && bHasGrid) {
      return (gridLookup[b.symbol]?.unrealized_pnl || 0) - (gridLookup[a.symbol]?.unrealized_pnl || 0)
    }
    // Non-grid : trier par grade
    const aGrade = gradesLookup[a.symbol]?.grade || 'F'
    const bGrade = gradesLookup[b.symbol]?.grade || 'F'
    return (GRADE_ORDER[bGrade] || 0) - (GRADE_ORDER[aGrade] || 0)
  })

  const handleRowClick = (symbol) => {
    setSelectedAsset(prev => prev === symbol ? null : symbol)
  }

  // Résumé pour ActivePositions
  const simPositions = wsData?.simulator_positions || []
  const execPositions = wsData?.executor?.positions || (wsData?.executor?.position ? [wsData.executor.position] : [])
  const gridState = wsData?.grid_state || null
  const hasGrids = gridState?.summary?.total_positions > 0
  const monoSimPositions = simPositions.filter(p => p.type !== 'grid')
  const hasActivePositions = monoSimPositions.length > 0 || execPositions.length > 0 || hasGrids
  const positionsSummary = hasActivePositions
    ? `${simPositions.length + execPositions.length} position${simPositions.length + execPositions.length > 1 ? 's' : ''}`
    : null

  return (
    <>
      <CollapsibleCard
        title="Positions actives"
        summary={positionsSummary}
        defaultOpen={true}
        storageKey="active-positions"
      >
        <ActivePositions wsData={wsData} />
      </CollapsibleCard>

      <div className="card">
        <h2>Scanner</h2>
        {strategyFilter && (
          <div className="text-xs muted" style={{ marginTop: -4, marginBottom: 8 }}>
            {strategyFilter}
            {wsData?.strategies?.[strategyFilter]?.leverage && (
              <> — x{wsData.strategies[strategyFilter].leverage}</>
            )}
            {watchedSymbols && <> — {watchedSymbols.size} assets</>}
            {wsData?.strategies?.[strategyFilter]?.circuit_breaker && (
              <span className="circuit-breaker-badge" title={`${wsData.strategies[strategyFilter].crash_count} crashes`}>
                DISABLED
              </span>
            )}
          </div>
        )}

      {loading && assets.length === 0 && (
        <div className="empty-state">
          <div className="skeleton skeleton-line" style={{ width: '80%', margin: '0 auto' }} />
          <div className="skeleton skeleton-line" style={{ width: '60%', margin: '8px auto 0' }} />
          <div className="skeleton skeleton-line" style={{ width: '70%', margin: '8px auto 0' }} />
        </div>
      )}

      {!loading && assets.length === 0 && (
        <div className="empty-state">En attente de données...</div>
      )}

      {sortedAssets.length > 0 && (
        <table className="scanner-table">
          <thead>
            <tr>
              <th style={{ width: '14%' }}>Actif</th>
              <th style={{ width: '10%' }}>Prix</th>
              <th style={{ width: '8%' }}><Tooltip content="Variation de prix entre les 2 dernières bougies 1 min">Var.</Tooltip></th>
              <th style={{ width: '7%' }}><Tooltip content="Direction suggérée par les indicateurs ou positions grid ouvertes">Dir.</Tooltip></th>
              <th style={{ width: '12%' }}><Tooltip content="Sparkline des 60 derniers prix de clôture (1 min)">Trend</Tooltip></th>
              {hasMonoStrategies && (
                <th style={{ width: '7%' }}><Tooltip content="Score = ratio de conditions remplies de la meilleure stratégie (100 = toutes remplies)">Score</Tooltip></th>
              )}
              <th style={{ width: '7%' }}><Tooltip content="Grade WFO de la stratégie (A=excellent, F=mauvais)">Grade</Tooltip></th>
              {hasMonoStrategies && (
                <th style={{ width: '20%' }}><Tooltip content="Pastilles par stratégie active">Signaux</Tooltip></th>
              )}
              {hasGridStrategies && (
                <th style={{ width: '9%' }}><Tooltip content="Distance du prix à la SMA (négatif = sous la SMA)">Dist.SMA</Tooltip></th>
              )}
              <th style={{ width: '7%' }}><Tooltip content="Niveaux grid DCA remplis / total. Coloré selon P&L non réalisé">Grid</Tooltip></th>
            </tr>
          </thead>
          <tbody>
            {sortedAssets.map(asset => {
              const isSelected = selectedAsset === asset.symbol
              const score = getAssetScore(asset)
              const gridInfo = gridLookup[asset.symbol]
              const direction = getDirection(asset.indicators, gridInfo)
              const changePct = asset.change_pct
              const gradeInfo = gradesLookup[asset.symbol]
              // Asset surveillé par la stratégie mais sans position ouverte
              const isInactive = !!strategyFilter && !!watchedSymbols && !inPositionSymbols.has(asset.symbol)

              return (
                <Fragment key={asset.symbol}>
                  <tr
                    className={`scanner-row ${isSelected ? 'selected' : ''} ${isInactive ? 'scanner-row--inactive' : ''}`}
                    onClick={isInactive ? undefined : () => handleRowClick(asset.symbol)}
                  >
                    <td style={{ fontWeight: 700 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
                        <span className="asset-dot" />
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{asset.symbol}</span>
                      </div>
                    </td>
                    <td>
                      {asset.price != null ? formatPrice(asset.price) : '--'}
                    </td>
                    <td>
                      {changePct != null ? (
                        <span style={{ color: changePct >= 0 ? 'var(--accent)' : 'var(--red)', fontWeight: 600 }}>
                          {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                        </span>
                      ) : '--'}
                    </td>
                    <td>
                      {direction ? (
                        <span className={`badge ${direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
                          {direction}
                        </span>
                      ) : (
                        <span className="dim text-xs">--</span>
                      )}
                    </td>
                    <td>
                      <Spark data={asset.sparkline} h={32} />
                    </td>
                    {hasMonoStrategies && (
                      <td>
                        <span className="score-number" style={{ color: scoreColor(score) }}>
                          {Math.round(score * 100)}
                        </span>
                      </td>
                    )}
                    <td>
                      {gradeInfo ? (
                        <span className={`grade-badge grade-${gradeInfo.grade}`}>
                          {gradeInfo.grade}
                        </span>
                      ) : (
                        <span className="dim text-xs">--</span>
                      )}
                    </td>
                    {hasMonoStrategies && (
                      <td>
                        <SignalDots strategies={asset.strategies} />
                      </td>
                    )}
                    {hasGridStrategies && (
                      <td className="mono" style={{ textAlign: 'center' }}>
                        {(() => {
                          // TP = SMA pour les stratégies grid → dist_sma = (price - sma) / sma
                          if (gridInfo?.tp_price && gridInfo?.current_price && gridInfo.tp_price > 0) {
                            const dist = ((gridInfo.current_price - gridInfo.tp_price) / gridInfo.tp_price * 100)
                            const color = dist >= 0 ? 'var(--accent)' : 'var(--red)'
                            return <span style={{ color }}>{dist >= 0 ? '+' : ''}{dist.toFixed(1)}%</span>
                          }
                          return <span className="muted">--</span>
                        })()}
                      </td>
                    )}
                    <td>
                      {gridInfo ? (
                        <span className={`grid-cell ${gridInfo.unrealized_pnl >= 0 ? 'grid-cell--profit' : 'grid-cell--loss'}`}>
                          {gridInfo.levels_open}/{gridInfo.levels_max}
                        </span>
                      ) : (
                        <span className="grid-cell grid-cell--empty">--</span>
                      )}
                    </td>
                  </tr>
                  {isSelected && (
                    <tr>
                      <td colSpan={colCount} style={{ padding: 0 }}>
                        {gridInfo ? (
                          <GridDetail
                            symbol={asset.symbol}
                            gridInfo={gridInfo}
                            indicators={asset.indicators}
                            regime={asset.regime}
                            price={asset.price}
                          />
                        ) : (
                          <SignalDetail asset={asset} />
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      )}
      </div>
    </>
  )
}
