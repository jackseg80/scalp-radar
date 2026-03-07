/**
 * Scanner — Table principale du scanner avec lignes cliquables et détail extensible.
 * Props : wsData
 * Utilise useApi('/api/simulator/conditions', 10000) pour les données de conditions.
 * Colonnes dynamiques : Score/Signaux masqués si aucune stratégie mono, Dist.SMA affiché si grid actif.
 */
import { useState, useMemo, Fragment, useEffect, useRef } from 'react'
import { useApi } from '../hooks/useApi'
import SignalDots from './SignalDots'
import SignalDetail from './SignalDetail'
import { formatPrice } from '../utils/format'
import GridDetail from './GridDetail'
import Spark from './Spark'
import GridChart from './GridChart'
import Tooltip from './Tooltip'
import ActivePositions from './ActivePositions'
import CollapsibleCard from './CollapsibleCard'
import { buildGridLookupBySymbol } from '../hooks/useFilteredWsData'
import { useStrategyContext } from '../contexts/StrategyContext'
import { GRID_STRATEGIES, GRADE_ORDER, SCANNER_COLUMNS } from '../constants'

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
  // Si grid actif avec positions ouvertes, montrer la direction des positions réelles
  if (gridInfo && gridInfo.levels_open > 0) {
    const dir = gridInfo.direction?.toLowerCase()
    if (dir === 'long') return 'LONG'
    if (dir === 'short') return 'SHORT'
  }
  
  // Suppression du fallback RSI/VWAP selon demande utilisateur
  // On ne retourne plus de signal directionnel s'il n'y a pas de position.
  return null
}

// Composant pour l'affichage du prix avec flash
function PriceCell({ symbol, price }) {
  const prevPriceRef = useRef(price)
  const [flash, setFlash] = useState(null) // 'up', 'down', null

  useEffect(() => {
    if (price > prevPriceRef.current) {
      setFlash('up')
      const timer = setTimeout(() => setFlash(null), 500)
      return () => clearTimeout(timer)
    } else if (price < prevPriceRef.current) {
      setFlash('down')
      const timer = setTimeout(() => setFlash(null), 500)
      return () => clearTimeout(timer)
    }
    prevPriceRef.current = price
  }, [price])

  const flashClass = flash === 'up' ? 'price-flash-up' : flash === 'down' ? 'price-flash-down' : ''

  return (
    <td className={`mono ${flashClass}`}>
      {price != null ? formatPrice(price) : '--'}
    </td>
  )
}

export default function Scanner({ wsData }) {
  const { data, loading } = useApi('/api/simulator/conditions', 10000)
  const { data: gradesData } = useApi('/api/optimization/results?latest_only=true&limit=500', 60000)
  const [selectedAsset, setSelectedAsset] = useState(null)
  const [sortKey, setSortKey] = useState(null) // null, 'symbol', 'change', 'dir', 'grade', 'dist_sma'
  const [sortDirection, setSortDirection] = useState('desc')
  const [showIgnored, setShowIgnored] = useState(false)
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
        lookup[r.asset] = { 
          grade: r.grade, 
          strategy: r.strategy_name, 
          score: r.total_score,
          created_at: r.created_at,
          grade_history: r.grade_history // Added this
        }
      }
    }
    return lookup
  }, [gradesData])

  // Lookup grid state par symbol (clés backend = strategy:symbol)
  const gridLookup = useMemo(
    () => buildGridLookupBySymbol(wsData?.grid_state?.grid_positions),
    [wsData?.grid_state?.grid_positions],
  )

  // Helper pour extraire la valeur de tri
  const getSortValue = (asset, key) => {
    const gridInfo = gridLookup[asset.symbol]
    switch (key) {
      case SCANNER_COLUMNS.SYMBOL: return asset.symbol
      case SCANNER_COLUMNS.CHANGE: return asset.change_pct ?? -999
      case SCANNER_COLUMNS.DIR: return getDirection(asset.indicators, gridInfo) || ''
      case SCANNER_COLUMNS.GRADE: return GRADE_ORDER[gradesLookup[asset.symbol]?.grade || 'F'] || 0
      case SCANNER_COLUMNS.DIST_SMA:
        if (gridInfo?.tp_price && gridInfo?.current_price && gridInfo.tp_price > 0) {
          return ((gridInfo.current_price - gridInfo.tp_price) / gridInfo.tp_price * 100)
        }
        const conds = asset.strategies?.[strategyFilter]?.conditions || []
        const firstLevel = conds.find(c => !c.gate && c.distance_pct != null)
        return firstLevel?.distance_pct ?? -999
      default: return 0
    }
  }

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDirection('desc')
    }
  }

  // Filtrer les assets : actifs vs ignorés
  const { activeAssets, ignoredAssets } = useMemo(() => {
    // 1. Symbols qui ont une position grid ouverte
    const inPosSet = new Set()
    for (const g of Object.values(wsData?.grid_state?.grid_positions || {})) {
      inPosSet.add(g.symbol)
    }

    // 2. Symbols surveillés par la stratégie (whitelist)
    let watchedSet = null
    if (strategyFilter) {
      const ws = wsData?.strategies?.[strategyFilter]?.watched_symbols
      if (ws && ws.length > 0) watchedSet = new Set(ws)
    }

    if (!strategyFilter) return { activeAssets: enrichedAssets, ignoredAssets: [] }
    
    const active = []
    const ignored = []
    
    enrichedAssets.forEach(a => {
      const isWatched = watchedSet?.has(a.symbol)
      const isInPosition = inPosSet.has(a.symbol)
      
      if (isWatched || isInPosition) {
        active.push(a)
      } else {
        ignored.push(a)
      }
    })
    
    return { activeAssets: active, ignoredAssets: ignored }
  }, [enrichedAssets, strategyFilter, wsData?.grid_state?.grid_positions, wsData?.strategies])

  // Détecter quels types de stratégies sont actives (basé sur les actifs actifs)
  const hasGridStrategies = Object.keys(gridLookup).length > 0
    || (strategyFilter && GRID_STRATEGIES.has(strategyFilter))
  const hasMonoStrategies = activeAssets.some(a => {
    const strats = a.strategies || {}
    return Object.keys(strats).some(name => !GRID_STRATEGIES.has(name) && (strats[name].conditions || []).length > 0)
  })

  // Nombre de colonnes dynamique pour colSpan du détail
  const colCount = 7 + (hasMonoStrategies ? 2 : 0) + (hasGridStrategies ? 1 : 0)

  // Helper de tri
  const sortAssets = (list) => {
    const result = [...list]
    if (sortKey) {
      result.sort((a, b) => {
        const valA = getSortValue(a, sortKey)
        const valB = getSortValue(b, sortKey)
        if (valA < valB) return sortDirection === 'asc' ? -1 : 1
        if (valA > valB) return sortDirection === 'asc' ? 1 : -1
        return 0
      })
    } else {
      result.sort((a, b) => {
        const aHasGrid = gridLookup[a.symbol] ? 1 : 0
        const bHasGrid = gridLookup[b.symbol] ? 1 : 0
        if (aHasGrid !== bHasGrid) return bHasGrid - aHasGrid
        if (aHasGrid && bHasGrid) {
          return (gridLookup[b.symbol]?.unrealized_pnl || 0) - (gridLookup[a.symbol]?.unrealized_pnl || 0)
        }
        const aGrade = gradesLookup[a.symbol]?.grade || 'F'
        const bGrade = gradesLookup[b.symbol]?.grade || 'F'
        return (GRADE_ORDER[bGrade] || 0) - (GRADE_ORDER[aGrade] || 0)
      })
    }
    return result
  }

  const sortedActive = useMemo(() => sortAssets(activeAssets), [activeAssets, sortKey, sortDirection, gridLookup, gradesLookup])
  const sortedIgnored = useMemo(() => sortAssets(ignoredAssets), [ignoredAssets, sortKey, sortDirection, gridLookup, gradesLookup])

  const handleRowClick = (symbol) => {
    setSelectedAsset(prev => prev === symbol ? null : symbol)
  }

  const renderAssetRow = (asset, isIgnored = false) => {
    const isSelected = selectedAsset === asset.symbol
    const score = getAssetScore(asset)
    const gridInfo = gridLookup[asset.symbol]
    const direction = getDirection(asset.indicators, gridInfo)
    const changePct = asset.change_pct
    const gradeInfo = gradesLookup[asset.symbol]
    
    // Détection backtest périmé (> 60 jours)
    const STALE_THRESHOLD_DAYS = 60
    let isStale = false
    let ageDays = 0
    if (gradeInfo?.created_at) {
      ageDays = (new Date() - new Date(gradeInfo.created_at)) / (1000 * 60 * 60 * 24)
      isStale = ageDays > STALE_THRESHOLD_DAYS
    }

    return (
      <Fragment key={asset.symbol}>
        <tr
          className={`scanner-row ${isSelected ? 'selected' : ''} ${isIgnored ? 'scanner-row--inactive' : ''}`}
          onClick={() => handleRowClick(asset.symbol)}
        >
          <td style={{ fontWeight: 700 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
              <span className="asset-dot" style={{ opacity: isIgnored ? 0.3 : 1 }} />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{asset.symbol}</span>
            </div>
          </td>
          <PriceCell price={asset.price} />
          <td>
            {changePct != null ? (
              <span style={{ color: changePct >= 0 ? 'var(--accent)' : 'var(--red)', fontWeight: 600 }}>
                {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
              </span>
            ) : '--'}
          </td>
          <td>
            {direction ? (
              <span className={`badge ${direction === 'LONG' ? 'badge-long' : 'badge-short'}`} style={{ opacity: isIgnored ? 0.5 : 1, fontSize: '11px', padding: '3px 8px' }}>
                {direction}
              </span>
            ) : (
              <span className="dim text-xs">--</span>
            )}
          </td>
          <td>
            <div style={{ opacity: isIgnored ? 0.4 : 1 }}>
              <Spark data={asset.sparkline} h={32} />
            </div>
          </td>
          {hasMonoStrategies && (
            <td>
              <span className="score-number" style={{ color: scoreColor(score), opacity: isIgnored ? 0.5 : 1 }}>
                {Math.round(score * 100)}
              </span>
            </td>
          )}
          <td>
            {gradeInfo ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Tooltip content={
                  gradeInfo.grade_history && gradeInfo.grade_history.length > 1
                    ? `Historique WFO : ${gradeInfo.grade_history.join(' → ')}`
                    : `Grade WFO actuel : ${gradeInfo.grade}`
                }>
                  <span className={`grade-badge grade-${gradeInfo.grade}`} style={{ opacity: isIgnored ? 0.5 : 1 }}>
                    {gradeInfo.grade}
                  </span>
                </Tooltip>
                
                {/* Indicateur de dégradation */}
                {gradeInfo.grade_history && gradeInfo.grade_history.length >= 2 && (
                  GRADE_ORDER[gradeInfo.grade] < GRADE_ORDER[gradeInfo.grade_history[gradeInfo.grade_history.length - 2]]
                ) && (
                  <Tooltip content={`Dégradation récente : était ${gradeInfo.grade_history[gradeInfo.grade_history.length - 2]}`}>
                    <span style={{ fontSize: '12px', cursor: 'help' }}>⚠️</span>
                  </Tooltip>
                )}

                {isStale && (
                  <span 
                    className="stale-badge" 
                    title={`Backtest périmé (${Math.floor(ageDays)} jours). Un re-run WFO est conseillé.`}
                  >
                    Périmé
                  </span>
                )}
              </div>
            ) : (
              <span className="dim text-xs">--</span>
            )}
          </td>
          {hasMonoStrategies && (
            <td>
              <div style={{ opacity: isIgnored ? 0.4 : 1 }}>
                <SignalDots strategies={asset.strategies} />
              </div>
            </td>
          )}
          {hasGridStrategies && (
            <td className="mono" style={{ textAlign: 'center' }}>
              {(() => {
                if (gridInfo?.tp_price && gridInfo?.current_price && gridInfo.tp_price > 0) {
                  const dist = ((gridInfo.current_price - gridInfo.tp_price) / gridInfo.tp_price * 100)
                  const color = dist >= 0 ? 'var(--accent)' : 'var(--red)'
                  return <span style={{ color, opacity: isIgnored ? 0.5 : 1 }}>{dist >= 0 ? '+' : ''}{dist.toFixed(1)}%</span>
                }
                const conds = asset.strategies?.[strategyFilter]?.conditions || []
                const firstLevel = conds.find(c => !c.gate && c.distance_pct != null)
                if (firstLevel) {
                  const d = firstLevel.distance_pct
                  const color = Math.abs(d) < 1 ? 'var(--accent)' : Math.abs(d) < 3 ? 'var(--yellow)' : 'var(--muted)'
                  return <span style={{ color, opacity: isIgnored ? 0.5 : 1 }}>{d >= 0 ? '+' : ''}{d.toFixed(1)}%</span>
                }
                return <span className="muted">--</span>
              })()}
            </td>
          )}
          <td>
            {gridInfo ? (
              <span className={`grid-cell ${gridInfo.unrealized_pnl >= 0 ? 'grid-cell--profit' : 'grid-cell--loss'}`} style={{ opacity: isIgnored ? 0.5 : 1 }}>
                {gridInfo.levels_open}/{gridInfo.levels_max}
              </span>
            ) : (
              <span className="grid-cell grid-cell--empty">--</span>
            )}
          </td>
        </tr>
        {isSelected && (
          <tr>
            <td colSpan={colCount} style={{ padding: 0, overflow: 'visible' }}>
              <div style={{ opacity: isIgnored ? 0.7 : 1 }}>
                {gridInfo || (strategyFilter && GRID_STRATEGIES.has(strategyFilter)) ? (
                  <GridDetail
                    symbol={asset.symbol}
                    gridInfo={gridInfo}
                    indicators={asset.indicators}
                    regime={asset.regime}
                    price={asset.price}
                    conditions={asset.strategies?.[strategyFilter]?.conditions}
                    strategyName={strategyFilter}
                    sparkline={asset.sparkline}
                    hasMono={hasMonoStrategies}
                    hasGrid={hasGridStrategies}
                    params={asset.strategies?.[strategyFilter]?.params}
                  />
                ) : (
                  <SignalDetail asset={asset} />
                )}
              </div>
            </td>
          </tr>
        )}
      </Fragment>
    )
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
            {wsData?.strategies?.[strategyFilter]?.watched_symbols && (
              <> — {wsData.strategies[strategyFilter].watched_symbols.length} assets</>
            )}
            {wsData?.strategies?.[strategyFilter]?.circuit_breaker && (
              <span className="circuit-breaker-badge" title={`${wsData.strategies[strategyFilter].crash_count} crashes`}>
                DISABLED
              </span>
            )}
          </div>
        )}

      {loading && assets.length === 0 && (
        <table className="scanner-table">
          <thead>
            <tr>
              <th style={{ width: '14%' }}>Actif</th>
              <th style={{ width: '10%' }}>Prix</th>
              <th style={{ width: '8%' }}>Var.</th>
              <th style={{ width: '7%' }}>Dir.</th>
              <th style={{ width: '12%' }}>Trend</th>
              <th style={{ width: '7%' }}>Grade</th>
              <th style={{ width: '7%' }}>Grid</th>
            </tr>
          </thead>
          <tbody>
            {[...Array(10)].map((_, i) => (
              <tr key={i} className="scanner-row">
                <td><div className="skeleton skeleton-cell" style={{ width: '60%' }} /></td>
                <td><div className="skeleton skeleton-cell" style={{ width: '70%' }} /></td>
                <td><div className="skeleton skeleton-cell" style={{ width: '50%' }} /></td>
                <td><div className="skeleton skeleton-cell" style={{ width: '40%' }} /></td>
                <td><div className="skeleton skeleton-spark" /></td>
                <td><div className="skeleton skeleton-cell" style={{ width: '30%' }} /></td>
                <td><div className="skeleton skeleton-cell" style={{ width: '40%' }} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {!loading && assets.length === 0 && (
        <div className="empty-state">En attente de données...</div>
      )}

      {(sortedActive.length > 0 || sortedIgnored.length > 0) && (
        <table className="scanner-table scanner-table--sticky">
          <thead>
            <tr>
              <th 
                style={{ width: '14%', cursor: 'pointer' }} 
                onClick={() => handleSort(SCANNER_COLUMNS.SYMBOL)}
                className={sortKey === SCANNER_COLUMNS.SYMBOL ? 'active-sort' : ''}
              >
                Actif {sortKey === SCANNER_COLUMNS.SYMBOL && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th style={{ width: '10%' }}>Prix</th>
              <th 
                style={{ width: '8%', cursor: 'pointer' }} 
                onClick={() => handleSort(SCANNER_COLUMNS.CHANGE)}
                className={sortKey === SCANNER_COLUMNS.CHANGE ? 'active-sort' : ''}
              >
                <Tooltip content="Variation de prix entre les 2 dernières bougies 1 min">
                  Var. {sortKey === SCANNER_COLUMNS.CHANGE && (sortDirection === 'asc' ? '↑' : '↓')}
                </Tooltip>
              </th>
              <th 
                style={{ width: '7%', cursor: 'pointer' }} 
                onClick={() => handleSort(SCANNER_COLUMNS.DIR)}
                className={sortKey === SCANNER_COLUMNS.DIR ? 'active-sort' : ''}
              >
                <Tooltip content="Direction suggérée par les indicateurs ou positions grid ouvertes">
                  Dir. {sortKey === SCANNER_COLUMNS.DIR && (sortDirection === 'asc' ? '↑' : '↓')}
                </Tooltip>
              </th>
              <th style={{ width: '12%' }}><Tooltip content="Sparkline des 60 derniers prix de clôture (1 min)">Trend</Tooltip></th>
              {hasMonoStrategies && (
                <th style={{ width: '7%' }}><Tooltip content="Score = ratio de conditions remplies de la meilleure stratégie (100 = toutes remplies)">Score</Tooltip></th>
              )}
              <th 
                style={{ width: '7%', cursor: 'pointer' }} 
                onClick={() => handleSort(SCANNER_COLUMNS.GRADE)}
                className={sortKey === SCANNER_COLUMNS.GRADE ? 'active-sort' : ''}
              >
                <Tooltip content="Grade WFO de la stratégie (A=excellent, F=mauvais)">
                  Grade {sortKey === SCANNER_COLUMNS.GRADE && (sortDirection === 'asc' ? '↑' : '↓')}
                </Tooltip>
              </th>
              {hasMonoStrategies && (
                <th style={{ width: '20%' }}><Tooltip content="Pastilles par stratégie active">Signaux</Tooltip></th>
              )}
              {hasGridStrategies && (
                <th 
                  style={{ width: '9%', cursor: 'pointer' }} 
                  onClick={() => handleSort(SCANNER_COLUMNS.DIST_SMA)}
                  className={sortKey === SCANNER_COLUMNS.DIST_SMA ? 'active-sort' : ''}
                >
                  <Tooltip content="Distance du prix à la SMA (négatif = sous la SMA)">
                    Dist.SMA {sortKey === SCANNER_COLUMNS.DIST_SMA && (sortDirection === 'asc' ? '↑' : '↓')}
                  </Tooltip>
                </th>
              )}
              <th style={{ width: '7%' }}><Tooltip content="Niveaux grid DCA remplis / total. Coloré selon P&L non réalisé">Grid</Tooltip></th>
            </tr>
          </thead>
          <tbody>
            {sortedActive.map(asset => renderAssetRow(asset, false))}
            
            {sortedIgnored.length > 0 && (
              <>
                <tr 
                  className="scanner-ignored-toggle"
                  onClick={() => setShowIgnored(!showIgnored)}
                  style={{ cursor: 'pointer', background: 'rgba(255,255,255,0.02)' }}
                >
                  <td colSpan={colCount} style={{ padding: '8px 16px', color: 'var(--text-muted)', fontSize: '11px', fontWeight: 600 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span>{showIgnored ? '▼' : '▶'}</span>
                      <span>AUTRES ACTIFS ({sortedIgnored.length})</span>
                      <div style={{ height: '1px', flex: 1, background: 'var(--border)', opacity: 0.5 }} />
                    </div>
                  </td>
                </tr>
                {showIgnored && sortedIgnored.map(asset => renderAssetRow(asset, true))}
              </>
            )}
          </tbody>
        </table>
      )}
      </div>
    </>
  )
}
