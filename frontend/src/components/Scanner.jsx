/**
 * Scanner — Table principale du scanner avec lignes cliquables et détail extensible.
 * Props : wsData
 * Utilise useApi('/api/simulator/conditions', 10000) pour les données de conditions.
 * Affiche un tableau d'assets avec prix, variation, direction, sparkline, score, grade, signaux, grid.
 */
import { useState, useMemo } from 'react'
import { useApi } from '../hooks/useApi'
import SignalDots from './SignalDots'
import SignalDetail from './SignalDetail'
import Spark from './Spark'
import Tooltip from './Tooltip'

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

function getDirection(indicators) {
  if (!indicators) return null
  const rsi = indicators.rsi_14
  const vwap = indicators.vwap_distance_pct
  if (rsi == null) return null
  // RSI extrême donne une direction claire
  if (rsi < 30) return 'LONG'
  if (rsi > 70) return 'SHORT'
  // VWAP distance comme confirmation secondaire
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

  // Lookup grades : pour chaque asset, meilleur grade (priorité envelope_dca)
  const gradesLookup = useMemo(() => {
    const lookup = {}
    if (!gradesData?.results) return lookup
    for (const r of gradesData.results) {
      const existing = lookup[r.asset]
      if (!existing ||
          r.strategy_name === 'envelope_dca' ||
          (existing.strategy !== 'envelope_dca' &&
           (GRADE_ORDER[r.grade] || 0) > (GRADE_ORDER[existing.grade] || 0))) {
        lookup[r.asset] = { grade: r.grade, strategy: r.strategy_name, score: r.total_score }
      }
    }
    return lookup
  }, [gradesData])

  // Lookup grid state depuis le WebSocket
  const gridLookup = wsData?.grid_state?.grid_positions || {}

  // Trier : positions grid ouvertes en premier, puis par score décroissant
  const sortedAssets = [...enrichedAssets].sort((a, b) => {
    const aHasGrid = gridLookup[a.symbol] ? 1 : 0
    const bHasGrid = gridLookup[b.symbol] ? 1 : 0
    if (aHasGrid !== bHasGrid) return bHasGrid - aHasGrid
    // Au sein des grids ouvertes, trier par P&L non réalisé desc
    if (aHasGrid && bHasGrid) {
      return (gridLookup[b.symbol]?.unrealized_pnl || 0) - (gridLookup[a.symbol]?.unrealized_pnl || 0)
    }
    return getAssetScore(b) - getAssetScore(a)
  })

  const handleRowClick = (symbol) => {
    setSelectedAsset(prev => prev === symbol ? null : symbol)
  }

  return (
    <div className="card">
      <h2>Scanner</h2>

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
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Actif</th>
                <th>Prix</th>
                <th><Tooltip content="Variation de prix entre les 2 dernières bougies 1 min">Var.</Tooltip></th>
                <th><Tooltip content="Direction suggérée par les indicateurs (RSI < 30 → LONG, RSI > 70 → SHORT)">Dir.</Tooltip></th>
                <th><Tooltip content="Sparkline des 60 derniers prix de clôture (1 min)">Trend</Tooltip></th>
                <th><Tooltip content="Score = ratio de conditions remplies de la meilleure stratégie (100 = toutes remplies)">Score</Tooltip></th>
                <th><Tooltip content="Grade WFO de la stratégie (A=excellent, F=mauvais)">Grade</Tooltip></th>
                <th><Tooltip content="Pastilles par stratégie : V=VWAP+RSI, M=Momentum, F=Funding, L=Liquidation">Signaux</Tooltip></th>
                <th><Tooltip content="Niveaux grid DCA remplis / total. Coloré selon P&L non réalisé">Grid</Tooltip></th>
              </tr>
            </thead>
            <tbody>
              {sortedAssets.map(asset => {
                const isSelected = selectedAsset === asset.symbol
                const score = getAssetScore(asset)
                const direction = getDirection(asset.indicators)
                const changePct = asset.change_pct
                const gradeInfo = gradesLookup[asset.symbol]
                const gridInfo = gridLookup[asset.symbol]

                return (
                  <tr key={asset.symbol}>
                    <td colSpan={9} style={{ padding: 0 }}>
                      <table style={{ width: '100%' }}>
                        <tbody>
                          <tr
                            className={`scanner-row ${isSelected ? 'selected' : ''}`}
                            onClick={() => handleRowClick(asset.symbol)}
                          >
                            <td style={{ fontWeight: 700, width: '13%' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span className="asset-dot" />
                                {asset.symbol}
                              </div>
                            </td>
                            <td style={{ width: '9%' }}>
                              {asset.price != null ? Number(asset.price).toFixed(2) : '--'}
                            </td>
                            <td style={{ width: '7%' }}>
                              {changePct != null ? (
                                <span style={{ color: changePct >= 0 ? 'var(--accent)' : 'var(--red)', fontWeight: 600 }}>
                                  {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                                </span>
                              ) : '--'}
                            </td>
                            <td style={{ width: '7%' }}>
                              {direction ? (
                                <span className={`badge ${direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
                                  {direction}
                                </span>
                              ) : (
                                <span className="dim text-xs">--</span>
                              )}
                            </td>
                            <td style={{ width: '12%' }}>
                              <Spark data={asset.sparkline} w={110} h={32} />
                            </td>
                            <td style={{ width: '8%' }}>
                              <span className="score-number" style={{ color: scoreColor(score) }}>
                                {Math.round(score * 100)}
                              </span>
                            </td>
                            <td style={{ width: '7%' }}>
                              {gradeInfo ? (
                                <span className={`grade-badge grade-${gradeInfo.grade}`}>
                                  {gradeInfo.grade}
                                </span>
                              ) : (
                                <span className="dim text-xs">--</span>
                              )}
                            </td>
                            <td style={{ width: '22%' }}>
                              <SignalDots strategies={asset.strategies} />
                            </td>
                            <td style={{ width: '7%' }}>
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
                              <td colSpan={9} style={{ padding: 0 }}>
                                <SignalDetail asset={asset} />
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
