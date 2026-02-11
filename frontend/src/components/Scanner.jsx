/**
 * Scanner — Table principale du scanner avec lignes cliquables et détail extensible.
 * Props : wsData
 * Utilise useApi('/api/simulator/conditions', 10000) pour les données de conditions.
 * Affiche un tableau d'assets avec prix, variation, direction, sparkline, score, signaux.
 */
import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import SignalDots from './SignalDots'
import SignalDetail from './SignalDetail'
import Spark from './Spark'

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

export default function Scanner({ wsData }) {
  const { data, loading } = useApi('/api/simulator/conditions', 10000)
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

  // Trier par score décroissant
  const sortedAssets = [...enrichedAssets].sort((a, b) => getAssetScore(b) - getAssetScore(a))

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
                <th>Var.</th>
                <th>Dir.</th>
                <th>Trend</th>
                <th>Score</th>
                <th>Signaux</th>
              </tr>
            </thead>
            <tbody>
              {sortedAssets.map(asset => {
                const isSelected = selectedAsset === asset.symbol
                const score = getAssetScore(asset)
                const direction = getDirection(asset.indicators)
                const changePct = asset.change_pct

                return (
                  <tr key={asset.symbol}>
                    <td colSpan={7} style={{ padding: 0 }}>
                      <table style={{ width: '100%' }}>
                        <tbody>
                          <tr
                            className={`scanner-row ${isSelected ? 'selected' : ''}`}
                            onClick={() => handleRowClick(asset.symbol)}
                          >
                            <td style={{ fontWeight: 700, width: '14%' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span className="asset-dot" />
                                {asset.symbol}
                              </div>
                            </td>
                            <td style={{ width: '11%' }}>
                              {asset.price != null ? Number(asset.price).toFixed(2) : '--'}
                            </td>
                            <td style={{ width: '9%' }}>
                              {changePct != null ? (
                                <span style={{ color: changePct >= 0 ? 'var(--accent)' : 'var(--red)', fontWeight: 600 }}>
                                  {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
                                </span>
                              ) : '--'}
                            </td>
                            <td style={{ width: '8%' }}>
                              {direction ? (
                                <span className={`badge ${direction === 'LONG' ? 'badge-long' : 'badge-short'}`}>
                                  {direction}
                                </span>
                              ) : (
                                <span className="dim text-xs">--</span>
                              )}
                            </td>
                            <td style={{ width: '14%' }}>
                              <Spark data={asset.sparkline} w={110} h={32} />
                            </td>
                            <td style={{ width: '10%' }}>
                              <span className="score-number" style={{ color: scoreColor(score) }}>
                                {Math.round(score * 100)}
                              </span>
                            </td>
                            <td style={{ width: '24%' }}>
                              <SignalDots strategies={asset.strategies} />
                            </td>
                          </tr>
                          {isSelected && (
                            <tr>
                              <td colSpan={7} style={{ padding: 0 }}>
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
