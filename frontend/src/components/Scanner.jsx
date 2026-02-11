/**
 * Scanner — Table principale du scanner avec lignes cliquables et détail extensible.
 * Props : wsData
 * Utilise useApi('/api/simulator/conditions', 10000) pour les données de conditions.
 * Affiche un tableau d'assets avec prix, régime, RSI, distance VWAP, et détail extensible.
 */
import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import SignalDots from './SignalDots'
import SignalDetail from './SignalDetail'
import Spark from './Spark'

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

      {enrichedAssets.length > 0 && (
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Asset</th>
                <th>Prix</th>
                <th>Sparkline</th>
                <th>Régime</th>
                <th>RSI</th>
                <th>VWAP dist</th>
                <th>Signaux</th>
              </tr>
            </thead>
            <tbody>
              {enrichedAssets.map(asset => {
                const isSelected = selectedAsset === asset.symbol
                const regime = asset.regime || '--'
                const rsi = asset.indicators?.rsi_14
                const vwapDist = asset.indicators?.vwap_distance_pct

                // Couleur RSI
                const rsiColor = rsi != null
                  ? rsi < 30 ? 'var(--accent)' : rsi > 70 ? 'var(--red)' : null
                  : null

                return (
                  <tr key={asset.symbol}>
                    <td colSpan={7} style={{ padding: 0 }}>
                      <table style={{ width: '100%' }}>
                        <tbody>
                          <tr
                            className={`scanner-row ${isSelected ? 'selected' : ''}`}
                            onClick={() => handleRowClick(asset.symbol)}
                          >
                            <td style={{ fontWeight: 600, width: '16%' }}>{asset.symbol}</td>
                            <td style={{ width: '14%' }}>
                              {asset.price != null ? Number(asset.price).toFixed(2) : '--'}
                            </td>
                            <td style={{ width: '14%' }}>
                              <Spark data={asset.sparkline} w={80} h={24} />
                            </td>
                            <td style={{ width: '12%' }}>
                              <span className={`badge ${
                                regime === 'RANGING' ? 'badge-ranging' : 'badge-trending'
                              }`}>
                                {regime}
                              </span>
                            </td>
                            <td style={{ width: '10%', color: rsiColor }}>
                              {rsi != null ? Number(rsi).toFixed(1) : '--'}
                            </td>
                            <td style={{ width: '14%' }}>
                              {vwapDist != null ? `${Number(vwapDist).toFixed(3)}%` : '--'}
                            </td>
                            <td style={{ width: '20%' }}>
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
