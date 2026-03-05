/**
 * GridChart — Graphique détaillé pour GridDetail.
 * Affiche la courbe de prix (sparkline) + niveaux Grid (horizontal lines).
 * 
 * Props :
 * - data (array) : Historique des prix (sparkline)
 * - levels (array) : Liste des niveaux { price, filled, direction }
 * - currentPrice (number) : Prix actuel
 * - tpPrice (number) : Prix de Take Profit
 * - slPrice (number) : Prix de Stop Loss
 * - width (number/string)
 * - height (number/string)
 */
import { useMemo } from 'react'

export default function GridChart({ data = [], levels = [], currentPrice, tpPrice, slPrice, width = 160, height = '100%', mini = false }) {
  // Calculer les bornes du graphique pour englober tous les niveaux importants
  const bounds = useMemo(() => {
    if (!data.length && !levels.length && !currentPrice) return { min: 0, max: 100 }
    
    let allPrices = [...data]
    if (currentPrice) allPrices.push(currentPrice)
    levels.forEach(l => { if (l.price) allPrices.push(l.price) })
    if (tpPrice) allPrices.push(tpPrice)
    if (slPrice) allPrices.push(slPrice)
    
    const min = Math.min(...allPrices)
    const max = Math.max(...allPrices)
    const range = max - min || (max * 0.01) // 1% par défaut si range nul
    
    // Ajouter 10% de padding en haut et en bas (moins en mini)
    const p = mini ? 0.05 : 0.1
    return {
      min: min - range * p,
      max: max + range * p,
      range: range * (1 + p * 2)
    }
  }, [data, levels, currentPrice, tpPrice, slPrice, mini])

  const getY = (price) => {
    if (!price || bounds.range === 0) return 0
    return ((bounds.max - price) / bounds.range) * 100 // en %
  }

  // Points pour la sparkline
  const points = useMemo(() => {
    if (!data.length) return ""
    return data.map((val, i) => {
      const x = (i / (data.length - 1)) * 100
      const y = getY(val)
      return `${x.toFixed(2)},${y.toFixed(2)}`
    }).join(' ')
  }, [data, bounds])

  return (
    <div style={{ 
      width, 
      height, 
      position: 'relative', 
      background: mini ? 'transparent' : 'rgba(255,255,255,0.02)', 
      borderRadius: 4, 
      overflow: 'hidden', 
      border: mini ? 'none' : '1px solid var(--border)' 
    }}>
      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" style={{ display: 'block' }}>
        {!mini && (
          <>
            <line x1="0" y1="25" x2="100" y2="25" stroke="var(--border)" strokeWidth="0.1" strokeDasharray="1,1" />
            <line x1="0" y1="50" x2="100" y2="50" stroke="var(--border)" strokeWidth="0.1" strokeDasharray="1,1" />
            <line x1="0" y1="75" x2="100" y2="75" stroke="var(--border)" strokeWidth="0.1" strokeDasharray="1,1" />
          </>
        )}

        {/* Niveaux Grid */}
        {levels.map((lvl, i) => {
          if (!lvl.price) return null
          const y = getY(lvl.price)
          const color = lvl.filled ? 'var(--accent)' : 'var(--text-dim)'
          return (
            <g key={i}>
              <line 
                x1="0" y1={y} x2="100" y2={y} 
                stroke={color} 
                strokeWidth={lvl.filled ? (mini ? "0.4" : "0.6") : (mini ? "0.2" : "0.3")} 
                strokeDasharray={lvl.filled ? "" : "2,1"}
              />
              {!mini && (
                <>
                  <rect x="0" y={y-2} width="8" height="4" fill={color} opacity="0.2" />
                  <text x="1" y={y+1} fontSize="3" fill={color} fontWeight="bold">L{i+1}</text>
                </>
              )}
            </g>
          )
        })}

        {/* Stop Loss */}
        {slPrice && (
          <g>
            <line x1="0" y1={getY(slPrice)} x2="100" y2={getY(slPrice)} stroke="var(--red)" strokeWidth={mini ? "0.4" : "0.8"} strokeDasharray="2,2" />
            {!mini && <text x="85" y={getY(slPrice)-1.5} fontSize="4" fill="var(--red)" fontWeight="bold">SL</text>}
          </g>
        )}

        {/* Take Profit */}
        {tpPrice && (
          <g>
            <line x1="0" y1={getY(tpPrice)} x2="100" y2={getY(tpPrice)} stroke="var(--accent)" strokeWidth={mini ? "0.4" : "0.8"} strokeDasharray="2,2" />
            {!mini && <text x="85" y={getY(tpPrice)-1.5} fontSize="4" fill="var(--accent)" fontWeight="bold">TP</text>}
          </g>
        )}

        {/* Courbe de prix */}
        {points && (
          <polyline
            points={points}
            fill="none"
            stroke={mini ? (data[data.length-1] > data[0] ? 'var(--accent)' : 'var(--red)') : 'var(--text-secondary)'}
            strokeWidth={mini ? "1.2" : "1"}
            strokeLinejoin="round"
            strokeLinecap="round"
            opacity={mini ? 1 : 0.8}
          />
        )}

        {/* Prix Actuel */}
        {currentPrice && (
          <g>
            <line x1="0" y1={getY(currentPrice)} x2="100" y2={getY(currentPrice)} stroke="var(--yellow)" strokeWidth={mini ? "0.3" : "0.5"} opacity="0.8" />
            <circle cx="100" cy={getY(currentPrice)} r={mini ? "1.2" : "2"} fill="var(--yellow)" />
          </g>
        )}
      </svg>
      
      {/* Label Prix Actuel (seulement en mode normal) */}
      {!mini && currentPrice && (
        <div style={{
          position: 'absolute',
          right: 2,
          top: `${getY(currentPrice)}%`,
          transform: 'translateY(-50%)',
          background: 'var(--yellow)',
          color: '#000',
          fontSize: '9px',
          padding: '1px 3px',
          borderRadius: 2,
          fontWeight: 800,
          pointerEvents: 'none',
          boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
          zIndex: 5
        }}>
          {currentPrice.toLocaleString()}
        </div>
      )}

      {/* Label TP (seulement en mode normal) */}
      {!mini && tpPrice && (
        <div style={{
          position: 'absolute',
          left: 2,
          top: `${getY(tpPrice)}%`,
          transform: 'translateY(-100%)',
          color: 'var(--accent)',
          fontSize: '8px',
          fontWeight: 700,
          pointerEvents: 'none',
          opacity: 0.8
        }}>
          TP {tpPrice.toLocaleString()}
        </div>
      )}

      {/* Label SL (seulement en mode normal) */}
      {!mini && slPrice && (
        <div style={{
          position: 'absolute',
          left: 2,
          top: `${getY(slPrice)}%`,
          transform: 'translateY(2px)',
          color: 'var(--red)',
          fontSize: '8px',
          fontWeight: 700,
          pointerEvents: 'none',
          opacity: 0.8
        }}>
          SL {slPrice.toLocaleString()}
        </div>
      )}
    </div>
  )
}
