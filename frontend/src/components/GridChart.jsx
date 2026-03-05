/**
 * GridChart — Graphique détaillé pour GridDetail.
 * Affiche la courbe de prix (sparkline) + niveaux Grid (horizontal lines).
 * 
 * Nouveau : Effet de zoom au survol pour voir les détails en grand.
 */
import { useMemo, useState } from 'react'

export default function GridChart({ data = [], levels = [], currentPrice, tpPrice, slPrice, width = 160, height = 32, mini = false }) {
  const [isHovered, setIsHovered] = useState(false)

  // Calculer les bornes du graphique
  const bounds = useMemo(() => {
    if (!data.length && !levels.length && !currentPrice) return { min: 0, max: 100 }
    
    let allPrices = [...data]
    if (currentPrice) allPrices.push(currentPrice)
    levels.forEach(l => { if (l.price) allPrices.push(l.price) })
    if (tpPrice) allPrices.push(tpPrice)
    if (slPrice) allPrices.push(slPrice)
    
    const min = Math.min(...allPrices)
    const max = Math.max(...allPrices)
    const range = max - min || (max * 0.01)
    
    const p = mini ? 0.05 : 0.1
    return {
      min: min - range * p,
      max: max + range * p,
      range: range * (1 + p * 2)
    }
  }, [data, levels, currentPrice, tpPrice, slPrice, mini])

  const getY = (price) => {
    if (!price || bounds.range === 0) return 0
    return ((bounds.max - price) / bounds.range) * 100
  }

  const points = useMemo(() => {
    if (!data.length) return ""
    return data.map((val, i) => {
      const x = (i / (data.length - 1)) * 100
      const y = getY(val)
      return `${x.toFixed(2)},${y.toFixed(2)}`
    }).join(' ')
  }, [data, bounds])

  // Rendu du contenu SVG (réutilisable pour le zoom)
  const renderSVG = (isZoomed = false) => (
    <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" style={{ display: 'block' }}>
      {!mini && (
        <>
          <line x1="0" y1="25" x2="100" y2="25" stroke="var(--border)" strokeWidth="0.1" strokeDasharray="1,1" />
          <line x1="0" y1="50" x2="100" y2="50" stroke="var(--border)" strokeWidth="0.1" strokeDasharray="1,1" />
          <line x1="0" y1="75" x2="100" y2="75" stroke="var(--border)" strokeWidth="0.1" strokeDasharray="1,1" />
        </>
      )}

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
            {(!mini || isZoomed) && (
              <g>
                <rect x="0" y={y-2} width="8" height="4" fill={color} opacity="0.2" />
                <text x="1" y={y+1} fontSize={isZoomed ? "2.5" : "3"} fill={color} fontWeight="bold">L{i+1}</text>
              </g>
            )}
          </g>
        )
      })}

      {slPrice && (
        <g>
          <line x1="0" y1={getY(slPrice)} x2="100" y2={getY(slPrice)} stroke="var(--red)" strokeWidth={mini ? "0.4" : "0.8"} strokeDasharray="2,2" />
          {(!mini || isZoomed) && <text x="88" y={getY(slPrice)-1.5} fontSize="4" fill="var(--red)" fontWeight="bold">SL</text>}
        </g>
      )}

      {tpPrice && (
        <g>
          <line x1="0" y1={getY(tpPrice)} x2="100" y2={getY(tpPrice)} stroke="var(--accent)" strokeWidth={mini ? "0.4" : "0.8"} strokeDasharray="2,2" />
          {(!mini || isZoomed) && <text x="88" y={getY(tpPrice)-1.5} fontSize="4" fill="var(--accent)" fontWeight="bold">TP</text>}
        </g>
      )}

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

      {currentPrice && (
        <g>
          <line x1="0" y1={getY(currentPrice)} x2="100" y2={getY(currentPrice)} stroke="var(--yellow)" strokeWidth={mini ? "0.3" : "0.5"} opacity="0.8" />
          <circle cx="100" cy={getY(currentPrice)} r={mini ? "1.2" : "2"} fill="var(--yellow)" />
        </g>
      )}
    </svg>
  )

  return (
    <div 
      style={{ 
        width, 
        height, 
        position: 'relative', 
        background: mini ? 'transparent' : 'rgba(255,255,255,0.02)', 
        borderRadius: 4, 
        border: mini ? 'none' : '1px solid var(--border)',
        cursor: mini ? 'default' : 'zoom-in'
      }}
      onMouseEnter={() => !mini && setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div style={{ width: '100%', height: '100%', overflow: 'hidden' }}>
        {renderSVG(false)}
      </div>
      
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

      {/* Overlay Grand Format au survol */}
      {isHovered && !mini && (
        <div style={{
          position: 'absolute',
          left: '50%',
          bottom: '100%',
          transform: 'translateX(-50%) translateY(-10px)',
          width: 400,
          height: 250,
          background: 'var(--bg-secondary)',
          border: '1px solid var(--accent)',
          borderRadius: 8,
          boxShadow: '0 10px 30px rgba(0,0,0,0.6)',
          zIndex: 2000,
          padding: 12,
          pointerEvents: 'none',
          animation: 'slideIn 0.15s ease-out'
        }}>
          <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            {renderSVG(true)}
            
            {/* Labels détaillés dans le zoom */}
            <div style={{ position: 'absolute', top: 0, left: 0, fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600 }}>
              {symbol} - GRID DETAIL
            </div>

            {currentPrice && (
              <div style={{
                position: 'absolute',
                right: -5,
                top: `${getY(currentPrice)}%`,
                transform: 'translateY(-50%)',
                background: 'var(--yellow)',
                color: '#000',
                fontSize: '11px',
                padding: '2px 5px',
                borderRadius: 3,
                fontWeight: 900,
                boxShadow: '0 2px 6px rgba(0,0,0,0.4)'
              }}>
                {currentPrice.toLocaleString()}
              </div>
            )}
            
            {tpPrice && (
              <div style={{ position: 'absolute', left: 0, top: `${getY(tpPrice)}%`, transform: 'translateY(-100%)', color: 'var(--accent)', fontSize: '10px', fontWeight: 700 }}>
                TP: {tpPrice.toLocaleString()}
              </div>
            )}
            {slPrice && (
              <div style={{ position: 'absolute', left: 0, top: `${getY(slPrice)}%`, transform: 'translateY(2px)', color: 'var(--red)', fontSize: '10px', fontWeight: 700 }}>
                SL: {slPrice.toLocaleString()}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
