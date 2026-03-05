/**
 * GridChart — Graphique détaillé pour GridDetail.
 * Affiche la courbe de prix (sparkline) + niveaux Grid (horizontal lines).
 * 
 * Nouveau : Utilisation de createPortal pour la modale afin d'éviter les problèmes de z-index et de clipping.
 */
import { useMemo, useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { formatPrice } from '../utils/format'

export default function GridChart({ symbol, data = [], levels = [], currentPrice, tpPrice, slPrice, width = 160, height = 32, mini = false }) {
  const [isModalOpen, setIsModalOpen] = useState(false)

  // Fermer la modale avec la touche Echap
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') setIsModalOpen(false)
    }
    if (isModalOpen) window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [isModalOpen])

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
    
    // Augmenter le padding si la modale est ouverte pour éviter de couper les paliers extrêmes
    const p = isModalOpen ? 0.15 : (mini ? 0.05 : 0.1)
    return {
      min: min - range * p,
      max: max + range * p,
      range: range * (1 + p * 2)
    }
  }, [data, levels, currentPrice, tpPrice, slPrice, mini, isModalOpen])

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

  // Rendu du contenu SVG
  const renderSVG = (isModal = false) => (
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
        const opacity = isModal ? 1 : 0.8
        return (
          <g key={i}>
            <line 
              x1="0" y1={y} x2="100" y2={y} 
              stroke={color} 
              strokeWidth={lvl.filled ? (mini ? "0.4" : "0.6") : (mini ? "0.2" : "0.3")} 
              strokeDasharray={lvl.filled ? "" : "2,1"}
              opacity={opacity}
            />
            {(!mini || isModal) && (
              <g opacity={opacity}>
                <rect x="0" y={y-2} width="8" height="4" fill={color} opacity="0.2" />
                <text x="1" y={y+1} fontSize={isModal ? "1.5" : "3"} fill={color} fontWeight="bold">L{i+1}</text>
              </g>
            )}
          </g>
        )
      })}

      {slPrice && (
        <g opacity={isModal ? 1 : 0.8}>
          <line x1="0" y1={getY(slPrice)} x2="100" y2={getY(slPrice)} stroke="var(--red)" strokeWidth={mini ? "0.4" : "0.8"} strokeDasharray="2,2" />
          {(!mini || isModal) && <text x={isModal ? "95" : "88"} y={getY(slPrice)-1.5} fontSize={isModal ? "2.5" : "4"} fill="var(--red)" fontWeight="bold">SL</text>}
        </g>
      )}

      {tpPrice && (
        <g opacity={isModal ? 1 : 0.8}>
          <line x1="0" y1={getY(tpPrice)} x2="100" y2={getY(tpPrice)} stroke="var(--accent)" strokeWidth={mini ? "0.4" : "0.8"} strokeDasharray="2,2" />
          {(!mini || isModal) && <text x={isModal ? "95" : "88"} y={getY(tpPrice)-1.5} fontSize={isModal ? "2.5" : "4"} fill="var(--accent)" fontWeight="bold">TP</text>}
        </g>
      )}

      {points ? (
        <polyline
          points={points}
          fill="none"
          stroke={mini ? ((data.length > 1 && data[data.length-1] > data[0]) ? 'var(--accent)' : 'var(--red)') : 'var(--text-secondary)'}
          strokeWidth={isModal ? "0.5" : (mini ? "1.2" : "1")}
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity={isModal ? 1 : (mini ? 1 : 0.8)}
        />
      ) : !mini && (
        <text x="50" y="50" fontSize="4" fill="var(--text-dim)" textAnchor="middle">En attente de données...</text>
      )}

      {currentPrice && (
        <g opacity={isModal ? 1 : 0.8}>
          <line x1="0" y1={getY(currentPrice)} x2="100" y2={getY(currentPrice)} stroke="var(--yellow)" strokeWidth={mini ? "0.3" : "0.5"} opacity={isModal ? 1 : 0.8} />
          <circle cx="100" cy={getY(currentPrice)} r={mini ? "1.2" : (isModal ? "1" : "2")} fill="var(--yellow)" />
        </g>
      )}
    </svg>
  )

  const modal = isModalOpen && createPortal(
    <div 
      style={{
        position: 'fixed',
        top: 0, left: 0, width: '100vw', height: '100vh',
        background: 'rgba(0,0,0,0.9)', 
        zIndex: 100000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2vh 5vw'
      }}
      onClick={() => setIsModalOpen(false)}
    >
      <div 
        style={{
          width: '80vw',
          height: '85vh',
          background: '#000', // OPAQUE TOTAL NOIR
          borderRadius: 12,
          border: '2px solid var(--accent)',
          boxShadow: '0 0 60px rgba(0, 0, 0, 1)',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          padding: '24px',
          overflow: 'hidden'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header unique */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, flexShrink: 0 }}>
          <div>
            <h2 style={{ margin: 0, color: 'var(--accent)', fontSize: '26px', fontWeight: 800 }}>{symbol}</h2>
            <div className="muted" style={{ fontSize: '13px' }}>Niveaux Grid & Prix en temps réel</div>
          </div>
          <button 
            style={{ 
              background: 'var(--accent)', 
              border: 'none', 
              color: '#000', 
              padding: '8px 20px', 
              borderRadius: 6, 
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 800
            }}
            onClick={() => setIsModalOpen(false)}
          >
            FERMER [ESC]
          </button>
        </div>

        {/* Graphique unique - Remplit tout l'espace */}
        <div style={{ flex: 1, position: 'relative', width: '100%', height: '100%', background: '#000' }}>
          {renderSVG(true)}

          {/* Labels superposés au graphique */}
          {currentPrice && (
            <div style={{
              position: 'absolute',
              right: 10,
              top: `${getY(currentPrice)}%`,
              transform: 'translateY(-50%)',
              background: 'var(--yellow)',
              color: '#000',
              fontSize: '14px',
              padding: '4px 10px',
              borderRadius: 4,
              fontWeight: 900,
              boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
              zIndex: 10
            }}>
              {currentPrice.toLocaleString()}
            </div>
          )}

          {tpPrice && (
            <div style={{ position: 'absolute', left: 20, top: `${getY(tpPrice)}%`, transform: 'translateY(-110%)', color: 'var(--accent)', fontSize: '13px', fontWeight: 800 }}>
              TAKE PROFIT: {tpPrice.toLocaleString()}
            </div>
          )}

          {slPrice && (
            <div style={{ position: 'absolute', left: 20, top: `${getY(slPrice)}%`, transform: 'translateY(10%)', color: 'var(--red)', fontSize: '13px', fontWeight: 800 }}>
              STOP LOSS: {slPrice.toLocaleString()}
            </div>
          )}

          {/* Liste des niveaux en bas à gauche */}
          <div style={{ position: 'absolute', left: 20, bottom: 20, display: 'flex', flexDirection: 'column', gap: 4, background: 'rgba(0,0,0,0.4)', padding: '8px', borderRadius: 4 }}>
            {levels.map((lvl, i) => lvl.price && (
              <div key={i} style={{ fontSize: '11px', color: lvl.filled ? 'var(--accent)' : 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                 L{i+1}: {formatPrice(lvl.price)} {lvl.filled ? '(FILLED)' : ''}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>,
    document.body
  )

  return (
    <>
      <div 
        style={{ 
          width, 
          height, 
          position: 'relative', 
          background: mini ? 'transparent' : 'rgba(255,255,255,0.02)', 
          borderRadius: 4, 
          border: mini ? 'none' : '1px solid var(--border)',
          cursor: mini ? 'default' : 'pointer'
        }}
        onClick={() => !mini && setIsModalOpen(true)}
      >
        <div style={{ width: '100%', height: '100%', overflow: 'hidden' }}>
          {renderSVG(false)}
        </div>
        
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
            zIndex: 10,
            boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            {currentPrice.toLocaleString()}
          </div>
        )}

        {/* Labels des niveaux Grid sur la droite */}
        {!mini && levels.map((lvl, i) => lvl.price && (
          <div key={i} style={{
            position: 'absolute',
            right: isModalOpen ? 10 : 2,
            top: `${getY(lvl.price)}%`,
            transform: 'translateY(-50%)',
            color: lvl.filled ? 'var(--accent)' : 'var(--text-dim)',
            fontSize: isModalOpen ? '11px' : '8px',
            fontFamily: 'var(--font-mono)',
            pointerEvents: 'none',
            fontWeight: lvl.filled ? 800 : 400,
            background: 'rgba(0,0,0,0.6)',
            padding: isModalOpen ? '2px 6px' : '0 2px',
            borderRadius: 3,
            zIndex: 5,
            border: isModalOpen ? `1px solid ${lvl.filled ? 'var(--accent-dim)' : 'var(--border)'}` : 'none'
          }}>
            {isModalOpen && `L${i+1} `}{lvl.price.toLocaleString()}
          </div>
        ))}

        {/* Labels TP/SL sur la droite */}
        {!mini && tpPrice && (
          <div style={{
            position: 'absolute',
            right: isModalOpen ? 10 : 2,
            top: `${getY(tpPrice)}%`,
            transform: 'translateY(-50%)',
            color: 'var(--accent)',
            fontSize: isModalOpen ? '12px' : '8px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 900,
            pointerEvents: 'none',
            background: 'rgba(0,0,0,0.7)',
            padding: isModalOpen ? '3px 8px' : '0 2px',
            borderRadius: 3,
            zIndex: 6,
            border: isModalOpen ? '1px solid var(--accent)' : 'none'
          }}>
            TP {tpPrice.toLocaleString()}
          </div>
        )}
        {!mini && slPrice && (
          <div style={{
            position: 'absolute',
            right: isModalOpen ? 10 : 2,
            top: `${getY(slPrice)}%`,
            transform: 'translateY(-50%)',
            color: 'var(--red)',
            fontSize: isModalOpen ? '12px' : '8px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 900,
            pointerEvents: 'none',
            background: 'rgba(0,0,0,0.7)',
            padding: isModalOpen ? '3px 8px' : '0 2px',
            borderRadius: 3,
            zIndex: 6,
            border: isModalOpen ? '1px solid var(--red)' : 'none'
          }}>
            SL {slPrice.toLocaleString()}
          </div>
        )}
      </div>

      {modal}
    </>
  )
}
