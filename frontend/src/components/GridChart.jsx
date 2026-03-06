/**
 * GridChart — Graphique détaillé pour GridDetail.
 * Affiche la courbe de prix (sparkline) + niveaux Grid (horizontal lines).
 * 
 * Nouveau : Utilisation de createPortal pour la modale afin d'éviter les problèmes de z-index et de clipping.
 */
import { useMemo, useState, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { formatPrice } from '../utils/format'

export default function GridChart({ symbol, data = [], levels = [], currentPrice, tpPrice, slPrice, width = 160, height = 32, mini = false, status = null }) {
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
          <line x1="0" y1="25" x2="100" y2="25" stroke="var(--border)" strokeWidth="0.05" strokeDasharray="1,1" opacity="0.5" />
          <line x1="0" y1="50" x2="100" y2="50" stroke="var(--border)" strokeWidth="0.05" strokeDasharray="1,1" opacity="0.5" />
          <line x1="0" y1="75" x2="100" y2="75" stroke="var(--border)" strokeWidth="0.05" strokeDasharray="1,1" opacity="0.5" />
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
              strokeWidth={lvl.filled ? (mini ? "0.4" : "0.4") : (mini ? "0.2" : "0.2")} 
              strokeDasharray={lvl.filled ? "" : "2,1"}
              opacity={opacity}
            />
          </g>
        )
      })}

      {slPrice && (
        <g opacity={isModal ? 1 : 0.8}>
          <line x1="0" y1={getY(slPrice)} x2="100" y2={getY(slPrice)} stroke="var(--red)" strokeWidth={mini ? "0.4" : "0.6"} strokeDasharray="2,2" />
        </g>
      )}

      {tpPrice && (
        <g opacity={isModal ? 1 : 0.8}>
          <line x1="0" y1={getY(tpPrice)} x2="100" y2={getY(tpPrice)} stroke="var(--accent)" strokeWidth={mini ? "0.4" : "0.6"} strokeDasharray="2,2" />
        </g>
      )}

      {points ? (
        <polyline
          points={points}
          fill="none"
          stroke={(data.length > 1 && data[data.length-1] > data[0]) ? 'var(--accent)' : 'var(--red)'}
          strokeWidth={isModal ? "0.4" : (mini ? "1.2" : "0.8")}
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity={isModal ? 1 : (mini ? 1 : 0.6)}
        />
      ) : !mini && (
        <text x="50" y="50" fontSize="4" fill="var(--text-dim)" textAnchor="middle">En attente de données...</text>
      )}

      {currentPrice && (
        <g opacity={isModal ? 1 : 0.8}>
          <line x1="0" y1={getY(currentPrice)} x2="100" y2={getY(currentPrice)} stroke="var(--yellow)" strokeWidth={mini ? "0.3" : "0.4"} opacity={isModal ? 1 : 0.8} />
          <circle cx="100" cy={getY(currentPrice)} r={mini ? "1.2" : (isModal ? "0.6" : "1.5")} fill="var(--yellow)" />
        </g>
      )}
    </svg>
  )

  const modal = isModalOpen && createPortal(
    <div 
      style={{
        position: 'fixed',
        top: 0, left: 0, width: '100vw', height: '100vh',
        background: 'rgba(0,0,0,0.92)', 
        zIndex: 100000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2vh 2vw'
      }}
      onClick={() => setIsModalOpen(false)}
    >
      <div 
        style={{
          width: '95vw',
          height: '90vh',
          background: '#0a0a0a', 
          borderRadius: 8,
          border: '1px solid var(--border)',
          boxShadow: '0 0 60px rgba(0, 0, 0, 1)',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          padding: '20px',
          overflow: 'hidden'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header unique */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15, flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 15 }}>
            <h2 style={{ margin: 0, color: 'var(--accent)', fontSize: '28px', fontWeight: 900, letterSpacing: '-0.5px' }}>{symbol}</h2>
            <div style={{ height: 24, width: 1, background: 'var(--border)' }} />
            <div className="mono" style={{ fontSize: '18px', color: 'var(--yellow)', fontWeight: 700 }}>
              {formatPrice(currentPrice)}
            </div>
            <div className="muted" style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: 1 }}>
              Chart Expert • {levels.length} Niveaux
            </div>
          </div>
          <button 
            style={{ 
              background: 'transparent', 
              border: '1px solid var(--border)', 
              color: 'var(--text-dim)', 
              padding: '6px 16px', 
              borderRadius: 4, 
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 600
            }}
            onClick={() => setIsModalOpen(false)}
          >
            FERMER [ESC]
          </button>
        </div>

        {/* Graphique unique - Remplit tout l'espace */}
        <div style={{ flex: 1, position: 'relative', width: '100%', height: '100%', background: '#050505', border: '1px solid #1a1a1a', borderRadius: 4 }}>
          {/* Badge de statut dans la modale */}
          {status && (
            <div style={{
              position: 'absolute',
              top: 20,
              right: 80, 
              zIndex: 20,
              background: status.color || 'var(--accent)',
              color: '#000',
              padding: '6px 14px',
              borderRadius: 4,
              fontSize: '14px',
              fontWeight: 900,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              boxShadow: '0 4px 20px rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.2)'
            }}>
              <span style={{ fontSize: '18px' }}>{status.icon}</span>
              <span>{status.label.toUpperCase()}</span>
            </div>
          )}

          {/* Grille de prix (Axe Y à droite) */}
          <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: 60, borderLeft: '1px solid #1a1a1a', zIndex: 1, pointerEvents: 'none' }}>
            {[0, 25, 50, 75, 100].map(p => {
              const priceAtY = bounds.max - (bounds.range * p / 100)
              return (
                <div key={p} style={{
                  position: 'absolute',
                  top: `${p}%`,
                  right: 5,
                  transform: 'translateY(-50%)',
                  fontSize: '10px',
                  color: 'var(--text-dim)',
                  fontFamily: 'var(--font-mono)'
                }}>
                  {formatPrice(priceAtY)}
                </div>
              )
            })}
          </div>

          <div style={{ position: 'absolute', inset: '0 60px 0 0' }}>
            {renderSVG(true)}

            {/* Labels à GAUCHE alignés sur les lignes */}
            {levels.map((lvl, i) => lvl.price && (
              <div key={i} style={{
                position: 'absolute',
                left: 0,
                top: `${getY(lvl.price)}%`,
                transform: 'translateY(-50%)',
                zIndex: 10,
                display: 'flex',
                alignItems: 'center'
              }}>
                <div style={{
                  background: lvl.filled ? 'var(--accent)' : '#1a1a1a',
                  color: lvl.filled ? '#000' : 'var(--text-dim)',
                  fontSize: '12px',
                  fontWeight: 900,
                  padding: '4px 8px',
                  borderRadius: '0 4px 4px 0',
                  boxShadow: '4px 0 10px rgba(0,0,0,0.5)',
                  minWidth: 100,
                  border: lvl.filled ? 'none' : '1px solid #333',
                  borderLeft: 'none',
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: 10
                }}>
                  <span>L{i+1}</span>
                  <span className="mono">{formatPrice(lvl.price)}</span>
                </div>
                {lvl.filled && <div style={{ width: 10, height: 1, background: 'var(--accent)' }} />}
              </div>
            ))}

            {tpPrice && (
              <div style={{
                position: 'absolute',
                left: 0,
                top: `${getY(tpPrice)}%`,
                transform: 'translateY(-50%)',
                zIndex: 11
              }}>
                <div style={{
                  background: 'var(--accent)',
                  color: '#000',
                  fontSize: '12px',
                  fontWeight: 900,
                  padding: '4px 8px',
                  borderRadius: '0 4px 4px 0',
                  boxShadow: '4px 0 10px rgba(0,0,0,0.5)',
                  minWidth: 120,
                  display: 'flex',
                  justifyContent: 'space-between'
                }}>
                  <span>TAKE PROFIT</span>
                  <span className="mono">{formatPrice(tpPrice)}</span>
                </div>
              </div>
            )}

            {slPrice && (
              <div style={{
                position: 'absolute',
                left: 0,
                top: `${getY(slPrice)}%`,
                transform: 'translateY(-50%)',
                zIndex: 11
              }}>
                <div style={{
                  background: 'var(--red)',
                  color: '#fff',
                  fontSize: '12px',
                  fontWeight: 900,
                  padding: '4px 8px',
                  borderRadius: '0 4px 4px 0',
                  boxShadow: '4px 0 10px rgba(0,0,0,0.5)',
                  minWidth: 120,
                  display: 'flex',
                  justifyContent: 'space-between'
                }}>
                  <span>STOP LOSS</span>
                  <span className="mono">{formatPrice(slPrice)}</span>
                </div>
              </div>
            )}

            {currentPrice && (
              <div style={{
                position: 'absolute',
                left: 0,
                top: `${getY(currentPrice)}%`,
                transform: 'translateY(-50%)',
                zIndex: 12
              }}>
                <div style={{
                  background: 'var(--yellow)',
                  color: '#000',
                  fontSize: '13px',
                  fontWeight: 900,
                  padding: '5px 10px',
                  borderRadius: '0 6px 6px 0',
                  boxShadow: '4px 0 15px rgba(0,0,0,0.6)',
                  minWidth: 130,
                  display: 'flex',
                  justifyContent: 'space-between',
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderLeft: 'none'
                }}>
                  <span>PRIX</span>
                  <span className="mono">{formatPrice(currentPrice)}</span>
                </div>
              </div>
            )}
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

        {/* Badge status en mode mini */}
        {!mini && status && (
          <div style={{
            position: 'absolute',
            top: 4,
            left: 4,
            background: status.color || 'var(--accent)',
            color: '#000',
            fontSize: '9px',
            fontWeight: 900,
            padding: '2px 6px',
            borderRadius: 3,
            zIndex: 20,
            boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            gap: 4
          }}>
            <span>{status.icon}</span>
            <span style={{ letterSpacing: 0.5 }}>{status.label.split(':')[0]}</span>
          </div>
        )}
        
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
            {formatPrice(currentPrice)}
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
            {isModalOpen && `L${i+1} `}{formatPrice(lvl.price)}
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
            TP {formatPrice(tpPrice)}
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
            SL {formatPrice(slPrice)}
          </div>
        )}
      </div>

      {modal}
    </>
  )
}
