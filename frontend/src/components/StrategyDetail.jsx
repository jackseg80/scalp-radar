/**
 * StrategyDetail — Vue détail d'une stratégie avec fiche + guide
 * Sprint Strategy Lab
 */

import { Suspense, lazy, useMemo } from 'react'
import { STRATEGIES } from '../data/strategyMeta'
import GenericGuide from './guides/GenericGuide'
import './StrategyDetail.css'

const GRADE_COLORS = {
  A: '#10b981',
  B: '#3b82f6',
  C: '#f59e0b',
  D: '#f97316',
  F: '#ef4444',
}

// Lazy-load des guides interactifs (Recharts = lourd)
const guideComponents = {
  grid_atr: lazy(() => import('./guides/GridAtrGuide')),
  grid_boltrend: lazy(() => import('./guides/GridBolTrendGuide')),
}

export default function StrategyDetail({ strategyId, onBack, onNavigate, setEvalStrategy }) {
  const strategy = useMemo(
    () => STRATEGIES.find(s => s.id === strategyId),
    [strategyId]
  )

  if (!strategy) {
    return (
      <div>
        <button className="sd-back" onClick={onBack}>← Retour</button>
        <p style={{ color: 'var(--text-muted)' }}>Stratégie introuvable : {strategyId}</p>
      </div>
    )
  }

  const handleNavigate = (tab) => {
    if (setEvalStrategy) {
      setEvalStrategy(strategy.id)
      window.dispatchEvent(new CustomEvent('eval-strategy-change', { detail: strategy.id }))
    }
    if (onNavigate) onNavigate(tab)
  }

  const GuideComponent = guideComponents[strategy.id]

  return (
    <div>
      <button className="sd-back" onClick={onBack}>← Retour au catalogue</button>

      {/* Header */}
      <div className="sd-header">
        <div className="sd-header-left">
          <h2 className="sd-title">{strategy.name}</h2>
          <StatusBadgeLarge status={strategy.status} />
        </div>
        {strategy.wfoGrade && (
          <span
            className="grade-badge"
            style={{ background: GRADE_COLORS[strategy.wfoGrade] || '#444' }}
          >
            Grade {strategy.wfoGrade}
          </span>
        )}
      </div>

      {/* Fiche résumé */}
      <div className="sd-section">
        <h3>Résumé</h3>
        <div className="sd-summary-grid">
          <div className="sd-summary-item">
            <span className="sd-summary-label">Type</span>
            <span className="sd-summary-value">{strategy.type}</span>
          </div>
          <div className="sd-summary-item">
            <span className="sd-summary-label">Direction</span>
            <span className="sd-summary-value">{strategy.direction}</span>
          </div>
          <div className="sd-summary-item">
            <span className="sd-summary-label">Timeframe</span>
            <span className="sd-summary-value">{strategy.timeframe}</span>
          </div>
          <div className="sd-summary-item">
            <span className="sd-summary-label">Famille</span>
            <span className="sd-summary-value">{strategy.family}</span>
          </div>
        </div>

        <p className="sd-edge">{strategy.edge}</p>

        <div className="sd-pros-cons">
          <div>
            <ul className="sd-list strengths">
              {strategy.strengths.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
          <div>
            <ul className="sd-list weaknesses">
              {strategy.weaknesses.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Paramètres clés */}
      {strategy.keyParams.length > 0 && (
        <div className="sd-section">
          <h3>Paramètres clés</h3>
          <table className="sd-params-table">
            <thead>
              <tr>
                <th>Paramètre</th>
                <th>Défaut</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {strategy.keyParams.map(p => (
                <tr key={p.name}>
                  <td>{p.name}</td>
                  <td>{p.default}</td>
                  <td>{p.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Logique entrée/sortie */}
      <div className="sd-section">
        <h3>Logique de trading</h3>
        <div className="sd-logic">
          <div className="sd-logic-item">
            <span className="sd-logic-label entry">Entrée</span>
            <div className="sd-logic-text">{strategy.entryLogic}</div>
          </div>
          <div className="sd-logic-item">
            <span className="sd-logic-label exit">Sortie</span>
            <div className="sd-logic-text">{strategy.exitLogic}</div>
          </div>
        </div>
      </div>

      {/* Guide interactif ou générique */}
      <div className="sd-section">
        <h3>{GuideComponent ? 'Tutoriel interactif' : 'Illustration'}</h3>
        {GuideComponent ? (
          <Suspense fallback={<div className="sd-guide-loading">Chargement du tutoriel...</div>}>
            <GuideComponent />
          </Suspense>
        ) : (
          <GenericGuide strategy={strategy} />
        )}
      </div>

      {/* Aller plus loin */}
      <div className="sd-section">
        <h3>Aller plus loin</h3>
        <div className="sd-links">
          <button className="sd-link-btn" onClick={() => handleNavigate('research')}>
            <span className="sd-link-icon">📊</span>
            <span className="sd-link-text">Voir les résultats WFO de {strategy.name}</span>
            <span className="sd-link-arrow">→</span>
          </button>
          <button className="sd-link-btn" onClick={() => handleNavigate('explorer')}>
            <span className="sd-link-icon">🔬</span>
            <span className="sd-link-text">Explorer les paramètres</span>
            <span className="sd-link-arrow">→</span>
          </button>
          <button className="sd-link-btn" onClick={() => handleNavigate('portfolio')}>
            <span className="sd-link-icon">💼</span>
            <span className="sd-link-text">Backtests portfolio</span>
            <span className="sd-link-arrow">→</span>
          </button>
        </div>
      </div>
    </div>
  )
}

function StatusBadgeLarge({ status }) {
  const config = {
    live:     { label: 'LIVE',      className: 'badge-active' },
    paper:    { label: 'PAPER',     className: 'badge-simulation' },
    disabled: { label: 'Désactivé', className: 'badge-stopped' },
    replaced: { label: 'Remplacé',  className: 'badge-stopped' },
  }
  const c = config[status] || config.disabled
  return <span className={`badge ${c.className}`}>{c.label}</span>
}
