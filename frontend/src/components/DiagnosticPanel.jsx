/**
 * DiagnosticPanel — Analyse intelligente des résultats WFO
 * Sprint 14c
 *
 * Produit des verdicts textuels en langage clair pour aider l'utilisateur
 * à comprendre rapidement la viabilité d'une stratégie.
 */

import { useMemo } from 'react'
import { analyzeResults, analyzeRegimes, REGIME_CONFIG } from '../utils/diagnosticUtils'
import './DiagnosticPanel.css'

// Composant icône cercle coloré
function StatusCircle({ level }) {
  const colors = { green: '#10b981', orange: '#f59e0b', red: '#ef4444' }
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" className="verdict-icon">
      <circle cx="5" cy="5" r="5" fill={colors[level]} />
    </svg>
  )
}

// Icône titre "Diagnostic"
function DiagnosticIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      {/* 3 barres verticales (graphique) */}
      <rect x="2" y="10" width="3" height="4" fill="#9ca3af" />
      <rect x="6.5" y="6" width="3" height="8" fill="#9ca3af" />
      <rect x="11" y="3" width="3" height="11" fill="#9ca3af" />
    </svg>
  )
}

/**
 * Composant DiagnosticPanel
 *
 * @param {Array} combos - Combos testées
 * @param {string} grade - Grade A-F
 * @param {number} totalScore - Score 0-100
 * @param {number} nWindows - Nombre de fenêtres WFO
 * @param {Object|null} regimeAnalysis - Analyse par régime (Sprint 15b)
 */
export default function DiagnosticPanel({ combos, grade, totalScore, nWindows, regimeAnalysis }) {
  // Guard : pas de données
  if (!combos || combos.length === 0 || !nWindows || nWindows <= 0) {
    return null
  }

  const verdicts = useMemo(() => {
    return analyzeResults(combos, grade, totalScore, nWindows)
  }, [combos, grade, totalScore, nWindows])

  const regimeVerdict = useMemo(() => {
    return analyzeRegimes(regimeAnalysis)
  }, [regimeAnalysis])

  // Déterminer la couleur de la bordure gauche (plus sévère)
  const allVerdicts = regimeVerdict ? [...verdicts, regimeVerdict] : verdicts
  const hasSevere = allVerdicts.some((v) => v.level === 'red')
  const hasWarning = allVerdicts.some((v) => v.level === 'orange')
  const borderColor = hasSevere ? '#ef4444' : hasWarning ? '#f59e0b' : '#10b981'

  return (
    <div className="diagnostic-panel" style={{ borderLeftColor: borderColor }}>
      <h4 className="diagnostic-title">
        <DiagnosticIcon />
        DIAGNOSTIC
      </h4>

      <div className="diagnostic-verdicts">
        {verdicts.map((verdict, idx) => (
          <div key={idx} className="verdict-item">
            <StatusCircle level={verdict.level} />
            <div>
              <div className="verdict-title">{verdict.title}</div>
              <div className="verdict-text">{verdict.text}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Sprint 15b : Section Analyse par Régime */}
      {regimeAnalysis && Object.keys(regimeAnalysis).length > 0 && (
        <div className="regime-section">
          <h5 className="regime-title">PERFORMANCE PAR REGIME</h5>

          <div className="regime-grid">
            {Object.entries(regimeAnalysis).map(([regime, data]) => {
              const config = REGIME_CONFIG[regime] || { color: '#6b7280', emoji: '?', label: regime }
              const sharpeColor =
                data.avg_oos_sharpe > 0.5
                  ? '#10b981'
                  : data.avg_oos_sharpe > 0
                    ? '#f59e0b'
                    : '#ef4444'

              return (
                <div key={regime} className="regime-card" style={{ borderTopColor: config.color }}>
                  <div className="regime-card-header">
                    <span className="regime-emoji">{config.emoji}</span>
                    <span className="regime-label">{config.label}</span>
                    <span className="regime-count">{data.n_windows} fen.</span>
                  </div>
                  <div className="regime-stats">
                    <div className="regime-stat">
                      <span className="regime-stat-label">Sharpe</span>
                      <span className="regime-stat-value" style={{ color: sharpeColor }}>
                        {data.avg_oos_sharpe.toFixed(2)}
                      </span>
                    </div>
                    <div className="regime-stat">
                      <span className="regime-stat-label">Consist.</span>
                      <span className="regime-stat-value">
                        {Math.round(data.consistency * 100)}%
                      </span>
                    </div>
                    <div className="regime-stat">
                      <span className="regime-stat-label">Return</span>
                      <span
                        className="regime-stat-value"
                        style={{ color: data.avg_return_pct > 0 ? '#10b981' : '#ef4444' }}
                      >
                        {data.avg_return_pct > 0 ? '+' : ''}
                        {data.avg_return_pct.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Verdict régime */}
          {regimeVerdict && (
            <div className="verdict-item regime-verdict">
              <StatusCircle level={regimeVerdict.level} />
              <div>
                <div className="verdict-title">{regimeVerdict.title}</div>
                <div className="verdict-text">{regimeVerdict.text}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
