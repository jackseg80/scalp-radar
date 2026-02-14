/**
 * DiagnosticPanel — Analyse intelligente des résultats WFO
 * Sprint 14c
 *
 * Produit des verdicts textuels en langage clair pour aider l'utilisateur
 * à comprendre rapidement la viabilité d'une stratégie.
 */

import { useMemo } from 'react'
import './DiagnosticPanel.css'

// Helper : calcul de la médiane
function median(arr) {
  if (!arr.length) return 0
  const sorted = [...arr].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

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
 * Analyse les résultats et produit un array de verdicts.
 *
 * @param {Array} combos - Combos testées
 * @param {string} grade - Grade A-F
 * @param {number} totalScore - Score 0-100
 * @param {number} nWindows - Nombre de fenêtres WFO
 * @returns {Array} - [{level, title, text}, ...]
 */
function analyzeResults(combos, grade, totalScore, nWindows) {
  const verdicts = []

  // Trouver le best combo
  const best = combos.find((c) => c.is_best) || combos[0]

  if (!best) return verdicts // Sécurité (ne devrait jamais arriver)

  // Métriques agrégées
  const allOosSharpe = combos.map((c) => c.oos_sharpe).filter((v) => v != null)
  const pctPositive = allOosSharpe.filter((v) => v > 0).length / allOosSharpe.length
  const pctAbove1 = allOosSharpe.filter((v) => v > 1).length / allOosSharpe.length
  const pctAbove05 = allOosSharpe.filter((v) => v > 0.5).length / allOosSharpe.length

  // ─── RÈGLE 1 : Grade global ────────────────────────────────────────

  if (['A', 'B'].includes(grade)) {
    verdicts.push({
      level: 'green',
      title: 'Stratégie viable',
      text: `Grade ${grade} (${totalScore}/100). Prête pour le paper trading.`,
    })
  } else if (grade === 'C') {
    verdicts.push({
      level: 'orange',
      title: 'Stratégie moyenne',
      text: `Grade ${grade} (${totalScore}/100). Déployable avec surveillance renforcée.`,
    })
  } else {
    verdicts.push({
      level: 'red',
      title: 'Stratégie non viable',
      text: `Grade ${grade} (${totalScore}/100). Non recommandée pour le déploiement.`,
    })
  }

  // ─── RÈGLE 2 : Consistance du best ────────────────────────────────

  const bestConsistency = best.consistency ?? 0
  const consistencyPct = Math.round(bestConsistency * 100)
  const consistencyWindows = Math.round(bestConsistency * nWindows)

  if (bestConsistency < 0.2) {
    verdicts.push({
      level: 'red',
      title: 'Consistance catastrophique',
      text: `Profitable dans seulement ${consistencyWindows}/${nWindows} fenêtres (${consistencyPct}%). C'est du bruit statistique.`,
    })
  } else if (bestConsistency < 0.5) {
    verdicts.push({
      level: 'red',
      title: 'Consistance faible',
      text: `Profitable dans ${consistencyWindows}/${nWindows} fenêtres (${consistencyPct}%). Instable — moins de la moitié des périodes.`,
    })
  } else if (bestConsistency < 0.8) {
    verdicts.push({
      level: 'orange',
      title: 'Consistance acceptable',
      text: `Profitable dans ${consistencyWindows}/${nWindows} fenêtres (${consistencyPct}%). Correct mais pas robuste.`,
    })
  } else {
    verdicts.push({
      level: 'green',
      title: 'Consistance excellente',
      text: `Profitable dans ${consistencyWindows}/${nWindows} fenêtres (${consistencyPct}%). Robuste sur toutes les conditions de marché.`,
    })
  }

  // ─── RÈGLE 3 : Transfert IS → OOS ────────────────────────────────

  const bestIsSharpe = best.is_sharpe ?? 0
  const bestOosSharpe = best.oos_sharpe ?? 0
  const bestRatio = best.oos_is_ratio ?? 0

  if (bestIsSharpe > 5 && bestOosSharpe < 1) {
    verdicts.push({
      level: 'red',
      title: 'Overfitting détecté',
      text: `IS Sharpe ${bestIsSharpe.toFixed(1)} mais OOS Sharpe ${bestOosSharpe.toFixed(
        1
      )}. Le modèle mémorise le passé sans généraliser.`,
    })
  } else if (bestRatio < 0.3) {
    verdicts.push({
      level: 'red',
      title: 'Dégradation sévère IS→OOS',
      text: `Ratio OOS/IS de ${bestRatio.toFixed(
        2
      )}. Moins de 30% de la performance d'entraînement se transfère.`,
    })
  } else if (bestRatio < 0.7) {
    verdicts.push({
      level: 'orange',
      title: 'Dégradation normale IS→OOS',
      text: `Ratio OOS/IS de ${bestRatio.toFixed(2)}. Dégradation typique, acceptable.`,
    })
  } else {
    verdicts.push({
      level: 'green',
      title: 'Bon transfert IS→OOS',
      text: `Ratio OOS/IS de ${bestRatio.toFixed(
        2
      )}. La performance se maintient hors échantillon.`,
    })
  }

  // ─── RÈGLE 4 : Edge structurel (distribution) ────────────────────

  if (pctAbove1 > 0.5) {
    verdicts.push({
      level: 'green',
      title: 'Edge structurel fort',
      text: `${Math.round(
        pctAbove1 * 100
      )}% des combos ont un OOS Sharpe > 1. L'edge ne dépend pas du choix de paramètres.`,
    })
  } else if (pctAbove05 > 0.5) {
    verdicts.push({
      level: 'orange',
      title: 'Edge modéré',
      text: `${Math.round(
        pctAbove05 * 100
      )}% des combos ont un OOS Sharpe > 0.5. L'edge existe mais est sensible aux paramètres.`,
    })
  } else if (pctPositive > 0.5) {
    verdicts.push({
      level: 'orange',
      title: 'Edge faible',
      text: `${Math.round(
        pctPositive * 100
      )}% des combos sont positives mais avec des Sharpe faibles. L'edge est fragile.`,
    })
  } else {
    verdicts.push({
      level: 'red',
      title: "Pas d'edge structurel",
      text: `Seulement ${Math.round(
        pctPositive * 100
      )}% des combos sont positives. La majorité perd de l'argent.`,
    })
  }

  // ─── RÈGLE 5 : Volume de trades ──────────────────────────────────

  const bestOosTrades = best.oos_trades ?? 0
  if (bestOosTrades < 30) {
    verdicts.push({
      level: 'orange',
      title: 'Données insuffisantes',
      text: `Le best combo n'a que ${bestOosTrades} trades OOS. Minimum 30 pour une signification statistique.`,
    })
  }

  // ─── RÈGLE 6 : Fenêtres partielles ──────────────────────────────

  const partialCombos = combos.filter((c) => (c.n_windows_evaluated ?? 0) < nWindows)
  if (partialCombos.length > combos.length * 0.3) {
    verdicts.push({
      level: 'orange',
      title: 'Combos partielles',
      text: `${partialCombos.length}/${combos.length} combos évaluées sur moins de ${nWindows} fenêtres (fine grid). Leurs stats sont moins fiables.`,
    })
  }

  return verdicts
}

// Couleurs et labels par régime (clés minuscules = format backend)
const REGIME_CONFIG = {
  bull: { color: '#10b981', emoji: '\u25b2', label: 'Bull' },
  bear: { color: '#ef4444', emoji: '\u25bc', label: 'Bear' },
  range: { color: '#f59e0b', emoji: '\u25c6', label: 'Range' },
  crash: { color: '#dc2626', emoji: '\u26a0', label: 'Crash' },
}

/**
 * Produit un verdict de conclusion basé sur l'analyse par régime.
 */
function analyzeRegimes(regimeAnalysis) {
  if (!regimeAnalysis) return null

  const regimes = Object.keys(regimeAnalysis)
  if (regimes.length === 0) return null

  // Identifier les régimes faibles (Sharpe < 0 ou consistance < 0.3)
  const weakRegimes = regimes.filter((r) => {
    const d = regimeAnalysis[r]
    return d.avg_oos_sharpe < 0 || d.consistency < 0.3
  })

  // Identifier les régimes forts (Sharpe > 0.5 et consistance > 0.5)
  const strongRegimes = regimes.filter((r) => {
    const d = regimeAnalysis[r]
    return d.avg_oos_sharpe > 0.5 && d.consistency > 0.5
  })

  if (weakRegimes.length === 0 && strongRegimes.length === regimes.length) {
    return {
      level: 'green',
      title: 'Robuste tous régimes',
      text: `Performante dans tous les régimes de marché testés (${regimes.join(', ')}).`,
    }
  }

  if (weakRegimes.length > 0) {
    const weakNames = weakRegimes.join(', ')
    if (weakRegimes.includes('crash')) {
      return {
        level: 'red',
        title: 'Vulnérable aux crashs',
        text: `La stratégie sous-performe en régime ${weakNames}. Risque élevé en conditions extrêmes.`,
      }
    }
    return {
      level: 'orange',
      title: 'Dépendante du régime',
      text: `Faible en régime ${weakNames}. Performances inégales selon les conditions de marché.`,
    }
  }

  return {
    level: 'orange',
    title: 'Performance mixte',
    text: `Résultats variables selon le régime. Aucun point faible critique identifié.`,
  }
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
