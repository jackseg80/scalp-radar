/**
 * diagnosticUtils.js — Fonctions d'analyse WFO partagées
 * Extraites de DiagnosticPanel (Sprint 14c) pour réutilisation dans ExportDiagnostic (Sprint 16a)
 */

// Helper : calcul de la médiane
export function median(arr) {
  if (!arr.length) return 0
  const sorted = [...arr].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

// Couleurs et labels par régime (clés minuscules = format backend)
export const REGIME_CONFIG = {
  bull: { color: '#10b981', emoji: '\u25b2', label: 'Bull' },
  bear: { color: '#ef4444', emoji: '\u25bc', label: 'Bear' },
  range: { color: '#f59e0b', emoji: '\u25c6', label: 'Range' },
  crash: { color: '#dc2626', emoji: '\u26a0', label: 'Crash' },
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
export function analyzeResults(combos, grade, totalScore, nWindows) {
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
  } else if (bestRatio < 0.5) {
    verdicts.push({
      level: 'red',
      title: 'Dégradation forte IS→OOS',
      text: `Ratio OOS/IS de ${bestRatio.toFixed(
        2
      )}. Overfitting probable — moins de 50% de la performance se transfère.`,
    })
  } else if (bestRatio < 0.7) {
    verdicts.push({
      level: 'orange',
      title: 'Dégradation modérée IS→OOS',
      text: `Ratio OOS/IS de ${bestRatio.toFixed(2)}. Possible léger overfitting.`,
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

/**
 * Produit un verdict de conclusion basé sur l'analyse par régime.
 */
export function analyzeRegimes(regimeAnalysis) {
  if (!regimeAnalysis) return null

  const regimes = Object.keys(regimeAnalysis)
  if (regimes.length === 0) return null

  const ALL_REGIMES = ['crash', 'bull', 'range', 'bear']
  const nTested = regimes.length
  const missing = ALL_REGIMES.filter((r) => !regimes.includes(r))

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

  // Régimes avec Sharpe négatif — prioritaire
  const negSharpe = regimes.filter((r) => regimeAnalysis[r].avg_oos_sharpe < 0)
  if (negSharpe.length > 0) {
    const names = negSharpe.join(', ')
    if (negSharpe.includes('crash')) {
      return {
        level: 'red',
        title: 'Vulnérable aux crashs',
        text: `La stratégie sous-performe en régime ${names}. Risque élevé en conditions extrêmes.`,
      }
    }
    return {
      level: 'red',
      title: `Faible en ${names}`,
      text: `Sharpe négatif en régime ${names}. La stratégie sous-performe dans ${negSharpe.length === 1 ? 'ce régime' : 'ces régimes'}.`,
    }
  }

  // 4/4 régimes testés + tous forts
  if (nTested === 4 && weakRegimes.length === 0 && strongRegimes.length === regimes.length) {
    return {
      level: 'green',
      title: 'Robuste tous régimes',
      text: `Performante dans les 4 régimes de marché testés (${regimes.join(', ')}).`,
    }
  }

  // 3/4 régimes testés + tous positifs
  if (nTested === 3 && weakRegimes.length === 0) {
    return {
      level: 'green',
      title: 'Robuste (3/4 régimes)',
      text: `Performante dans les 3 régimes testés. ${missing.join(', ')} non couvert dans les données.`,
    }
  }

  // Couverture partielle (1/4 ou 2/4)
  if (nTested <= 2 && weakRegimes.length === 0) {
    return {
      level: 'orange',
      title: 'Couverture partielle',
      text: `Testé sur ${nTested}/4 régimes seulement (${missing.join(', ')} non couverts). Résultats non validés en conditions de ${missing.join(', ')}.`,
    }
  }

  // Faibles (consistance < 0.3 mais Sharpe >= 0)
  if (weakRegimes.length > 0) {
    const weakNames = weakRegimes.join(', ')
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
