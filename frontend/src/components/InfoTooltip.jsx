/**
 * InfoTooltip — Tooltip d'aide pour les termes techniques
 * Sprint 14b Bloc G
 */

import { useState, useEffect, useRef } from 'react'
import './InfoTooltip.css'

// Glossaire complet des termes
const GLOSSARY = {
  oos_sharpe: {
    title: 'OOS Sharpe Ratio',
    description: 'Ratio de Sharpe mesuré sur les périodes Out-of-Sample (OOS) uniquement.',
    interpretation:
      'Plus le Sharpe OOS est élevé, meilleure est la performance ajustée au risque en données inconnues. Un Sharpe > 1 est bon, > 2 est excellent.',
  },
  is_sharpe: {
    title: 'IS Sharpe Ratio',
    description: "Ratio de Sharpe mesuré sur les périodes In-Sample (IS) d'optimisation.",
    interpretation:
      "L'IS Sharpe est souvent plus élevé que l'OOS car les paramètres sont optimisés dessus. Sert de référence de performance maximale théorique.",
  },
  oos_is_ratio: {
    title: 'Ratio OOS/IS',
    description:
      "Rapport entre le Sharpe OOS et le Sharpe IS. Mesure la dégradation de performance sur données inconnues.",
    interpretation:
      "Plus proche de 1 = meilleure robustesse. Un ratio > 0.8 indique peu de sur-optimisation. Un ratio < 0.5 signale un overfitting sévère.",
  },
  consistency: {
    title: 'Consistance',
    description:
      'Pourcentage de fenêtres WFO où le Sharpe OOS est positif.',
    interpretation:
      'Mesure la fiabilité de la stratégie. Une consistance > 70% indique que la stratégie performe régulièrement, même en conditions variables.',
  },
  dsr: {
    title: 'DSR (Degradation-Stability Ratio)',
    description:
      "Score combinant la dégradation IS→OOS et la stabilité des performances OOS. Métrique d'overfitting développée par Harvey & Liu (2015).",
    interpretation:
      'DSR > 0.85 = faible overfitting (Grade A/B). DSR < 0.5 = overfitting sévère (Grade F). Pénalise les stratégies fragiles.',
  },
  monte_carlo_pvalue: {
    title: 'Monte Carlo p-value',
    description:
      'Probabilité que le Sharpe observé soit dû au hasard, calculée via 1000 simulations aléatoires (permutations de trades).',
    interpretation:
      'p-value < 0.05 = résultat statistiquement significatif (Grade A/B). p-value > 0.1 = résultat probablement aléatoire (Grade D/F).',
  },
  param_stability: {
    title: 'Stabilité Paramétrique',
    description:
      'Écart-type normalisé des paramètres optimaux à travers les fenêtres WFO. Mesure la convergence.',
    interpretation:
      'Stabilité > 0.8 = paramètres convergents (même zone optimale). Stabilité < 0.5 = paramètres erratiques → stratégie non robuste.',
  },
  grade: {
    title: 'Grade (A-F)',
    description:
      "Note finale basée sur 10 critères pondérés : Sharpe OOS, consistance, ratio OOS/IS, DSR, stabilité, Monte Carlo, validation Bitget, etc.",
    interpretation:
      'Grade A (score ≥ 85) = production immédiate. Grade B (≥ 70) = paper trading. Grade C (≥ 55) = surveillance. Grade D/F = rejet.',
  },
  total_score: {
    title: 'Score Total',
    description:
      'Somme pondérée des 10 critères de notation (0-100). Détermine le grade A-F.',
    interpretation:
      "Plus le score est élevé, meilleure est la stratégie sur l'ensemble des critères de robustesse, performance et validation.",
  },
  ci_sharpe: {
    title: 'Intervalle de Confiance Sharpe',
    description:
      "Plage [min, max] dans laquelle se situe le vrai Sharpe avec 95% de certitude (basé sur l'écart-type des rendements).",
    interpretation:
      'Un IC large indique une grande incertitude. Un IC étroit signale une performance stable. Utilisé pour comparer Bitget vs Binance.',
  },
  transfer_ratio: {
    title: 'Ratio de Transfert',
    description:
      "Rapport Sharpe Bitget / Sharpe Binance (OOS moyen). Mesure si la stratégie performe de manière cohérente sur 2 exchanges.",
    interpretation:
      "Ratio proche de 1 = excellente transférabilité. Ratio < 0.6 ou > 1.4 = dépendance à l'exchange (risque de surfit micro-structure).",
  },
  wfo: {
    title: 'Walk-Forward Optimization',
    description:
      "Méthode d'optimisation robuste : divise l'historique en fenêtres glissantes IS (120j) + OOS (30j). Réoptimise à chaque fenêtre.",
    interpretation:
      "Simule la vraie utilisation d'une stratégie : optimisation périodique + validation sur données futures. Gold standard anti-overfitting.",
  },
  is_vs_oos_chart: {
    title: 'Equity Curve IS vs OOS',
    description:
      'Graphique montrant les performances In-Sample (bleu) et Out-of-Sample (orange) fenêtre par fenêtre.',
    interpretation:
      "Courbes parallèles = bonne robustesse. Divergence croissante = dégradation OOS (overfitting). Pentes similaires = transfert réussi.",
  },
  oos_return_pct: {
    title: 'OOS Return %',
    description:
      'Rendement total cumulé sur les périodes Out-of-Sample, exprimé en pourcentage.',
    interpretation:
      'Mesure brute de profit. À combiner avec le Sharpe (ajusté au risque) et la consistance (régularité).',
  },
}

export default function InfoTooltip({ term }) {
  const [isOpen, setIsOpen] = useState(false)
  const tooltipRef = useRef(null)
  const iconRef = useRef(null)

  const glossaryEntry = GLOSSARY[term]

  // Fermer le tooltip si clic ailleurs
  useEffect(() => {
    if (!isOpen) return

    const handleClickOutside = (event) => {
      if (
        tooltipRef.current &&
        !tooltipRef.current.contains(event.target) &&
        iconRef.current &&
        !iconRef.current.contains(event.target)
      ) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])

  if (!glossaryEntry) {
    return null
  }

  return (
    <span className="info-tooltip-container">
      {/* Icône (i) */}
      <span
        ref={iconRef}
        className="info-icon"
        onClick={() => setIsOpen(!isOpen)}
        title="Cliquer pour plus d'infos"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <circle cx="7" cy="7" r="6.5" stroke="#666" strokeWidth="1" />
          <text
            x="7"
            y="10.5"
            textAnchor="middle"
            fontSize="10"
            fontWeight="600"
            fill="#666"
          >
            i
          </text>
        </svg>
      </span>

      {/* Popover */}
      {isOpen && (
        <div ref={tooltipRef} className="info-tooltip-popover">
          <div className="tooltip-title">{glossaryEntry.title}</div>
          <div className="tooltip-desc">{glossaryEntry.description}</div>
          <div className="tooltip-interp">
            <strong>Interprétation :</strong> {glossaryEntry.interpretation}
          </div>
        </div>
      )}
    </span>
  )
}
