/**
 * useRecommendations — moteur de règles déterministes pour recommandations stratégie
 * Sprint 36
 */

import { useMemo } from 'react'

const PRIORITY = { error: 0, recommended: 1, warning: 2, info: 3, success: 4 }

export function useRecommendations(summary) {
  return useMemo(() => {
    if (!summary || !summary.total_assets) return []

    const recos = []

    // --- Règles WFO ---

    if (summary.total_assets > 0 && summary.ab_count === 0) {
      recos.push({
        level: 'error',
        text: `Aucun asset Grade A ou B sur ${summary.total_assets} — Stratégie à abandonner`,
      })
    }

    if (summary.underpowered_pct >= 50) {
      recos.push({
        level: 'error',
        text: `${summary.underpowered_count}/${summary.total_assets} assets underpowered — Non viable statistiquement`,
      })
    }

    if (summary.red_flags.oos_is_ratio_suspect > summary.total_assets * 0.5) {
      recos.push({
        level: 'warning',
        text: `${summary.red_flags.oos_is_ratio_suspect}/${summary.total_assets} OOS/IS suspects — Vérifiez dans Recherche`,
        action: { label: 'Voir détails', tab: 'research' },
      })
    }

    // --- Règles WFO → Portfolio ---

    if (summary.ab_count >= 5 && summary.portfolio_runs.length === 0) {
      recos.push({
        level: 'recommended',
        text: `${summary.ab_count} assets A+B validés — Lancez un portfolio backtest`,
        action: { label: 'Lancer', tab: 'portfolio' },
      })
    }

    // --- Règles Portfolio ---

    if (summary.portfolio_runs.length > 0) {
      const hasForward = summary.portfolio_runs.some(r => r.days >= 300 && r.days <= 400)

      if (!hasForward) {
        recos.push({
          level: 'recommended',
          text: 'Backtest complet OK — Lancez le forward test 365j',
          action: { label: 'Lancer forward', tab: 'portfolio' },
        })
      }

      // Évaluer le dernier forward test
      const latestForward = summary.portfolio_runs.find(r => r.days >= 300 && r.days <= 400)
      if (latestForward) {
        if (latestForward.return_pct < 0) {
          recos.push({
            level: 'error',
            text: `Forward test négatif (${latestForward.return_pct.toFixed(1)}%) — Stratégie non viable en conditions récentes`,
            action: { label: 'Voir le run', tab: 'portfolio' },
          })
        } else if (latestForward.return_pct < 20) {
          recos.push({
            level: 'warning',
            text: `Forward 365j : +${latestForward.return_pct.toFixed(1)}% — Performance modeste`,
            action: { label: 'Comparer', tab: 'portfolio' },
          })
        } else {
          recos.push({
            level: 'success',
            text: `Forward 365j : +${latestForward.return_pct.toFixed(1)}% — Prêt pour paper trading`,
            action: { label: 'Voir le run', tab: 'portfolio' },
          })
        }
      }
    }

    // --- Règles fraîcheur ---

    if (summary.latest_wfo_date) {
      const daysSince = Math.floor(
        (Date.now() - new Date(summary.latest_wfo_date).getTime()) / 86400000
      )
      if (daysSince > 30) {
        recos.push({
          level: 'info',
          text: `WFO date de ${daysSince} jours — Relancer avec données récentes ?`,
          action: { label: 'Explorer', tab: 'explorer' },
        })
      }
    }

    // Tri par priorité
    recos.sort((a, b) => PRIORITY[a.level] - PRIORITY[b.level])

    return recos
  }, [summary])
}
