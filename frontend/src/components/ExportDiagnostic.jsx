/**
 * ExportDiagnostic — Bouton "Copier le diagnostic" pour l'Explorer
 * Sprint 16a
 *
 * Construit un résumé textuel structuré d'un run WFO et le copie dans le presse-papier.
 */

import { useState, useCallback } from 'react'
import { analyzeResults, analyzeRegimes, median, REGIME_CONFIG } from '../utils/diagnosticUtils'

const LEVEL_EMOJI = { green: '\uD83D\uDFE2', orange: '\uD83D\uDFE0', red: '\uD83D\uDD34' }

/**
 * Construit le texte diagnostic complet pour un run WFO.
 */
export function buildDiagnosticText({ strategy, asset, selectedRun, combos, regimeAnalysis }) {
  const lines = []
  const best = combos.find((c) => c.is_best) || combos[0]
  if (!best || !selectedRun) return ''

  const nWindows = Math.max(...combos.map((c) => c.n_windows_evaluated || 0))
  const grade = selectedRun.grade || '?'
  const totalScore = selectedRun.total_score || 0
  const date = new Date(selectedRun.created_at)
  const dateStr = date.toLocaleDateString('fr-FR') + ' ' + date.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })

  // ═══ HEADER ═══
  lines.push('\u2550\u2550\u2550 WFO REPORT \u2550\u2550\u2550')
  lines.push(`Stratégie : ${strategy}`)
  lines.push(`Asset : ${asset}`)
  lines.push(`Date : ${dateStr}`)
  lines.push(`Grade : ${grade} (${Math.round(totalScore)}/100)`)
  lines.push('')

  // ─── PARAMÈTRES OPTIMAUX ───
  lines.push('\u2500\u2500\u2500 PARAMÈTRES OPTIMAUX \u2500\u2500\u2500')
  const params = selectedRun.recommended_params || best.params || {}
  for (const [key, value] of Object.entries(params)) {
    lines.push(`${key}: ${value}`)
  }
  lines.push('')

  // ─── MÉTRIQUES ───
  lines.push('\u2500\u2500\u2500 MÉTRIQUES \u2500\u2500\u2500')
  lines.push(`OOS Sharpe (best combo) : ${(best.oos_sharpe ?? 0).toFixed(2)}`)
  lines.push(`IS Sharpe (best combo) : ${(best.is_sharpe ?? 0).toFixed(2)}`)
  const consistPct = Math.round((best.consistency ?? 0) * 100)
  const consistWin = Math.round((best.consistency ?? 0) * nWindows)
  lines.push(`Consistance : ${consistPct}% (${consistWin}/${nWindows} fenêtres)`)
  lines.push(`Trades OOS : ${best.oos_trades ?? 0}`)
  lines.push(`Ratio OOS/IS : ${(best.oos_is_ratio ?? 0).toFixed(2)}`)
  lines.push('')

  // ─── DIAGNOSTIC ───
  lines.push('\u2500\u2500\u2500 DIAGNOSTIC \u2500\u2500\u2500')
  const verdicts = analyzeResults(combos, grade, totalScore, nWindows)
  for (const v of verdicts) {
    const emoji = LEVEL_EMOJI[v.level] || '\u2B55'
    lines.push(`- ${emoji} ${v.title} \u2014 ${v.text}`)
  }
  lines.push('')

  // ─── RÉGIMES DE MARCHÉ ───
  if (regimeAnalysis && Object.keys(regimeAnalysis).length > 0) {
    lines.push('\u2500\u2500\u2500 RÉGIMES DE MARCHÉ \u2500\u2500\u2500')
    for (const [regime, data] of Object.entries(regimeAnalysis)) {
      const config = REGIME_CONFIG[regime] || { label: regime }
      const label = config.label.padEnd(5)
      const sharpe = data.avg_oos_sharpe.toFixed(2)
      const consist = Math.round(data.consistency * 100)
      const ret = `${data.avg_return_pct > 0 ? '+' : ''}${data.avg_return_pct.toFixed(1)}%`
      lines.push(`${label} : Sharpe ${sharpe} | Consist. ${consist}% | Return ${ret} (${data.n_windows} fen.)`)
    }
    const regimeVerdict = analyzeRegimes(regimeAnalysis)
    if (regimeVerdict) {
      const emoji = LEVEL_EMOJI[regimeVerdict.level] || '\u2B55'
      lines.push(`\u2192 ${emoji} ${regimeVerdict.title} \u2014 ${regimeVerdict.text}`)
    }
    lines.push('')
  }

  // ─── TOP 5 COMBOS (score composite) ───
  lines.push('\u2500\u2500\u2500 TOP 5 COMBOS (score composite) \u2500\u2500\u2500')
  // Même formule que backend combo_score(): sharpe × (0.4 + 0.6 × consistency) × min(1, trades/100) × window_factor
  const maxWin = Math.max(...combos.map(c => c.n_windows_evaluated ?? 1), 1)
  const comboScore = (c) => {
    const s = Math.max(c.oos_sharpe ?? 0, 0)
    const cons = c.consistency ?? 0
    const trades = c.oos_trades ?? 0
    const nWin = c.n_windows_evaluated ?? maxWin
    const windowFactor = Math.min(1, nWin / maxWin)
    return s * (0.4 + 0.6 * cons) * Math.min(1, trades / 100) * windowFactor
  }
  const sorted = [...combos].sort((a, b) => comboScore(b) - comboScore(a))
  const top5 = sorted.slice(0, 5)
  top5.forEach((combo, idx) => {
    const p = combo.params || {}
    const paramStr = Object.entries(p).map(([k, v]) => `${k}=${v}`).join(' ')
    const consistPctCombo = Math.round((combo.consistency ?? 0) * 100)
    const ratio = (combo.oos_is_ratio ?? 0).toFixed(2)
    lines.push(`#${idx + 1}: ${paramStr}`)
    lines.push(`    OOS Sharpe: ${(combo.oos_sharpe ?? 0).toFixed(2)} | Consist: ${consistPctCombo}% | Trades: ${combo.oos_trades ?? 0} | OOS/IS: ${ratio}`)
  })
  lines.push('')

  // ─── DISTRIBUTION ───
  lines.push('\u2500\u2500\u2500 DISTRIBUTION \u2500\u2500\u2500')
  const allOos = combos.map((c) => c.oos_sharpe).filter((v) => v != null)
  const total = allOos.length
  if (total > 0) {
    const pctPos = Math.round((allOos.filter((v) => v > 0).length / total) * 100)
    const pctAbove05 = Math.round((allOos.filter((v) => v > 0.5).length / total) * 100)
    const pctAbove1 = Math.round((allOos.filter((v) => v > 1).length / total) * 100)
    const med = median(allOos)
    lines.push(`Combos OOS > 0 : ${pctPos}%`)
    lines.push(`Combos OOS > 0.5 : ${pctAbove05}%`)
    lines.push(`Combos OOS > 1 : ${pctAbove1}%`)
    lines.push(`Médiane OOS Sharpe : ${med.toFixed(2)}`)
  }

  return lines.join('\n')
}

/**
 * Bouton Export avec feedback visuel.
 */
export default function ExportButton({ strategy, asset, selectedRun, combos, regimeAnalysis }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    const text = buildDiagnosticText({ strategy, asset, selectedRun, combos, regimeAnalysis })
    if (!text) return

    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback pour les contextes sans clipboard API
      const textarea = document.createElement('textarea')
      textarea.value = text
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }, [strategy, asset, selectedRun, combos, regimeAnalysis])

  return (
    <button
      onClick={handleCopy}
      className="btn btn-secondary"
      style={{ fontSize: '0.85rem', padding: '4px 12px' }}
    >
      {copied ? '\u2705 Copié !' : '\uD83D\uDCCB Copier le diagnostic'}
    </button>
  )
}
