/**
 * StrategySummaryPanel — Résumé WFO d'une stratégie dans la page Recherche
 * Sprint 36
 *
 * Sections : grade distribution, red flags, convergence params, verdict, bouton action.
 */

import { useState, useEffect } from 'react'

const GRADE_COLORS = {
  A: '#10b981',
  B: '#3b82f6',
  C: '#f59e0b',
  D: '#f97316',
  F: '#ef4444',
}

function generateVerdict(s) {
  const parts = []

  parts.push(`${s.ab_count}/${s.total_assets} Grade A+B`)

  if (s.underpowered_count === 0)
    parts.push('MC significatif')
  else
    parts.push(`${s.underpowered_count} underpowered`)

  if (s.red_flags_total > 0)
    parts.push(`${s.red_flags_total} red flags`)
  else
    parts.push('aucun red flag')

  if (s.ab_count === 0) return parts.join('. ') + ' \u2192 Stratégie à abandonner'
  if (s.underpowered_pct >= 50) return parts.join('. ') + ' \u2192 Non viable statistiquement'
  if (s.ab_count >= 5 && s.portfolio_runs.length === 0) return parts.join('. ') + ' \u2192 Forward test recommandé'
  if (s.ab_count >= 5) return parts.join('. ') + ' \u2192 Prêt pour évaluation portfolio'
  return parts.join('. ') + ' \u2192 Paper trading uniquement'
}

function convergenceIcon(modePct) {
  if (modePct >= 60) return '\u2705'
  if (modePct >= 40) return '\u26A0\uFE0F'
  return '\u274C'
}

export default function StrategySummaryPanel({ strategyName, onNavigatePortfolio }) {
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!strategyName) {
      setSummary(null)
      return
    }
    setLoading(true)
    fetch(`/api/optimization/strategy-summary?strategy=${encodeURIComponent(strategyName)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setSummary(data))
      .catch(() => setSummary(null))
      .finally(() => setLoading(false))
  }, [strategyName])

  if (!strategyName || loading) return null
  if (!summary || !summary.total_assets) return null

  const grades = summary.grades || {}
  const redFlags = summary.red_flags || {}
  const convergence = summary.param_convergence || []

  return (
    <div style={{
      background: '#111',
      border: '1px solid #333',
      borderRadius: 8,
      padding: 16,
      marginBottom: 16,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 15, color: '#eee' }}>
          {strategyName} — Résumé WFO
        </h3>
        <span style={{ color: '#888', fontSize: 12 }}>{summary.latest_wfo_date}</span>
      </div>

      {/* Grade distribution */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
        {['A', 'B', 'C', 'D', 'F'].map(g => (
          <span
            key={g}
            className={`grade-badge grade-${g}`}
            style={{
              padding: '4px 10px',
              borderRadius: 6,
              fontSize: 13,
              fontWeight: 600,
              opacity: grades[g] > 0 ? 1 : 0.3,
            }}
          >
            {g}: {grades[g] || 0}
          </span>
        ))}
        <span style={{ color: '#888', fontSize: 12, alignSelf: 'center', marginLeft: 8 }}>
          {summary.total_assets} total | Sharpe moy {summary.avg_oos_sharpe} | Consist. {(summary.avg_consistency * 100).toFixed(0)}%
        </span>
      </div>

      {/* Red flags */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ color: '#aaa', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Red Flags</div>
        {summary.red_flags_total === 0 ? (
          <div style={{ color: '#10b981', fontSize: 13 }}>{'\u2705'} Aucun red flag détecté</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {redFlags.oos_is_ratio_suspect > 0 && (
              <div style={{ color: '#f59e0b', fontSize: 13 }}>
                {'\u26A0\uFE0F'} {redFlags.oos_is_ratio_suspect}/{summary.total_assets} OOS/IS ratio {'>'} 1.5 (suspect)
              </div>
            )}
            {redFlags.sharpe_anomalous > 0 && (
              <div style={{ color: '#ef4444', fontSize: 13 }}>
                {'\u274C'} {redFlags.sharpe_anomalous}/{summary.total_assets} Sharpe {'>'} 20 (anomal)
              </div>
            )}
            {redFlags.underpowered > 0 && (
              <div style={{ color: '#f97316', fontSize: 13 }}>
                {'\u26A0\uFE0F'} {redFlags.underpowered}/{summary.total_assets} underpowered (MC non significatif)
              </div>
            )}
            {redFlags.low_consistency > 0 && (
              <div style={{ color: '#f59e0b', fontSize: 13 }}>
                {'\u26A0\uFE0F'} {redFlags.low_consistency}/{summary.total_assets} consistance {'<'} 50%
              </div>
            )}
            {redFlags.low_stability > 0 && (
              <div style={{ color: '#f59e0b', fontSize: 13 }}>
                {'\u26A0\uFE0F'} {redFlags.low_stability}/{summary.total_assets} stabilité params {'<'} 0.3
              </div>
            )}
          </div>
        )}
      </div>

      {/* Convergence params */}
      {convergence.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: '#aaa', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Convergence Paramètres</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 16px' }}>
            {convergence.map(c => (
              <span key={c.param} style={{ fontSize: 13, color: '#ccc' }}>
                {convergenceIcon(c.mode_pct)} {c.param}: mode={c.mode} ({c.mode_pct}%)
                {c.n_unique > 1 && <span style={{ color: '#888' }}> — {c.n_unique} valeurs</span>}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Verdict */}
      <div style={{
        background: '#1a1a1a',
        borderRadius: 6,
        padding: '10px 14px',
        marginBottom: 12,
        fontSize: 13,
        color: '#ddd',
        borderLeft: '3px solid #3b82f6',
      }}>
        {generateVerdict(summary)}
      </div>

      {/* Action button */}
      {summary.ab_count >= 1 && onNavigatePortfolio && (
        <button
          onClick={onNavigatePortfolio}
          style={{
            padding: '6px 16px',
            background: '#2a2a2a',
            border: '1px solid #444',
            borderRadius: 4,
            color: '#ccc',
            cursor: 'pointer',
            fontSize: 13,
          }}
        >
          Tester en portfolio {'\u2192'}
        </button>
      )}
    </div>
  )
}
