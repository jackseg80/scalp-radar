/**
 * StrategyEvalBar — Bandeau global d'évaluation stratégie
 * Sprint 36
 *
 * Visible uniquement sur Explorer, Recherche, Portfolio.
 * Dropdown stratégie + pills d'état + badge recommandations.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useRecommendations } from '../hooks/useRecommendations'
import './StrategyEvalBar.css'

const VISIBLE_TABS = new Set(['explorer', 'research', 'portfolio'])

const LEVEL_ICONS = {
  error: '\u{1F534}',
  recommended: '\u{1F7E0}',
  warning: '\u{1F7E1}',
  info: '\u{1F535}',
  success: '\u{1F7E2}',
}

export default function StrategyEvalBar({ activeTab, onNavigate, evalStrategy, setEvalStrategy }) {
  const [strategies, setStrategies] = useState([])
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [recoOpen, setRecoOpen] = useState(false)
  const panelRef = useRef(null)

  const recos = useRecommendations(summary)

  // Fetch strategies list (once)
  useEffect(() => {
    fetch('/api/optimization/strategies')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.strategies) setStrategies(data.strategies)
      })
      .catch(() => {})
  }, [])

  // Fetch summary when strategy changes
  useEffect(() => {
    if (!evalStrategy) {
      setSummary(null)
      return
    }
    setLoading(true)
    fetch(`/api/optimization/strategy-summary?strategy=${encodeURIComponent(evalStrategy)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setSummary(data))
      .catch(() => setSummary(null))
      .finally(() => setLoading(false))
  }, [evalStrategy])

  // Close reco panel on outside click
  useEffect(() => {
    if (!recoOpen) return
    const handler = (e) => {
      if (panelRef.current && !panelRef.current.contains(e.target)) {
        setRecoOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [recoOpen])

  // Écouter les changements venant des autres composants
  useEffect(() => {
    const handler = (e) => {
      const val = e.detail || ''
      if (val !== evalStrategy) setEvalStrategy(val)
    }
    window.addEventListener('eval-strategy-change', handler)
    return () => window.removeEventListener('eval-strategy-change', handler)
  }, [evalStrategy, setEvalStrategy])

  const handleStrategyChange = useCallback((e) => {
    const val = e.target.value
    setEvalStrategy(val)
    setRecoOpen(false)
    window.dispatchEvent(new CustomEvent('eval-strategy-change', { detail: val }))
  }, [setEvalStrategy])

  if (!VISIBLE_TABS.has(activeTab)) return null

  // Pills
  const wfoPill = summary ? (() => {
    const { total_assets, grades, ab_count } = summary
    const gradeStr = ['A', 'B', 'C', 'D', 'F']
      .filter(g => grades[g] > 0)
      .map(g => `${grades[g]}${g}`)
      .join(' ')
    const color = ab_count >= 5 ? 'green' : ab_count >= 1 ? 'yellow' : 'gray'
    return { label: `WFO: ${total_assets} (${gradeStr})`, color }
  })() : null

  const auditPill = summary ? (() => {
    const n = summary.red_flags_total
    const color = n === 0 ? 'green' : n <= 5 ? 'yellow' : 'orange'
    return { label: `Audit: ${n} flag${n !== 1 ? 's' : ''}`, color }
  })() : null

  const portfolioPill = summary ? (() => {
    const n = summary.portfolio_runs.length
    const color = n >= 1 ? 'green' : 'gray'
    return { label: `Portf: ${n} run${n !== 1 ? 's' : ''}`, color }
  })() : null

  return (
    <div className="eval-bar">
      <select
        className="eval-strategy-select"
        value={evalStrategy}
        onChange={handleStrategyChange}
      >
        <option value="">-- Stratégie --</option>
        {strategies.map(s => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>

      {!evalStrategy && (
        <span className="eval-bar__placeholder">
          Sélectionnez une stratégie pour voir son état
        </span>
      )}

      {loading && <span className="eval-bar--loading">Chargement...</span>}

      {summary && (
        <div className="eval-pills">
          {wfoPill && (
            <button
              className={`eval-pill eval-pill--${wfoPill.color}`}
              onClick={() => onNavigate('research')}
            >
              {wfoPill.label}
            </button>
          )}
          {auditPill && (
            <button
              className={`eval-pill eval-pill--${auditPill.color}`}
              onClick={() => onNavigate('research')}
            >
              {auditPill.label}
            </button>
          )}
          {portfolioPill && (
            <button
              className={`eval-pill eval-pill--${portfolioPill.color}`}
              onClick={() => onNavigate('portfolio')}
            >
              {portfolioPill.label}
            </button>
          )}
        </div>
      )}

      {recos.length > 0 && (
        <div className="eval-reco-wrapper" ref={panelRef}>
          <button
            className="eval-reco-badge"
            onClick={() => setRecoOpen(prev => !prev)}
          >
            {recos.length} reco{recos.length > 1 ? 's' : ''}
          </button>
          {recoOpen && (
            <div className="eval-reco-panel">
              {recos.map((r, i) => (
                <div key={i} className="eval-reco-item">
                  <span className="eval-reco-icon">{LEVEL_ICONS[r.level] || ''}</span>
                  <span className="eval-reco-text">{r.text}</span>
                  {r.action && (
                    <button
                      className="eval-reco-action"
                      onClick={() => { onNavigate(r.action.tab); setRecoOpen(false) }}
                    >
                      {r.action.label}
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
