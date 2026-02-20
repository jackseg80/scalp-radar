/**
 * StrategiesPage — Catalogue interactif des 16 stratégies
 * Sprint Strategy Lab
 */

import { useState, useMemo } from 'react'
import { STRATEGIES, STRATEGY_FAMILIES } from '../data/strategyMeta'
import StrategyDetail from './StrategyDetail'
import './StrategiesPage.css'

const GRADE_COLORS = {
  A: '#10b981',
  B: '#3b82f6',
  C: '#f59e0b',
  D: '#f97316',
  F: '#ef4444',
}

function StatusBadge({ status }) {
  const labels = {
    live: 'LIVE',
    paper: 'PAPER',
    disabled: 'Off',
    replaced: 'Remplacé',
  }
  return (
    <span className={`status-badge ${status}`}>
      {labels[status] || status}
    </span>
  )
}

function GradeBadgeMini({ grade }) {
  if (!grade) {
    return <span className="strategy-grade-mini grade-null">—</span>
  }
  return (
    <span
      className="strategy-grade-mini"
      style={{ background: GRADE_COLORS[grade] || '#444' }}
    >
      {grade}
    </span>
  )
}

function StrategyCard({ strategy, onClick }) {
  return (
    <div
      className={`strategy-card status-${strategy.status}`}
      onClick={() => onClick(strategy.id)}
    >
      <div className="strategy-card-header">
        <span className="strategy-card-name">{strategy.name}</span>
        <div className="strategy-card-badges">
          <GradeBadgeMini grade={strategy.wfoGrade} />
          <StatusBadge status={strategy.status} />
        </div>
      </div>

      <div className="strategy-card-meta">
        <span>{strategy.type}</span>
        <span>{strategy.direction}</span>
      </div>

      <p className="strategy-card-desc">{strategy.shortDesc}</p>

      <div className="strategy-card-footer">
        <span className="strategy-card-cta">Comprendre →</span>
      </div>
    </div>
  )
}

export default function StrategiesPage({ onNavigate, setEvalStrategy }) {
  const [selectedStrategy, setSelectedStrategy] = useState(null)
  const [familyFilter, setFamilyFilter] = useState('all')

  // Grouper les stratégies par famille, triées par ordre de famille
  const groupedStrategies = useMemo(() => {
    const filtered = familyFilter === 'all'
      ? STRATEGIES
      : STRATEGIES.filter(s => s.family === familyFilter)

    const groups = {}
    for (const s of filtered) {
      if (!groups[s.family]) groups[s.family] = []
      groups[s.family].push(s)
    }

    // Trier les groupes par STRATEGY_FAMILIES[family].order
    return Object.entries(groups)
      .sort(([a], [b]) => (STRATEGY_FAMILIES[a]?.order || 99) - (STRATEGY_FAMILIES[b]?.order || 99))
  }, [familyFilter])

  const familyCounts = useMemo(() => {
    const counts = { all: STRATEGIES.length }
    for (const s of STRATEGIES) {
      counts[s.family] = (counts[s.family] || 0) + 1
    }
    return counts
  }, [])

  // Vue détail
  if (selectedStrategy) {
    return (
      <div className="strategies-page">
        <StrategyDetail
          strategyId={selectedStrategy}
          onBack={() => setSelectedStrategy(null)}
          onNavigate={onNavigate}
          setEvalStrategy={setEvalStrategy}
        />
      </div>
    )
  }

  // Vue catalogue
  const filterButtons = [
    { key: 'all', label: 'Toutes' },
    { key: 'grid', label: STRATEGY_FAMILIES.grid.label },
    { key: 'swing', label: STRATEGY_FAMILIES.swing.label },
    { key: 'scalp', label: STRATEGY_FAMILIES.scalp.label },
  ]

  return (
    <div className="strategies-page">
      <div className="strategies-header">
        <h2>Stratégies</h2>
        <p>Documentation interactive des 16 stratégies de Scalp Radar</p>
        <div className="strategies-filters">
          {filterButtons.map(f => (
            <button
              key={f.key}
              className={`strategies-filter-btn ${familyFilter === f.key ? 'active' : ''}`}
              onClick={() => setFamilyFilter(f.key)}
            >
              {f.label} ({familyCounts[f.key] || 0})
            </button>
          ))}
        </div>
      </div>

      {groupedStrategies.map(([family, strategies]) => (
        <div key={family} className="strategies-family-group">
          <div className="strategies-family-label">
            <span
              className="strategies-family-dot"
              style={{ background: STRATEGY_FAMILIES[family]?.color || '#888' }}
            />
            <span className="strategies-family-name">
              {STRATEGY_FAMILIES[family]?.label || family}
            </span>
          </div>
          <div className="strategies-grid">
            {strategies.map(s => (
              <StrategyCard
                key={s.id}
                strategy={s}
                onClick={setSelectedStrategy}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
