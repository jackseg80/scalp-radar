/**
 * RegimeWidget — Sidebar card affichant le régime de marché BTC.
 * Sprint 61 — Regime Monitor.
 */
import { useApi } from '../hooks/useApi'
import Spark from './Spark'
import './RegimeWidget.css'

const REGIME_COLORS = {
  BULL: 'var(--accent)',
  BEAR: 'var(--red)',
  RANGE: 'var(--yellow)',
  CRASH: 'var(--red)',
}

const REGIME_LABELS = {
  BULL: 'Haussier',
  BEAR: 'Baissier',
  RANGE: 'Range',
  CRASH: 'Crash',
}

export default function RegimeWidget() {
  const { data: snapData, loading } = useApi('/api/regime/snapshot', 60000)
  const { data: histData } = useApi('/api/regime/history?days=30', 60000)

  if (loading && !snapData) {
    return <div className="text-xs muted">Chargement regime...</div>
  }

  const snap = snapData?.snapshot
  if (!snap) {
    return <div className="text-xs muted">Regime non disponible</div>
  }

  const color = REGIME_COLORS[snap.regime] || 'var(--text-muted)'
  const label = REGIME_LABELS[snap.regime] || snap.regime

  // ATR bar (0-6% range, clamped)
  const atrClamped = Math.min(snap.btc_atr_14d_pct, 6)
  const atrPct = (atrClamped / 6) * 100

  // Sparkline ATR 30j
  const sparkData = histData?.history?.map(h => h.btc_atr_14d_pct) || []

  return (
    <div className="regime-widget">
      {/* Regime badge */}
      <div className="regime-header">
        <span className="regime-dot" style={{ background: color }} />
        <span className="regime-label" style={{ color }}>
          {label}
        </span>
        <span className="regime-days text-xs muted">
          {snap.regime_days}j
        </span>
      </div>

      {/* Metriques */}
      <div className="regime-metrics">
        <div className="regime-metric">
          <span className="regime-metric-label">BTC 30j</span>
          <span className={`regime-metric-value ${snap.btc_change_30d_pct >= 0 ? 'positive' : 'negative'}`}>
            {snap.btc_change_30d_pct >= 0 ? '+' : ''}{snap.btc_change_30d_pct.toFixed(1)}%
          </span>
        </div>
        <div className="regime-metric">
          <span className="regime-metric-label">ATR 14j</span>
          <span className="regime-metric-value">{snap.btc_atr_14d_pct.toFixed(2)}%</span>
        </div>
        <div className="regime-metric">
          <span className="regime-metric-label">Leverage</span>
          <span className="regime-metric-value">x{snap.suggested_leverage}</span>
        </div>
      </div>

      {/* ATR bar */}
      <div className="regime-atr-bar">
        <div className="regime-atr-track">
          <div
            className="regime-atr-fill"
            style={{ width: `${atrPct}%` }}
          />
        </div>
        <div className="regime-atr-labels">
          <span>LOW</span>
          <span>MED</span>
          <span>HIGH</span>
        </div>
      </div>

      {/* Sparkline ATR 30j */}
      {sparkData.length >= 3 && (
        <div className="regime-sparkline">
          <div className="regime-sparkline-label">ATR 30j</div>
          <Spark data={sparkData} h={28} />
        </div>
      )}
    </div>
  )
}
