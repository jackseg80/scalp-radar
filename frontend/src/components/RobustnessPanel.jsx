import { useState, useEffect } from 'react'
import './RobustnessPanel.css'

const API = ''

// ─── Helpers ─────────────────────────────────────────────────────

function fmtPct(v) {
  if (v == null) return '--'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`
}

function fmtPct2(v) {
  if (v == null) return '--'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
}

// ─── Verdict Badge ───────────────────────────────────────────────

function VerdictBadge({ verdict }) {
  if (!verdict) return null
  const cls = {
    VIABLE: 'verdict-viable',
    CAUTION: 'verdict-caution',
    FAIL: 'verdict-fail',
  }[verdict] || 'verdict-fail'
  return <span className={`verdict-badge ${cls}`}>{verdict}</span>
}

// ─── GO/NO-GO Criteria ──────────────────────────────────────────

function CriteriaList({ criteria }) {
  if (!criteria || criteria.length === 0) return null
  return (
    <div className="rob-criteria">
      {criteria.map((c, i) => (
        <div key={i} className={`rob-criterion ${c.pass ? 'criterion-pass' : 'criterion-fail'}`}>
          <span className="criterion-icon">{c.pass ? '\u2705' : '\u274C'}</span>
          <span className="criterion-name">{c.name}</span>
          <span className="criterion-value mono">{c.value}</span>
        </div>
      ))}
    </div>
  )
}

// ─── Bootstrap Section ──────────────────────────────────────────

function BootstrapSection({ data }) {
  if (data.bootstrap_n_sims == null) {
    return <div className="rob-skip">Block Bootstrap — donnees insuffisantes</div>
  }
  return (
    <div className="rob-section">
      <div className="rob-section-title">
        Block Bootstrap ({data.bootstrap_n_sims} sims, blocs {data.bootstrap_block_size}j)
      </div>
      <div className="rob-metrics-row">
        <RobMetric label="Return median" value={fmtPct(data.bootstrap_median_return)} />
        <RobMetric
          label="CI95 Return"
          value={`[${fmtPct(data.bootstrap_ci95_return_low)}, ${fmtPct(data.bootstrap_ci95_return_high)}]`}
        />
        <RobMetric label="DD median" value={fmtPct(data.bootstrap_median_dd)} />
        <RobMetric
          label="Prob. perte"
          value={fmtPct(data.bootstrap_prob_loss)}
          warn={data.bootstrap_prob_loss > 0.10}
        />
        <RobMetric label="Prob. DD > 30%" value={fmtPct(data.bootstrap_prob_dd_30)} />
        <RobMetric label="Prob. DD > KS" value={fmtPct(data.bootstrap_prob_dd_ks)} />
      </div>
    </div>
  )
}

// ─── CVaR Section ───────────────────────────────────────────────

function CvarSection({ data }) {
  if (data.cvar_5_daily == null) {
    return <div className="rob-skip">CVaR — donnees insuffisantes</div>
  }
  const byRegime = data.cvar_by_regime || {}
  return (
    <div className="rob-section">
      <div className="rob-section-title">Value at Risk / CVaR</div>
      <div className="rob-metrics-row">
        <RobMetric label="VaR 5% jour" value={fmtPct2(data.var_5_daily)} />
        <RobMetric label="CVaR 5% jour" value={fmtPct2(data.cvar_5_daily)} />
        <RobMetric label="CVaR 5% 30j" value={fmtPct(data.cvar_30d)} />
        <RobMetric label="CVaR 5% annuel" value={fmtPct(data.cvar_5_annualized)} />
      </div>
      {Object.keys(byRegime).length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="rob-sub-title">CVaR par regime</div>
          <div className="rob-metrics-row">
            {['RANGE', 'BULL', 'BEAR', 'CRASH'].filter(r => r in byRegime).map(r => (
              <RobMetric key={r} label={r} value={fmtPct2(byRegime[r])} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Regime Stress Table ────────────────────────────────────────

function RegimeStressSection({ scenarios }) {
  if (!scenarios || Object.keys(scenarios).length === 0) {
    return <div className="rob-skip">Regime Stress — non disponible</div>
  }
  return (
    <div className="rob-section">
      <div className="rob-section-title">Regime Stress Scenarios</div>
      <table className="rob-table">
        <thead>
          <tr>
            <th>Scenario</th>
            <th>Return (med)</th>
            <th>Max DD (med)</th>
            <th>Prob. perte</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(scenarios).map(([name, info]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className={`mono ${info.median_return >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
                {fmtPct(info.median_return)}
              </td>
              <td className="mono pnl-neg">{fmtPct(info.median_dd)}</td>
              <td className={`mono ${info.prob_loss > 0.5 ? 'pnl-neg' : ''}`}>
                {fmtPct(info.prob_loss)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Historical Stress Table ────────────────────────────────────

function HistoricalStressSection({ events }) {
  if (!events || Object.keys(events).length === 0) {
    return <div className="rob-skip">Historical Stress — non disponible</div>
  }
  return (
    <div className="rob-section">
      <div className="rob-section-title">Historical Stress — Crashes reels</div>
      <table className="rob-table">
        <thead>
          <tr>
            <th>Evenement</th>
            <th>Periode</th>
            <th>Portfolio DD</th>
            <th>BTC DD</th>
            <th>Recovery</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(events).map(([name, info]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="mono muted">
                {info.status === 'N/A' ? (info.reason || 'N/A') : info.period}
              </td>
              <td className="mono">
                {info.status === 'N/A' ? '--' : fmtPct(info.portfolio_dd)}
              </td>
              <td className="mono">
                {info.btc_dd != null ? fmtPct(info.btc_dd) : '--'}
              </td>
              <td className="mono">
                {info.status === 'N/A'
                  ? '--'
                  : info.recovery_days != null
                    ? `${info.recovery_days}j`
                    : '>365j'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Metric Card ────────────────────────────────────────────────

function RobMetric({ label, value, warn }) {
  return (
    <div className="rob-metric">
      <div className="rob-metric-label">{label}</div>
      <div className={`rob-metric-value mono ${warn ? 'pnl-neg' : ''}`}>
        {value}
      </div>
    </div>
  )
}

// ─── Main Component ─────────────────────────────────────────────

export default function RobustnessPanel({ backtestId }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (backtestId == null) { setData(null); return }
    let cancelled = false
    setLoading(true)
    ;(async () => {
      try {
        const res = await fetch(`${API}/api/portfolio/backtests/${backtestId}/robustness`)
        if (res.ok && !cancelled) {
          const json = await res.json()
          setData(json.robustness)
        }
      } catch { /* ignore */ }
      finally { if (!cancelled) setLoading(false) }
    })()
    return () => { cancelled = true }
  }, [backtestId])

  if (loading) return <div className="rob-loading">Chargement robustesse...</div>
  if (!data) return null

  const verdictDetails = data.verdict_details || {}
  const criteria = verdictDetails.criteria || []

  return (
    <div className="robustness-panel">
      <div className="rob-header">
        <span className="rob-title">Analyse de Robustesse</span>
        <VerdictBadge verdict={data.verdict} />
      </div>

      <CriteriaList criteria={criteria} />
      <BootstrapSection data={data} />
      <CvarSection data={data} />
      <RegimeStressSection scenarios={data.regime_stress_results} />
      <HistoricalStressSection events={data.historical_stress_results} />

      <div className="rob-footer">
        Analyse le {new Date(data.created_at).toLocaleDateString('fr-FR')}
        {data.label && ` — ${data.label}`}
      </div>
    </div>
  )
}
