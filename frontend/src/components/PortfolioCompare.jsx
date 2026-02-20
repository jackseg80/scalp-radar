import { useMemo } from 'react'
import CompareVerdict from './CompareVerdict'

function fmtDuration(hours) {
  const d = Math.floor(hours / 24)
  const h = Math.round(hours % 24)
  return d > 0 ? `${d}j ${h}h` : `${h}h`
}

function computeLosersPct(run) {
  const par = run.per_asset_results
  if (!par) return 0
  const entries = typeof par === 'string' ? Object.entries(JSON.parse(par)) : Object.entries(par)
  if (entries.length === 0) return 0
  const losers = entries.filter(([, v]) => (v.net_pnl ?? 0) < 0).length
  return (losers / entries.length) * 100
}

const METRICS = [
  { label: 'Return total', key: 'total_return_pct', fmt: v => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` },
  { label: 'Max drawdown', key: 'max_drawdown_pct', fmt: v => `${v.toFixed(1)}%`, inverted: true },
  { label: 'Return/DD ratio', key: '_return_dd_ratio', fmt: v => v.toFixed(1), computed: true },
  { label: 'DD duree max', key: 'max_drawdown_duration_hours', fmt: v => fmtDuration(v), inverted: true },
  { label: 'Win rate', key: 'win_rate', fmt: v => `${v.toFixed(1)}%` },
  { label: 'Trades', key: 'total_trades', fmt: v => v },
  { label: 'Margin peak', key: 'peak_margin_ratio', fmt: v => `${(v * 100).toFixed(0)}%`, inverted: true },
  { label: 'Kill switch', key: 'kill_switch_triggers', fmt: v => v, inverted: true },
  { label: 'P&L realise', key: 'realized_pnl', fmt: v => `${v >= 0 ? '+' : ''}${v.toFixed(0)}$` },
  { label: '% runners perte', key: '_losers_pct', fmt: v => `${v.toFixed(0)}%`, inverted: true, computed: true },
]

function enrichRun(run) {
  const dd = Math.abs(run.max_drawdown_pct || 0.01)
  return {
    ...run,
    _return_dd_ratio: dd > 0 ? (run.total_return_pct || 0) / dd : 0,
    _losers_pct: computeLosersPct(run),
  }
}

export default function PortfolioCompare({ runs }) {
  if (!runs || runs.length < 2) return null

  const enriched = useMemo(() => runs.map(enrichRun), [runs])
  const run1 = enriched[0]
  const run2 = enriched[1]

  return (
    <>
      <div className="compare-section">
        <h4>Comparaison</h4>
        <table className="compare-table">
          <thead>
            <tr>
              <th>Metrique</th>
              <th>{run1.label || `Run #${run1.id}`}</th>
              <th>{run2.label || `Run #${run2.id}`}</th>
              <th style={{ textAlign: 'center' }}>&Delta;</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map(m => {
              const v1 = run1[m.key] ?? 0
              const v2 = run2[m.key] ?? 0
              const delta = v2 - v1
              const better = m.inverted ? delta < 0 : delta > 0
              const cls = delta === 0 ? 'muted' : better ? 'pnl-pos' : 'pnl-neg'
              return (
                <tr key={m.key}>
                  <td style={{ color: '#aaa' }}>{m.label}</td>
                  <td className="mono">{m.fmt(v1)}</td>
                  <td className="mono">{m.fmt(v2)}</td>
                  <td className={`mono ${cls}`} style={{ textAlign: 'center' }}>
                    {delta >= 0 ? '+' : ''}{typeof m.fmt(delta) === 'number' ? delta : m.fmt(delta)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <CompareVerdict runs={enriched} />
    </>
  )
}
