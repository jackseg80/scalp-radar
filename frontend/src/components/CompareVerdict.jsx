/**
 * CompareVerdict — Verdict automatique sous le tableau de comparaison portfolio
 * Sprint 36
 */

export default function CompareVerdict({ runs }) {
  if (!runs || runs.length < 2) return null

  const r1 = runs[0]
  const r2 = runs[1]
  const flags = []

  // Return/DD ratio : flag si best > worst * 1.5
  const rd1 = r1._return_dd_ratio ?? 0
  const rd2 = r2._return_dd_ratio ?? 0
  const rdBest = Math.max(rd1, rd2)
  const rdWorst = Math.min(rd1, rd2)
  if (rdWorst > 0 && rdBest > rdWorst * 1.5) {
    const winner = rd1 > rd2 ? r1 : r2
    flags.push({
      level: 'warning',
      text: `Return/DD ratio diverge : ${(winner.label || `Run #${winner.id}`)} est nettement meilleur (${rdBest.toFixed(1)} vs ${rdWorst.toFixed(1)})`,
    })
  }

  // DD duree > 720h (30j)
  for (const r of [r1, r2]) {
    const ddh = r.max_drawdown_duration_hours ?? 0
    if (ddh > 720) {
      flags.push({
        level: 'warning',
        text: `${r.label || `Run #${r.id}`} : drawdown > 30j (${Math.round(ddh / 24)}j)`,
      })
    }
  }

  // % losers > 20%
  for (const r of [r1, r2]) {
    const lp = r._losers_pct ?? 0
    if (lp > 20) {
      flags.push({
        level: 'info',
        text: `${r.label || `Run #${r.id}`} : ${lp.toFixed(0)}% des assets en perte`,
      })
    }
  }

  // Verdict final : un run domine tous les criteres cles
  const keys = ['total_return_pct', '_return_dd_ratio', 'win_rate']
  const r1Wins = keys.filter(k => (r1[k] ?? 0) > (r2[k] ?? 0)).length
  const r2Wins = keys.filter(k => (r2[k] ?? 0) > (r1[k] ?? 0)).length
  if (r1Wins === keys.length || r2Wins === keys.length) {
    const winner = r1Wins > r2Wins ? r1 : r2
    flags.push({
      level: 'success',
      text: `${winner.label || `Run #${winner.id}`} domine sur tous les criteres cles`,
    })
  }

  if (flags.length === 0) return null

  const LEVEL_STYLE = {
    warning: { color: '#f59e0b', icon: '\u26A0\uFE0F' },
    info: { color: '#3b82f6', icon: '\u2139\uFE0F' },
    success: { color: '#10b981', icon: '\u2705' },
  }

  return (
    <div style={{
      background: '#1a1a1a',
      border: '1px solid #333',
      borderRadius: 8,
      padding: '12px 16px',
      marginTop: 8,
    }}>
      <div style={{ color: '#aaa', fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Verdict</div>
      {flags.map((f, i) => {
        const s = LEVEL_STYLE[f.level] || LEVEL_STYLE.info
        return (
          <div key={i} style={{ color: s.color, fontSize: 13, marginBottom: 4 }}>
            {s.icon} {f.text}
          </div>
        )
      })}
    </div>
  )
}
