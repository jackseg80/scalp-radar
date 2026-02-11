/**
 * Heatmap — Matrice assets x stratégies.
 * Utilise useApi('/api/simulator/conditions', 10000).
 * Grille CSS avec cellules colorées selon le ratio conditions_met / conditions_total.
 */
import { useApi } from '../hooks/useApi'

export default function Heatmap() {
  const { data, loading } = useApi('/api/simulator/conditions', 10000)

  // Backend renvoie assets comme dict {symbol: data}, convertir en tableau
  const assetsObj = data?.assets || {}
  const assets = Object.entries(assetsObj).map(([symbol, a]) => ({ symbol, ...a }))

  // Extraire la liste de toutes les stratégies uniques
  const strategyNames = []
  assets.forEach(a => {
    Object.keys(a.strategies || {}).forEach(name => {
      if (!strategyNames.includes(name)) strategyNames.push(name)
    })
  })

  if (loading && assets.length === 0) {
    return (
      <div className="card">
        <h2>Heatmap</h2>
        <div className="empty-state">
          <div className="skeleton skeleton-line" style={{ width: '90%', margin: '0 auto' }} />
          <div className="skeleton skeleton-line" style={{ width: '90%', margin: '8px auto 0' }} />
          <div className="skeleton skeleton-line" style={{ width: '90%', margin: '8px auto 0' }} />
        </div>
      </div>
    )
  }

  if (assets.length === 0 || strategyNames.length === 0) {
    return (
      <div className="card">
        <h2>Heatmap</h2>
        <div className="empty-state">En attente de données...</div>
      </div>
    )
  }

  const columns = strategyNames.length + 1
  const gridTemplate = `100px repeat(${strategyNames.length}, 1fr)`

  return (
    <div className="card">
      <h2>Heatmap — Conditions</h2>

      <div className="heatmap-grid" style={{ gridTemplateColumns: gridTemplate }}>
        {/* En-tête : cellule vide + noms des stratégies */}
        <div className="heatmap-header" />
        {strategyNames.map(name => (
          <div key={name} className="heatmap-header">{name}</div>
        ))}

        {/* Lignes : un asset par ligne */}
        {assets.map(asset => (
          <AssetRow
            key={asset.symbol}
            symbol={asset.symbol}
            strategies={asset.strategies || {}}
            strategyNames={strategyNames}
          />
        ))}
      </div>
    </div>
  )
}

function AssetRow({ symbol, strategies, strategyNames }) {
  return (
    <>
      <div className="heatmap-cell" style={{ fontWeight: 600, textAlign: 'left', fontSize: 12 }}>
        {symbol}
      </div>
      {strategyNames.map(name => {
        const s = strategies[name]
        if (!s) {
          return (
            <div
              key={name}
              className="heatmap-cell"
              style={{ background: 'transparent', color: 'var(--text-dim)' }}
            >
              --
            </div>
          )
        }

        const conditions = s.conditions || []
        const total = conditions.length || 1
        const met = conditions.filter(c => c.met).length
        const ratio = met / total

        const { bg, fg } = getCellColor(ratio)

        return (
          <div
            key={name}
            className="heatmap-cell"
            style={{ background: bg, color: fg }}
            title={`${symbol} / ${name}: ${met}/${total}`}
          >
            {met}/{total}
          </div>
        )
      })}
    </>
  )
}

function getCellColor(ratio) {
  if (ratio >= 0.75) return { bg: 'var(--accent-dim)', fg: 'var(--accent)' }
  if (ratio >= 0.55) return { bg: 'var(--yellow-dim)', fg: 'var(--yellow)' }
  if (ratio >= 0.35) return { bg: 'var(--orange-dim)', fg: 'var(--orange)' }
  if (ratio > 0)     return { bg: 'var(--red-dim)', fg: 'var(--red)' }
  return { bg: 'transparent', fg: 'var(--text-dim)' }
}
