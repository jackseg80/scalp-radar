/**
 * SignalDetail — Panneau de détail extensible enrichi pour un asset.
 * Props : asset (objet avec strategies, indicators, price, regime)
 * Affiche les conditions par stratégie avec barres de progression,
 * indicateurs, et résumé des conditions manquantes.
 */
import ScoreRing from './ScoreRing'
import Tooltip from './Tooltip'

const STRATEGY_NAMES = {
  vwap_rsi: 'VWAP + RSI',
  momentum: 'Momentum',
  funding: 'Funding',
  liquidation: 'Liquidation',
  bollinger_mr: 'Bollinger MR',
  donchian_breakout: 'Donchian Breakout',
  supertrend: 'Supertrend',
  boltrend: 'Bollinger Trend',
  envelope_dca: 'Envelope DCA',
  envelope_dca_short: 'Envelope DCA Short',
  grid_atr: 'Grid ATR',
  grid_range_atr: 'Grid Range ATR',
  grid_multi_tf: 'Grid Multi-TF',
  grid_funding: 'Grid Funding',
  grid_trend: 'Grid Trend',
  grid_boltrend: 'Grid BolTrend',
}

function conditionBarColor(met) {
  return met ? 'var(--accent)' : 'var(--red)'
}

function conditionFillPct(condition) {
  const { value, threshold } = condition
  if (value == null || threshold == null) return 0
  const v = Math.abs(Number(value))
  const t = Math.abs(Number(threshold))
  if (t === 0) return condition.met ? 100 : 0
  return Math.min(100, (v / t) * 100)
}

export default function SignalDetail({ asset = {} }) {
  const strategies = asset.strategies || {}
  const indicators = asset.indicators || {}
  const price = asset.price

  // Score global = moyenne des ratios de toutes les stratégies
  const entries = Object.entries(strategies)
  const avgScore = entries.length > 0
    ? entries.reduce((sum, [, s]) => {
        const conditions = s.conditions || []
        const total = conditions.length || 1
        const met = conditions.filter(c => c.met).length
        return sum + met / total
      }, 0) / entries.length
    : 0

  // Conditions non remplies (toutes stratégies)
  const allMissing = []
  entries.forEach(([stratName, s]) => {
    const conditions = s.conditions || []
    conditions.forEach(c => {
      if (!c.met) allMissing.push(c.name)
    })
  })
  const uniqueMissing = [...new Set(allMissing)]

  return (
    <div className="scanner-expand">
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        {/* Score ring */}
        <div style={{ flexShrink: 0, textAlign: 'center' }}>
          <Tooltip content="Score global = moyenne des ratios de toutes les stratégies">
            <ScoreRing score={avgScore} size={72} />
          </Tooltip>
          <div className="text-xs muted" style={{ marginTop: 4 }}>Score global</div>
        </div>

        {/* Conditions par stratégie avec barres */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="text-xs dim" style={{ marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Conditions par stratégie
          </div>
          {entries.map(([stratName, s]) => {
            const conditions = s.conditions || []
            const met = conditions.filter(c => c.met).length
            const total = conditions.length
            return (
              <div key={stratName} style={{ marginBottom: 10 }}>
                <div className="flex-between" style={{ marginBottom: 4 }}>
                  <span className="text-xs" style={{ fontWeight: 600 }}>
                    {STRATEGY_NAMES[stratName] || stratName}
                  </span>
                  <span className="text-xs mono muted">{met}/{total}</span>
                </div>
                {conditions.map((cond, i) => (
                  <div className="condition-row" key={i}>
                    <span className="condition-label">{cond.name}</span>
                    <div className="condition-bar">
                      <div
                        className="condition-bar__fill"
                        style={{
                          width: `${conditionFillPct(cond)}%`,
                          background: conditionBarColor(cond.met),
                        }}
                      />
                    </div>
                    <span className="condition-value">
                      {cond.value != null ? (typeof cond.value === 'number' ? Number(cond.value).toFixed(1) : cond.value) : '--'}
                    </span>
                    <span className="condition-status">{cond.met ? '\u2713' : '\u2717'}</span>
                  </div>
                ))}
              </div>
            )
          })}

          {/* Résumé manquant */}
          {uniqueMissing.length > 0 && (
            <div className="text-xs" style={{ marginTop: 4, color: 'var(--orange)' }}>
              Manque : {uniqueMissing.join(', ')}
            </div>
          )}
        </div>

        {/* Indicateurs */}
        <div style={{ flexShrink: 0, minWidth: 140 }}>
          <div className="text-xs dim" style={{ marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Indicateurs
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {price != null && (
              <IndicatorRow label="Prix" value={Number(price).toFixed(2)} />
            )}
            {indicators.rsi_14 != null && (
              <IndicatorRow
                label="RSI"
                value={Number(indicators.rsi_14).toFixed(1)}
                color={indicators.rsi_14 < 30 ? 'var(--accent)' : indicators.rsi_14 > 70 ? 'var(--red)' : null}
                bar={{ value: indicators.rsi_14, max: 100 }}
                tooltip="RSI (14 périodes) : < 30 = survente (LONG), > 70 = surachat (SHORT)"
              />
            )}
            {indicators.vwap_distance_pct != null && (
              <IndicatorRow
                label="VWAP dist"
                value={`${Number(indicators.vwap_distance_pct).toFixed(3)}%`}
                tooltip="Distance au VWAP : négatif = sous le VWAP (LONG probable), positif = au-dessus (SHORT probable)"
              />
            )}
            {indicators.adx != null && (
              <IndicatorRow
                label="ADX"
                value={Number(indicators.adx).toFixed(1)}
                bar={{ value: indicators.adx, max: 60 }}
                tooltip="Force de la tendance : > 25 = tendance confirmée, < 20 = marché latéral"
              />
            )}
            {indicators.atr_pct != null && (
              <IndicatorRow
                label="ATR %"
                value={`${Number(indicators.atr_pct).toFixed(2)}%`}
                tooltip="Volatilité moyenne (ATR / prix). Plus élevé = mouvements plus larges"
              />
            )}
            {asset.regime && (
              <IndicatorRow
                label="Régime"
                value={asset.regime}
                badge
                tooltip="RANGING = latéral (mean reversion), TRENDING = directionnel (momentum)"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function IndicatorRow({ label, value, color, badge, tooltip, bar }) {
  const row = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
      <span className="muted" style={{ width: 55, flexShrink: 0 }}>{label}</span>
      {bar && (
        <div className="condition-bar" style={{ width: 40, height: 4 }}>
          <div
            className="condition-bar__fill"
            style={{
              width: `${Math.min(100, (bar.value / bar.max) * 100)}%`,
              background: color || 'var(--blue)',
            }}
          />
        </div>
      )}
      {badge ? (
        <span className={`badge ${
          value === 'RANGING' ? 'badge-ranging' : 'badge-trending'
        }`}>
          {value}
        </span>
      ) : (
        <span className="mono" style={{ color: color || 'var(--text-primary)', fontWeight: 500, marginLeft: 'auto' }}>
          {value}
        </span>
      )}
    </div>
  )
  if (!tooltip) return row
  return <Tooltip content={tooltip} inline={false}>{row}</Tooltip>
}
