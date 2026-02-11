/**
 * SignalDetail — Panneau de détail extensible pour un asset.
 * Props : asset (objet avec strategies, indicators, price)
 * Affiche ScoreRing, SignalBreakdown, et indicateurs.
 */
import ScoreRing from './ScoreRing'
import SignalBreakdown from './SignalBreakdown'
import Tooltip from './Tooltip'

export default function SignalDetail({ asset = {} }) {
  const strategies = asset.strategies || {}
  const indicators = asset.indicators || {}
  const price = asset.price

  // Score global = moyenne des ratios de toutes les stratégies
  const entries = Object.values(strategies)
  const avgScore = entries.length > 0
    ? entries.reduce((sum, s) => {
        const conditions = s.conditions || []
        const total = conditions.length || 1
        const met = conditions.filter(c => c.met).length
        return sum + met / total
      }, 0) / entries.length
    : 0

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

        {/* Breakdown des stratégies */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="text-xs dim" style={{ marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Conditions par stratégie
          </div>
          <SignalBreakdown strategies={strategies} />
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
                label="Regime"
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

function IndicatorRow({ label, value, color, badge, tooltip }) {
  const row = (
    <div className="flex-between" style={{ fontSize: 11, gap: 8 }}>
      <span className="muted">{label}</span>
      {badge ? (
        <span className={`badge ${
          value === 'RANGING' ? 'badge-ranging' : 'badge-trending'
        }`}>
          {value}
        </span>
      ) : (
        <span className="mono" style={{ color: color || 'var(--text-primary)', fontWeight: 500 }}>
          {value}
        </span>
      )}
    </div>
  )
  if (!tooltip) return row
  return <Tooltip content={tooltip} inline={false}>{row}</Tooltip>
}
