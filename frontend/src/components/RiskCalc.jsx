/**
 * RiskCalc — Calculatrice de risque interactive (client-side uniquement).
 * Inputs : capital (default 10000), leverage (slider 1-50, default 10), stop loss % (default 0.3)
 * Outputs : taille de position, perte max, distance de liquidation.
 */
import { useState } from 'react'
import Tooltip from './Tooltip'

export default function RiskCalc() {
  const [capital, setCapital] = useState(10000)
  const [leverage, setLeverage] = useState(10)
  const [stopLoss, setStopLoss] = useState(0.3)

  // Calculs
  const positionSize = capital * leverage
  const maxLoss = capital * (stopLoss / 100)
  const liqDistance = 100 / leverage
  const riskReward = stopLoss > 0 ? (liqDistance / stopLoss).toFixed(1) : '--'

  return (
    <div className="card">
      <h2>Calculatrice de Risque</h2>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {/* Capital */}
        <div>
          <Tooltip content="Capital initial en USDT, base pour le calcul de la taille de position" inline={false}>
            <label className="text-xs muted" style={{ display: 'block', marginBottom: 4 }}>
              Capital (USDT)
            </label>
          </Tooltip>
          <input
            type="number"
            className="risk-input"
            value={capital}
            onChange={e => setCapital(Number(e.target.value) || 0)}
            min={0}
            step={100}
          />
        </div>

        {/* Levier */}
        <div>
          <div className="flex-between" style={{ marginBottom: 4 }}>
            <Tooltip content="Multiplicateur de marge (x1 = spot, x50 max). Plus le levier est haut, plus la liquidation est proche">
              <label className="text-xs muted">Levier</label>
            </Tooltip>
            <span className="mono text-xs" style={{ color: 'var(--accent)' }}>x{leverage}</span>
          </div>
          <input
            type="range"
            className="risk-slider"
            min={1}
            max={50}
            value={leverage}
            onChange={e => setLeverage(Number(e.target.value))}
          />
          <div className="flex-between text-xs dim" style={{ marginTop: 2 }}>
            <span>x1</span>
            <span>x50</span>
          </div>
        </div>

        {/* Stop Loss */}
        <div>
          <div className="flex-between" style={{ marginBottom: 4 }}>
            <Tooltip content="Distance du stop loss en % depuis l'entrée. Détermine la perte max par trade">
              <label className="text-xs muted">Stop Loss (%)</label>
            </Tooltip>
            <span className="mono text-xs">{stopLoss}%</span>
          </div>
          <input
            type="number"
            className="risk-input"
            value={stopLoss}
            onChange={e => setStopLoss(Number(e.target.value) || 0)}
            min={0}
            max={100}
            step={0.1}
          />
        </div>

        {/* Séparateur */}
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }}>
          <div className="text-xs dim" style={{ textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
            Résultats
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <ResultRow
              label="Taille position"
              value={`${positionSize.toLocaleString('fr-FR')} USDT`}
              tooltip="= Capital × Levier"
            />
            <ResultRow
              label="Perte max (SL)"
              value={`${maxLoss.toFixed(2)} USDT`}
              color="var(--red)"
              tooltip="= Capital × (Stop Loss % / 100)"
            />
            <ResultRow
              label="Distance liquidation"
              value={`${liqDistance.toFixed(2)}%`}
              color={liqDistance < 2 ? 'var(--red)' : liqDistance < 5 ? 'var(--orange)' : 'var(--accent)'}
              tooltip="= 100 / Levier. À x20 = 5%. Minimum recommandé : 2-3%"
            />
            <ResultRow
              label="Ratio SL / Liq"
              value={`${riskReward}x`}
              color="var(--text-secondary)"
              tooltip="Marge de sécurité entre le SL et la liquidation. Plus élevé = plus sûr"
            />
          </div>

          {/* Alerte si distance de liquidation faible */}
          {liqDistance < 2 && (
            <div style={{
              marginTop: 10,
              padding: '6px 10px',
              background: 'var(--red-dim)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--red)',
              fontSize: 11,
              fontWeight: 500,
            }}>
              Attention : distance de liquidation tres faible ({liqDistance.toFixed(2)}%)
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ResultRow({ label, value, color, tooltip }) {
  const row = (
    <div className="flex-between" style={{ fontSize: 12 }}>
      <span className="muted">{label}</span>
      <span className="mono" style={{ fontWeight: 600, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
  if (!tooltip) return row
  return <Tooltip content={tooltip} inline={false}>{row}</Tooltip>
}
