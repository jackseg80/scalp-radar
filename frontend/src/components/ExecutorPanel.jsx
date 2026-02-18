/**
 * ExecutorPanel — Panneau de statut de l'executor (multi-positions).
 * Props : wsData
 * Conçu pour être wrappé par CollapsibleCard dans App.jsx.
 */
import { useState } from 'react'
import Tooltip from './Tooltip'
import { formatPrice } from '../utils/format'

const STORAGE_KEY = 'scalp-radar-collapse-executor-positions'

export default function ExecutorPanel({ wsData }) {
  const storedOpen = localStorage.getItem(STORAGE_KEY)
  const [positionsOpen, setPositionsOpen] = useState(storedOpen === null ? true : storedOpen === 'true')

  function togglePositions() {
    const next = !positionsOpen
    setPositionsOpen(next)
    localStorage.setItem(STORAGE_KEY, String(next))
  }

  const executor = wsData?.executor

  // Pas d'executor = mode simulation
  if (!executor) {
    return (
      <div style={{ textAlign: 'center', padding: '12px 0' }}>
        <span className="badge badge-simulation" style={{ fontSize: 12, padding: '4px 12px' }}>
          SIMULATION ONLY
        </span>
        <p className="text-xs muted" style={{ marginTop: 8 }}>
          Aucun executor actif. Les trades sont simulés.
        </p>
      </div>
    )
  }

  const { enabled, connected, risk_manager: rm, selector } = executor
  // Multi-positions (nouveau) avec fallback sur position (ancien)
  const positions = executor.positions || (executor.position ? [executor.position] : [])

  const isLive = enabled
  const sessionPnl = rm?.session_pnl ?? 0
  // Solde Bitget réel en priorité, fallback sur capital initial configuré
  const balance = executor.exchange_balance ?? rm?.initial_capital

  return (
    <>
      {/* Status global */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: positions.length > 0 || selector ? 12 : 0 }}>
        <div className="flex-between" style={{ marginBottom: 2 }}>
          <span className="text-xs muted">Mode</span>
          <Tooltip content={isLive ? 'LIVE = ordres réels Bitget mainnet' : 'OFF = executor désactivé'}>
            <span className={`badge ${isLive ? 'badge-active' : 'badge-stopped'}`}>
              {isLive ? 'LIVE' : 'OFF'}
            </span>
          </Tooltip>
        </div>
        <StatusRow
          label="Connexion"
          value={connected ? 'connecté' : 'déconnecté'}
          color={connected ? 'var(--accent)' : 'var(--red)'}
        />
        {balance != null && (
          <StatusRow
            label="Solde Bitget"
            value={`${Number(balance).toFixed(2)} USDT`}
          />
        )}
        <StatusRow
          label="P&L Session"
          value={`${sessionPnl >= 0 ? '+' : ''}${Number(sessionPnl).toFixed(2)}$`}
          color={sessionPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
        />
        {rm?.total_orders != null && rm.total_orders > 0 && (
          <StatusRow label="Ordres passés" value={rm.total_orders} />
        )}
        {rm?.kill_switch && (
          <Tooltip content="Trading stoppé : perte session ≥ 5% du capital" inline={false}>
            <div className="badge badge-stopped" style={{ textAlign: 'center', marginTop: 4, padding: '6px 8px' }}>
              KILL SWITCH LIVE ACTIF
            </div>
          </Tooltip>
        )}
      </div>

      {/* Selector — stratégies autorisées en live */}
      {selector && (
        <div style={{ marginBottom: positions.length > 0 ? 12 : 0 }}>
          <div className="text-xs muted" style={{ marginBottom: 4 }}>Stratégies live autorisées</div>
          {selector.allowed_strategies && selector.allowed_strategies.length > 0 ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {selector.allowed_strategies.map((s) => (
                <span key={s} className="badge badge-active" style={{ fontSize: 10, padding: '2px 8px' }}>
                  {s}
                </span>
              ))}
            </div>
          ) : (
            <Tooltip content="L'adaptive selector attend 3+ trades simulés profitables avant d'autoriser une stratégie en live">
              <span className="text-xs muted">aucune (en évaluation)</span>
            </Tooltip>
          )}
        </div>
      )}

      {/* Positions ouvertes */}
      {positions.length > 0 && (
        <div>
          <button
            onClick={togglePositions}
            style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'none', border: 'none', cursor: 'pointer', padding: 0, marginBottom: positionsOpen ? 8 : 0, width: '100%' }}
          >
            <span className="text-xs muted">
              {positions.length} position{positions.length > 1 ? 's' : ''} ouverte{positions.length > 1 ? 's' : ''}
            </span>
            <span className={`collapsible-arrow${positionsOpen ? ' open' : ''}`} style={{ marginLeft: 'auto', fontSize: 10 }}>▼</span>
          </button>
          {positionsOpen && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {positions.map((pos, idx) => {
                // Symbole spot pour le prix live : BTC/USDT:USDT → BTC/USDT
                const spotSymbol = (pos.symbol || '').split(':')[0]
                const currentPrice = wsData?.prices?.[spotSymbol]?.last
                return <PositionCard key={pos.symbol || idx} position={pos} currentPrice={currentPrice} />
              })}
            </div>
          )}
        </div>
      )}

      {/* Pas de position */}
      {positions.length === 0 && (
        <div className="text-xs muted text-center" style={{ paddingTop: 4 }}>
          Aucune position ouverte
        </div>
      )}
    </>
  )
}

// Summary pour CollapsibleCard
ExecutorPanel.getSummary = function(wsData) {
  const executor = wsData?.executor
  if (!executor) return 'OFF'
  return executor.enabled ? 'LIVE' : 'OFF'
}

function PositionCard({ position, currentPrice }) {
  const isLong = position.direction === 'LONG'
  const leverage = position.leverage ?? null
  const notional = position.notional ?? null
  const margin = (notional != null && leverage != null && leverage > 0)
    ? notional / leverage
    : null

  // P&L non réalisé
  let unrealizedPnl = null
  if (currentPrice != null && position.entry_price != null && position.quantity != null) {
    unrealizedPnl = isLong
      ? (currentPrice - position.entry_price) * position.quantity
      : (position.entry_price - currentPrice) * position.quantity
  }

  return (
    <div className="executor-position">
      <div className="flex-between" style={{ marginBottom: 6 }}>
        <span style={{ fontWeight: 600, fontSize: 12 }}>{position.symbol}</span>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          {position.strategy_name && (
            <span className="badge" style={{ fontSize: 9, padding: '1px 6px', opacity: 0.7 }}>
              {position.strategy_name}
            </span>
          )}
          {leverage != null && (
            <span className="badge" style={{ fontSize: 9, padding: '1px 6px', background: 'var(--surface-2)', color: 'var(--text-dim)' }}>
              x{leverage}
            </span>
          )}
          <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`}>
            {position.direction}
          </span>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <PosRow label="Entrée" value={formatPrice(position.entry_price)} />
        {position.sl_price != null && (
          <PosRow label="Stop Loss" value={formatPrice(position.sl_price)} color="var(--red)" />
        )}
        {position.tp_price != null && (
          <PosRow
            label="Take Profit"
            value={position.tp_price === 0 ? 'SMA dynamique' : formatPrice(position.tp_price)}
            color="var(--accent)"
          />
        )}
        {position.quantity != null && (
          <PosRow label="Quantité" value={position.quantity} />
        )}
        {notional != null && (
          <PosRow label="Notionnel" value={`${notional.toFixed(2)} USDT`} />
        )}
        {margin != null && (
          <PosRow label="Marge" value={`${margin.toFixed(2)} USDT`} />
        )}
        {unrealizedPnl != null && (
          <PosRow
            label="P&L latent"
            value={`${unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toFixed(2)}$`}
            color={unrealizedPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
          />
        )}
      </div>
    </div>
  )
}

function StatusRow({ label, value, color }) {
  return (
    <div className="flex-between" style={{ fontSize: 12 }}>
      <span className="muted">{label}</span>
      <span className="mono" style={{ fontWeight: 500, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
}

function PosRow({ label, value, color }) {
  return (
    <div className="flex-between" style={{ fontSize: 11 }}>
      <span className="muted">{label}</span>
      <span className="mono" style={{ fontWeight: 500, color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
}
