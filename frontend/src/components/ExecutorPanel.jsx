/**
 * ExecutorPanel — Panneau de statut de l'executor (multi-positions).
 * Props : wsData
 * Conçu pour être wrappé par CollapsibleCard dans App.jsx.
 *
 * Cas d'affichage :
 *  - executor absent           → SIMULATION ONLY
 *  - executor.mode === 'paper' → PAPER ONLY (stratégie paper sélectionnée)
 *  - executor.enabled === true → LIVE (mode normal)
 *  - executor.enabled === false && !mode → OFF (executor inactif)
 *
 * Sprint 39 : P&L backend, TP/SL distances, durée, P&L par niveau.
 */
import { useState } from 'react'
import { useApi } from '../hooks/useApi'
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

  // Mode PAPER ONLY (stratégie paper sélectionnée dans la StrategyBar)
  if (executor.mode === 'paper') {
    return <PaperOnlyPanel executor={executor} />
  }

  const { enabled, connected, risk_manager: rm, selector } = executor
  // Multi-positions (nouveau) avec fallback sur position (ancien)
  const positions = executor.positions || (executor.position ? [executor.position] : [])

  const isLive = enabled
  // Sprint 46 : P&L Jour + P&L Total (remplace P&L Session)
  const { data: pnlData } = useApi('/api/journal/daily-pnl-summary', 60000)
  const dailyPnl = pnlData?.daily_pnl ?? null
  const totalPnl = pnlData?.total_pnl ?? null
  const firstTradeDate = pnlData?.first_trade_date
    ? new Date(pnlData.first_trade_date).toLocaleDateString('fr-FR')
    : null
  // Solde Bitget réel en priorité, fallback sur capital initial configuré
  const balance = executor.exchange_balance ?? rm?.initial_capital

  // Stratégies paper (pour résumé en mode overview)
  const allStrategyNames = Object.keys(wsData?.strategies || {})
  const liveStrats = selector?.allowed_strategies || []
  const paperStrats = allStrategyNames.filter(s => !liveStrats.includes(s))

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
        {(() => {
          const grids = Object.values(wsData?.grid_state?.grid_positions || {})
          const lev = grids.length > 0 ? grids[0].leverage : null
          return lev != null ? <StatusRow label="Leverage" value={`x${lev}`} /> : null
        })()}
        {dailyPnl != null && (
          <StatusRow
            label="P&L Jour"
            value={`${dailyPnl >= 0 ? '+' : ''}${Number(dailyPnl).toFixed(2)}$`}
            color={dailyPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
          />
        )}
        {totalPnl != null && (
          <Tooltip content={firstTradeDate ? `Depuis le ${firstTradeDate}` : 'P&L total depuis le début des trades en DB'}>
            <StatusRow
              label="P&L Total"
              value={`${totalPnl >= 0 ? '+' : ''}${Number(totalPnl).toFixed(2)}$`}
              color={totalPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
            />
          </Tooltip>
        )}
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
        <div style={{ marginBottom: (paperStrats.length > 0 || positions.length > 0) ? 12 : 0 }}>
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

      {/* Stratégies paper (visible en mode overview quand des stratégies paper tournent) */}
      {paperStrats.length > 0 && (
        <div style={{ marginBottom: positions.length > 0 ? 12 : 0 }}>
          <div className="text-xs muted" style={{ marginBottom: 4 }}>Stratégies paper</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {paperStrats.map((s) => (
              <span key={s} className="badge badge-simulation" style={{ fontSize: 10, padding: '2px 8px' }}>
                {s}
              </span>
            ))}
          </div>
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
                // Prix : préférer backend (enrichi), fallback WS prices
                const spotSymbol = (pos.symbol || '').split(':')[0]
                const currentPrice = pos.current_price ?? wsData?.prices?.[spotSymbol]?.last
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
  if (executor.mode === 'paper') return 'PAPER'
  return executor.enabled ? 'LIVE' : 'OFF'
}

/** Vue PAPER ONLY — stratégie paper sélectionnée dans la StrategyBar */
function PaperOnlyPanel({ executor }) {
  const assets = executor.paper_assets || []
  const numPositions = executor.paper_num_positions ?? 0
  const isWarmingUp = executor.paper_is_warming_up ?? false

  // Noms courts des assets : "BTC/USDT" → "BTC"
  const assetNames = assets.map(s => s.split('/')[0])
  // Si trop long, tronquer à 6 + "…"
  const displayNames = assetNames.length > 6
    ? assetNames.slice(0, 6).join(', ') + ', …'
    : assetNames.join(', ')

  return (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div className="flex-between" style={{ marginBottom: 2 }}>
          <span className="text-xs muted">Mode</span>
          <Tooltip content="Cette stratégie tourne en paper trading (simulation). Pas d'ordres réels.">
            <span className="badge badge-simulation" style={{ fontSize: 10, padding: '2px 8px' }}>
              PAPER ONLY
            </span>
          </Tooltip>
        </div>
        <div className="flex-between" style={{ fontSize: 12 }}>
          <span className="muted">Statut</span>
          <span style={{ color: 'var(--accent)', fontWeight: 500 }}>
            {isWarmingUp ? '⏳ Warm-up' : '✅ Actif'}
          </span>
        </div>
        {assets.length > 0 && (
          <div style={{ fontSize: 12 }}>
            <div className="flex-between" style={{ marginBottom: 4 }}>
              <span className="muted">Assets</span>
              <span className="mono" style={{ fontWeight: 500 }}>{assets.length}</span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-dim)', textAlign: 'right', lineHeight: 1.4 }}>
              {displayNames}
            </div>
          </div>
        )}
        <StatusRow label="Positions" value={`${numPositions} ouverte${numPositions !== 1 ? 's' : ''}`} />
      </div>
      <div className="text-xs muted text-center" style={{ paddingTop: 12 }}>
        {isWarmingUp ? 'Warm-up en cours, patience...' : numPositions === 0 ? 'En attente de signaux...' : ''}
      </div>
    </>
  )
}

function PositionCard({ position, currentPrice }) {
  const isLong = position.direction === 'LONG'
  const leverage = position.leverage ?? null
  const notional = position.notional ?? null
  const margin = position.margin_used ?? (
    (notional != null && leverage != null && leverage > 0) ? notional / leverage : null
  )

  // P&L : préférer backend (enrichi), fallback client-side
  let unrealizedPnl = position.unrealized_pnl ?? null
  if (unrealizedPnl == null && currentPrice != null && position.entry_price != null && position.quantity != null) {
    unrealizedPnl = isLong
      ? (currentPrice - position.entry_price) * position.quantity
      : (position.entry_price - currentPrice) * position.quantity
  }
  const unrealizedPnlPct = position.unrealized_pnl_pct ?? null

  const tpPrice = position.tp_price
  const tpDistPct = position.tp_distance_pct
  const slDistPct = position.sl_distance_pct
  const durationHours = position.duration_hours

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
          {position.type === 'grid' && (
            <span className="badge" style={{ fontSize: 9, padding: '1px 6px', background: 'var(--surface)', color: 'var(--text-dim)' }}>
              {position.levels}/{position.levels_max}
            </span>
          )}
          <span className={`badge ${isLong ? 'badge-long' : 'badge-short'}`}>
            {position.direction}
          </span>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <PosRow label="Entrée" value={formatPrice(position.entry_price)} />
        {currentPrice != null && (
          <PosRow label="Prix actuel" value={formatPrice(currentPrice)} />
        )}
        {position.sl_price != null && position.sl_price > 0 && (
          <PosRow
            label="Stop Loss"
            value={`${formatPrice(position.sl_price)}${slDistPct != null ? ` (${slDistPct > 0 ? '+' : ''}${slDistPct.toFixed(1)}%)` : ''}`}
            color="var(--red)"
          />
        )}
        <PosRow
          label="Take Profit"
          value={
            tpPrice && tpPrice > 0
              ? `${formatPrice(tpPrice)}${tpDistPct != null ? ` (${tpDistPct > 0 ? '+' : ''}${tpDistPct.toFixed(1)}%)` : ''}`
              : 'SMA dynamique'
          }
          color="var(--accent)"
        />
        {notional != null && (
          <PosRow label="Notionnel" value={`${notional.toFixed(2)} USDT`} />
        )}
        {margin != null && (
          <PosRow label="Marge" value={`${Number(margin).toFixed(2)} USDT`} />
        )}
        {unrealizedPnl != null && (
          <PosRow
            label="P&L latent"
            value={`${unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toFixed(2)}$${unrealizedPnlPct != null ? ` (${unrealizedPnlPct >= 0 ? '+' : ''}${unrealizedPnlPct.toFixed(1)}%)` : ''}`}
            color={unrealizedPnl >= 0 ? 'var(--accent)' : 'var(--red)'}
          />
        )}
        {durationHours != null && (
          <PosRow
            label="Durée"
            value={durationHours >= 24 ? `${(durationHours / 24).toFixed(1)}j` : `${durationHours.toFixed(1)}h`}
          />
        )}
      </div>

      {/* P&L par niveau pour les grids */}
      {position.type === 'grid' && position.positions?.length > 0 && (
        <div style={{ marginTop: 6, borderTop: '1px solid var(--border)', paddingTop: 4 }}>
          <div className="text-xs muted" style={{ marginBottom: 2 }}>Niveaux</div>
          {position.positions.map(p => (
            <div key={p.level} className="flex-between" style={{ fontSize: 10, padding: '1px 0' }}>
              <span className="muted">Niv.{p.level + 1} @ {formatPrice(p.entry_price)}</span>
              {p.pnl_usd != null && (
                <span className={`mono ${p.pnl_usd >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
                  {p.pnl_usd >= 0 ? '+' : ''}{p.pnl_usd.toFixed(2)}$
                  {p.pnl_pct != null && (
                    <span className="muted"> ({p.pnl_pct >= 0 ? '+' : ''}{p.pnl_pct.toFixed(1)}%)</span>
                  )}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
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
