import { useState, useEffect, useCallback, useMemo } from 'react'
import EquityCurveSVG from './EquityCurveSVG'
import DrawdownChart from './DrawdownChart'
import PortfolioCompare from './PortfolioCompare'
import { usePersistedState } from '../hooks/usePersistedState'
import './PortfolioPage.css'

const API = ''

function useApi(url) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`${API}${url}`)
      if (res.ok) setData(await res.json())
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [url])
  useEffect(() => { fetchData() }, [fetchData])
  return { data, loading, refetch: fetchData }
}

// ─── Composants internes ────────────────────────────────────────────────

function MetricsGrid({ run }) {
  const metrics = [
    { label: 'Return', value: `${run.total_return_pct >= 0 ? '+' : ''}${run.total_return_pct.toFixed(1)}%`, positive: run.total_return_pct >= 0 },
    { label: 'Equity finale', value: `${run.final_equity.toLocaleString('fr-FR', { maximumFractionDigits: 0 })} $` },
    { label: 'Trades', value: run.total_trades },
    { label: 'Win rate', value: `${run.win_rate.toFixed(1)}%` },
    { label: 'Max drawdown', value: `${run.max_drawdown_pct.toFixed(1)}%`, negative: run.max_drawdown_pct < -5 },
    { label: 'Margin peak', value: `${(run.peak_margin_ratio * 100).toFixed(0)}%` },
    { label: 'Kill switch', value: run.kill_switch_triggers },
    { label: 'Levier', value: run.leverage ? `${run.leverage}x` : '—' },
    { label: 'P&L réalisé', value: `${run.realized_pnl >= 0 ? '+' : ''}${run.realized_pnl.toFixed(0)} $`, positive: run.realized_pnl >= 0 },
  ]
  return (
    <div className="metrics-grid">
      {metrics.map(m => (
        <div key={m.label} className="metric-card">
          <div className="metric-label">{m.label}</div>
          <div className={`metric-value ${m.positive ? 'pnl-pos' : m.negative ? 'pnl-neg' : ''}`}>
            {m.value}
          </div>
        </div>
      ))}
    </div>
  )
}

function BenchmarkBTC({ run }) {
  const btcReturn = run.btc_benchmark_return_pct
  if (btcReturn == null) return null

  const alpha = run.alpha_vs_btc ?? 0
  const btcDD = run.btc_benchmark_max_dd_pct ?? 0
  const btcSharpe = run.btc_benchmark_sharpe ?? 0
  const outperforms = alpha >= 0

  return (
    <div style={{ marginTop: 12, padding: '12px 16px', background: '#111', borderRadius: 8, border: '1px solid #333' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: '#aaa' }}>Benchmark BTC Buy-Hold</span>
        <span style={{
          fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
          background: outperforms ? '#065f4620' : '#7f1d1d20',
          color: outperforms ? '#10b981' : '#ef4444',
          border: `1px solid ${outperforms ? '#10b98140' : '#ef444440'}`,
        }}>
          {outperforms ? '▲' : '▼'} {outperforms ? '+' : ''}{alpha.toFixed(1)}% alpha
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
        {[
          {
            label: 'Return BTC',
            value: `${btcReturn >= 0 ? '+' : ''}${btcReturn.toFixed(1)}%`,
            sub: `Portfolio : ${run.total_return_pct >= 0 ? '+' : ''}${run.total_return_pct.toFixed(1)}%`,
            color: btcReturn >= 0 ? '#10b981' : '#ef4444',
          },
          {
            label: 'Max DD BTC',
            value: `${btcDD.toFixed(1)}%`,
            sub: `Portfolio : ${run.max_drawdown_pct.toFixed(1)}%`,
            color: '#f59e0b',
          },
          {
            label: 'Sharpe BTC',
            value: btcSharpe.toFixed(2),
            sub: null,
            color: '#888',
          },
        ].map(item => (
          <div key={item.label} style={{ background: '#1a1a1a', borderRadius: 6, padding: '8px 10px' }}>
            <div style={{ fontSize: 11, color: '#666', marginBottom: 3 }}>{item.label}</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: item.color, fontFamily: 'monospace' }}>{item.value}</div>
            {item.sub && <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>{item.sub}</div>}
          </div>
        ))}
      </div>
    </div>
  )
}

function AssetTable({ data }) {
  if (!data) return null
  const sorted = Object.entries(data).sort((a, b) => (b[1].net_pnl || 0) - (a[1].net_pnl || 0))
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="asset-table">
        <thead>
          <tr>
            <th>Asset</th>
            <th>Trades</th>
            <th>Win Rate</th>
            <th>P&L Net</th>
            <th>P&L Réalisé</th>
            <th>Force-closed</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(([symbol, s]) => (
            <tr key={symbol}>
              <td style={{ fontWeight: 600 }}>{symbol}</td>
              <td className="mono">{s.trades}</td>
              <td className="mono">{s.win_rate.toFixed(1)}%</td>
              <td className={`mono ${s.net_pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
                {s.net_pnl >= 0 ? '+' : ''}{s.net_pnl.toFixed(2)} $
              </td>
              <td className="mono">{s.realized_pnl.toFixed(2)} $</td>
              <td className="mono muted">{s.force_closed_trades}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Page principale ────────────────────────────────────────────────────

export default function PortfolioPage({ wsData, lastEvent, evalStrategy }) {
  // Presets
  const { data: presetsData } = useApi('/api/portfolio/presets')
  const presets = presetsData?.presets || []

  // Stratégie sélectionnée
  const [selectedStrategy, setSelectedStrategy] = usePersistedState('portfolio-strategy', 'grid_atr')
  const [strategies, setStrategies] = useState([])

  // Fetch strategies list
  useEffect(() => {
    fetch('/api/optimization/strategies')
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data?.strategies) setStrategies(data.strategies) })
      .catch(() => {})
  }, [])

  // Config (persisté)
  const [activePreset, setActivePreset] = usePersistedState('portfolio-preset', 'balanced')
  const [capital, setCapital] = usePersistedState('portfolio-capital', 5000)
  const [days, setDays] = usePersistedState('portfolio-days', 90)
  const [assetsMode, setAssetsMode] = usePersistedState('portfolio-assets-mode', 'auto')
  const [selectedAssetsArray, setSelectedAssetsArray] = usePersistedState('portfolio-selected-assets', [])
  const selectedAssets = useMemo(() => new Set(selectedAssetsArray), [selectedAssetsArray])
  const setSelectedAssets = useCallback((setOrUpdater) => {
    if (typeof setOrUpdater === 'function') {
      setSelectedAssetsArray(prev => Array.from(setOrUpdater(new Set(prev))))
    } else {
      setSelectedAssetsArray(Array.from(setOrUpdater))
    }
  }, [setSelectedAssetsArray])
  const [killSwitchPct, setKillSwitchPct] = usePersistedState('portfolio-killswitch', 30)
  const [leverage, setLeverage] = usePersistedState('portfolio-leverage', null)  // null = auto depuis strategies.yaml
  const [label, setLabel] = usePersistedState('portfolio-label', '')

  // Sélection runs (persisté)
  const [selectedId, setSelectedId] = usePersistedState('portfolio-selected-id', null)
  const [compareIdsArray, setCompareIdsArray] = usePersistedState('portfolio-compare-ids', [])
  const compareIds = useMemo(() => new Set(compareIdsArray), [compareIdsArray])
  const setCompareIds = useCallback((setOrUpdater) => {
    if (typeof setOrUpdater === 'function') {
      setCompareIdsArray(prev => Array.from(setOrUpdater(new Set(prev))))
    } else {
      setCompareIdsArray(Array.from(setOrUpdater))
    }
  }, [setCompareIdsArray])

  // Job (non-persisté - état temporaire)
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressPhase, setProgressPhase] = useState('')

  // Data (non-persisté - rechargé depuis API)
  const [backtests, setBacktests] = useState([])
  const [detail, setDetail] = useState(null)
  const [compareDetails, setCompareDetails] = useState([])

  // Charger la liste des backtests
  const fetchBacktests = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/portfolio/backtests?limit=50`)
      if (res.ok) {
        const data = await res.json()
        setBacktests(data.backtests || [])
      }
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { fetchBacktests() }, [fetchBacktests])

  // Charger le détail d'un run
  useEffect(() => {
    if (selectedId == null) { setDetail(null); return }
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch(`${API}/api/portfolio/backtests/${selectedId}`)
        if (res.ok && !cancelled) setDetail(await res.json())
      } catch { /* ignore */ }
    })()
    return () => { cancelled = true }
  }, [selectedId])

  // Baseline grid_atr (dernier run, pour comparaison auto)
  const baselineRun = useMemo(() => {
    if (!backtests || backtests.length === 0) return null
    return backtests.find(b => b.strategy_name === 'grid_atr') || null
  }, [backtests])

  // IDs compare effectifs (ajouter baseline si nécessaire)
  const effectiveCompareIds = useMemo(() => {
    const ids = new Set(compareIds)
    if (baselineRun && ids.size >= 1 && selectedStrategy !== 'grid_atr' && !ids.has(baselineRun.id)) {
      ids.add(baselineRun.id)
    }
    return ids
  }, [compareIds, baselineRun, selectedStrategy])

  // Charger les détails pour la comparaison
  useEffect(() => {
    if (effectiveCompareIds.size < 2) { setCompareDetails([]); return }
    let cancelled = false
    ;(async () => {
      const idsStr = Array.from(effectiveCompareIds).join(',')
      try {
        const res = await fetch(`${API}/api/portfolio/compare?ids=${idsStr}`)
        if (res.ok && !cancelled) {
          const data = await res.json()
          setCompareDetails(data.runs || [])
        }
      } catch { /* ignore */ }
    })()
    return () => { cancelled = true }
  }, [effectiveCompareIds])

  // WebSocket progress (via lastEvent, séparé de wsData type=update)
  useEffect(() => {
    if (!lastEvent) return
    if (lastEvent.type === 'portfolio_progress') {
      setIsRunning(true)
      setProgress(lastEvent.progress_pct || 0)
      setProgressPhase(lastEvent.phase || '')
    }
    if (lastEvent.type === 'portfolio_completed') {
      setIsRunning(false)
      setProgress(100)
      fetchBacktests()
      // Auto-sélectionner le nouveau run
      if (lastEvent.result_id) setSelectedId(lastEvent.result_id)
    }
    if (lastEvent.type === 'portfolio_failed') {
      setIsRunning(false)
      setProgress(0)
    }
  }, [lastEvent, fetchBacktests])

  // Appliquer un preset
  const applyPreset = useCallback((preset) => {
    setActivePreset(preset.name)
    setCapital(preset.capital)
    setDays(preset.days)
    if (preset.assets) {
      setAssetsMode('custom')
      setSelectedAssets(new Set(preset.assets))
    } else {
      setAssetsMode('auto')
      setSelectedAssets(new Set())
    }
  }, [])

  // Lancer un backtest (overrides optionnels pour Forward Test)
  const launchBacktest = useCallback(async (overrides = {}) => {
    setIsRunning(true)
    setProgress(0)
    try {
      const body = {
        strategy_name: selectedStrategy || 'grid_atr',
        initial_capital: capital,
        days: overrides.days || days,
        assets: assetsMode === 'custom' && selectedAssets.size > 0
          ? Array.from(selectedAssets)
          : null,
        kill_switch_pct: killSwitchPct,
        leverage: leverage || null,
        label: overrides.label || label || activePreset || undefined,
      }
      const res = await fetch(`${API}/api/portfolio/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json()
        alert(err.detail || 'Erreur')
        setIsRunning(false)
      }
    } catch (e) {
      alert('Erreur réseau: ' + e.message)
      setIsRunning(false)
    }
  }, [capital, days, assetsMode, selectedAssets, killSwitchPct, leverage, label, activePreset, selectedStrategy])

  // Supprimer un run
  const deleteRun = useCallback(async (id, e) => {
    e.stopPropagation()
    if (!confirm('Supprimer ce run ?')) return
    try {
      await fetch(`${API}/api/portfolio/backtests/${id}`, { method: 'DELETE' })
      fetchBacktests()
      if (selectedId === id) setSelectedId(null)
      setCompareIds(prev => { const n = new Set(prev); n.delete(id); return n })
    } catch { /* ignore */ }
  }, [selectedId, fetchBacktests])

  // Toggle comparaison
  const toggleCompare = useCallback((id, e) => {
    e.stopPropagation()
    setCompareIds(prev => {
      const n = new Set(prev)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })
  }, [])

  // Equity curves pour le chart (+ overlay BTC si disponible)
  const equityCurves = useMemo(() => {
    if (!detail?.equity_curve) return []
    const curves = [{
      label: detail.label || `Run #${detail.id}`,
      points: detail.equity_curve,
      initialCapital: detail.initial_capital,
    }]
    if (detail.btc_equity_curve && detail.btc_equity_curve.length > 0) {
      curves.push({
        label: 'BTC Buy-Hold',
        color: '#f59e0b',
        dashed: true,
        points: detail.btc_equity_curve,
        initialCapital: detail.initial_capital,
      })
    }
    return curves
  }, [detail])

  // Equity curves pour la comparaison
  const compareCurves = useMemo(() => {
    if (compareDetails.length < 2) return []
    const colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
    return compareDetails.map((r, i) => ({
      label: r.label || `Run #${r.id}`,
      color: colors[i % colors.length],
      points: r.equity_curve || [],
      initialCapital: r.initial_capital,
    }))
  }, [compareDetails])

  // Filtrer les backtests par stratégie
  const filteredBacktests = useMemo(() => {
    if (!selectedStrategy) return backtests
    return backtests.filter(b => !b.strategy_name || b.strategy_name === selectedStrategy)
  }, [backtests, selectedStrategy])

  // Assets disponibles : watched_symbols de la stratégie en priorité, sinon presets
  const allAssets = useMemo(() => {
    const fromWs = wsData?.strategies?.[selectedStrategy]?.watched_symbols
    if (fromWs && fromWs.length > 0) return [...fromWs].sort()
    const set = new Set()
    for (const p of presets) {
      if (p.assets) p.assets.forEach(a => set.add(a))
    }
    return Array.from(set).sort()
  }, [presets, wsData, selectedStrategy])

  // Collapse des runs précédents (réduit par défaut)
  const [runsCollapsed, setRunsCollapsed] = useState(true)

  return (
    <div className="portfolio-page">
      <div className="portfolio-main">
        {/* ─── Config Panel (gauche) ─── */}
        <div className="portfolio-config">
          <h3>Portfolio Backtest</h3>

          {/* Stratégie */}
          <div className="pf-form-group">
            <label>Strategie</label>
            <select className="pf-input" value={selectedStrategy}
              onChange={e => setSelectedStrategy(e.target.value)}>
              {strategies.length === 0 && (
                <option value="grid_atr">grid_atr</option>
              )}
              {strategies.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Presets */}
          <h4>Presets</h4>
          <div className="preset-group">
            {presets.map(p => (
              <label
                key={p.name}
                className={`preset-card ${activePreset === p.name ? 'active' : ''}`}
                onClick={() => applyPreset(p)}
              >
                <input type="radio" name="preset" value={p.name} checked={activePreset === p.name} readOnly />
                <div className="preset-label">{p.label}</div>
                <div className="preset-desc">{p.description}</div>
                <div className="preset-meta">
                  {p.capital.toLocaleString()}$ | {p.assets ? `${p.assets.length} assets` : 'Tous'} | {p.days}j
                </div>
              </label>
            ))}
          </div>

          <div className="divider" />

          {/* Capital */}
          <div className="pf-form-group">
            <label>Capital ($)</label>
            <input type="number" className="pf-input" value={capital}
              onChange={e => { setCapital(+e.target.value); setActivePreset('custom') }}
              min={100} max={100000} step={500} />
          </div>

          {/* Période */}
          <div className="pf-form-group">
            <label>Période (jours)</label>
            <div className="pf-row">
              <input type="range" className="pf-input" style={{ padding: 0, height: 6 }}
                min={30} max={365} value={days}
                onChange={e => { setDays(+e.target.value); setActivePreset('custom') }} />
              <span className="mono">{days}j</span>
            </div>
          </div>

          {/* Kill switch */}
          <div className="pf-form-group">
            <label>Kill switch (%)</label>
            <input type="number" className="pf-input" value={killSwitchPct}
              onChange={e => setKillSwitchPct(+e.target.value)}
              min={5} max={50} step={5} />
          </div>

          {/* Leverage */}
          <div className="pf-form-group">
            <label>Levier{leverage ? ` — ${leverage}x` : ' — auto (YAML)'}</label>
            <div className="pf-row">
              <input type="range" className="pf-input" style={{ padding: 0, height: 6 }}
                min={1} max={20} value={leverage || 6}
                onChange={e => setLeverage(+e.target.value)} />
              <button
                style={{ fontSize: 11, padding: '2px 6px', marginLeft: 4, cursor: 'pointer' }}
                onClick={() => setLeverage(null)}
                title="Réinitialiser (utilise le levier de strategies.yaml)"
              >auto</button>
            </div>
          </div>

          {/* Assets mode */}
          <div className="pf-form-group">
            <label>Assets</label>
            <select className="pf-input" value={assetsMode}
              onChange={e => { setAssetsMode(e.target.value); setActivePreset('custom') }}>
              <option value="auto">Automatique (tous per_asset)</option>
              <option value="custom">Sélection manuelle</option>
            </select>
          </div>

          {assetsMode === 'custom' && (
            <div className="assets-list">
              {allAssets.map(symbol => (
                <label key={symbol} className="asset-checkbox">
                  <input type="checkbox"
                    checked={selectedAssets.has(symbol)}
                    onChange={() => {
                      setSelectedAssets(prev => {
                        const n = new Set(prev)
                        if (n.has(symbol)) n.delete(symbol)
                        else n.add(symbol)
                        return n
                      })
                      setActivePreset('custom')
                    }} />
                  <span>{symbol}</span>
                </label>
              ))}
            </div>
          )}

          {/* Label */}
          <div className="pf-form-group">
            <label>Label (optionnel)</label>
            <input type="text" className="pf-input" value={label}
              onChange={e => setLabel(e.target.value)}
              placeholder={activePreset !== 'custom' ? activePreset : ''} />
          </div>

          {/* Bouton Lancer */}
          <button
            className="pf-launch-btn"
            onClick={launchBacktest}
            disabled={isRunning || (assetsMode === 'custom' && selectedAssets.size === 0)}
          >
            {isRunning ? `En cours... ${progress.toFixed(0)}%` : 'Lancer le backtest'}
          </button>
          <button
            className="pf-forward-btn"
            onClick={() => launchBacktest({ days: 365, label: `${selectedStrategy}_forward_365` })}
            disabled={isRunning || !selectedStrategy}
            title="Lance un backtest 365 jours avec les parametres actuels de strategies.yaml (appliquer les A/B depuis la page Recherche au prealable)"
          >
            {'\u26A1'} Forward Test 365j
          </button>
          {isRunning && (
            <>
              <div className="pf-progress">
                <div className="pf-progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <div className="pf-progress-text">{progressPhase}</div>
            </>
          )}

          <div className="divider" />

          {/* Runs précédents (collapsible) */}
          <h4
            className="runs-header"
            onClick={() => setRunsCollapsed(v => !v)}
          >
            <span>Runs précédents ({filteredBacktests.length})</span>
            <span className="runs-collapse-icon">{runsCollapsed ? '▶' : '▼'}</span>
          </h4>
          {!runsCollapsed && (
            <div className="runs-history">
              {filteredBacktests.map(run => (
                <div
                  key={run.id}
                  className={`run-item ${selectedId === run.id ? 'selected' : ''}`}
                  onClick={() => setSelectedId(run.id)}
                >
                  <input
                    type="checkbox"
                    className="run-compare-check"
                    checked={compareIds.has(run.id)}
                    onChange={e => toggleCompare(run.id, e)}
                    title="Comparer"
                  />
                  <div>
                    <div className="run-label">{run.label || `Run #${run.id}`}</div>
                    <div className="run-meta">
                      {run.initial_capital.toLocaleString()}$ | {run.n_assets} assets | {run.period_days}j{run.leverage ? ` | ${run.leverage}x` : ''}
                    </div>
                  </div>
                  <span className={`run-pnl ${run.total_return_pct >= 0 ? 'pnl-pos' : 'pnl-neg'}`}>
                    {run.total_return_pct >= 0 ? '+' : ''}{run.total_return_pct.toFixed(1)}%
                  </span>
                  <button className="run-delete" onClick={e => deleteRun(run.id, e)} title="Supprimer">
                    &times;
                  </button>
                </div>
              ))}
              {filteredBacktests.length === 0 && (
                <div className="muted" style={{ fontSize: 12, textAlign: 'center', padding: 12 }}>
                  Aucun run sauvegardé
                </div>
              )}
            </div>
          )}
        </div>

        {/* ─── Résultats (droite) ─── */}
        <div className="portfolio-results">
          {!detail && (
            <div className="empty-state">
              <p style={{ fontSize: 24 }}>&#128202;</p>
              <p>Sélectionnez un preset et lancez un backtest,</p>
              <p>ou cliquez sur un run précédent.</p>
            </div>
          )}

          {detail && (
            <>
              <h3>{detail.label || `Run #${detail.id}`}</h3>

              {/* Equity curve */}
              <div>
                <div className="chart-title">Equity Curve</div>
                <EquityCurveSVG curves={equityCurves} height={300} />
              </div>

              {/* Drawdown */}
              <div>
                <div className="chart-title">Drawdown</div>
                <DrawdownChart
                  curves={equityCurves}
                  height={120}
                  killSwitchPct={detail.kill_switch_pct || 30}
                />
              </div>

              {/* Métriques */}
              <MetricsGrid run={detail} />

              {/* Benchmark BTC */}
              <BenchmarkBTC run={detail} />

              {/* Table par asset */}
              <div>
                <div className="chart-title">Performance par asset</div>
                <AssetTable data={detail.per_asset_results} />
              </div>
            </>
          )}
        </div>
      </div>

      {/* ─── Comparaison (pleine largeur) ─── */}
      {compareDetails.length >= 2 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {effectiveCompareIds.size > compareIds.size && baselineRun && (
            <div className="baseline-notice">
              <span className="baseline-notice-icon">ℹ</span>
              Baseline <strong>grid_atr</strong> (Run #{baselineRun.id} — {baselineRun.label || `${baselineRun.period_days}j`}) auto-ajoutée pour comparaison
            </div>
          )}
          <div className="compare-section">
            <h4>Comparaison — Equity Curves (% return)</h4>
            <EquityCurveSVG curves={compareCurves} height={280} />
          </div>
          <PortfolioCompare runs={compareDetails} />
        </div>
      )}
    </div>
  )
}
