import { useState, useCallback, useEffect, useRef } from 'react'
import Header from './components/Header'
import Scanner from './components/Scanner'
import ResearchPage from './components/ResearchPage'
import ExplorerPage from './components/ExplorerPage'
import PortfolioPage from './components/PortfolioPage'
import JournalPage from './components/JournalPage'
import LogViewer from './components/LogViewer'
import CollapsibleCard from './components/CollapsibleCard'
import ExecutorPanel from './components/ExecutorPanel'
import SessionStats from './components/SessionStats'
import EquityCurve from './components/EquityCurve'
import ActivityFeed from './components/ActivityFeed'
import LogMini from './components/LogMini'
import TradeHistory from './components/TradeHistory'
import ArenaRankingMini from './components/ArenaRankingMini'
import OverviewPage from './components/OverviewPage'
import StrategyEvalBar from './components/StrategyEvalBar'
import { useWebSocket } from './hooks/useWebSocket'
import { usePersistedState } from './hooks/usePersistedState'
import { StrategyProvider, useStrategyContext } from './contexts/StrategyContext'
import useFilteredWsData from './hooks/useFilteredWsData'

const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/live`

const TABS = [
  { id: 'scanner', label: 'Scanner' },
  { id: 'research', label: 'Recherche' },
  { id: 'explorer', label: 'Explorer' },
  { id: 'portfolio', label: 'Portfolio' },
  { id: 'journal', label: 'Journal' },
  { id: 'logs', label: 'Logs' },
]

// Tabs qui ne sont PAS filtrés par stratégie
const UNFILTERED_TABS = new Set(['research', 'explorer', 'portfolio', 'journal', 'logs'])

const SIDEBAR_MIN = 280
const SIDEBAR_MAX_PCT = 0.50
const SIDEBAR_DEFAULT = 340

function loadSidebarWidth() {
  const saved = localStorage.getItem('scalp-radar-sidebar-width')
  if (saved) {
    const n = parseInt(saved, 10)
    if (n >= SIDEBAR_MIN && n <= 900) return n
  }
  return SIDEBAR_DEFAULT
}

function loadActiveTab() {
  const saved = localStorage.getItem('scalp-radar-active-tab')
  const validTabs = ['scanner', 'research', 'explorer', 'portfolio', 'journal', 'logs']
  if (saved && validTabs.includes(saved)) return saved
  return 'scanner'
}

function AppContent() {
  const [activeTab, setActiveTab] = useState(loadActiveTab)
  const { lastUpdate, lastEvent, logAlerts, connected } = useWebSocket(wsUrl)
  const [sidebarWidth, setSidebarWidth] = useState(loadSidebarWidth)
  const [unseenLogErrors, setUnseenLogErrors] = useState(0)
  const dragging = useRef(false)
  const containerRef = useRef(null)
  const prevLogCountRef = useRef(0)

  const { activeStrategy, strategyFilter } = useStrategyContext()
  const filteredWsData = useFilteredWsData(lastUpdate, strategyFilter)
  const [evalStrategy, setEvalStrategy] = usePersistedState('eval-strategy', '')

  // Sauvegarder l'onglet actif dans localStorage
  useEffect(() => {
    localStorage.setItem('scalp-radar-active-tab', activeTab)
  }, [activeTab])

  // Compter les erreurs non vues quand on n'est pas sur l'onglet Logs
  useEffect(() => {
    if (logAlerts.length > prevLogCountRef.current && activeTab !== 'logs') {
      setUnseenLogErrors(prev => prev + (logAlerts.length - prevLogCountRef.current))
    }
    prevLogCountRef.current = logAlerts.length
  }, [logAlerts.length, activeTab])

  // Reset au clic sur l'onglet Logs
  const handleTabChange = useCallback((tab) => {
    setActiveTab(tab)
    if (tab === 'logs') {
      setUnseenLogErrors(0)
    }
  }, [])

  // Resize handler
  const onMouseDown = useCallback((e) => {
    e.preventDefault()
    dragging.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  useEffect(() => {
    const onMouseMove = (e) => {
      if (!dragging.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const totalWidth = rect.width
      const newSidebar = Math.max(
        SIDEBAR_MIN,
        Math.min(totalWidth * SIDEBAR_MAX_PCT, rect.right - e.clientX)
      )
      setSidebarWidth(Math.round(newSidebar))
    }

    const onMouseUp = () => {
      if (dragging.current) {
        dragging.current = false
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
        setSidebarWidth(prev => {
          localStorage.setItem('scalp-radar-sidebar-width', String(prev))
          return prev
        })
      }
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [])

  // Données filtrées pour les tabs filtrables, brutes pour les autres
  const isFilteredTab = !UNFILTERED_TABS.has(activeTab)
  const wsData = isFilteredTab ? filteredWsData : lastUpdate

  // Summaries pour les sections collapsibles (utilisent les données filtrées)
  const killSwitch = filteredWsData?.kill_switch || false
  const simSummary = SessionStats.getSummary(filteredWsData)
  const execSummary = ExecutorPanel.getSummary(filteredWsData)
  const arenaSummary = ArenaRankingMini.getSummary(lastUpdate)
  const logSummary = LogMini.getSummary(logAlerts)
  const tradeCount = filteredWsData?.strategies
    ? Object.values(filteredWsData.strategies).reduce((sum, s) => sum + (s.total_trades || 0), 0)
    : 0

  // Scanner vs OverviewPage
  const showOverview = activeTab === 'scanner' && activeStrategy === 'overview'

  return (
    <div className="app">
      <Header
        wsConnected={connected}
        tabs={TABS}
        activeTab={activeTab}
        onTabChange={handleTabChange}
        unseenLogErrors={unseenLogErrors}
        wsData={lastUpdate}
      />
      <StrategyEvalBar
        activeTab={activeTab}
        onNavigate={handleTabChange}
        evalStrategy={evalStrategy}
        setEvalStrategy={setEvalStrategy}
      />
      <div
        className="main-grid"
        ref={containerRef}
        style={{ gridTemplateColumns: `1fr auto ${sidebarWidth}px` }}
      >
        <div className="content">
          {activeTab === 'scanner' && (showOverview
            ? <OverviewPage wsData={lastUpdate} />
            : <Scanner wsData={wsData} />
          )}
          {activeTab === 'research' && (
            <ResearchPage
              onTabChange={handleTabChange}
              evalStrategy={evalStrategy}
              setEvalStrategy={setEvalStrategy}
            />
          )}
          {activeTab === 'explorer' && <ExplorerPage wsData={lastUpdate} lastEvent={lastEvent} />}
          {activeTab === 'portfolio' && (
            <PortfolioPage
              wsData={lastUpdate}
              lastEvent={lastEvent}
              evalStrategy={evalStrategy}
            />
          )}
          {activeTab === 'journal' && <JournalPage wsData={lastUpdate} onTabChange={handleTabChange} />}
          {activeTab === 'logs' && <LogViewer />}
        </div>

        <div className="resize-handle" onMouseDown={onMouseDown} />

        <aside className="sidebar">
          <CollapsibleCard
            title="Executor"
            summary={execSummary}
            defaultOpen={true}
            storageKey="executor"
          >
            <ExecutorPanel wsData={filteredWsData} />
          </CollapsibleCard>

          <CollapsibleCard
            title="Simulator (Paper)"
            summary={simSummary}
            defaultOpen={false}
            storageKey="simulator"
            cardClassName={killSwitch ? 'card--kill-switch' : ''}
          >
            <SessionStats wsData={filteredWsData} />
          </CollapsibleCard>

          <CollapsibleCard
            title="Equity Curve"
            defaultOpen={true}
            storageKey="equity"
          >
            <EquityCurve strategyFilter={strategyFilter} />
          </CollapsibleCard>

          <CollapsibleCard
            title="Activité"
            defaultOpen={true}
            storageKey="activity"
          >
            <ActivityFeed wsData={filteredWsData} onTabChange={handleTabChange} />
          </CollapsibleCard>

          <CollapsibleCard
            title="Log Alerts"
            summary={logSummary}
            defaultOpen={true}
            storageKey="log-alerts"
          >
            <LogMini logAlerts={logAlerts} onTabChange={handleTabChange} />
          </CollapsibleCard>

          <CollapsibleCard
            title="Trades Récents"
            summary={tradeCount > 0 ? `${tradeCount} trades` : null}
            defaultOpen={false}
            storageKey="trades"
          >
            <TradeHistory />
          </CollapsibleCard>

          <CollapsibleCard
            title="Arena"
            summary={arenaSummary}
            defaultOpen={false}
            storageKey="arena"
          >
            <ArenaRankingMini wsData={lastUpdate} />
          </CollapsibleCard>
        </aside>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <StrategyProvider>
      <AppContent />
    </StrategyProvider>
  )
}
