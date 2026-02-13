import { useState, useCallback, useEffect, useRef } from 'react'
import Header from './components/Header'
import Scanner from './components/Scanner'
import Heatmap from './components/Heatmap'
import RiskCalc from './components/RiskCalc'
import ResearchPage from './components/ResearchPage'
import ActivePositions from './components/ActivePositions'
import CollapsibleCard from './components/CollapsibleCard'
import ExecutorPanel from './components/ExecutorPanel'
import SessionStats from './components/SessionStats'
import EquityCurve from './components/EquityCurve'
import ActivityFeed from './components/ActivityFeed'
import TradeHistory from './components/TradeHistory'
import ArenaRankingMini from './components/ArenaRankingMini'
import { useWebSocket } from './hooks/useWebSocket'

const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/live`

const TABS = [
  { id: 'scanner', label: 'Scanner' },
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'risk', label: 'Risque' },
  { id: 'research', label: 'Recherche' },
]

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

export default function App() {
  const [activeTab, setActiveTab] = useState('scanner')
  const { lastMessage, connected } = useWebSocket(wsUrl)
  const [sidebarWidth, setSidebarWidth] = useState(loadSidebarWidth)
  const dragging = useRef(false)
  const containerRef = useRef(null)

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

  // Summaries pour les sections collapsibles
  const wsData = lastMessage
  const killSwitch = wsData?.kill_switch || false
  const simSummary = SessionStats.getSummary(wsData)
  const execSummary = ExecutorPanel.getSummary(wsData)
  const arenaSummary = ArenaRankingMini.getSummary(wsData)
  const tradeCount = wsData?.strategies
    ? Object.values(wsData.strategies).reduce((sum, s) => sum + (s.total_trades || 0), 0)
    : 0

  return (
    <div className="app">
      <Header
        wsConnected={connected}
        tabs={TABS}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
      <div
        className="main-grid"
        ref={containerRef}
        style={{ gridTemplateColumns: `1fr auto ${sidebarWidth}px` }}
      >
        <div className="content">
          <ActivePositions wsData={wsData} />
          {activeTab === 'scanner' && <Scanner wsData={wsData} />}
          {activeTab === 'heatmap' && <Heatmap />}
          {activeTab === 'risk' && <RiskCalc />}
          {activeTab === 'research' && <ResearchPage />}
        </div>

        <div className="resize-handle" onMouseDown={onMouseDown} />

        <aside className="sidebar">
          <CollapsibleCard
            title="Executor"
            summary={execSummary}
            defaultOpen={true}
            storageKey="executor"
          >
            <ExecutorPanel wsData={wsData} />
          </CollapsibleCard>

          <CollapsibleCard
            title="Simulator (Paper)"
            summary={simSummary}
            defaultOpen={false}
            storageKey="simulator"
            cardClassName={killSwitch ? 'card--kill-switch' : ''}
          >
            <SessionStats wsData={wsData} />
          </CollapsibleCard>

          <CollapsibleCard
            title="Equity Curve"
            defaultOpen={true}
            storageKey="equity"
          >
            <EquityCurve />
          </CollapsibleCard>

          <CollapsibleCard
            title="Activité"
            defaultOpen={true}
            storageKey="activity"
          >
            <ActivityFeed wsData={wsData} />
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
            <ArenaRankingMini wsData={wsData} />
          </CollapsibleCard>
        </aside>
      </div>
    </div>
  )
}
