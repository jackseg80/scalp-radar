import { useState } from 'react'
import Header from './components/Header'
import Scanner from './components/Scanner'
import Heatmap from './components/Heatmap'
import RiskCalc from './components/RiskCalc'
import ExecutorPanel from './components/ExecutorPanel'
import SessionStats from './components/SessionStats'
import EquityCurve from './components/EquityCurve'
import AlertFeed from './components/AlertFeed'
import TradeHistory from './components/TradeHistory'
import ArenaRankingMini from './components/ArenaRankingMini'
import { useWebSocket } from './hooks/useWebSocket'

const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/live`

const TABS = [
  { id: 'scanner', label: 'Scanner' },
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'risk', label: 'Risque' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('scanner')
  const { lastMessage, connected } = useWebSocket(wsUrl)

  return (
    <div className="app">
      <Header
        wsConnected={connected}
        tabs={TABS}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
      <div className="main-grid">
        <div className="content">
          {activeTab === 'scanner' && <Scanner wsData={lastMessage} />}
          {activeTab === 'heatmap' && <Heatmap />}
          {activeTab === 'risk' && <RiskCalc />}
        </div>
        <aside className="sidebar">
          <ExecutorPanel wsData={lastMessage} />
          <SessionStats wsData={lastMessage} />
          <EquityCurve />
          <AlertFeed wsData={lastMessage} />
          <TradeHistory />
          <ArenaRankingMini wsData={lastMessage} />
        </aside>
      </div>
    </div>
  )
}
