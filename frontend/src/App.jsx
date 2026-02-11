import Header from './components/Header'
import ArenaRanking from './components/ArenaRanking'
import SignalFeed from './components/SignalFeed'
import SessionStats from './components/SessionStats'
import TradeHistory from './components/TradeHistory'
import { useWebSocket } from './hooks/useWebSocket'

const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/live`

export default function App() {
  const { lastMessage, connected } = useWebSocket(wsUrl)

  return (
    <div className="app">
      <Header wsConnected={connected} />
      <div className="main-grid">
        <div className="content">
          <ArenaRanking />
          <SignalFeed wsData={lastMessage} />
          <TradeHistory />
        </div>
        <aside>
          <SessionStats />
        </aside>
      </div>
    </div>
  )
}
