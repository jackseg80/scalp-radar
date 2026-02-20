/**
 * GenericGuide — Schéma SVG illustratif par type de stratégie
 * Fallback pour les stratégies sans tutoriel interactif dédié
 */

// Mapping type → schéma
const TYPE_TO_SCHEMA = {
  'Mean Reversion': 'mean-reversion',
  'Trend Following': 'trend-following',
  'Trend DCA': 'trend-following',
  'Trend + DCA': 'trend-following',
  'Trend Following DCA': 'trend-following',
  'Breakout': 'breakout',
  'Event-Driven': 'breakout',
  'Range Trading': 'range-trading',
  'Funding Arbitrage': 'funding-arbitrage',
}

function MeanReversionSVG() {
  // Prix oscillant autour d'une SMA avec flèches achat/vente aux extrêmes
  const smaY = 125
  const pricePath = 'M 40,125 C 80,125 100,70 140,65 C 180,60 200,110 240,130 C 280,150 300,185 340,190 C 380,195 400,140 440,120 C 480,100 500,60 540,55'
  const smaPath = `M 40,${smaY} L 540,${smaY}`

  return (
    <svg viewBox="0 0 600 250" width="100%" style={{ display: 'block' }}>
      <defs>
        <marker id="arrow-buy" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 10 L 5 0 L 10 10" fill="#00e68a" />
        </marker>
        <marker id="arrow-sell" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 5 10 L 10 0" fill="#ff4466" />
        </marker>
      </defs>

      {/* Zones extrêmes */}
      <rect x="40" y="40" width="500" height="35" fill="rgba(255,68,102,0.08)" rx="4" />
      <rect x="40" y="175" width="500" height="35" fill="rgba(0,230,138,0.08)" rx="4" />
      <text x="545" y="60" fill="#ff4466" fontSize="10" textAnchor="end" opacity="0.6">Zone de vente</text>
      <text x="545" y="200" fill="#00e68a" fontSize="10" textAnchor="end" opacity="0.6">Zone d'achat</text>

      {/* SMA */}
      <path d={smaPath} stroke="#ffc53d" strokeWidth="2" strokeDasharray="6,4" fill="none" />
      <text x="545" y={smaY - 8} fill="#ffc53d" fontSize="11" textAnchor="end">SMA</text>

      {/* Prix */}
      <path d={pricePath} stroke="#4da6ff" strokeWidth="2.5" fill="none" />
      <text x="545" y="48" fill="#4da6ff" fontSize="11" textAnchor="end">Prix</text>

      {/* Flèches achat (aux creux) */}
      <line x1="340" y1="220" x2="340" y2="195" stroke="#00e68a" strokeWidth="2" markerEnd="url(#arrow-buy)" />
      <text x="340" y="235" fill="#00e68a" fontSize="10" textAnchor="middle">ACHAT</text>

      {/* Flèches vente (aux pics) */}
      <line x1="140" y1="35" x2="140" y2="60" stroke="#ff4466" strokeWidth="2" markerEnd="url(#arrow-sell)" />
      <text x="140" y="28" fill="#ff4466" fontSize="10" textAnchor="middle">VENTE</text>

      <line x1="540" y1="30" x2="540" y2="50" stroke="#ff4466" strokeWidth="2" markerEnd="url(#arrow-sell)" />
      <text x="540" y="23" fill="#ff4466" fontSize="10" textAnchor="middle">VENTE</text>
    </svg>
  )
}

function TrendFollowingSVG() {
  // Prix en tendance haussière avec entrées dans la direction
  const trendPath = 'M 40,200 C 100,190 130,170 170,155 C 210,140 230,150 260,135 C 290,120 310,125 340,105 C 370,85 400,95 430,75 C 460,55 490,65 540,45'
  const emaFast = 'M 40,195 C 100,185 140,168 180,155 C 220,142 250,148 290,130 C 330,112 360,115 400,95 C 440,75 480,72 540,55'
  const emaSlow = 'M 40,205 C 110,198 150,185 200,175 C 250,165 290,160 340,145 C 390,130 420,120 470,105 C 520,90 540,80 540,75'

  return (
    <svg viewBox="0 0 600 250" width="100%" style={{ display: 'block' }}>
      {/* EMAs */}
      <path d={emaSlow} stroke="#ffc53d" strokeWidth="1.5" strokeDasharray="6,4" fill="none" />
      <path d={emaFast} stroke="#ff8c42" strokeWidth="1.5" fill="none" />

      {/* Prix */}
      <path d={trendPath} stroke="#4da6ff" strokeWidth="2.5" fill="none" />

      {/* Entrées sur pullbacks */}
      <circle cx="230" cy="150" r="6" fill="none" stroke="#00e68a" strokeWidth="2" />
      <text x="230" y="170" fill="#00e68a" fontSize="10" textAnchor="middle">LONG</text>

      <circle cx="400" cy="95" r="6" fill="none" stroke="#00e68a" strokeWidth="2" />
      <text x="400" y="115" fill="#00e68a" fontSize="10" textAnchor="middle">LONG</text>

      {/* Labels */}
      <text x="545" y="50" fill="#4da6ff" fontSize="11" textAnchor="end">Prix</text>
      <text x="545" y="65" fill="#ff8c42" fontSize="11" textAnchor="end">EMA rapide</text>
      <text x="545" y="80" fill="#ffc53d" fontSize="11" textAnchor="end">EMA lente</text>

      {/* Flèche tendance */}
      <line x1="60" y1="220" x2="520" y2="30" stroke="rgba(0,230,138,0.15)" strokeWidth="1.5" strokeDasharray="8,6" />
      <text x="300" y="240" fill="var(--text-muted, #666)" fontSize="11" textAnchor="middle">Direction de la tendance</text>
    </svg>
  )
}

function BreakoutSVG() {
  // Channel horizontal avec sortie violente
  const upperBound = 80
  const lowerBound = 170
  const rangePath = 'M 40,125 L 100,110 L 140,140 L 180,115 L 220,135 L 260,120 L 300,130 L 340,125'
  const breakoutPath = 'M 340,125 L 370,110 L 400,85 L 430,60 L 470,40 L 520,30'

  return (
    <svg viewBox="0 0 600 250" width="100%" style={{ display: 'block' }}>
      {/* Zone de range */}
      <rect x="40" y={upperBound} width="310" height={lowerBound - upperBound} fill="rgba(77,166,255,0.06)" rx="4" />

      {/* Bornes du channel */}
      <line x1="40" y1={upperBound} x2="350" y2={upperBound} stroke="#666" strokeWidth="1" strokeDasharray="4,4" />
      <line x1="40" y1={lowerBound} x2="350" y2={lowerBound} stroke="#666" strokeWidth="1" strokeDasharray="4,4" />
      <text x="35" y={upperBound - 5} fill="#888" fontSize="10" textAnchor="end">Résistance</text>
      <text x="35" y={lowerBound + 15} fill="#888" fontSize="10" textAnchor="end">Support</text>

      {/* Prix en range */}
      <path d={rangePath} stroke="#4da6ff" strokeWidth="2" fill="none" />

      {/* Breakout */}
      <path d={breakoutPath} stroke="#00e68a" strokeWidth="2.5" fill="none" />

      {/* Point d'entrée */}
      <circle cx="370" cy="110" r="7" fill="none" stroke="#00e68a" strokeWidth="2.5" />
      <text x="370" y="100" fill="#00e68a" fontSize="11" fontWeight="600" textAnchor="middle">LONG</text>

      {/* Volume bar */}
      <rect x="355" y="195" width="20" height="40" fill="rgba(0,230,138,0.3)" rx="2" />
      <rect x="315" y="210" width="20" height="25" fill="rgba(77,166,255,0.2)" rx="2" />
      <rect x="275" y="215" width="20" height="20" fill="rgba(77,166,255,0.2)" rx="2" />
      <text x="365" y="245" fill="var(--text-muted, #666)" fontSize="9" textAnchor="middle">Volume spike</text>

      <text x="470" y="25" fill="#00e68a" fontSize="11" fontWeight="600" textAnchor="middle">Breakout !</text>
    </svg>
  )
}

function RangeTradingSVG() {
  // Canal avec achats en bas, ventes en haut, des deux côtés
  const upper = 70
  const lower = 180
  const mid = 125
  const pricePath = 'M 40,125 L 80,90 L 120,75 L 160,110 L 200,165 L 240,180 L 280,140 L 320,80 L 360,70 L 400,120 L 440,175 L 480,170 L 520,130'

  return (
    <svg viewBox="0 0 600 250" width="100%" style={{ display: 'block' }}>
      {/* Canal */}
      <rect x="40" y={upper} width="500" height={lower - upper} fill="rgba(77,166,255,0.04)" rx="4" />
      <line x1="40" y1={upper} x2="540" y2={upper} stroke="#666" strokeWidth="1" strokeDasharray="4,4" />
      <line x1="40" y1={lower} x2="540" y2={lower} stroke="#666" strokeWidth="1" strokeDasharray="4,4" />
      <line x1="40" y1={mid} x2="540" y2={mid} stroke="#ffc53d" strokeWidth="1.5" strokeDasharray="6,4" />
      <text x="545" y={mid - 5} fill="#ffc53d" fontSize="10" textAnchor="end">SMA</text>

      {/* Prix */}
      <path d={pricePath} stroke="#4da6ff" strokeWidth="2" fill="none" />

      {/* Shorts en haut */}
      <circle cx="120" cy="75" r="5" fill="#ff4466" />
      <text x="120" y="65" fill="#ff4466" fontSize="9" textAnchor="middle">SHORT</text>
      <circle cx="360" cy="70" r="5" fill="#ff4466" />
      <text x="360" y="60" fill="#ff4466" fontSize="9" textAnchor="middle">SHORT</text>

      {/* Longs en bas */}
      <circle cx="240" cy="180" r="5" fill="#00e68a" />
      <text x="240" y="200" fill="#00e68a" fontSize="9" textAnchor="middle">LONG</text>
      <circle cx="440" cy="175" r="5" fill="#00e68a" />
      <text x="440" y="195" fill="#00e68a" fontSize="9" textAnchor="middle">LONG</text>

      {/* TP au centre */}
      <text x="300" y="240" fill="var(--text-muted, #666)" fontSize="10" textAnchor="middle">TP = retour au centre du range</text>
    </svg>
  )
}

function FundingArbitrageSVG() {
  // Histogramme de funding rate avec zones de signal
  const bars = [
    { x: 60, h: 20, positive: true },
    { x: 100, h: 35, positive: true },
    { x: 140, h: 15, positive: true },
    { x: 180, h: -10, positive: false },
    { x: 220, h: -30, positive: false },
    { x: 260, h: -55, positive: false },
    { x: 300, h: -40, positive: false },
    { x: 340, h: -15, positive: false },
    { x: 380, h: 10, positive: true },
    { x: 420, h: 25, positive: true },
    { x: 460, h: 30, positive: true },
    { x: 500, h: 10, positive: true },
  ]
  const baseline = 125

  return (
    <svg viewBox="0 0 600 250" width="100%" style={{ display: 'block' }}>
      {/* Zone de signal (funding négatif) */}
      <rect x="170" y={baseline} width="180" height="70" fill="rgba(0,230,138,0.06)" rx="4" />
      <text x="260" y={baseline + 80} fill="#00e68a" fontSize="10" textAnchor="middle" opacity="0.6">Zone signal LONG</text>

      {/* Baseline */}
      <line x1="40" y1={baseline} x2="540" y2={baseline} stroke="#666" strokeWidth="1" />
      <text x="35" y={baseline + 4} fill="#888" fontSize="10" textAnchor="end">0%</text>

      {/* Barres */}
      {bars.map((b, i) => (
        <rect
          key={i}
          x={b.x}
          y={b.h >= 0 ? baseline - b.h : baseline}
          width="25"
          height={Math.abs(b.h)}
          fill={b.positive ? 'rgba(77,166,255,0.4)' : 'rgba(0,230,138,0.5)'}
          rx="2"
        />
      ))}

      {/* Flèche d'entrée */}
      <circle cx="260" cy={baseline + 55 + 8} r="0" />
      <text x="260" y={baseline - 65} fill="#00e68a" fontSize="11" fontWeight="600" textAnchor="middle">LONG ici</text>
      <line x1="260" y1={baseline - 58} x2="260" y2={baseline - 10} stroke="#00e68a" strokeWidth="1.5" strokeDasharray="3,3" />

      {/* Labels */}
      <text x="100" y="40" fill="#4da6ff" fontSize="10" textAnchor="middle">Funding positif</text>
      <text x="100" y="52" fill="#4da6ff" fontSize="9" textAnchor="middle">(shorts paient)</text>
      <text x="260" y="230" fill="#00e68a" fontSize="10" textAnchor="middle">Funding négatif</text>
      <text x="260" y="242" fill="#00e68a" fontSize="9" textAnchor="middle">(longs reçoivent)</text>
    </svg>
  )
}

const SCHEMA_MAP = {
  'mean-reversion': MeanReversionSVG,
  'trend-following': TrendFollowingSVG,
  'breakout': BreakoutSVG,
  'range-trading': RangeTradingSVG,
  'funding-arbitrage': FundingArbitrageSVG,
}

export default function GenericGuide({ strategy }) {
  const schemaKey = TYPE_TO_SCHEMA[strategy.type] || 'mean-reversion'
  const SvgComponent = SCHEMA_MAP[schemaKey] || MeanReversionSVG

  return (
    <div style={{ width: '100%', maxWidth: 700, margin: '0 auto' }}>
      <SvgComponent />
      <div style={{
        textAlign: 'center',
        marginTop: 12,
        fontSize: 12,
        color: 'var(--text-muted, #666)',
      }}>
        Schéma illustratif — type : {strategy.type}
      </div>
    </div>
  )
}
