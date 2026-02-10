import { useState, useEffect, useCallback, useRef, useMemo } from "react";

// ─── CONFIG ────────────────────────────────────────────────────────────────
const STRATEGIES = {
  VWAP: { name: "VWAP", full: "VWAP Bounce/Reject", weight: 0.25, icon: "◎" },
  RSI: { name: "RSI", full: "RSI Extrême (<20 / >80)", weight: 0.20, icon: "◆" },
  VOL: { name: "Volume", full: "Volume Spike 3x+", weight: 0.20, icon: "▮" },
  LIQ: { name: "Liquidations", full: "Zone de liquidation proche", weight: 0.20, icon: "⚡" },
  FUND: { name: "Funding", full: "Funding Rate extrême", weight: 0.15, icon: "∿" },
};

const ASSETS = [
  { symbol: "BTC/USDT", exchange: "Binance", type: "crypto", basePrice: 97500, vol: 0.0025, color: "#f7931a" },
  { symbol: "ETH/USDT", exchange: "Binance", type: "crypto", basePrice: 3850, vol: 0.0035, color: "#627eea" },
  { symbol: "SOL/USDT", exchange: "Binance", type: "crypto", basePrice: 195, vol: 0.005, color: "#9945ff" },
  { symbol: "BNB/USDT", exchange: "Binance", type: "crypto", basePrice: 685, vol: 0.003, color: "#f3ba2f" },
  { symbol: "XRP/USDT", exchange: "Bitget", type: "crypto", basePrice: 2.45, vol: 0.004, color: "#00aae4" },
  { symbol: "DOGE/USDT", exchange: "Bitget", type: "crypto", basePrice: 0.325, vol: 0.007, color: "#c2a633" },
  { symbol: "NQ100", exchange: "Saxo", type: "index", basePrice: 21450, vol: 0.0015, color: "#00d4aa" },
  { symbol: "DAX40", exchange: "Saxo", type: "index", basePrice: 20850, vol: 0.0015, color: "#ff6b35" },
];

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * t;
const fmtPrice = (p) => p >= 1000 ? p.toLocaleString("fr-FR", { maximumFractionDigits: 1 }) : p < 1 ? p.toFixed(4) : p.toFixed(2);
const fmtTime = (d) => d.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

// Seeded-ish signal generation with some persistence
function evolveSignals(prev, drift = 0.15) {
  const out = {};
  Object.keys(STRATEGIES).forEach(k => {
    const base = prev?.[k] ?? Math.random();
    out[k] = clamp(base + (Math.random() - 0.5) * drift, 0, 1);
  });
  return out;
}

function scoreFromSignals(sig) {
  let s = 0;
  Object.entries(sig).forEach(([k, v]) => { s += v * (STRATEGIES[k]?.weight || 0); });
  return clamp(s, 0, 1);
}

function directionFromSignals(sig) {
  const bull = (sig.RSI < 0.35 ? 1 : 0) + (sig.VWAP > 0.6 ? 1 : 0) + (sig.FUND < 0.35 ? 1 : 0);
  const bear = (sig.RSI > 0.65 ? 1 : 0) + (sig.VWAP < 0.4 ? 1 : 0) + (sig.FUND > 0.65 ? 1 : 0);
  return bull > bear ? "LONG" : bear > bull ? "SHORT" : Math.random() > 0.5 ? "LONG" : "SHORT";
}

// ─── COLORS ────────────────────────────────────────────────────────────────
const C = {
  bg: "#06080d",
  card: "rgba(255,255,255,0.018)",
  border: "rgba(255,255,255,0.055)",
  text: "#e8eaed",
  muted: "rgba(255,255,255,0.35)",
  dim: "rgba(255,255,255,0.18)",
  green: "#00e68a",
  red: "#ff4466",
  yellow: "#ffc53d",
  orange: "#ff8c42",
  blue: "#4da6ff",
  accent: "#00e68a",
};
const scoreColor = (s) => s >= 0.75 ? C.green : s >= 0.55 ? C.yellow : s >= 0.35 ? C.orange : C.red;
const scoreLabel = (s) => s >= 0.75 ? "EXCELLENT" : s >= 0.55 ? "BON" : s >= 0.35 ? "MOYEN" : "FAIBLE";

// ─── SVG SPARKLINE ─────────────────────────────────────────────────────────
function Spark({ data, w = 110, h = 32, stroke = 1.5 }) {
  if (!data || data.length < 3) return <div style={{ width: w, height: h }} />;
  const mn = Math.min(...data), mx = Math.max(...data), rng = mx - mn || 1;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - 2 - ((v - mn) / rng) * (h - 4)}`).join(" ");
  const up = data[data.length - 1] >= data[0];
  const c = up ? C.green : C.red;
  const id = `sg${Math.random().toString(36).slice(2, 6)}`;
  return (
    <svg width={w} height={h} style={{ display: "block", overflow: "visible" }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={c} stopOpacity="0.18" />
          <stop offset="100%" stopColor={c} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon fill={`url(#${id})`} points={`0,${h} ${pts} ${w},${h}`} />
      <polyline fill="none" stroke={c} strokeWidth={stroke} points={pts} />
      <circle cx={w} cy={parseFloat(pts.split(" ").pop().split(",")[1])} r="2" fill={c}>
        <animate attributeName="opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

// ─── CIRCULAR SCORE ────────────────────────────────────────────────────────
function ScoreRing({ score, size = 72 }) {
  const c = scoreColor(score);
  const pct = score * 100;
  const circ = 2 * Math.PI * 30;
  return (
    <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
      <svg width={size} height={size} viewBox="0 0 68 68">
        <circle cx="34" cy="34" r="30" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="4" />
        <circle cx="34" cy="34" r="30" fill="none" stroke={c} strokeWidth="4" strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={circ - (pct / 100) * circ}
          transform="rotate(-90 34 34)" style={{ transition: "all 0.7s cubic-bezier(.4,0,.2,1)" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 17, fontWeight: 800, color: c, fontFamily: "var(--mono)" }}>{Math.round(pct)}</span>
      </div>
    </div>
  );
}

// ─── SIGNAL HEATMAP ROW ────────────────────────────────────────────────────
function SignalDots({ signals }) {
  return (
    <div style={{ display: "flex", gap: 3 }}>
      {Object.entries(signals).map(([k, v]) => {
        const c = v > 0.7 ? C.green : v > 0.45 ? C.yellow : v > 0.25 ? C.orange : "rgba(255,255,255,0.08)";
        return (
          <div key={k} title={`${STRATEGIES[k].full}: ${Math.round(v * 100)}%`} style={{
            width: 18, height: 18, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center",
            background: `${c}18`, border: `1px solid ${c}30`, fontSize: 9, color: c, fontWeight: 700,
            cursor: "default", transition: "all 0.3s ease"
          }}>
            {STRATEGIES[k].icon}
          </div>
        );
      })}
    </div>
  );
}

// ─── SIGNAL BREAKDOWN BARS ─────────────────────────────────────────────────
function SignalBreakdown({ signals }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {Object.entries(signals).map(([k, v]) => {
        const c = v > 0.7 ? C.green : v > 0.45 ? C.yellow : C.red;
        return (
          <div key={k}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontSize: 10, color: C.muted }}>{STRATEGIES[k].icon} {STRATEGIES[k].full}</span>
              <span style={{ fontSize: 10, fontFamily: "var(--mono)", color: c, fontWeight: 600 }}>{Math.round(v * 100)}%</span>
            </div>
            <div style={{ height: 3, background: "rgba(255,255,255,0.04)", borderRadius: 2, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${v * 100}%`, background: c, borderRadius: 2, transition: "width 0.5s ease" }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── RISK CALCULATOR ───────────────────────────────────────────────────────
function RiskCalc() {
  const [cap, setCap] = useState(1000);
  const [lev, setLev] = useState(15);
  const [sl, setSl] = useState(0.5);
  const pos = cap * lev;
  const maxLoss = cap * (sl / 100) * lev;
  const liq = (100 / lev).toFixed(2);
  const danger = lev > 20;

  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 2, color: C.muted, fontWeight: 600, marginBottom: 14 }}>
        ⚡ Risque & Position
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
        <div>
          <label style={{ fontSize: 8, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>Capital $</label>
          <input type="number" value={cap} onChange={e => setCap(+e.target.value)} style={{
            width: "100%", background: "rgba(255,255,255,0.04)", border: `1px solid ${C.border}`, borderRadius: 6,
            padding: "7px 8px", color: "#fff", fontSize: 13, fontFamily: "var(--mono)", marginTop: 3, boxSizing: "border-box"
          }} />
        </div>
        <div>
          <label style={{ fontSize: 8, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>Levier</label>
          <input type="range" min={2} max={50} value={lev} onChange={e => setLev(+e.target.value)}
            style={{ width: "100%", marginTop: 8, accentColor: danger ? C.red : C.yellow }} />
          <span style={{ fontSize: 13, fontWeight: 800, color: danger ? C.red : C.yellow, fontFamily: "var(--mono)" }}>{lev}x</span>
        </div>
        <div>
          <label style={{ fontSize: 8, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>Stop Loss %</label>
          <input type="number" value={sl} step={0.1} min={0.1} max={5} onChange={e => setSl(+e.target.value)} style={{
            width: "100%", background: "rgba(255,255,255,0.04)", border: `1px solid ${C.border}`, borderRadius: 6,
            padding: "7px 8px", color: "#fff", fontSize: 13, fontFamily: "var(--mono)", marginTop: 3, boxSizing: "border-box"
          }} />
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, background: "rgba(0,0,0,0.25)", borderRadius: 8, padding: 10 }}>
        {[
          { l: "Position", v: `$${pos.toLocaleString()}`, c: "#fff" },
          { l: "Perte Max", v: `-$${maxLoss.toFixed(0)}`, c: C.red },
          { l: "Liquidation", v: `±${liq}%`, c: danger ? C.red : C.yellow },
          { l: "R:R (TP 2×SL)", v: "1:2", c: C.green },
        ].map((x, i) => (
          <div key={i} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 7, color: C.dim, textTransform: "uppercase", letterSpacing: 1, marginBottom: 3 }}>{x.l}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: x.c, fontFamily: "var(--mono)" }}>{x.v}</div>
          </div>
        ))}
      </div>
      {danger && (
        <div style={{
          marginTop: 10, padding: "6px 10px", background: "rgba(255,68,102,0.08)", border: "1px solid rgba(255,68,102,0.15)",
          borderRadius: 6, fontSize: 10, color: "#ff8899", lineHeight: 1.5
        }}>
          ⚠️ À {lev}x, tu es liquidé à seulement {liq}% de mouvement. Avec un SL de {sl}%, ta perte réelle est de ${maxLoss.toFixed(0)} ({((maxLoss / cap) * 100).toFixed(0)}% du capital).
        </div>
      )}
    </div>
  );
}

// ─── HEATMAP OVERVIEW ──────────────────────────────────────────────────────
function Heatmap({ opportunities }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 14 }}>
      <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 2, color: C.muted, fontWeight: 600, marginBottom: 10 }}>
        🗺️ Heatmap Signaux
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `80px repeat(${Object.keys(STRATEGIES).length}, 1fr) 50px`, gap: 2, fontSize: 9 }}>
        <div style={{ color: C.dim }}></div>
        {Object.values(STRATEGIES).map(s => (
          <div key={s.name} style={{ color: C.dim, textAlign: "center", fontWeight: 600 }}>{s.icon}</div>
        ))}
        <div style={{ color: C.dim, textAlign: "center", fontWeight: 600 }}>Σ</div>

        {opportunities.slice(0, 8).map(opp => (
          <React.Fragment key={opp.symbol}>
            <div style={{ fontSize: 10, color: C.text, fontWeight: 600, display: "flex", alignItems: "center" }}>
              <span style={{ width: 6, height: 6, borderRadius: 2, background: opp.color, marginRight: 5, flexShrink: 0 }} />
              {opp.symbol.replace("/USDT", "")}
            </div>
            {Object.entries(opp.signals).map(([k, v]) => {
              const intensity = v;
              const bg = v > 0.7 ? `rgba(0,230,138,${intensity * 0.35})` : v > 0.45 ? `rgba(255,197,61,${intensity * 0.3})` : `rgba(255,68,102,${intensity * 0.2})`;
              return (
                <div key={k} style={{
                  background: bg, borderRadius: 3, textAlign: "center", padding: "4px 0",
                  fontFamily: "var(--mono)", fontWeight: 600, fontSize: 9,
                  color: v > 0.7 ? C.green : v > 0.45 ? C.yellow : C.red
                }}>
                  {Math.round(v * 100)}
                </div>
              );
            })}
            <div style={{
              textAlign: "center", padding: "4px 0", fontFamily: "var(--mono)", fontWeight: 800, fontSize: 10,
              color: scoreColor(opp.score), background: `${scoreColor(opp.score)}10`, borderRadius: 3
            }}>
              {Math.round(opp.score * 100)}
            </div>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// ─── ALERT FEED ────────────────────────────────────────────────────────────
function AlertFeed({ alerts }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 14, maxHeight: 300, overflowY: "auto" }}>
      <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 2, color: C.muted, fontWeight: 600, marginBottom: 10 }}>
        📡 Alertes Live
      </div>
      {alerts.length === 0 ? (
        <div style={{ color: C.dim, fontSize: 11, textAlign: "center", padding: 24 }}>Scan en cours...</div>
      ) : alerts.map((a, i) => (
        <div key={i} style={{
          display: "flex", gap: 8, alignItems: "flex-start", padding: "7px 0",
          borderBottom: i < alerts.length - 1 ? `1px solid rgba(255,255,255,0.03)` : "none",
          animation: i === 0 ? "slideIn 0.25s ease" : "none"
        }}>
          <div style={{
            width: 5, height: 5, borderRadius: "50%", marginTop: 5, flexShrink: 0,
            background: scoreColor(a.score),
            boxShadow: `0 0 6px ${scoreColor(a.score)}50`
          }} />
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
                {a.symbol}
                <span style={{
                  marginLeft: 5, fontSize: 8, padding: "1px 4px", borderRadius: 3, fontWeight: 700,
                  background: a.direction === "LONG" ? `${C.green}15` : `${C.red}15`,
                  color: a.direction === "LONG" ? C.green : C.red
                }}>{a.direction}</span>
              </span>
              <span style={{ fontSize: 8, color: C.dim, fontFamily: "var(--mono)" }}>{a.time}</span>
            </div>
            <div style={{ fontSize: 9, color: C.muted, marginTop: 2 }}>
              Score {Math.round(a.score * 100)} · {a.trigger} · Entry {a.price}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── MAIN ──────────────────────────────────────────────────────────────────
export default function ScalpRadar() {
  const [opps, setOpps] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState("all");
  const [live, setLive] = useState(true);
  const [tab, setTab] = useState("scanner"); // scanner | heatmap | risk
  const prevSignals = useRef({});
  const priceHist = useRef({});

  useEffect(() => {
    ASSETS.forEach(a => {
      if (!priceHist.current[a.symbol]) {
        const prices = [a.basePrice];
        for (let i = 1; i < 50; i++) prices.push(prices[i - 1] + prices[i - 1] * a.vol * (Math.random() - 0.5) * 2);
        priceHist.current[a.symbol] = prices;
      }
    });
  }, []);

  useEffect(() => {
    if (!live) return;
    const iv = setInterval(() => {
      ASSETS.forEach(a => {
        const h = priceHist.current[a.symbol];
        if (h) {
          const last = h[h.length - 1];
          h.push(last + last * a.vol * (Math.random() - 0.5) * 2);
          if (h.length > 60) h.shift();
        }
      });

      const newOpps = ASSETS.map(a => {
        const sig = evolveSignals(prevSignals.current[a.symbol]);
        prevSignals.current[a.symbol] = sig;
        const score = scoreFromSignals(sig);
        const dir = directionFromSignals(sig);
        const h = priceHist.current[a.symbol] || [];
        const price = h[h.length - 1] || a.basePrice;
        const prev = h[h.length - 2] || a.basePrice;
        return { ...a, signals: sig, score, direction: dir, price, change: ((price - prev) / prev) * 100, history: [...h] };
      }).sort((a, b) => b.score - a.score);

      setOpps(newOpps);

      // Alert generation
      const top = newOpps.find(o => o.score > 0.68);
      if (top && Math.random() > 0.55) {
        const best = Object.entries(top.signals).sort((a, b) => b[1] - a[1])[0];
        setAlerts(p => [{
          symbol: top.symbol, score: top.score, direction: top.direction,
          trigger: STRATEGIES[best[0]]?.full || "Multi", time: fmtTime(new Date()), price: fmtPrice(top.price)
        }, ...p].slice(0, 25));
      }
    }, 2500);
    return () => clearInterval(iv);
  }, [live]);

  const filtered = filter === "all" ? opps : opps.filter(o => filter === "crypto" ? o.type === "crypto" : o.type === "index");
  const sel = selected ? opps.find(o => o.symbol === selected) : null;
  const topScore = opps.length ? Math.max(...opps.map(o => o.score)) : 0;

  return (
    <div style={{
      minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'Satoshi', 'DM Sans', -apple-system, sans-serif",
      padding: "16px 20px", lineHeight: 1.5,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');
        :root { --mono: 'JetBrains Mono', monospace; --sans: 'DM Sans', sans-serif; }
        * { box-sizing: border-box; margin: 0; }
        body { background: ${C.bg}; }
        @keyframes slideIn { from { opacity:0; transform:translateY(-3px); } to { opacity:1; transform:translateY(0); } }
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:.4; } }
        ::-webkit-scrollbar { width:3px; } ::-webkit-scrollbar-track { background:transparent; } ::-webkit-scrollbar-thumb { background:rgba(255,255,255,.08); border-radius:3px; }
        input[type=range] { -webkit-appearance:none; height:2px; background:rgba(255,255,255,.08); border-radius:2px; outline:none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:12px; height:12px; border-radius:50%; background:${C.yellow}; cursor:pointer; }
        input[type=number] { outline:none; }
        input[type=number]:focus { border-color: ${C.accent}40; }
      `}</style>

      {/* HEADER */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18, paddingBottom: 14, borderBottom: `1px solid ${C.border}` }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h1 style={{ fontSize: 20, fontWeight: 800, letterSpacing: -0.5, color: C.accent }}>SCALP RADAR</h1>
          <span style={{ fontSize: 10, color: C.dim, fontFamily: "var(--mono)" }}>v0.2 · prototype</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {/* Top signal indicator */}
          <div style={{
            display: "flex", alignItems: "center", gap: 5, padding: "4px 10px",
            background: `${scoreColor(topScore)}08`, border: `1px solid ${scoreColor(topScore)}20`,
            borderRadius: 6, fontSize: 10, fontFamily: "var(--mono)", color: scoreColor(topScore)
          }}>
            <span style={{ fontSize: 7, textTransform: "uppercase", letterSpacing: 1, color: C.muted }}>Best</span>
            <span style={{ fontWeight: 800 }}>{Math.round(topScore * 100)}</span>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 2, background: "rgba(255,255,255,0.03)", borderRadius: 7, padding: 2 }}>
            {[
              { id: "scanner", label: "Scanner" },
              { id: "heatmap", label: "Heatmap" },
              { id: "risk", label: "Risque" },
            ].map(t => (
              <button key={t.id} onClick={() => setTab(t.id)} style={{
                padding: "4px 10px", fontSize: 10, fontWeight: 600, border: "none", borderRadius: 5, cursor: "pointer",
                background: tab === t.id ? "rgba(255,255,255,0.08)" : "transparent",
                color: tab === t.id ? C.text : C.muted, transition: "all .15s ease"
              }}>{t.label}</button>
            ))}
          </div>

          {/* Filters */}
          <div style={{ display: "flex", gap: 2, background: "rgba(255,255,255,0.03)", borderRadius: 7, padding: 2 }}>
            {["all", "crypto", "indices"].map(f => (
              <button key={f} onClick={() => setFilter(f)} style={{
                padding: "4px 10px", fontSize: 10, fontWeight: 600, border: "none", borderRadius: 5, cursor: "pointer",
                background: filter === f ? "rgba(255,255,255,0.08)" : "transparent",
                color: filter === f ? C.text : C.muted, transition: "all .15s ease"
              }}>{f === "all" ? "Tous" : f === "crypto" ? "Crypto" : "Indices"}</button>
            ))}
          </div>

          {/* Live toggle */}
          <button onClick={() => setLive(!live)} style={{
            display: "flex", alignItems: "center", gap: 5, padding: "4px 12px", border: `1px solid ${live ? `${C.green}30` : C.border}`,
            borderRadius: 6, background: live ? `${C.green}08` : "transparent", color: live ? C.green : C.muted,
            fontSize: 10, fontWeight: 700, cursor: "pointer", fontFamily: "var(--mono)"
          }}>
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: live ? C.green : C.muted, animation: live ? "blink 1.2s ease infinite" : "none" }} />
            {live ? "LIVE" : "PAUSED"}
          </button>
        </div>
      </div>

      {/* CONTENT */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 16 }}>

        {/* LEFT */}
        <div>
          {tab === "scanner" && (
            <>
              {/* Table */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, overflow: "hidden", marginBottom: 16 }}>
                <div style={{
                  display: "grid", gridTemplateColumns: "130px 85px 65px 52px 110px 60px 1fr",
                  padding: "8px 14px", fontSize: 8, textTransform: "uppercase", letterSpacing: 1.5,
                  color: C.dim, borderBottom: `1px solid ${C.border}`, fontWeight: 600
                }}>
                  <span>Actif</span><span>Prix</span><span>Var.</span><span>Dir.</span><span>Trend</span><span>Score</span><span>Signaux</span>
                </div>
                {filtered.map((o, i) => (
                  <div key={o.symbol} onClick={() => setSelected(selected === o.symbol ? null : o.symbol)} style={{
                    display: "grid", gridTemplateColumns: "130px 85px 65px 52px 110px 60px 1fr",
                    padding: "8px 14px", alignItems: "center", cursor: "pointer",
                    borderBottom: `1px solid rgba(255,255,255,0.02)`,
                    background: selected === o.symbol ? `${C.accent}06` : "transparent",
                    transition: "background .15s ease",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ width: 6, height: 6, borderRadius: 2, background: o.color, flexShrink: 0 }} />
                      <span style={{ fontSize: 12, fontWeight: 700 }}>{o.symbol}</span>
                      <span style={{ fontSize: 7, color: C.dim, background: "rgba(255,255,255,0.04)", padding: "1px 4px", borderRadius: 3 }}>{o.exchange}</span>
                    </div>
                    <span style={{ fontSize: 12, fontFamily: "var(--mono)", color: "rgba(255,255,255,0.75)" }}>{fmtPrice(o.price)}</span>
                    <span style={{ fontSize: 11, fontFamily: "var(--mono)", fontWeight: 600, color: o.change >= 0 ? C.green : C.red }}>
                      {o.change >= 0 ? "+" : ""}{o.change.toFixed(2)}%
                    </span>
                    <span style={{
                      fontSize: 9, fontWeight: 800, padding: "2px 5px", borderRadius: 3, textAlign: "center",
                      background: o.direction === "LONG" ? `${C.green}12` : `${C.red}12`,
                      color: o.direction === "LONG" ? C.green : C.red
                    }}>{o.direction}</span>
                    <Spark data={o.history} />
                    <span style={{ fontSize: 13, fontWeight: 800, fontFamily: "var(--mono)", color: scoreColor(o.score) }}>
                      {Math.round(o.score * 100)}
                    </span>
                    <SignalDots signals={o.signals} />
                  </div>
                ))}
              </div>

              {/* Detail panel */}
              {sel && (
                <div style={{
                  background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 18,
                  animation: "slideIn 0.2s ease"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ width: 10, height: 10, borderRadius: 3, background: sel.color }} />
                        <h2 style={{ fontSize: 18, fontWeight: 800, margin: 0 }}>{sel.symbol}</h2>
                        <span style={{
                          fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 4,
                          background: sel.direction === "LONG" ? `${C.green}12` : `${C.red}12`,
                          color: sel.direction === "LONG" ? C.green : C.red
                        }}>{sel.direction}</span>
                        <span style={{
                          fontSize: 9, padding: "2px 6px", borderRadius: 4,
                          background: `${scoreColor(sel.score)}12`, color: scoreColor(sel.score), fontWeight: 700
                        }}>{scoreLabel(sel.score)}</span>
                      </div>
                      <p style={{ fontSize: 10, color: C.muted, marginTop: 4 }}>
                        {sel.exchange} · {fmtPrice(sel.price)} · {sel.change >= 0 ? "+" : ""}{sel.change.toFixed(3)}%
                      </p>
                    </div>
                    <ScoreRing score={sel.score} />
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
                    <SignalBreakdown signals={sel.signals} />
                    <div>
                      <div style={{ marginBottom: 12 }}>
                        <div style={{ fontSize: 9, color: C.dim, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>Trend (60 ticks)</div>
                        <Spark data={sel.history} w={260} h={80} stroke={2} />
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
                        {[
                          { l: "Entry", v: fmtPrice(sel.price), c: C.text },
                          { l: "TP (0.5%)", v: fmtPrice(sel.price * (sel.direction === "LONG" ? 1.005 : 0.995)), c: C.green },
                          { l: "SL (0.25%)", v: fmtPrice(sel.price * (sel.direction === "LONG" ? 0.9975 : 1.0025)), c: C.red },
                        ].map((x, i) => (
                          <div key={i} style={{ background: "rgba(0,0,0,0.2)", borderRadius: 6, padding: "6px 8px", textAlign: "center" }}>
                            <div style={{ fontSize: 7, color: C.dim, textTransform: "uppercase", letterSpacing: 1 }}>{x.l}</div>
                            <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--mono)", color: x.c, marginTop: 2 }}>{x.v}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {tab === "heatmap" && <Heatmap opportunities={opps} />}
          {tab === "risk" && <RiskCalc />}
        </div>

        {/* RIGHT SIDEBAR */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <AlertFeed alerts={alerts} />

          {/* Strategies */}
          <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 14 }}>
            <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 2, color: C.muted, fontWeight: 600, marginBottom: 10 }}>
              🎯 Stratégies
            </div>
            {Object.entries(STRATEGIES).map(([k, s]) => (
              <div key={k} style={{
                display: "flex", alignItems: "center", gap: 8, padding: "5px 0",
                borderBottom: `1px solid rgba(255,255,255,0.025)`
              }}>
                <span style={{ fontSize: 12, width: 18, textAlign: "center" }}>{s.icon}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 10, fontWeight: 600, color: "rgba(255,255,255,0.7)" }}>{s.name}</div>
                  <div style={{ fontSize: 8, color: C.dim }}>{s.full}</div>
                </div>
                <span style={{
                  fontSize: 9, fontFamily: "var(--mono)", color: C.muted, background: "rgba(255,255,255,0.04)",
                  padding: "1px 5px", borderRadius: 3
                }}>{Math.round(s.weight * 100)}%</span>
              </div>
            ))}
          </div>

          {/* Session stats */}
          <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 14 }}>
            <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: 2, color: C.muted, fontWeight: 600, marginBottom: 10 }}>
              📊 Session
            </div>
            {[
              { l: "Alertes", v: alerts.length, c: C.text },
              { l: "Score > 70", v: alerts.filter(a => a.score >= 0.7).length, c: C.green },
              { l: "Ratio L/S", v: opps.length ? `${opps.filter(o => o.direction === "LONG").length}/${opps.filter(o => o.direction === "SHORT").length}` : "-", c: C.blue },
              { l: "Actifs scannés", v: ASSETS.length, c: C.muted },
            ].map((x, i) => (
              <div key={i} style={{
                display: "flex", justifyContent: "space-between", padding: "5px 0",
                borderBottom: i < 3 ? `1px solid rgba(255,255,255,0.025)` : "none"
              }}>
                <span style={{ fontSize: 10, color: C.muted }}>{x.l}</span>
                <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--mono)", color: x.c }}>{x.v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* FOOTER */}
      <div style={{ marginTop: 18, padding: "10px 0", borderTop: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", fontSize: 9, color: C.dim }}>
        <span>Prototype — données simulées · Les signaux ne constituent pas un conseil financier</span>
        <span style={{ fontFamily: "var(--mono)" }}>Phase 1/5 · Next → Backtesting Engine</span>
      </div>
    </div>
  );
}