# Grid Multi-TF Paper Execution Realism Audit

**Date:** 2026-07-20
**Environment inspected:** `robot2` (`192.168.1.200`)
**Decision:** Existing `grid_multi_tf` paper performance is invalid and must not
be used for strategy selection or live sizing.

## Production finding

The dashboard reported:

- Total P&L: **+$30,091.93**
- Realized P&L: **+$30,137.80**
- Unrealized P&L: **-$45.87**
- Equity: **$31,703 (+1,826.1%)**
- Trades: **2,553**
- Wins/losses: **2,466 / 87**
- Win rate: **97%**

The trade history decomposition explains the anomaly:

| Trade class | Count | Wins | Net P&L |
|---|---:|---:|---:|
| Entry and exit on the same H1 candle | 2,315 | 2,311 | +$37,700.95 |
| Positive holding duration | 236 | 155 | -$3,569.71 |
| Negative holding duration | 2 | 0 | -$3,993.44 |

The apparently exceptional result is therefore entirely carried by impossible
or economically unverifiable same-candle executions. Trades that genuinely
survived beyond their entry candle lost money in aggregate.

## Root causes

### 1. Repeated execution on an open candle

`DataEngine.on_candle` emits updates while the current candle is still forming.
The paper simulator was making trading decisions on every update, unlike the
live executor, which consumes closed candles.

### 2. Retrospective grid creation

The runner calculated a grid from the completed candle and then used the same
candle's high/low to decide that a newly created level had already been filled.
This is look-ahead bias: the order did not exist when the price touched it.

### 3. Retrospective exits

TP/SL values recalculated at candle close could be tested against the same
candle's full high/low range. The threshold was not necessarily known during
the move that supposedly triggered it.

### 4. Incomplete chronology guard

Duplicate and out-of-order candles could reach trading logic. Two persisted
trades even had an exit timestamp earlier than their entry timestamp.

### 5. Accounting inconsistencies

- Journal equity omitted margin already committed to open positions.
- The dashboard displayed margin usage relative to initial capital instead of
  current equity.
- Funding reduced cash but was not included in realized P&L or persisted.
- Total P&L presentation could diverge from the equity source of truth.

## Implemented execution model

The production paper simulator now uses `closed_bar_v2`:

1. Intrabar updates refresh indicators and mark-to-market prices only.
2. Trading decisions run once per closed strategy candle.
3. A grid calculated at candle close becomes a set of virtual orders active
   from the next candle.
4. Only orders that existed before a candle may be filled by that candle's
   range.
5. TP/SL thresholds must also exist before the candle that triggers them.
6. Duplicate and out-of-order closed candles are rejected.
7. Chronological exits use candle close time, guaranteeing
   `exit_time > entry_time`.
8. Pending orders, planned exits, execution clocks, and funding costs survive
   restart.
9. Legacy intrabar snapshots are rejected, so invalid historical capital is
   not restored into the new paper engine.

Accounting now follows:

```text
equity = free capital + used margin + unrealized P&L
total P&L = equity - initial capital
realized P&L = total P&L - unrealized P&L
margin usage = used margin / equity
```

Funding is included in realized and total P&L.

## Validation

- 9 dedicated paper-realism tests:
  - next-candle activation;
  - duplicate/out-of-order rejection;
  - first post-warmup candle idempotency;
  - pre-existing TP/SL thresholds;
  - strictly positive trade duration;
  - realtime updates cannot trade;
  - restart persistence;
  - legacy state rejection;
  - funding accounting.
- Full backend suite: **2,269 passed**.
- Frontend production build: passed.
- Ruff and whitespace validation: passed.

## Remaining model limitations

This correction removes the material look-ahead bias, but H1 OHLC data cannot
reconstruct the exact path inside a candle. When several prices are touched in
one candle, execution ordering remains an approximation. Paper trading also
cannot reproduce exchange latency, queue position, partial fills, transient
spread, or order-book depth exactly.

Tick/order-book replay with recorded exchange events would be required for the
highest possible fidelity. The closed-bar model is the safest realistic model
available with the current candle data and is deliberately conservative about
what was known before each candle.

## Rollout note

The first startup with this change will ignore the old intrabar runner snapshot
and reset paper capital/statistics to configured initial capital. The global
kill switch remains independent and is still restored. Historical database
rows remain available for audit but all `grid_multi_tf` paper results produced
before `closed_bar_v2` must be treated as invalid.
