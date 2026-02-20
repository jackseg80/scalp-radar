"""Fixtures pytest partagées."""

import os
os.environ["PYTHON_JIT"] = "0"

import numpy as np
import pytest

from backend.optimization.indicator_cache import IndicatorCache


@pytest.fixture
def config_dir(tmp_path):
    """Crée un répertoire de config temporaire avec des YAML valides."""
    import yaml

    assets = {
        "assets": [
            {
                "symbol": "BTC/USDT",
                "exchange": "bitget",
                "type": "futures",
                "timeframes": ["1m", "5m", "15m", "1h"],
                "max_leverage": 20,
                "min_order_size": 0.001,
                "tick_size": 0.1,
                "correlation_group": "crypto_major",
            },
            {
                "symbol": "ETH/USDT",
                "exchange": "bitget",
                "type": "futures",
                "timeframes": ["1m", "5m", "15m", "1h"],
                "max_leverage": 20,
                "min_order_size": 0.01,
                "tick_size": 0.01,
                "correlation_group": "crypto_major",
            },
        ],
        "correlation_groups": {
            "crypto_major": {
                "max_concurrent_same_direction": 2,
                "max_exposure_percent": 60,
            }
        },
    }

    strategies = {
        "vwap_rsi": {"enabled": True, "timeframe": "5m", "weight": 0.25},
        "liquidation": {"enabled": False},
        "momentum": {"enabled": False},
        "funding": {"enabled": False},
        "custom_strategies": {
            "swing_baseline": {"enabled": False, "timeframe": "1h"},
        },
    }

    risk = {
        "kill_switch": {
            "max_session_loss_percent": 5.0,
            "max_daily_loss_percent": 10.0,
        },
        "position": {
            "max_risk_per_trade_percent": 2.0,
            "max_concurrent_positions": 3,
            "default_leverage": 15,
            "max_leverage": 30,
        },
        "fees": {"maker_percent": 0.02, "taker_percent": 0.06},
        "slippage": {"default_estimate_percent": 0.05},
        "margin": {"mode": "cross", "min_free_margin_percent": 20},
        "sl_tp": {
            "mode": "server_side",
            "sl_type": "market",
            "sl_real_cost_includes": ["distance", "taker_fee", "slippage"],
        },
    }

    exchanges = {
        "bitget": {
            "name": "Bitget",
            "websocket": {
                "url": "wss://ws.bitget.com/v2/ws/public",
                "ping_interval": 25,
                "reconnect_delay": 5,
                "max_reconnect_attempts": 10,
            },
            "rate_limits": {
                "market_data": {"requests_per_second": 20},
                "trade": {"requests_per_second": 10},
                "account": {"requests_per_second": 10},
                "position": {"requests_per_second": 10},
            },
            "api": {
                "base_url": "https://api.bitget.com",
                "futures_type": "USDT-M",
                "price_type": "mark_price",
            },
        }
    }

    for name, data in [
        ("assets.yaml", assets),
        ("strategies.yaml", strategies),
        ("risk.yaml", risk),
        ("exchanges.yaml", exchanges),
    ]:
        (tmp_path / name).write_text(yaml.dump(data, allow_unicode=True))

    return tmp_path


@pytest.fixture
def make_indicator_cache():
    """Factory pour créer un IndicatorCache avec des valeurs par défaut.

    Usage::

        cache = make_indicator_cache(n=50, bb_sma={7: sma_array})
        cache = make_indicator_cache(n=10, closes=np.full(10, 105.0), rsi={14: np.full(10, 55.0)})
    """

    def _make(
        n: int = 100,
        *,
        closes: np.ndarray | None = None,
        opens: np.ndarray | None = None,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
        volumes: np.ndarray | None = None,
        total_days: float | None = None,
        rsi: dict | None = None,
        vwap: np.ndarray | None = None,
        vwap_distance_pct: np.ndarray | None = None,
        adx_arr: np.ndarray | None = None,
        di_plus: np.ndarray | None = None,
        di_minus: np.ndarray | None = None,
        atr_arr: np.ndarray | None = None,
        atr_sma: np.ndarray | None = None,
        volume_sma_arr: np.ndarray | None = None,
        regime: np.ndarray | None = None,
        rolling_high: dict | None = None,
        rolling_low: dict | None = None,
        filter_adx: np.ndarray | None = None,
        filter_di_plus: np.ndarray | None = None,
        filter_di_minus: np.ndarray | None = None,
        bb_sma: dict | None = None,
        bb_upper: dict | None = None,
        bb_lower: dict | None = None,
        supertrend_direction: dict | None = None,
        atr_by_period: dict | None = None,
        supertrend_dir_4h: dict | None = None,
        funding_rates_1h: np.ndarray | None = None,
        candle_timestamps: np.ndarray | None = None,
        ema_by_period: dict | None = None,
        adx_by_period: dict | None = None,
    ) -> IndicatorCache:
        if closes is None:
            closes = np.full(n, 100.0)
        if opens is None:
            opens = closes.copy()
        if highs is None:
            highs = closes + 1.0
        if lows is None:
            lows = closes - 1.0
        if volumes is None:
            volumes = np.full(n, 100.0)
        if total_days is None:
            total_days = n / 24

        return IndicatorCache(
            n_candles=n,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            total_days=total_days,
            rsi=rsi if rsi is not None else {14: np.full(n, 50.0)},
            vwap=vwap if vwap is not None else np.full(n, np.nan),
            vwap_distance_pct=vwap_distance_pct if vwap_distance_pct is not None else np.full(n, np.nan),
            adx_arr=adx_arr if adx_arr is not None else np.full(n, 25.0),
            di_plus=di_plus if di_plus is not None else np.full(n, 15.0),
            di_minus=di_minus if di_minus is not None else np.full(n, 10.0),
            atr_arr=atr_arr if atr_arr is not None else np.full(n, 1.0),
            atr_sma=atr_sma if atr_sma is not None else np.full(n, 1.0),
            volume_sma_arr=volume_sma_arr if volume_sma_arr is not None else np.full(n, 100.0),
            regime=regime if regime is not None else np.zeros(n, dtype=np.int8),
            rolling_high=rolling_high if rolling_high is not None else {},
            rolling_low=rolling_low if rolling_low is not None else {},
            filter_adx=filter_adx if filter_adx is not None else np.full(n, np.nan),
            filter_di_plus=filter_di_plus if filter_di_plus is not None else np.full(n, np.nan),
            filter_di_minus=filter_di_minus if filter_di_minus is not None else np.full(n, np.nan),
            bb_sma=bb_sma if bb_sma is not None else {},
            bb_upper=bb_upper if bb_upper is not None else {},
            bb_lower=bb_lower if bb_lower is not None else {},
            supertrend_direction=supertrend_direction if supertrend_direction is not None else {},
            atr_by_period=atr_by_period if atr_by_period is not None else {},
            supertrend_dir_4h=supertrend_dir_4h if supertrend_dir_4h is not None else {},
            funding_rates_1h=funding_rates_1h,
            candle_timestamps=candle_timestamps,
            ema_by_period=ema_by_period if ema_by_period is not None else {},
            adx_by_period=adx_by_period if adx_by_period is not None else {},
        )

    return _make
