"""Fixtures pytest partagées."""

import pytest


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
        "orderflow": {"enabled": False},
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
            "sandbox": False,
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
