
import json
import numpy as np
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from backend.backtesting.simulator import Simulator, DiagnosticEncoder, GridStrategyRunner
from backend.core.config import AppConfig

def test_diagnostic_encoder_numpy():
    """Vérifie que l'encoder gère correctement les types numpy."""
    encoder = DiagnosticEncoder()
    
    data = {
        "int": np.int64(42),
        "float": np.float64(3.14),
        "array": np.array([1.0, 2.0, 3.0]),
        "normal": "hello"
    }
    
    encoded = json.dumps(data, cls=DiagnosticEncoder)
    decoded = json.loads(encoded)
    
    assert decoded["int"] == 42
    assert isinstance(decoded["int"], int)
    assert decoded["float"] == 3.14
    assert isinstance(decoded["float"], float)
    assert decoded["array"] == [1.0, 2.0, 3.0]
    assert decoded["normal"] == "hello"

def test_diagnostic_encoder_fallback():
    """Vérifie que l'encoder ne crash pas sur un objet complexe inconnu."""
    class ComplexObj:
        pass
    
    data = {"obj": ComplexObj()}
    encoded = json.dumps(data, cls=DiagnosticEncoder)
    assert "<<NON-SERIALIZABLE" in encoded

def test_get_conditions_per_asset_resolution():
    """Vérifie que get_conditions résout correctement les paramètres par actif."""
    # Setup DataEngine Mock
    data_engine = MagicMock()
    data_engine.get_all_symbols.return_value = ["BTC/USDT"]
    
    # Mock des bougies avec des valeurs réelles (pas des mocks)
    mock_candle = MagicMock()
    mock_candle.close = 100.0
    data_engine.get_data.return_value.candles = {"1h": [mock_candle]}
    
    # Setup Config Mock
    config = MagicMock(spec=AppConfig)
    config.assets = {"BTC/USDT": MagicMock()}
    
    sim = Simulator(data_engine, config)
    
    # Setup Runner Mock
    runner = MagicMock(spec=GridStrategyRunner)
    runner.name = "grid_atr"
    runner.strategy.min_candles = {"1h": 10}
    runner.strategy.get_params.return_value = {"min_atr_pct": 0.5, "min_grid_spacing_pct": 1.0}
    
    # Simulation d'un override per_asset = 1.2%
    def mock_get_per_asset(symbol, param, default):
        if param == "min_atr_pct" and symbol == "BTC/USDT":
            return 1.2
        return default
        
    runner._get_per_asset_float.side_effect = mock_get_per_asset
    runner.current_regime.value = "RANGING"
    
    # Mock context avec indicateurs réels (sérialisables)
    ctx = MagicMock()
    ctx.indicators = {"1h": {"rsi": 50.0, "adx": 20.0, "atr": 1.0, "close": 100.0, "vwap": 95.0}}
    runner.build_context.return_value = ctx
    
    runner.strategy.get_current_conditions.return_value = []
    runner._position = None
    runner._position_symbol = None
    
    sim._runners = [runner]
    sim._indicator_engine = None 
    
    # Action
    res = sim.get_conditions()
    
    # Assert
    assert "BTC/USDT" in res["assets"]
    strat_data = res["assets"]["BTC/USDT"]["strategies"]["grid_atr"]
    params = strat_data["params"]
    assert params["min_atr_pct"] == 1.2 
    assert params["min_grid_spacing_pct"] == 1.0
