"""Tests pour le parsing et l'application de --params dans portfolio_backtest."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from backend.core.config import GridATRConfig


# ---------------------------------------------------------------------------
# Helpers : logique extraite du script pour tester indépendamment
# ---------------------------------------------------------------------------


def _parse_params(raw: str) -> dict:
    """Même logique que dans main() de portfolio_backtest."""
    result = {}
    for item in raw.split(","):
        key, val_str = item.strip().split("=", 1)
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
        result[key.strip()] = val
    return result


def _apply_params(strat_cfg, params_override: dict, logger=None) -> list[str]:
    """Même logique que dans main() de portfolio_backtest. Retourne les warnings."""
    warnings = []
    for k, v in params_override.items():
        if hasattr(strat_cfg, k):
            setattr(strat_cfg, k, v)
        else:
            warnings.append(k)
    if hasattr(strat_cfg, "per_asset"):
        for asset_params in strat_cfg.per_asset.values():
            for k, v in params_override.items():
                if k in asset_params or hasattr(strat_cfg, k):
                    asset_params[k] = v
    return warnings


# ---------------------------------------------------------------------------
# test 1 : parsing correct (int, float, string)
# ---------------------------------------------------------------------------


def test_params_parsing_types():
    """Vérifie le cast auto : int, float, string."""
    parsed = _parse_params("max_hold_candles=48,sl_percent=15.0,sides=long")
    assert parsed["max_hold_candles"] == 48
    assert isinstance(parsed["max_hold_candles"], int)
    assert parsed["sl_percent"] == 15.0
    assert isinstance(parsed["sl_percent"], float)
    assert parsed["sides"] == "long"
    assert isinstance(parsed["sides"], str)


def test_params_parsing_single():
    """Parsing d'un seul paramètre."""
    parsed = _parse_params("max_hold_candles=96")
    assert parsed == {"max_hold_candles": 96}


# ---------------------------------------------------------------------------
# test 2 : application sur config de base
# ---------------------------------------------------------------------------


def test_params_applied_to_config():
    """Override max_hold_candles sur GridATRConfig."""
    cfg = GridATRConfig()
    assert cfg.max_hold_candles == 0  # valeur par défaut

    params = _parse_params("max_hold_candles=48")
    warnings = _apply_params(cfg, params)

    assert cfg.max_hold_candles == 48
    assert warnings == []


# ---------------------------------------------------------------------------
# test 3 : per_asset aussi patché
# ---------------------------------------------------------------------------


def test_params_applied_to_per_asset():
    """Override propagé dans les per_asset overrides."""
    cfg = GridATRConfig(
        per_asset={
            "BTC/USDT": {"sl_percent": 12.0},
            "ETH/USDT": {"sl_percent": 10.0, "num_levels": 3},
        }
    )

    params = _parse_params("max_hold_candles=48,sl_percent=15.0")
    _apply_params(cfg, params)

    # Config de base patchée
    assert cfg.max_hold_candles == 48

    # per_asset aussi patchés (clé existante OU champ de strat_cfg)
    assert cfg.per_asset["BTC/USDT"]["max_hold_candles"] == 48
    assert cfg.per_asset["BTC/USDT"]["sl_percent"] == 15.0
    assert cfg.per_asset["ETH/USDT"]["max_hold_candles"] == 48
    assert cfg.per_asset["ETH/USDT"]["sl_percent"] == 15.0


# ---------------------------------------------------------------------------
# test 4 : param inconnu → warning, pas crash
# ---------------------------------------------------------------------------


def test_params_unknown_key_warning():
    """Un param inconnu génère un warning (retourné dans la liste) sans crash."""
    cfg = GridATRConfig()
    params = _parse_params("max_hold_candles=48,param_bidon=999")
    warnings = _apply_params(cfg, params)

    assert cfg.max_hold_candles == 48  # param valide appliqué
    assert "param_bidon" in warnings   # param inconnu retourné
