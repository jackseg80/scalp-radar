"""
Tests de cohérence pour config/assets.yaml et config/strategies.yaml.
Vérifie que les assets retirés sont absents et que les nouveaux sont présents.
"""

import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
ASSETS_YAML = ROOT / "config" / "assets.yaml"
STRATEGIES_YAML = ROOT / "config" / "strategies.yaml"

REMOVED_ASSETS = {"ENJ/USDT", "SUSHI/USDT", "IMX/USDT", "SAND/USDT", "AR/USDT", "APE/USDT", "XTZ/USDT", "JUP/USDT"}
NEW_ASSETS = {"XRP/USDT", "BCH/USDT", "BNB/USDT", "AAVE/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT"}
EXPECTED_COUNT = 21


@pytest.fixture(scope="module")
def assets_config():
    with open(ASSETS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def strategies_config():
    with open(STRATEGIES_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def asset_symbols(assets_config):
    return {a["symbol"] for a in assets_config["assets"]}


def test_assets_count(assets_config):
    """Le fichier doit contenir exactement 20 assets."""
    count = len(assets_config["assets"])
    assert count == EXPECTED_COUNT, f"Attendu {EXPECTED_COUNT} assets, trouvé {count}"


def test_no_removed_assets(asset_symbols):
    """Les assets retirés ne doivent plus être présents."""
    still_present = REMOVED_ASSETS & asset_symbols
    assert not still_present, f"Assets retirés encore présents : {still_present}"


def test_new_assets_present(asset_symbols):
    """Les 6 nouveaux assets doivent être présents."""
    missing = NEW_ASSETS - asset_symbols
    assert not missing, f"Nouveaux assets absents : {missing}"


def test_grid_atr_no_enj(strategies_config):
    """ENJ/USDT ne doit pas être dans le per_asset de grid_atr."""
    per_asset = strategies_config.get("grid_atr", {}).get("per_asset", {}) or {}
    assert "ENJ/USDT" not in per_asset, "ENJ/USDT encore présent dans grid_atr.per_asset"


def test_grid_boltrend_no_sand(strategies_config):
    """SAND/USDT ne doit pas être dans le per_asset de grid_boltrend."""
    per_asset = strategies_config.get("grid_boltrend", {}).get("per_asset", {}) or {}
    assert "SAND/USDT" not in per_asset, "SAND/USDT encore présent dans grid_boltrend.per_asset"


def test_all_per_asset_symbols_in_assets(strategies_config, asset_symbols):
    """
    Pour chaque stratégie avec per_asset, tous les symbols référencés
    doivent exister dans assets.yaml. Détecte les références orphelines.
    """
    orphans = []
    for strat_name, strat_cfg in strategies_config.items():
        if not isinstance(strat_cfg, dict):
            continue
        per_asset = strat_cfg.get("per_asset") or {}
        for sym in per_asset:
            if sym not in asset_symbols:
                orphans.append(f"{strat_name}.per_asset.{sym}")
    assert not orphans, f"Références orphelines dans strategies.yaml :\n" + "\n".join(f"  - {o}" for o in orphans)


# ─── Timeframes ──────────────────────────────────────────────────────────

# Assets avec timeframes 5m/15m (scalp actif) — BTC/ETH/SOL/DOGE/LINK/XRP + ARB/SUI
TOP_SCALP_ASSETS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "LINK/USDT", "XRP/USDT", "ARB/USDT", "SUI/USDT"}
TOP6_ASSETS = TOP_SCALP_ASSETS  # alias rétrocompat


def test_timeframes_all_have_1h_4h_1d(assets_config):
    """Tous les 21 assets ont au moins [1h, 4h, 1d] dans leurs timeframes."""
    required = {"1h", "4h", "1d"}
    for asset in assets_config["assets"]:
        tfs = set(asset["timeframes"])
        missing = required - tfs
        assert not missing, f"{asset['symbol']} manque timeframes: {missing}"


def test_timeframes_top6_have_5m_15m(assets_config):
    """Les assets scalp (BTC/ETH/SOL/DOGE/LINK/XRP/ARB/SUI) ont aussi 5m et 15m."""
    for asset in assets_config["assets"]:
        if asset["symbol"] in TOP6_ASSETS:
            tfs = set(asset["timeframes"])
            assert "5m" in tfs, f"{asset['symbol']} manque 5m"
            assert "15m" in tfs, f"{asset['symbol']} manque 15m"


def test_timeframes_no_1m(assets_config):
    """Aucun asset n'a 1m dans ses timeframes."""
    for asset in assets_config["assets"]:
        assert "1m" not in asset["timeframes"], f"{asset['symbol']} a encore 1m!"


def test_timeframes_others_no_5m_15m(assets_config):
    """Les assets hors scalp n'ont PAS 5m ni 15m."""
    for asset in assets_config["assets"]:
        if asset["symbol"] not in TOP6_ASSETS:
            tfs = set(asset["timeframes"])
            assert "5m" not in tfs, f"{asset['symbol']} a 5m (ne devrait pas)"
            assert "15m" not in tfs, f"{asset['symbol']} a 15m (ne devrait pas)"
