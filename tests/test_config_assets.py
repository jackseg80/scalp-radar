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
NEW_ASSETS = {"XRP/USDT", "SUI/USDT", "BCH/USDT", "BNB/USDT", "AAVE/USDT", "ARB/USDT", "OP/USDT"}
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
    """Le fichier doit contenir exactement 21 assets."""
    count = len(assets_config["assets"])
    assert count == EXPECTED_COUNT, f"Attendu {EXPECTED_COUNT} assets, trouvé {count}"


def test_no_removed_assets(asset_symbols):
    """Les assets retirés ne doivent plus être présents."""
    still_present = REMOVED_ASSETS & asset_symbols
    assert not still_present, f"Assets retirés encore présents : {still_present}"


def test_new_assets_present(asset_symbols):
    """Les 7 nouveaux assets doivent être présents."""
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
