"""Tests du chargeur de configuration."""

import pytest
import yaml

from backend.core.config import AppConfig


class TestConfigLoading:
    def test_load_valid_config(self, config_dir):
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert len(config.assets) == 2
        assert config.assets[0].symbol == "BTC/USDT"
        assert config.exchange.name == "Bitget"

    def test_assets_have_1h_timeframe(self, config_dir):
        config = AppConfig(config_dir=config_dir, env_file=None)
        for asset in config.assets:
            assert "1h" in asset.timeframes

    def test_correlation_groups_loaded(self, config_dir):
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert "crypto_major" in config.correlation_groups
        group = config.correlation_groups["crypto_major"]
        assert group.max_concurrent_same_direction == 2
        assert group.max_exposure_percent == 60

    def test_strategies_loaded(self, config_dir):
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert config.strategies.vwap_rsi.enabled is True
        assert config.strategies.liquidation.enabled is False

    def test_custom_strategies_loaded(self, config_dir):
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert "swing_baseline" in config.strategies.custom_strategies
        assert config.strategies.custom_strategies["swing_baseline"].timeframe == "1h"

    def test_risk_loaded(self, config_dir):
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert config.risk.kill_switch.max_session_loss_percent == 5.0
        assert config.risk.position.default_leverage == 15
        assert config.risk.fees.taker_percent == 0.06

    def test_exchange_rate_limits(self, config_dir):
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert config.exchange.rate_limits.market_data.requests_per_second == 20
        assert config.exchange.rate_limits.trade.requests_per_second == 10

    def test_sl_tp_config(self, config_dir):
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert config.risk.sl_tp.mode == "server_side"
        assert config.risk.sl_tp.sl_type == "market"
        assert "taker_fee" in config.risk.sl_tp.sl_real_cost_includes


class TestConfigValidation:
    def test_invalid_leverage(self, config_dir):
        """default_leverage > max_leverage doit échouer."""
        risk_path = config_dir / "risk.yaml"
        data = yaml.safe_load(risk_path.read_text())
        data["position"]["default_leverage"] = 50
        data["position"]["max_leverage"] = 30
        risk_path.write_text(yaml.dump(data))

        with pytest.raises(ValueError, match="default_leverage"):
            AppConfig(config_dir=config_dir, env_file=None)

    def test_missing_correlation_group(self, config_dir):
        """Référencer un groupe inexistant doit échouer."""
        assets_path = config_dir / "assets.yaml"
        data = yaml.safe_load(assets_path.read_text())
        data["assets"][0]["correlation_group"] = "nonexistent"
        assets_path.write_text(yaml.dump(data))

        with pytest.raises(ValueError, match="nonexistent"):
            AppConfig(config_dir=config_dir, env_file=None)

    def test_missing_config_file(self, tmp_path):
        """Un répertoire vide doit échouer."""
        with pytest.raises(Exception):
            AppConfig(config_dir=tmp_path, env_file=None)

    def test_selector_bypass_env_override_true(self, config_dir, monkeypatch, tmp_path):
        """SELECTOR_BYPASS_AT_BOOT=true dans .env override risk.yaml (false)."""
        env_file = tmp_path / ".env"
        env_file.write_text("SELECTOR_BYPASS_AT_BOOT=true\n")
        config = AppConfig(config_dir=config_dir, env_file=str(env_file))
        assert config.risk.selector_bypass_at_boot is True

    def test_selector_bypass_env_not_set(self, config_dir):
        """Sans env var, risk.yaml value est utilisée (false)."""
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert config.risk.selector_bypass_at_boot is False

    def test_force_strategies_env_override_single(self, config_dir, tmp_path):
        """FORCE_STRATEGIES=grid_atr dans .env override risk.yaml ([])."""
        env_file = tmp_path / ".env"
        env_file.write_text("FORCE_STRATEGIES=grid_atr\n")
        config = AppConfig(config_dir=config_dir, env_file=str(env_file))
        assert config.risk.adaptive_selector.force_strategies == ["grid_atr"]

    def test_force_strategies_env_override_multiple(self, config_dir, tmp_path):
        """FORCE_STRATEGIES avec plusieurs stratégies (comma-separated)."""
        env_file = tmp_path / ".env"
        env_file.write_text("FORCE_STRATEGIES=grid_atr,grid_trend,vwap_rsi\n")
        config = AppConfig(config_dir=config_dir, env_file=str(env_file))
        assert config.risk.adaptive_selector.force_strategies == [
            "grid_atr",
            "grid_trend",
            "vwap_rsi",
        ]

    def test_force_strategies_env_whitespace_handling(self, config_dir, tmp_path):
        """FORCE_STRATEGIES gère les espaces autour des virgules."""
        env_file = tmp_path / ".env"
        env_file.write_text("FORCE_STRATEGIES= grid_atr , grid_trend , \n")
        config = AppConfig(config_dir=config_dir, env_file=str(env_file))
        assert config.risk.adaptive_selector.force_strategies == ["grid_atr", "grid_trend"]

    def test_force_strategies_env_not_set(self, config_dir):
        """Sans env var, risk.yaml value est utilisée ([])."""
        config = AppConfig(config_dir=config_dir, env_file=None)
        assert config.risk.adaptive_selector.force_strategies == []

    def test_force_strategies_env_empty_string(self, config_dir, tmp_path):
        """FORCE_STRATEGIES vide → liste vide."""
        env_file = tmp_path / ".env"
        env_file.write_text("FORCE_STRATEGIES=\n")
        config = AppConfig(config_dir=config_dir, env_file=str(env_file))
        assert config.risk.adaptive_selector.force_strategies == []
