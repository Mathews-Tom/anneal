"""Tests for anneal.engine.client — pricing configuration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_pricing_default_without_config_uses_hardcoded() -> None:
    """Without a pricing.toml, defaults are used."""
    from anneal.engine.client import get_model_costs

    costs = get_model_costs("gpt-4.1")
    assert costs == (2.0, 8.0)


def test_pricing_unknown_model_returns_default() -> None:
    """Unknown model returns default costs."""
    from anneal.engine.client import _DEFAULT_COSTS, get_model_costs

    costs = get_model_costs("unknown-model-xyz")
    assert costs == _DEFAULT_COSTS


def test_pricing_from_config_file_overrides_defaults(tmp_path: Path) -> None:
    """A pricing.toml file overrides hardcoded defaults."""
    from anneal.engine.client import _load_pricing

    config_content = """
[models.custom-model]
input = 1.5
output = 6.0

[models."gpt-4.1"]
input = 3.0
output = 12.0
"""
    config_file = tmp_path / "pricing.toml"
    config_file.write_text(config_content)

    with patch("anneal.engine.client._PRICING_CONFIG_PATH", config_file):
        costs = _load_pricing()

    assert costs["custom-model"] == (1.5, 6.0)
    assert costs["gpt-4.1"] == (3.0, 12.0)
    # Non-overridden models keep defaults
    assert costs["gemini-2.5-flash"] == (0.15, 0.60)


def test_pricing_config_missing_file_returns_defaults() -> None:
    """When config file doesn't exist, _load_pricing returns defaults."""
    from anneal.engine.client import _load_pricing

    with patch("anneal.engine.client._PRICING_CONFIG_PATH", Path("/nonexistent/pricing.toml")):
        costs = _load_pricing()

    assert "gpt-4.1" in costs
    assert costs["gpt-4.1"] == (2.0, 8.0)


def test_safety_uses_client_pricing() -> None:
    """safety._get_costs delegates to client.get_model_costs."""
    from anneal.engine.client import get_model_costs
    from anneal.engine.safety import _get_costs

    # They should return the same values
    assert _get_costs("gpt-4.1") == get_model_costs("gpt-4.1")
