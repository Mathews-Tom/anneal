"""Eval criteria templates for common optimization domains.

Templates are TOML files in this package directory. Stochastic templates
are directly usable as eval_criteria.toml files. Deterministic templates
document the run-cmd and parse-cmd values to pass to ``anneal register``.

Usage::

    from anneal.templates import list_templates, load_template

    for name in list_templates():
        print(name)

    config = load_template("prompt-quality")
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent


def list_templates() -> list[str]:
    """Return names of available eval criteria templates.

    Names are the TOML file stems, e.g. ``"prompt-quality"``.
    """
    return sorted(p.stem for p in _TEMPLATES_DIR.glob("*.toml"))


def load_template(name: str) -> dict[str, object]:
    """Load a template by name and return the parsed TOML dict.

    Parameters
    ----------
    name:
        Template stem without the ``.toml`` extension, e.g.
        ``"prompt-quality"``.

    Raises
    ------
    FileNotFoundError
        If no template with the given name exists.
    """
    path = _TEMPLATES_DIR / f"{name}.toml"
    if not path.exists():
        available = ", ".join(list_templates())
        raise FileNotFoundError(
            f"Template '{name}' not found. Available: {available}"
        )
    return tomllib.loads(path.read_text(encoding="utf-8"))
