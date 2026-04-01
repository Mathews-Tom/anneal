"""Typed evaluator protocol for custom plugin-based evaluation.

Custom evaluators let you replace shell pipeline eval with a Python callable
that has full access to the Python ecosystem.

Usage — writing a plugin::

    # eval_plugin.py
    from pathlib import Path
    from anneal.eval_protocol import EvalResult

    def evaluate(artifact_path: str) -> EvalResult:
        content = Path(artifact_path).read_text()
        score = run_my_custom_benchmark(content)
        return EvalResult(score=score, metadata={"detail": "..."})

Usage — loading a plugin at runtime::

    from anneal.eval_protocol import load_evaluator

    evaluator = load_evaluator("eval_plugin.py:evaluate")
    result = evaluator.evaluate("/path/to/artifact.py")

The module:callable spec supports both file paths and dotted module names::

    load_evaluator("eval_plugin.py:evaluate")        # file path
    load_evaluator("mypackage.evaluators:evaluate")  # importable module
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result returned by a custom evaluator.

    Attributes
    ----------
    score:
        Numeric score for the artifact. The direction (higher/lower is better)
        is determined by the target registration, not by this value.
    metadata:
        Optional key-value pairs with diagnostic information. Values must be
        strings, floats, or ints — no nested structures.
    details:
        Optional free-form string with human-readable evaluation notes.
        Surfaced in the anneal dashboard and experiment logs.
    """

    score: float
    metadata: dict[str, str | float | int] | None = field(default=None)
    details: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for custom evaluator objects.

    Any object with an ``evaluate(artifact_path: str) -> EvalResult`` method
    satisfies this protocol. This includes plain functions wrapped via
    :func:`load_evaluator`.

    Example implementation::

        class MyEvaluator:
            def evaluate(self, artifact_path: str) -> EvalResult:
                content = Path(artifact_path).read_text()
                score = len(content.splitlines())
                return EvalResult(score=float(score))
    """

    def evaluate(self, artifact_path: str) -> EvalResult:
        """Evaluate an artifact and return a scored result.

        Parameters
        ----------
        artifact_path:
            Absolute path to the artifact file to evaluate.

        Returns
        -------
        EvalResult
            Score and optional metadata for this evaluation.
        """
        ...


# ---------------------------------------------------------------------------
# _FunctionEvaluator adapter
# ---------------------------------------------------------------------------


class _FunctionEvaluator:
    """Wraps a bare callable into an Evaluator-compatible object."""

    def __init__(self, fn: object) -> None:
        if not callable(fn):
            raise TypeError(f"Expected a callable, got {type(fn).__name__}")
        self._fn = fn

    def evaluate(self, artifact_path: str) -> EvalResult:
        result = self._fn(artifact_path)  # type: ignore[call-arg]
        if not isinstance(result, EvalResult):
            raise TypeError(
                f"Evaluator function must return EvalResult, got {type(result).__name__}"
            )
        return result


# ---------------------------------------------------------------------------
# load_evaluator
# ---------------------------------------------------------------------------


def load_evaluator(module_path: str) -> Evaluator:
    """Load a Python callable from a module:callable spec and return an Evaluator.

    The spec format is ``<module>:<callable>`` where ``<module>`` is either:

    - A file path ending in ``.py`` (e.g., ``eval_plugin.py:evaluate``)
    - A dotted importable module name (e.g., ``mypackage.evals:evaluate``)

    The callable must accept a single positional argument (``artifact_path:
    str``) and return an :class:`EvalResult`.

    Parameters
    ----------
    module_path:
        Module-colon-callable spec, e.g. ``"eval_plugin.py:evaluate"``.

    Returns
    -------
    Evaluator
        An object satisfying the :class:`Evaluator` protocol.

    Raises
    ------
    ValueError
        If the spec is malformed (missing ``:``, empty module, or empty
        callable name).
    FileNotFoundError
        If the spec refers to a ``.py`` file path that does not exist.
    ImportError
        If the module cannot be imported.
    AttributeError
        If the callable name does not exist in the module.
    TypeError
        If the resolved attribute is not callable.
    """
    if ":" not in module_path:
        raise ValueError(
            f"Invalid evaluator spec '{module_path}': expected 'module:callable' format"
        )

    module_spec, _, callable_name = module_path.partition(":")
    module_spec = module_spec.strip()
    callable_name = callable_name.strip()

    if not module_spec:
        raise ValueError(
            f"Invalid evaluator spec '{module_path}': module part is empty"
        )
    if not callable_name:
        raise ValueError(
            f"Invalid evaluator spec '{module_path}': callable name is empty"
        )

    if module_spec.endswith(".py"):
        module = _load_module_from_file(module_spec)
    else:
        module = importlib.import_module(module_spec)

    fn = getattr(module, callable_name)
    return _FunctionEvaluator(fn)


def _load_module_from_file(file_path: str) -> object:
    """Load a Python module from a file path.

    Adds the file's parent directory to ``sys.path`` so that relative imports
    within the plugin file resolve correctly.

    Parameters
    ----------
    file_path:
        Path to the ``.py`` file, absolute or relative to the current
        working directory.

    Returns
    -------
    module
        The loaded module object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If the file cannot be loaded as a module.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Evaluator file not found: {file_path}")

    module_name = path.stem
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module
