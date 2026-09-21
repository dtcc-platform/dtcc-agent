"""dtcc-core is a hard dependency and its absence must be loud.

Before this, dtcc-core was undeclared and every consumer caught the ImportError
and degraded: registry.py skipped dataset registration, serializers.py built an
empty dispatch table, dispatcher.py stopped coercing Bounds. A server with no
Core started cleanly and served an empty catalogue, which looks like success.

These tests pin the two halves of the fix: the dependency is declared with a
commit pin, and importing the package without Core raises rather than degrades.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _dependencies() -> list[str]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["dependencies"]


def test_dtcc_core_is_declared():
    """The dependency exists at all — it was absent entirely."""
    assert any(d.startswith("dtcc-core") for d in _dependencies()), (
        "dtcc-core must be a declared dependency; without it a fresh install "
        "serves an empty catalogue with no error"
    )


def test_dtcc_core_is_pinned_to_a_full_sha():
    """A branch or tag pin is not enough: dtcc_core exposes no __version__.

    A full 40-character SHA is the only handle that identifies what a given
    install actually ran against.
    """
    import re

    dep = next(d for d in _dependencies() if d.startswith("dtcc-core"))
    assert re.search(r"\.git@[0-9a-f]{40}$", dep), (
        f"dtcc-core must be pinned to a full 40-char commit SHA, got: {dep}"
    )


def test_import_without_core_raises_with_a_remedy():
    """Importing the package with Core unavailable fails loudly, not silently.

    Run in a subprocess so the blocked import cannot affect this interpreter,
    and assert on the remedy text: an error that does not say what to do sends
    the reader back to catching the ImportError, which is the original bug.
    """
    program = """
import sys
from importlib.abc import MetaPathFinder


class Blocker(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dtcc_core" or fullname.startswith("dtcc_core."):
            raise ImportError("simulated: dtcc_core not installed")
        return None


sys.meta_path.insert(0, Blocker())
assert "dtcc_core" not in sys.modules, "core already imported; probe invalid"

try:
    import dtcc_agent
except ImportError as exc:
    print("RAISED")
    print(str(exc))
else:
    print("NO_RAISE")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "RAISED" in result.stdout, (
        "importing dtcc_agent without dtcc-core must raise, but it succeeded. "
        f"stdout={result.stdout!r} stderr={result.stderr[-2000:]!r}"
    )
    assert "uv sync --locked" in result.stdout, (
        "the error must name a remedy, otherwise the reader's next move is to "
        f"catch it. Got: {result.stdout!r}"
    )


@pytest.mark.parametrize(
    "module",
    ["dtcc_agent.registry", "dtcc_agent.serializers", "dtcc_agent.dispatcher"],
)
def test_core_backed_modules_import(module):
    """The modules that used to degrade silently now import for real."""
    __import__(module)
